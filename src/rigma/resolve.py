from __future__ import annotations

import math
import re

from pydantic import ValidationError

from .models import (CACHE_BYTES, ENGINE_DEFAULT_UBATCH, LAUNCH_PARALLEL,
                     MIN_LAUNCH_CTX, CachePolicy, ComboFlags, GgufFile,
                     HardwareProfile, ModelSpec, RunPlan)
from .registry import Registry

VRAM_RESERVE_MB = {"windows": 1200, "linux": 400, "darwin": 0}
RAM_RESERVE_MB = 2048
# llama.cpp's own scratch allocation, on top of weights and KV. Measured
# 2026-08-21 on a dense 27B at ubatch 512, across five fully-resident rungs:
# 37, 43, 44, 55 and 81 MiB. The original 900 was a guess made when the Windows
# desktop reserve was also a guess and the two were covering for each other;
# with the desktop now measured, this can be an estimate of the thing it names.
#
# Dropped 400 -> 150 once draft_cache_mb landed: that figure was obtained by
# DIFFERENCING measured VRAM, so it already contains the draft head's compute
# buffers, and reserving 400 on top double-counted them. The double-count cost
# real residency — a 32K MTP plan that measured 14,298 MiB was refused against
# a budget that had 360 MiB of imaginary scratch in it.
#
# DR2-3: this is a FLOOR for the measured point, NOT "~2x the largest
# observation at 512". The only `sched_reserve ... compute buffer size` line
# this repository has for a real load at ubatch 512 is 410.28 MiB
# (`.scratch/prism-v.log:4669`, second load, `n_ubatch = 512` at :4499; the
# fitting pass at :2337 reads 400.28), so 150 was BELOW the engine's own number,
# not above it. The 37-81 rungs are an older differencing with no log in this
# tree. At or below the measured point the constant still stands untouched;
# above it, MEASURED_COMPUTE_BUFFER_MB governs.
COMPUTE_BUFFER_MB = 150

# The engine's OWN compute-buffer allocation at the engine's default physical
# batch, from this repository's real load log: `.scratch/prism-v.log:4669`
# `sched_reserve: ROCm0 compute buffer size = 410.28 MiB`, for the load whose
# `n_ubatch = 512` is logged at :4499. That segment carries no draft/speculation
# lines, so it is the non-draft compute buffer; `draft_cache_mb` charges the
# draft head separately, and scaling this base therefore does not re-introduce
# the double count the 150 was cut for. Used ONLY above
# ENGINE_DEFAULT_UBATCH — at or below it COMPUTE_BUFFER_MB stands unchanged, so
# every existing plan stays byte-identical.
MEASURED_COMPUTE_BUFFER_MB = 410.28


def compute_buffer_mb(ubatch: int = 0) -> float:
    """llama.cpp's scratch allocation for a physical batch of `ubatch`, in MB.

    At or below the engine's default 512 the answer is COMPUTE_BUFFER_MB
    exactly, unchanged: there is no measurement under 512, and shrinking the
    reserve on an unmeasured extrapolation would make the fit optimistic in the
    unsafe direction (the launch OOM this whole module exists to prevent).

    Above 512 the base is the engine's own line for this machine,
    MEASURED_COMPUTE_BUFFER_MB = 410.28 MiB at ubatch 512. llama.cpp sizes its
    compute buffer by the PHYSICAL batch — the KQ/KQV and matmul scratch scale
    with `n_ubatch`, not the logical `n_batch` (independent corroboration of the
    direction, not the constant: "the compute buffer ... scales with -ub and is
    essentially independent of -b" — multigrid.ai/learn/llamacpp-batch-ubatch)
    — so a requested ubatch is charged proportionally.

    DR2-3: the previous base of 150 charged 600 MiB at ubatch 2048 where this
    same machine's own data point implies ~1641 MiB, i.e. it UNDER-reserved by
    ~1 GB on the single lever whose stated purpose is that the fit still rules.
    The C10 verifier called that direction "safe"; the engine's own log says
    otherwise, and the engine's own log wins: an under-reserve is exactly the
    silent WDDM paging the fit exists to prevent.

    PREDICTION, not a measurement: the scaling above 512 is linear from that one
    measured point, and the compute buffer also depends on ctx and graph shape.
    There is a deliberate STEP at 512 (150 -> 411 MiB) — the safe direction, and
    the acceptance rule is "nothing changes at or below 512".

    0 = no opinion = the engine default, so a plan with no override is
    byte-identical to before this lever existed.
    """
    if ubatch <= ENGINE_DEFAULT_UBATCH:
        return float(COMPUTE_BUFFER_MB)
    return MEASURED_COMPUTE_BUFFER_MB * ubatch / ENGINE_DEFAULT_UBATCH


def launch_ubatch(spec: ModelSpec) -> int:
    """The `-ub` this model will actually launch with (0 = engine default).

    The launch default IS the request — there is no other route into `-ub` —
    so the fit reads it from the spec rather than being told separately by each
    caller, which is what keeps `resolve`, `fit_for_launch` and the Models-page
    explorer pricing the same launch."""
    launch = getattr(spec, "launch", None)
    if launch is None:
        return 0
    return max(0, int(getattr(launch, "ubatch", 0) or 0))


def launch_ngl(spec: ModelSpec) -> int:
    """The requested `-ngl` cap, or -1 when the model has no opinion."""
    launch = getattr(spec, "launch", None)
    if launch is None:
        return -1
    return int(getattr(launch, "ngl", -1))


# CACHE_BYTES lives in models.py: the fit math here and ComboFlags' K/V
# normalisation must read ONE table, and the validators that reject an unknown
# cache type live next to it. Re-exported here because this module is where the
# cache arithmetic lives and callers import it from `rigma.resolve`.
CTX_DEFAULT = {"coding": 32768}
# The PLANNING floor: the smallest context the planner will plan for a model
# whose own window is at least this big. It is NOT `models.MIN_LAUNCH_CTX`
# (2048), which is the smallest context a LAUNCH will ask the engine for — the
# two are different quantities, so they are named apart (this was `CTX_FLOOR`,
# which the UI also spells `CTX_FLOOR` for the 2048 launch floor; W5F5B-N3).
# `_ctx_floor` clamps it down to the model's own window when that is smaller.
PLAN_CTX_FLOOR = 8192


class ResolveError(RuntimeError):
    pass


def _engine_now(plan: RunPlan) -> str:
    """The llama.cpp build this plan will launch on, or "" if unknowable.

    Asks for the PLAN's binary, not the backend's pin: a model whose tensor types
    need a registered engine (the PrismML fork for PQ2_0) runs that engine, and its
    calibration is stamped with that build — comparing it to the pin's build would
    call every such calibration stale, and miss a change to the fork.
    """
    try:
        from .server_ops import plan_engine_identity
        return plan_engine_identity(plan.gguf, plan.backend)
    except Exception:
        return ""


def _desktop_vram_mb(profile: HardwareProfile | None) -> float | None:
    """What everything OTHER than the model we are about to load is holding.

    The profile's reading is preferred over a fresh counter read for two
    reasons: it is the same number `_budgets` planned against, so the plan and
    the staleness test describe one machine; and `server_ops._free_current`
    has already credited the OUTGOING engine back into it, where the raw
    adapter counter would count our own 12GB model as desktop pressure and
    call every calibration stale mid-switch.
    """
    # AUDIT F17: docs/audit-2026-09-04-full.md — apply the same sanity clamp
    # `_budgets` uses. A reading at or above the card's own capacity is a broken
    # counter, not a busy desktop (Windows once reported 359,777MB on a 16GB
    # card). Unclamped, one bogus read is an enormous apparent drift that
    # silently retires every calibration on the machine, with the reason buried
    # in plan.explain. Returning None means "unknown", which skips the check.
    def _sane(mb: float | None) -> float | None:
        if mb is None:
            return None
        total = sum(g.vram_mb for g in (profile.gpus or [])) if profile else 0
        return None if total and mb >= total else mb

    if profile is not None and profile.vram_used_mb is not None:
        return _sane(profile.vram_used_mb)
    try:
        from .probe import gpu_used_mb
        return _sane(gpu_used_mb())
    except Exception:
        return None


# C10-nits N2: the placement keys a calibration row may move. Reported by
# `_apply_calibration` so an explicit stored value it overrode is visible in
# `explain` rather than silently contradicted by an earlier fit note.
_CALIBRATION_PLACEMENT_KEYS = ("ngl", "n_cpu_moe", "cache_type_k",
                               "cache_type_v", "spec_type", "batch", "ubatch")


def _without_quality_env_levers(flags: dict) -> tuple[dict, list[str]]:
    """`flags` with any quality-degrading engine env lever removed, and their names.

    C11-read: C11 closed the WRITE path — `bench.crowned_row` no longer crowns a
    trial that turns off a quality-preserving engine default, so `run_sweep`
    cannot persist one. But a `calibration.json` written BEFORE that fix, or
    hand-edited, is still merged by `_apply_calibration` and reaches the child's
    environment on every later launch. A sweep scores tokens/sec only, so the
    read path must refuse the same lever the write path refuses to crown.

    `bench._QUALITY_ENV_LEVERS` is the ONE list of such levers (the C11 verifier
    found `LLAMA_ATTN_ROT_DISABLE` applied end to end), and this reuses it
    rather than inventing a second one. The match is case-insensitive, exactly
    as `bench._quality_env_levers_in` is, because the engine's `getenv` ignores
    case on Windows — a lowercase hand-edited key must not slip through. The
    rest of the row is kept: only the quality lever is dropped, so a
    corrupt/hostile key cannot disable the gate by riding in with a legitimate
    one.
    """
    from .bench import _QUALITY_ENV_LEVERS, _quality_env_levers_in
    if not isinstance(flags, dict):
        # A hand-edited row can be a list/string. Leave it alone so the merge's
        # existing TypeError handler keeps reporting "flags is not an object"
        # rather than this helper raising AttributeError first.
        return flags, []
    env = flags.get("env")
    if not isinstance(env, dict):
        return flags, []
    dropped = _quality_env_levers_in(env)
    if not dropped:
        return flags, []
    levers = {name.upper() for name in _QUALITY_ENV_LEVERS}
    cleaned = {k: v for k, v in flags.items() if k != "env"}
    rest = {k: v for k, v in env.items() if str(k).upper() not in levers}
    if rest:
        cleaned["env"] = rest
    return cleaned, dropped


def _apply_calibration(plan: RunPlan,
                       profile: HardwareProfile | None = None) -> RunPlan:
    from .bench import calibration_entry, calibration_stale, current_identity, load_calibration
    key, entry = calibration_entry(load_calibration(), plan.model_slug,
                                   plan.gguf.quant, plan.backend)
    if entry and entry.get("flags"):
        # AUDIT F17: docs/audit-2026-09-04-full.md
        # The key is model:quant:backend and nothing else, so a stored
        # placement was replayed onto a machine in a different state. The
        # reachable flags are the ones that decide whether the weights fit at
        # all — n_cpu_moe, cache_type_k/v (q8_0 crowned over q4_0 nearly
        # doubles the KV cache) and spec_type (~565MB of draft cache at 32K).
        # Measured 2026-08-21: 2,828MB of desktop drift took the same combo
        # from 37.59 to 9.95 tok/s. The entry already records the engine, the
        # ctx and the VRAM held when it was measured, and calibration_stale
        # already reads all three — this simply was not asking, while
        # server_ops._measured_placement gated the very same keys on an exact
        # ctx match. Two policies for one set of flags; this is the strict one.
        reason = calibration_stale(entry, _desktop_vram_mb(profile),
                                   _engine_now(plan), plan.flags.ctx,
                                   current_identity(plan.backend))
        if reason:
            plan.explain.append(f"calibration override skipped: {reason}")
            return plan
        # C11-read: re-gate the quality-degrading env levers at MERGE time too.
        # The write path (bench.crowned_row) cannot crown one, but a row
        # written before that fix — or hand-edited — is still applied here.
        stored, dropped = _without_quality_env_levers(entry["flags"])
        # AUDIT F06-4: model_copy(update=...) does not validate, so a stored
        # entry replayed the flash_attn enum, the spec_type whitelist, the
        # cache-type table and _symmetric_kv straight onto the plan — `-fa
        # sideways` and `--cache-type-k q3_k_bogus` reached llama.cpp. The
        # calibration file is user-editable and is treated as untrusted
        # everywhere else (server_ops: a corrupt file "is not an error"), so
        # validate the merge and keep the fresh plan when it fails.
        # All-or-nothing on purpose: a half-applied placement is not a
        # placement anyone measured.
        before = plan.flags
        try:
            merged = ComboFlags.model_validate(
                {**plan.flags.model_dump(), **stored})
        except ValidationError as exc:
            plan.explain.append(
                f"calibration override ignored: {exc.error_count()} invalid "
                f"field(s)")
            return plan
        except TypeError:
            plan.explain.append("calibration override ignored: flags is not an "
                                "object")
            return plan
        plan.flags = merged
        plan.origin += "+calibrated"
        plan.explain.append(f"calibration override applied: {stored} "
                            f"(measured {entry.get('date', '?')})")
        # C10-nits N2: a measured placement silently beat an explicit stored
        # `ngl` (and could leave the fit's own note, "launch default ngl 40 (the
        # fit allows 59)", standing while the plan carried the measured 63).
        # Name every placement key the measurement moved, so the override is
        # visible and the earlier fit note is plainly superseded. The
        # measurement WINS rather than being clamped: it came from a stopwatch
        # on this machine (AUDIT F17), and the merge above is all-or-nothing.
        moved = [f"{k} {getattr(before, k)} -> {getattr(merged, k)}"
                 for k in _CALIBRATION_PLACEMENT_KEYS
                 if getattr(before, k) != getattr(merged, k)]
        if moved:
            plan.explain.append(
                "calibration override changed " + "; ".join(moved)
                + " (a measured placement wins over the fit and any launch "
                  "default; an earlier note about those values is superseded)")
        if dropped:
            plan.explain.append(
                "calibration override dropped quality-degrading env lever(s) "
                + ", ".join(dropped) + ": a sweep scores speed only, and the "
                "engine turns the quality-preserving default ON for a reason "
                "(see bench._QUALITY_ENV_LEVERS)")
    return plan


# AUDIT F21: docs/audit-2026-09-04-full.md
def _ctx_floor(spec: ModelSpec) -> int:
    """The smallest context worth trying for THIS model.

    PLAN_CTX_FLOOR was a global 8192, so a model trained on 4096 (a Llama-2
    derivative) or 2048 (Phi-2, TinyLlama, or any gguf whose header omits
    context_length) never entered the fit loop at all: `_calculate` starts at
    min(CTX_DEFAULT, native_ctx) and the `while ctx >= PLAN_CTX_FLOOR` body never
    ran, so fit_gguf was never called and the plan fell through to the absolute
    floor — CPU, ngl=0 — on a card the model fits in four times over, while
    quant_verdicts (which probes 8192/4096/2048) told the Models page the same
    model runs on the GPU. A model cannot be asked for more context than it
    has, so its own window is the floor.
    """
    return (min(PLAN_CTX_FLOOR, spec.native_ctx) if spec.native_ctx > 0
            else PLAN_CTX_FLOOR)


# AUDIT F06-5
def _next_ctx_rung(ctx: int, floor: int) -> int:
    """The next smaller context to try, never stepping PAST the floor.

    `ctx //= 2` skipped the floor rung whenever native_ctx was not a power-of-two
    multiple of two above it (12288, 10000, 24576, ...): the sequence jumped from
    the native rung straight below `_ctx_floor`, which is the one rung that
    function exists to guarantee. The Models page probes 8192/4096/2048 and
    reported the same model fitting at 8192 while `rigma up` never asked.

    Returns a value below the floor — ending the caller's `while ctx >= floor`
    loop — when `ctx` is already at or below the floor, so the loop terminates.
    """
    nxt = ctx // 2
    if nxt < floor and ctx > floor:
        nxt = floor
    return nxt


def kv_bytes_per_token(spec: ModelSpec, k: str, v: str) -> float:
    per_side = spec.full_attn_layers * spec.kv_heads * spec.head_dim
    return per_side * CACHE_BYTES[k] + per_side * CACHE_BYTES[v]


# Sliding-window models (Gemma 2/3/4, …) run TWO KV caches: the global layers
# keep one that grows with ctx, and the windowed layers keep a second one
# bounded by the window. gguf_meta identifies the windowed layers in order to
# keep them OUT of the growing cache — correct, and it left their second cache
# budgeted at exactly zero. For a 27B-class SWA model (52 windowed layers, 16
# kv heads, 128 wide, 1K window) that is 416 MiB of f16 the planner does not
# know exists: a fixed term, proportionally worst at small contexts, and the
# only approximation in this file that errs toward overcommitting the card
# rather than refusing a plan. It replaced a previous 6x OVERestimate with a
# zero, so the direction of the error flipped without anyone choosing that.
def swa_kv_bytes(spec: ModelSpec, k: str, v: str, ctx: int) -> float:
    """Bytes of the second, window-sized KV cache. Zero unless the spec carries
    a sliding window.

    Read with getattr because the three fields (`swa_layers`, `swa_kv_heads`,
    `swa_window`) are what gguf_meta now reports in spec_fields and belong on
    ModelSpec next to full_attention_interval; a spec without them is charged
    nothing, which is exactly the behaviour before this term existed. Positive
    evidence only: a hybrid's SSM or linear-attention layers (Qwen3.5/3.8,
    DeltaNet) hold a fixed state rather than a windowed cache, and they carry
    no window size, so they are never charged for one.
    """
    n = int(getattr(spec, "swa_layers", 0) or 0)
    heads = int(getattr(spec, "swa_kv_heads", 0) or 0)
    window = int(getattr(spec, "swa_window", 0) or 0)
    if n <= 0 or heads <= 0 or window <= 0 or spec.head_dim <= 0:
        return 0.0
    per_side = n * heads * spec.head_dim
    # llama.cpp sizes this cache by the window, not by ctx; a context shorter
    # than the window cannot fill it.
    return min(ctx, window) * (per_side * CACHE_BYTES[k]
                               + per_side * CACHE_BYTES[v])


def recurrent_state_mb(spec: ModelSpec) -> float:
    """Per-SEQUENCE MiB llama.cpp allocates for a hybrid's recurrent state.

    A Mamba/DeltaNet layer holds a fixed-size state instead of a growing KV
    cache, and llama.cpp gives it its own buffer — `llama_memory_recurrent`,
    sized `n_embd_r() + n_embd_s()` per recurrent layer per sequence, both f32
    (llama-memory-recurrent.cpp builds one `r` and one `s` tensor of
    `mem_size * (1 + n_rs_seq)` rows; llama-hparams.cpp sizes the rows):

        n_embd_s() = ssm_d_state * ssm_d_inner
        n_embd_r() = (ssm_d_conv - 1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)

    MEASURED, not predicted: a real load of Ternary-Bonsai-2-27B (qwen35, 64
    layers, 48 recurrent) at n_seq_max=1 logged `RS buffer size = 149.62 MiB`
    and `R (f32): 5.62 MiB, S (f32): 144.00 MiB` (.scratch/prism-v.log). The 48
    layers are what `recurrent_layers` counts and the four ssm_* numbers are the
    file's own header keys. Zero for a dense model, and zero — with
    `recurrent_state_unknown` true — for a hybrid whose header omitted them.
    """
    n = spec.recurrent_layers
    if n <= 0:
        return 0.0
    state, inner = spec.ssm_state_size, spec.ssm_inner_size
    if state <= 0 or inner <= 0:
        return 0.0
    s_bytes = state * inner * 4
    r_bytes = max(spec.ssm_conv_kernel - 1, 0) * (
        inner + 2 * spec.ssm_group_count * state) * 4
    return n * (s_bytes + r_bytes) / 2**20


def recurrent_state_unknown(spec: ModelSpec) -> bool:
    """True when the model gives evidence of recurrent layers but there is no
    geometry to size them with, so `recurrent_state_mb` returns 0 for lack of
    evidence rather than because there is nothing to allocate. The fit says so in
    its explain line instead of silently reading the model as dense.

    Two ways to know that, and both are the same statement:
      * `recurrent_layers > 0` with no usable `ssm.*` geometry (A2);
      * the probe could not derive the count or the buffer at all — a pure-Mamba
        header, an explicit `attention.recurrent_layers` array, or a partial
        `ssm.*` set (A2d). Before the flag existed those shapes charged 0 with no
        note, which is a confident wrong number.
    """
    if getattr(spec, "rs_geometry_unknown", False):
        return True
    return spec.recurrent_layers > 0 and recurrent_state_mb(spec) == 0.0


def kv_geometry_unknown(spec: ModelSpec) -> bool:
    """True when the model gives evidence of a KV cache whose WIDTH or LAYER
    SPLIT could not be derived, so `kv_bytes_per_token` (and `swa_kv_bytes`)
    return a number that is a guess — often exactly 0 — rather than a measured
    charge. The fit says `kv=unknown` in its explain line instead of presenting
    that number as confident.

    Set by `gguf_meta` while the header is parsed (`kv_geometry_unknown`). The
    derived clause is the same statement for a spec written by an older probe,
    which stored no flag: attention layers are declared but the growing cache's
    width computes to zero. `head_dim > 0` is the attention EVIDENCE that
    separates that shape from a pure-Mamba spec — the probe reads a Mamba header
    as `full_attn_layers == n_layers` (there is no attention pattern, so every
    layer falls through to full attention) with `head_dim == 0`, and that spec
    genuinely has no KV cache; its missing count is what `rs_unknown` already
    reports. Without the head_dim guard this clause flagged every Mamba file
    `kv=unknown` (A2d-kv verifier FAIL, 2026-09-30).
    """
    if getattr(spec, "kv_geometry_unknown", False):
        return True
    return (spec.full_attn_layers > 0 and spec.head_dim > 0
            and spec.kv_heads <= 0)


# Speculative decoding's draft head needs its own KV cache and compute buffers.
# MEASURED on an RX 9070 XT, 2026-08-21, five repeats per point, by differencing
# dedicated VRAM against the same model without the head:
#
#     ctx 16K  ->  465 MiB        ctx 32K  ->  565 MiB
#
# which is 365 MiB fixed plus 6.25 KiB/token. Derived from geometry it "should"
# be ~1.1 KiB/token for one draft block; it is not, because the head carries
# four nextn blocks and its own buffers. The measurement wins.
#
# Planning without this term produced configurations that fit on paper and paged
# in practice: the live server came up at ngl=62 and ran 30.63 tok/s where the
# identical settings benched at 49.46.
DRAFT_FIXED_MB = 365.0
DRAFT_KIB_PER_TOKEN = 6.25


def draft_cache_mb(spec: ModelSpec, ctx: int,
                   spec_type: str, n_max: int,
                   draft_bytes: int = 0) -> float:
    """VRAM the speculative draft head needs on top of weights and KV.

    Zero when speculation is off — the head's WEIGHTS are in the file either
    way, but its caches are only allocated when it is asked to draft.

    `draft_bytes` is the size of a SEPARATE draft gguf (DFlash, DFlash2, DSpark,
    EAGLE-3). That distinction is the whole reason the parameter exists: for
    `draft-mtp` the head is inside the target file, so the fixed term below is
    the measured truth. For an external draft the file's own weights are loaded
    BESIDE the target and are the dominant term — and they were budgeted nowhere,
    which is the same class of error as the projector that was charged after the
    fit.

    AUDIT F06-7: this took a `kv` argument and never used it. DELETED rather
    than scaled, because the two constants below are a MEASURED TOTAL obtained
    by differencing, not a KV term: 6.25 KiB/token is ~5.7x the geometry-derived
    KV size for one draft block, so most of it is the head's own four nextn
    blocks and compute buffers, which do not scale with cache precision.
    Scaling the whole term by CACHE_BYTES[kv]/CACHE_BYTES["f16"] would
    under-reserve for a quantised cache — the unsafe direction — and would
    contradict the one measured point this repo has (465 MiB at 16K). The
    parameter implied a precision dependence the function does not have; the
    honest fix is to stop claiming it.
    """
    if not spec_type or spec_type == "none":
        return 0.0
    # Deeper drafts need a slot per drafted token. Measured only at n=1; n=2
    # more than doubled the total, so scaling by depth is the conservative
    # reading rather than an established fit.
    depth = max(1, n_max)
    return (DRAFT_FIXED_MB + DRAFT_KIB_PER_TOKEN * ctx / 1024 * depth
            + max(0, int(draft_bytes)) / 2 ** 20)


def draft_bytes_for(flags) -> int:
    """On-disk size of the draft artefact a launch will load, or 0 for none.

    An unreadable or not-yet-downloaded path is 0: the fit must not invent a
    charge for a file Rigma cannot measure, and `spec_decode.head_mismatch`
    is what refuses that launch on other grounds.
    """
    value = getattr(flags, "spec_draft", "") or ""
    if not value:
        return 0
    try:
        from .hangar import draft_file_bytes
        return draft_file_bytes(value)
    except Exception:
        return 0


def with_launch_overheads(spec: ModelSpec, *, vision: bool, ctx: int,
                          spec_type: str = "", n_max: int = 0,
                          draft_bytes: int = 0) -> ModelSpec:
    """A copy of `spec` whose mmproj slot holds what will ACTUALLY be resident.

    `fit_gguf` treats mmproj as memory that sits on the GPU and cannot be
    offloaded. Two other things behave identically and were both getting the
    wrong treatment:

      * a projector that will NOT be loaded (vision off) was still reserved —
        600 MiB of a 16GB card, which cost two layers of GPU residency and took
        a live server from 49.46 tok/s on the bench to 30.63 in practice;
      * the speculative draft cache was not reserved at all, so plans that fit
        on paper paged the moment speculation was switched on.

    Folding both into that one slot fixes the arithmetic without threading a new
    parameter through every fit function.
    """
    mm_bytes = spec.mmproj.bytes if (vision and spec.mmproj) else 0
    draft = draft_cache_mb(spec, ctx, spec_type, n_max, draft_bytes)
    total = mm_bytes + int(draft * 2**20)
    if total == 0:
        return spec.model_copy(update={"mmproj": None})
    slot = (spec.mmproj.model_copy(update={"bytes": total}) if spec.mmproj
            else GgufFile(repo="local", file="__overhead__", bytes=total,
                          quant="overhead"))
    return spec.model_copy(update={"mmproj": slot})


def _budgets(profile: HardwareProfile,
             other_vram_mb: float | None = None,
             ubatch: int = 0) -> tuple[float, float]:
    """Usable VRAM and RAM for a plan.

    `other_vram_mb` is what the desktop is measured to be holding. Without it
    the reserve is a constant, and on Windows that constant was a fiction:
    1200MB budgeted against 3,955MB actually held by a browser and an editor.
    Windows does not refuse the resulting overcommit — it pages the difference
    to system RAM, so the plan reports 0% offload while 29% of the weights ride
    PCIe on every token (measured 2026-08-21, decode at 21% of card bandwidth).

    The constant stays a FLOOR. A measurement may only make the budget
    smaller, never larger: it is a snapshot, the user can open something a
    second later, and being optimistic here is what caused the bug.

    `ubatch` is the model's requested physical batch (0 = engine default). It
    only moves the COMPUTE term — `compute_buffer_mb` — because that is the one
    part of the reserve llama.cpp sizes from it.
    """
    # llama.cpp splits tensors across all GPUs, so budget the SUM of their
    # VRAM (reserving per-card overhead), not just the primary
    gpus = profile.gpus or []
    total_vram = sum(g.vram_mb for g in gpus)
    floor = VRAM_RESERVE_MB[profile.os] * max(1, len(gpus))
    # A reading at or above the card's own capacity is a broken counter, not a
    # busy desktop (Windows' per-process counter once reported 359,777MB on a
    # 16GB card). Discard it rather than budget zero.
    measured = (profile.vram_used_mb if other_vram_mb is None
                else other_vram_mb) or 0.0
    if measured >= total_vram:
        measured = 0.0
    reserve = max(floor, measured) + compute_buffer_mb(ubatch)
    vram = total_vram - reserve
    return max(vram, 0), max(profile.ram_free_mb - RAM_RESERVE_MB, 0)


def fit_gguf(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
             ctx: int, explain: list[str], backend: str = "",
             ubatch: int | None = None) -> ComboFlags | None:
    # The physical batch the launch will use, unless the caller is pricing a
    # different one. None = read the model's own launch default (the only route
    # into -ub), so `resolve`, `fit_for_launch` and `quant_verdicts` all price
    # the same launch; an explicit value exists for callers comparing ubatches.
    if ubatch is None:
        ubatch = launch_ubatch(spec)
    usable_vram, usable_ram = _budgets(profile, ubatch=ubatch)
    # Two passes, and the order matters: try EVERY cache type fully on the GPU
    # before letting any of them spill weights to RAM. Quantising the cache
    # costs ~8.5 effective bits; pushing layers (or experts) to system RAM costs
    # real tokens/sec. Doing this in one pass picked f16-with-offload over
    # q8_0-fully-resident, which is strictly the worse trade.
    for strict in (True, False):
        best = None
        for k, v in _cache_candidates(spec, backend):
            got = _fit_with_cache(spec, gguf, profile, ctx, k, v,
                                  usable_vram, usable_ram, explain, strict)
            if got is None:
                continue
            if strict:
                return got      # fully resident: the most precise cache wins
            # Offloading is unavoidable at this ctx, so now the cache's job is
            # to MINIMISE it. Taking the first that merely fits picked f16 and
            # spilled 22% of a dense model's weights, where q8_0 spills 8% —
            # ~0.06% perplexity against a PCIe round trip on every token, for
            # every offloaded layer. The pass above already guaranteed nothing
            # here can be fully resident, so precision is no longer free.
            if best is None or _spilled(spec, got) < _spilled(spec, best):
                best = got
        if best is not None:
            if best.cache_type_k != spec.cache_type_policy.k:
                explain.append(
                    f"cache {best.cache_type_k} over "
                    f"{spec.cache_type_policy.k}: keeps "
                    f"{_spilled(spec, best):.0%} of the weights off system RAM")
            # A PINNED policy (the Models-page explorer's "what if the cache
            # were q4_0") is allowed to spill — that is the question it was
            # asked. Say so when a rung the pin removed would have fit fully,
            # because the alternative is a silent slowdown. The measured size of
            # that slowdown is model-specific and lives in the findings doc, not
            # in this string: this line reports the CPU layer count and lets the
            # number speak for the plan in front of the user.
            if spec.cache_type_policy.pinned and spec.moe is None \
                    and _spilled(spec, best) > 0:
                rung = _resident_rung(spec, gguf, profile, ctx, usable_vram,
                                      usable_ram, backend)
                if rung is not None:
                    cpu = _cpu_layers(spec, best)
                    explain.append(
                        f"pinned {spec.cache_type_policy.k} at ctx {ctx} puts "
                        f"{cpu} of {spec.n_layers} layers on the CPU; "
                        f"{rung.cache_type_k} fits fully on the GPU")
            return best
    return None


def _cpu_layers(spec: ModelSpec, flags: ComboFlags) -> int:
    """Transformer layers llama.cpp leaves on the CPU under these flags.

    `ngl` counts the OUTPUT layer as one of the offloaded layers
    (llama-model.cpp: `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`,
    and the output layer is assigned through the same list), so `-ngl 58` on a
    64-layer model pins layers 0-6 — seven — not six. `_spilled` reports the
    weight fraction and counts the same seven, so the two agree.
    """
    n = spec.n_layers or 0
    if n <= 0:
        return 0
    ngl = min(max(flags.ngl, 0), n + 1)
    return max(0, (n + 1) - ngl)


def _resident_rung(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
                   ctx: int, usable_vram: float,
                   usable_ram: float, backend: str = "") -> ComboFlags | None:
    """The most precise UNPINNED cache that is fully resident at `ctx`, or None.

    Only used to explain a pinned spill, so it takes the strict pass directly
    and never offloads."""
    unpinned = spec.model_copy(update={
        "cache_type_policy": spec.cache_type_policy.model_copy(
            update={"pinned": False})})
    for k, v in _cache_candidates(unpinned, backend):
        got = _fit_with_cache(unpinned, gguf, profile, ctx, k, v, usable_vram,
                              usable_ram, [], strict=True)
        if got is not None:
            return got
    return None


def _spilled(spec: ModelSpec, flags: ComboFlags) -> float:
    """Fraction of the model's weights left in system RAM under this plan.

    Dense counts the layers the engine actually keeps on the CPU — the same
    number `_cpu_layers` reports. `ngl` counts the OUTPUT layer, so `-ngl 58` on
    64 layers is 7/64, not 6/64; an explain line and an offload percentage that
    disagreed by a layer would both be distrusted. MoE counts only the expert
    share of an offloaded layer, since sparse activation makes that far cheaper,
    and `n_cpu_moe` has no output-layer off-by-one to mirror.
    """
    n = spec.n_layers or 0
    if not n:
        return 0.0
    if spec.moe is None:
        return min(1.0, ((n + 1) - min(max(flags.ngl, 0), n + 1)) / n)
    return (min(flags.n_cpu_moe, n) / n) * spec.moe.expert_weight_fraction


# Fused flash-attention KV cache types, per backend, from the engines' source.
#
# The ladder below steps q8_0 -> q5_1 -> q4_0, but a rung is only a WIN if the
# backend can run attention over that cache on the GPU. Where it cannot,
# llama.cpp does not fail: with `-fa on` (Rigma's default) the unsupported node
# is assigned to the CPU backend and attention runs there SILENTLY (the
# scheduler falls back; the CPU backend's supports_op returns true), so a plan
# that reads "fully on the GPU" can still be paying CPU attention on every
# token.
#
#   vulkan   ggml-vulkan.cpp at 87268f77: two INDEPENDENT `fa_kv_ok` calls
#            (18321-18339) accept {F32,F16,BF16,Q8_0,Q5_1,Q5_0,Q4_1,Q4_0}; only a
#            BF16/non-BF16 mix is rejected. Mainline b9867's Vulkan build has
#            the same list (verified) and is the other engine installed here, so
#            the full ladder is safe.
#   cuda/hip ggml-cuda/fattn.cu at 87268f77: `ggml_cuda_fattn_kv_type_supported`
#            returns false for Q4_1/Q5_0/Q5_1 (338-356), and without
#            GGML_CUDA_FA_ALL_QUANTS it returns NONE for K->type != V->type
#            (442-446). That macro defaults OFF (ggml/CMakeLists.txt) and the
#            shipped PrismML HIP binary does not define it (checked in the
#            binary). Rigma never builds an engine — it downloads a prebuilt
#            release (runtime.ENGINE_URL_ALLOWLIST) or uses a registered one —
#            so it cannot turn the macro on. Only f16/bf16/q8_0/q4_0 fuse.
#   metal    ggml-metal-device.m at 87268f77: `ggml_metal_device_supports_op`
#            accepts {F32,F16,Q8_0,Q4_0,Q4_1,Q5_0,Q5_1, BF16 iff has_bfloat}
#            (1618-1635) and REQUIRES K.type == V.type (1636-1638). Wider than a
#            default CUDA build, and Rigma is always symmetric, so the full
#            ladder is safe there.
#   cpu      no separate FA backend — attention is CPU work either way.
#   ""       unknown/not threaded: keep the historical ladder, so a caller that
#            does not know its backend cannot make a plan WORSE than before.
_FA_KV_FULL = ("f16", "bf16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0")
_FA_KV_CUDA = ("f16", "bf16", "q8_0", "q4_0")
_FA_KV_BY_BACKEND = {"rocm": _FA_KV_CUDA, "hip": _FA_KV_CUDA, "cuda": _FA_KV_CUDA}


def _fa_kv_types(backend: str) -> tuple[str, ...]:
    """KV cache types `backend`'s fused flash-attention kernel accepts."""
    return _FA_KV_BY_BACKEND.get((backend or "").lower(), _FA_KV_FULL)


def fa_kv_supported(backend: str, kv: str) -> bool:
    """True when `kv` has a fused flash-attention kernel on `backend`."""
    return (kv or "").lower() in _fa_kv_types(backend)


def step_down_notice(stepped: str, backend: str, used: str, ctx: int) -> str:
    """One sentence for a cache type the fit could not honour as asked.

    Two different reasons, because the fix the user reaches for is different:
    a type that does not FIT is a memory question, a type the backend cannot
    FUSE is a backend question."""
    if not fa_kv_supported(backend, stepped):
        return (f"{stepped} has no fused flash-attention on {backend}; using "
                f"{used} with every layer on the GPU")
    return (f"{stepped} does not fit fully at ctx {ctx:,}; using {used} with "
            f"every layer on the GPU")


def _cache_candidates(spec: ModelSpec, backend: str = ""):
    """Cache types to try, best quality first, on THIS backend.

    The policy default (f16) is tried first, then q8_0. q8_0 stores 32 values as
    int8 plus one f16 scale — 1.0625 bytes/element vs 2.0, so it HALVES the KV
    cache for ~8.5 effective bits. That is far more precision than the weights
    themselves carry (Q6_K ~6.5 bits, IQ3_M ~3.5), so it is not the accuracy
    bottleneck — but dropping context to 8K to protect it very much is a real
    cost. Trying it before giving up context is close to free.

    A rung the backend cannot FUSE is not a rung: it moves attention to the CPU
    without saying so (see the table above). Those are dropped, INCLUDING a
    requested type — the fit then lands on the most precise rung that actually
    runs on the GPU, and the caller reports the step-down (`step_down_notice`).
    """
    k, v = spec.cache_type_policy.k, spec.cache_type_policy.v
    allowed = _fa_kv_types(backend)
    ladder = [(a, b) for a, b in (("q8_0", "q8_0"), ("q5_1", "q5_1"),
                                  ("q4_0", "q4_0")) if a in allowed]
    # The requested pair can be answered as asked only if this backend can fuse
    # it. When it cannot, the request is not answerable at all — pinning the
    # SUBSTITUTE would ask "what if q8_0" of a user who asked for q5_1, and on a
    # tight context would then report a spill the ladder exists to avoid. So an
    # unsupported request falls through to the ladder, which steps to the most
    # precise rung that actually fits.
    requested_ok = k in allowed and v in allowed
    if spec.cache_type_policy.pinned and requested_ok:
        return [(k, v)]          # explorer: answer the question that was asked
    out = [(k, v)] if requested_ok else []
    # Down to q5_1 and q4_0 before giving up and spilling weights. The ladder
    # used to stop at q8_0, so a 27B at 64K "did not fit" and nine of its
    # sixty-four layers went to the CPU — while q5_1 fit on the GPU with room
    # to spare and ran at 38.27 tok/s. Measured on the same machine the same
    # day, offloading half a gigabyte cost 60% of throughput (32.47 -> 15.86
    # tok/s) and 80% of prefill. One more step of cache quantisation costs a
    # fraction of a percent of perplexity. The trade is not close.
    for step in ladder:
        if step not in out:
            out.append(step)
    return out


def _fit_with_cache(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
                    ctx: int, k: str, v: str, usable_vram: float,
                    usable_ram: float, explain: list[str],
                    strict: bool = False) -> ComboFlags | None:
    file_mb = gguf.bytes / 2**20
    # vision projector loads alongside the weights and can't be offloaded;
    # it counts against VRAM but not the MoE expert math below
    mm_mb = spec.mmproj.bytes / 2**20 if spec.mmproj else 0.0
    # AUDIT F19: docs/audit-2026-09-04-full.md — the windowed layers' own cache
    # is resident too, and until now was budgeted as zero.
    swa_mb = swa_kv_bytes(spec, k, v, ctx) / 2**20
    kv_mb = ctx * kv_bytes_per_token(spec, k, v) / 2**20 + swa_mb
    # A hybrid's recurrent state is allocated PER SEQUENCE, and the launch runs
    # LAUNCH_PARALLEL sequences. --kv-unified keeps the KV pool at ctx, so only
    # this term multiplies. Zero for every dense model.
    #
    # A2d-gap: `recurrent_state_unknown` is tested FIRST, not as the fallback
    # for a zero. An unrecognised geometry can still yield a nonzero
    # interval-derived count AND a size — the upstream header that carries BOTH
    # an explicit `attention.recurrent_layers` array and a complete `ssm.*` set
    # — and that estimate must not read as a confident number: the interval is
    # only right for a uniform layout, which is exactly what is not known here.
    # The estimate is still CHARGED. Dropping it would free VRAM llama.cpp is
    # about to allocate, which is the launch OOM this fit exists to prevent; the
    # label says what the number is instead.
    rs_mb = recurrent_state_mb(spec) * LAUNCH_PARALLEL
    if recurrent_state_unknown(spec):
        rs_txt = (f"rs=unknown(est {rs_mb:.0f}MB) " if rs_mb
                  else "rs=unknown ")
    elif rs_mb:
        rs_txt = f"rs={rs_mb:.0f}MB "
    else:
        rs_txt = ""
    # A2d-kv: the KV charge gets the same provenance label the RS charge has.
    # `kv_geometry_unknown` means the cache's width or layer split could not be
    # derived, so the number is a guess — and for a missing
    # `attention.head_count_kv` it is exactly 0. The charge is NOT dropped:
    # freeing VRAM llama.cpp is about to allocate is the launch OOM this fit
    # exists to prevent. The label says what the number is.
    if kv_geometry_unknown(spec):
        kv_txt = (f"kv=unknown(est {kv_mb:.0f}MB)" if kv_mb else "kv=unknown")
    else:
        kv_txt = f"kv={kv_mb:.0f}MB"
    explain.append(f"{gguf.quant}@ctx{ctx} kv={k}: file={file_mb:.0f}MB "
                   + kv_txt + " "
                   + (f"(incl. {swa_mb:.0f}MB windowed) " if swa_mb else "")
                   + (f"mmproj={mm_mb:.0f}MB " if mm_mb else "")
                   + rs_txt
                   + f"vs vram={usable_vram:.0f}MB ram={usable_ram:.0f}MB")
    if spec.moe is None:
        if file_mb + mm_mb + kv_mb + rs_mb <= usable_vram:
            return ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=v)
        if strict:
            return None            # try the next cache type before offloading
        # partial offload: keep as many layers on the GPU as fit, spill the
        # rest to system RAM. A 24B dense model on 16GB runs at ~15 t/s this
        # way instead of returning "doesn't fit" and dropping to CPU/a smaller
        # model. mmproj + kv must stay on the GPU, so they eat the VRAM budget.
        if spec.n_layers <= 0:
            return None
        per_layer = file_mb / spec.n_layers
        gpu_room = usable_vram - mm_mb - kv_mb - rs_mb
        n_gpu = int(gpu_room // per_layer) if per_layer else 0
        if n_gpu <= 0:
            return None                      # not even one layer + kv fits
        n_gpu = min(n_gpu, spec.n_layers)
        spilled = (spec.n_layers - n_gpu) * per_layer
        if spilled > usable_ram:
            return None
        explain.append(f"dense partial offload: {n_gpu}/{spec.n_layers} layers "
                       f"on GPU ({spilled:.0f}MB to RAM)")
        return ComboFlags(ctx=ctx, ngl=n_gpu, cache_type_k=k, cache_type_v=v)
    if strict and file_mb + mm_mb + kv_mb + rs_mb > usable_vram:
        return None                # ditto for MoE expert offload
    # AUDIT F06-1: a MoE header that omits block_count reports n_layers = 0, and
    # this divide ran before the `if need_off` test, so even a fully-resident
    # model raised ZeroDivisionError out of resolve(). The dense branch above
    # guards the same case; without a layer count there is nothing to place.
    if spec.n_layers <= 0:
        return None
    expert_mb = file_mb * spec.moe.expert_weight_fraction
    per_layer = expert_mb / spec.n_layers
    need_off = max(0.0, file_mb + mm_mb + kv_mb + rs_mb - usable_vram)
    n_off = math.ceil(need_off / per_layer) if need_off else 0
    if n_off <= spec.n_layers and n_off * per_layer <= usable_ram:
        return ComboFlags(ctx=ctx, n_cpu_moe=n_off, cache_type_k=k, cache_type_v=v)
    return None


def _apply_launch_ngl(spec: ModelSpec, flags: ComboFlags, allowed: int,
                      ctx: int, cache_type: str,
                      explain: list[str]) -> ComboFlags:
    """Fold a requested `-ngl` into a fitted plan: CLAMP DOWN, with a note.

    The request is a CAP, never a pin. Putting MORE layers on the GPU than the
    fit allows is the exact overcommit the fit exists to prevent (Windows/WDDM
    accepts the allocation and pages it, so the launch "succeeds" and runs at
    PCIe speed with nothing printed), so it is clamped to what fits. Asking for
    FEWER layers is honoured as-is: that direction only frees VRAM.

    Refusing instead was considered and rejected. The value is stored per model
    and outlives the machine state it was chosen in; a stored `ngl: 63` that
    stops fitting because a browser opened would then make the model
    UNLAUNCHABLE, where every other memory lever here degrades (the cache ladder
    steps down, `_grow_ctx` stops growing, the dense fit spills layers). Clamping
    is the same policy, and `allowed` is the fit's own number, so the plan and
    the argv cannot disagree about the placement.

    `allowed` is the fit's raw ngl (99 = "all"). The note is emitted only when
    there IS a request, so a plan with no override stays byte-identical.
    """
    want = launch_ngl(spec)
    if want < 0 or flags is None:
        return flags
    used = min(want, allowed)
    if used < want:
        explain.append(
            f"launch default ngl {want} exceeds what fits at ctx {ctx:,} "
            f"({cache_type}): using {used}")
    else:
        explain.append(f"launch default ngl {want} (the fit allows {allowed})")
    return flags.model_copy(update={"ngl": used})


def _with_launch_defaults(plan: RunPlan, spec: ModelSpec | None,
                          profile: HardwareProfile) -> RunPlan:
    """Fold a model's stored launch defaults into a resolved plan.

    `batch`/`ubatch` are copied straight on: they are a request, not a
    placement, and nothing in the fit can contradict them (the pair was checked
    launchable when it was stored, and `ComboFlags` re-checks it). `ngl` is a
    CAP and is priced against the fit at THIS plan's ctx and cache, so the plan
    says what it actually used rather than what was asked for.

    The raw fit is obtained by neutralising only the request (`ngl` -1) on a
    copy of the spec: `fit_for_launch` would itself clamp, and clamping twice
    would report the clamped number as "what the fit allows". The copy keeps
    `ubatch`, so the compute buffer charged here is the one that will launch.
    """
    if spec is None:
        return plan
    launch = getattr(spec, "launch", None)
    if launch is None:
        return plan
    upd = {}
    if launch.batch > 0:
        upd["batch"] = launch.batch
    if launch.ubatch > 0:
        upd["ubatch"] = launch.ubatch
    if upd:
        plan.flags = plan.flags.model_copy(update=upd)
    want = launch_ngl(spec)
    if want < 0:
        return plan
    bare = spec.model_copy(update={
        "launch": launch.model_copy(update={"ngl": -1})})
    # Price the vision setting that will actually launch: a model pinned
    # text-only frees the projector, which is worth GPU layers, and pricing the
    # projector anyway would report a smaller cap than the launch will use.
    allowed, _ = fit_for_launch(
        bare, plan.gguf, profile, plan.flags.ctx,
        kv=plan.flags.cache_type_k, vision=(True if launch.vision is None
                                            else launch.vision),
        spec_type=plan.flags.spec_type,
        n_max=plan.flags.spec_n_max, backend=plan.backend, explain=[],
        draft_bytes=draft_bytes_for(plan.flags))
    if allowed is None:
        plan.explain.append(
            f"launch default ngl {want} cannot be placed at ctx "
            f"{plan.flags.ctx:,}; keeping the resolver's ngl {plan.flags.ngl}")
        return plan
    plan.flags = _apply_launch_ngl(spec, plan.flags, allowed.ngl,
                                   plan.flags.ctx, allowed.cache_type_k,
                                   plan.explain)
    return plan


def fit_for_launch(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
                   ctx: int, *, kv: str = "", vision: bool = True,
                   spec_type: str = "", n_max: int = 0, backend: str = "",
                   draft_bytes: int = 0,
                   explain: list[str] | None = None
                   ) -> tuple[ComboFlags | None, str]:
    """The fit a LAUNCH will actually run: `(flags, stepped_down_from)`.

    A requested cache type — an explicit `kv`, or the model's stored launch
    default — is a CEILING, not a hard pin. The two passes in `fit_gguf` take
    the requested type whenever it fits FULLY on the GPU and step DOWN the
    ladder (q8_0 -> q5_1 -> q4_0) rather than spilling weights when it does
    not. That is the whole point of the ladder, and the launch paths used to
    defeat it by applying the type AFTER the fit: on the owner's 16GB card,
    ctx 262144 with a stored `kv: q8_0` is 15,576MB against a 14,954MB budget,
    so the resolver fitted q5_1 (fully resident), the post-fit force put q8_0
    back, and `-ngl 99` paged 622MB to system RAM with no error printed
    (Windows/WDDM). Fitting it in removes the window in which the plan and the
    argv can disagree about the cache.

    `backend` also removes rungs the backend cannot FUSE (ROCm/CUDA: q5_1 is
    not a flash-attention type without GGML_CUDA_FA_ALL_QUANTS, and with
    `-fa on` the unsupported node runs on the CPU silently). A requested type
    that is unsupported steps to the most precise supported rung, and the
    caller reports it with `step_down_notice`.

    A stored spec is never PINNED — `CachePolicy.pinned` is the Models-page
    explorer's knob ("answer the question that was asked") and a real launch
    keeps the ladder so a too-large cache degrades instead of failing. A stored
    spec that carries the flag anyway is honoured as "no opinion" here.

    A requested `-ngl` (launch default) is applied LAST, as a CAP: clamped down
    to what the fit allows, never above it. The launch paths re-fit here at a
    requested ctx, so without this the fit's own ngl would silently overwrite
    the request — and the request would be what the caller believes is running.

    Returns "" for the second value when the requested type was used as-is, or
    the requested type when the fit stepped down from it.
    """
    expl = explain if explain is not None else []
    fit = with_launch_overheads(spec, vision=vision, ctx=ctx,
                                spec_type=spec_type, n_max=n_max,
                                draft_bytes=draft_bytes)
    if kv:
        fit = fit.model_copy(update={"cache_type_policy":
                                     CachePolicy(k=kv, v=kv)})
    elif fit.cache_type_policy.pinned:
        fit = fit.model_copy(update={
            "cache_type_policy": fit.cache_type_policy.model_copy(
                update={"pinned": False})})
    flags = fit_gguf(fit, gguf, profile, ctx, expl, backend=backend)
    if flags is not None:
        flags = _apply_launch_ngl(spec, flags, flags.ngl, ctx,
                                  flags.cache_type_k, expl)
    stepped = kv if (kv and flags is not None
                     and flags.cache_type_k != kv) else ""
    return flags, stepped


def _backend(profile: HardwareProfile, override: str | None = None) -> str:
    """Which compute backend to launch on.

    The card's table row lists what it CAN run, best-first, and taking [0] was
    the whole selection policy — so an RX 9070 XT, whose row has read
    ["vulkan", "rocm"] since it was added, could never run ROCm. ROCm 7 brought
    rocWMMA flash-attention to RDNA4 and the answer for dense models is now an
    open question, which cannot be measured without being selectable first
    (owner request, 2026-08-21).

    An override the card does not list is an ERROR, never a quiet fallback:
    launching Vulkan while the UI says ROCm would file the resulting benchmark
    under the wrong backend, which is worse than not running it.
    """
    gpu = profile.primary_gpu
    if not gpu or not gpu.backends:
        return "cpu"
    if not override:
        return gpu.backends[0]
    if override not in gpu.backends:
        raise ResolveError(
            f"{gpu.name} cannot run {override} on {profile.os} — it supports "
            f"{', '.join(gpu.backends)}")
    return override


def _grow_ctx(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
              flags: ComboFlags, explain: list[str],
              layer_budget: float = 0.0, backend: str = "") -> ComboFlags:
    """Calculator plans only: double ctx while it still fits, up to native.

    CTX_DEFAULT is a starting probe, not a ceiling (owner finding 2026-07-16:
    the old cap silently wasted VRAM that could hold 4-8x more context)."""
    best = start = flags
    ctx = best.ctx * 2
    while ctx <= spec.native_ctx:
        grown = fit_gguf(spec, gguf, profile, ctx, explain, backend=backend)
        if grown is None:
            break
        # dense: fewer GPU layers is a real per-token cost, so stop before the
        # larger window forces weights off the GPU. MoE: expert offload is
        # cheap (sparse activation) and users want the context — grow to native
        # as long as it still fits (the 35B is verified healthy at 262K ctx
        # with expert offload: 56.7 t/s). Guarding MoE here capped qwen3.6 at
        # 32K when it runs fine at 256K (owner, 2026-07-18).
        # layer_budget > 0 buys the doubling anyway, up to that share of the
        # layers. The default 0.0 is the historical rule: never trade a layer.
        # It was set when a KV cache cost 4x what a hybrid-attention model's
        # actually does, and it silently reports the smaller window — UD-Q3_K_XL
        # missed 16K by 46MB and simply displayed 8K (owner, 2026-07-30).
        #
        # Measured against `start`, not the previous step: a per-doubling budget
        # is spent again at every rung, so 8K->64K quietly offloaded 26 layers
        # on a 15% allowance. And ngl is a SENTINEL (99 = "all"), so it has to
        # be clamped to n_layers before differencing or "all -> 62 of 65" reads
        # as 37 layers lost instead of 3.
        cap = spec.n_layers or 99
        lost = max(0, min(start.ngl, cap) - min(grown.ngl, cap))
        allowed = int(cap * layer_budget)
        if (lost > allowed) or (spec.moe is None
                                and grown.n_cpu_moe > best.n_cpu_moe):
            explain.append(f"grow-to-fit: stop at ctx {best.ctx} "
                           f"(ctx {ctx} would push weights off the GPU)")
            break
        if lost:
            # Price it. "15% of the layers" reads as a small number; on the
            # owner's 27B PQ2_0 hybrid a CPU-resident layer measured ~9 ms/token
            # against ~0.3 ms on the GPU (R3-ENG-13, findings-r3/37), so 7 of 64
            # layers cost 4.3x. The constants are that measurement, not a
            # universal law — the line says so.
            explain.append(
                f"grow-to-fit: spending {lost} of {cap} GPU layers to reach "
                f"ctx {ctx} (budget {allowed}); measured on a 27B PQ2_0 hybrid, "
                f"a CPU-resident layer costs ~9 ms/token vs ~0.3 ms on the GPU")
        explain.append(f"grow-to-fit: ctx {best.ctx} -> {ctx} "
                       f"(n_cpu_moe {best.n_cpu_moe} -> {grown.n_cpu_moe})")
        best = grown
        ctx *= 2
    return best


def quant_verdicts(spec: ModelSpec, profile: HardwareProfile, *,
                   kv: str = "", vision: bool = True,
                   grow: str = "speed", backend: str = "") -> list[dict]:
    """Per-quant verdicts, with the three things that were fixed constants.

    Every number this returned was computed under one hidden configuration —
    the spec's own cache policy, the vision projector always resident, and a
    growth rule that never trades a GPU layer for context. Each of those is a
    real choice with a real cost, and presenting one of them as the answer made
    the page read as a wall (owner, 2026-07-30).

      kv      "" keeps the spec's policy ladder; or force f16/q8_0/q5_1/q4_0,
              or "k,v" for an asymmetric pair such as "q8_0,q4_0".
      vision  False drops the mmproj. It is permanently resident and counted
              whether or not an image is ever sent — 888MB on Qwen3.8-27B,
              which is 4x the context on a 16GB card.
      grow    "speed" stops growing context at the first GPU layer it would
              cost (the long-standing default). "context" spends up to
              _GROW_LAYER_BUDGET of the layers for each doubling.
    ONE implementation, deliberately: this ran only inside hf_browse's
    pre-download browser, so the Models page — the page you actually pick a
    quant from — showed a bare size and nothing else. Two copies of this
    arithmetic would drift, and a fit verdict that disagrees with itself
    between two screens is worse than none.
    """
    spec = _configured(spec, kv=kv, vision=vision)
    # The explorer answers for a BACKEND, because the cache types that can be
    # fused differ per backend and the page is one click from a launch. Default
    # to the one a launch would pick.
    be = backend or _backend(profile)
    # Price the SAME physical batch the fit below will price, or the row's
    # "headroom / OVER by" arithmetic would describe a launch this model is not
    # going to run (the A2b/A2d failure mode, one term over).
    usable_vram, _ = _budgets(profile, ubatch=launch_ubatch(spec))
    mm_mb = spec.mmproj.bytes / 2**20 if spec.mmproj else 0.0
    out = []
    # AUDIT F21: docs/audit-2026-09-04-full.md — the probe ladder is capped by
    # the model's own window. It used to report a 2048-token model as fitting
    # "at 8192", a window it was never trained on, while the resolver refused
    # the same model outright: the two screens disagreed about one model
    # because only one of them had a floor. The bottom rung IS the launch floor
    # (`models.MIN_LAUNCH_CTX`), so the page cannot price a context below the
    # one a launch would ask the engine for (W5F5B-N3).
    ladder = ([c for c in (8192, 4096, MIN_LAUNCH_CTX) if c <= spec.native_ctx]
              or [spec.native_ctx or MIN_LAUNCH_CTX])
    for g in spec.ggufs:
        flags = None
        for ctx in ladder:
            flags = fit_gguf(spec, g, profile, ctx, [], backend=be)
            if flags:
                flags = _grow_ctx(spec, g, profile, flags, [],
                                  layer_budget=(_GROW_LAYER_BUDGET
                                                if grow == "context" else 0.0),
                                  backend=be)
                break
        if flags is None:
            out.append({"ok": False, "speed": "no", "offload_pct": 100,
                        "budget": _budget_rows(spec, g, mm_mb, ladder[0],
                                               usable_vram)})
            continue
        # The spill fraction comes from the PLAN the resolver actually made —
        # ngl for dense, n_cpu_moe for MoE — not from comparing the file to
        # VRAM. The old file-size guess called a quant "gpu" while the very
        # same verdict carried ngl=56 of 65 layers: it ignored the KV cache,
        # which is precisely what a big context window spends VRAM on.
        # ONE implementation, `_spilled`, deliberately: a duplicate here read
        # 6/64 where `_cpu_layers` said 7, so the page and the fit could flip
        # opposite sides of the 0.15 "light" boundary for the same plan.
        spill = _spilled(spec, flags)
        speed = "gpu" if spill <= 0.001 else ("light" if spill <= 0.15
                                              else "offload")
        # The explorer pins the requested type on purpose, so a spill here is
        # the answer — but if a rung the pin removed would have been fully
        # resident, the page must say so rather than showing "offload" with no
        # reason. Same fact as the explain line in fit_gguf, on the surface the
        # owner actually reads.
        note = ""
        # A requested type this backend cannot fuse is answered with the rung it
        # stepped to, and said out loud — otherwise the page shows a q8_0
        # verdict under a q5_1 heading.
        if kv and not fa_kv_supported(be, kv.split(",", 1)[0].strip()):
            note = (f"{kv} has no fused flash-attention on {be}; showing "
                    f"{flags.cache_type_k}")
        if spec.cache_type_policy.pinned and spec.moe is None and spill > 0:
            rung = _resident_rung(spec, g, profile, flags.ctx, usable_vram,
                                  _budgets(profile,
                                           ubatch=launch_ubatch(spec))[1], be)
            if rung is not None:
                note = (f"{flags.cache_type_k} spills at ctx {flags.ctx} "
                        f"({_cpu_layers(spec, flags)} of {spec.n_layers} layers "
                        f"on the CPU); {rung.cache_type_k} fits fully on the GPU")
        row = {"ok": True, "ctx": flags.ctx, "n_cpu_moe": flags.n_cpu_moe,
               "ngl": flags.ngl, "kv": flags.cache_type_k,
               "kv_v": flags.cache_type_v,
               "offload_pct": round(spill * 100), "speed": speed,
               "budget": _budget_rows(spec, g, mm_mb, flags.ctx,
                                      usable_vram,
                                      flags.cache_type_k,
                                      flags.cache_type_v)}
        if note:
            row["note"] = note
        out.append(row)
    return out


# How much of the model the "context" growth policy may push off the GPU for
# each doubling of the window. 0.15 = up to 15% of the layers; the "speed"
# policy uses 0.0, which is the historical behaviour (never trade a layer).
#
# MEASURED, and much more expensive than "15% of the layers" sounds. On a
# 64-layer dense hybrid (Qwen3.5-family, 48 SSM + 16 attention,
# full_attention_interval=4) with a 6872 MB PQ2_0 file on a 16 GB card, at a
# FIXED ctx 131072 (findings-r3/36, re-derived in findings-r3/37):
#
#     -ngl 99 (0 CPU layers)   54.5 t/s    18.3 ms/token
#     -ngl 58 (7 CPU layers)   12.6 t/s    79.4 ms   (mean of six, sd 1.0)
#     -ngl 48 (17 CPU layers)   5.9 t/s   169.5 ms
#
# `ngl` counts the OUTPUT layer (llama-model.cpp:
# `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`, output placed by the
# same rule), so -ngl 58 leaves layers 0-6 — SEVEN — on the CPU, not six.
# `_spilled` reports the same seven as the weight fraction, 7/64.
#
# The cost is a straight line, not a context-sized cliff:
#     ms/token ~= 0.28 x GPU layers + 9.18 x CPU layers   (R^2 0.99992)
# Each CPU-resident layer streams ~107 MB of PQ2_0 weights through the CPU
# 2-bit unpack + AVX2/VNNI `dpbusd` kernels at ~11.4 GiB/s — a COMPUTE-bound
# kernel (desktop DRAM is 40-80 GB/s), not DRAM bandwidth and not the KV
# cache. llama.cpp attends only over the OCCUPIED, padded cells, not the
# allocated window: `get_n_kv` pads to 256, so a ~260-token run sees n_kv=512,
# and llama-kv-cache.h calls n_kv "a heuristic, to avoid attending the full
# cache if it is not yet utilized". The CPU attention term is <=0.3 ms of the
# 9.18. An earlier note blamed "the spilled layer's context-proportional KV
# cache"; KV-on-layer-device is real (`llama-kv-cache.cpp`: `if (offload) {
# auto * dev = model.dev_layer(il); ... }`) but with a barely-filled cache it
# is not what costs the time.
#
# 262144 FITS FULLY on this card with a smaller cache: q8_0/q8_0 is 8704 MB of
# KV (15576 MB total, plus 299 MB of recurrent state at --parallel 2, against a
# 14954 MB budget, so it spills 10 layers), while q5_1/q5_1 is 6144 MB (13315 MB
# total, 1639 MB of headroom) and keeps every layer on the GPU. A pinned q8_0
# policy is what turns a fully-resident 262K plan into a 10-layer spill (~4x
# slower); `fit_for_launch` now treats the requested type as a CEILING so that
# cannot happen silently.
#
# The user selects this policy (Models page -> Growth policy), so it is not
# forced — but the dropdown prices it as "more context" and never as "4x
# slower". Left at 0.15 deliberately: other models genuinely prefer the window,
# and changing a global default on one machine's measurement would be worse
# than documenting it.
_GROW_LAYER_BUDGET = 0.15


def _configured(spec: ModelSpec, *, kv: str = "",
                vision: bool = True) -> ModelSpec:
    """A copy of `spec` with the cache policy and vision projector overridden.

    A copy, not a mutation: these come from a query string and must not leak
    into the registry object the rest of the process is sharing."""
    if not kv and vision:
        return spec
    s = spec.model_copy(deep=True)
    if kv:
        # ONE type: K and V always match (ComboFlags._symmetric_kv). Vulkan at
        # 87268f77 would fuse a mixed pair, but the CUDA/HIP path would not, and
        # Rigma runs both — see the comment on that validator. A "k,v" pair used
        # to be parsed here and then silently normalised downstream, so an
        # asymmetric request looked honoured and wasn't. The API rejects that
        # form now; this only has to not crash on one.
        k = kv.split(",", 1)[0].strip()
        if k:
            s.cache_type_policy.k = s.cache_type_policy.v = k
        # asked for explicitly, so do not quietly fall back to q8_0 — that made
        # f16 / q5_1 / q4_0 all report the identical verdict
        s.cache_type_policy.pinned = True
    if not vision:
        s.mmproj = None
    return s


def _budget_rows(spec: ModelSpec, gguf: GgufFile, mm_mb: float, ctx: int,
                 usable_vram: float, k: str = "", v: str = "") -> dict:
    """Where the VRAM actually goes, in MB. The whole point of the explorer is
    that a single "8K" tells you nothing about WHY — this is the arithmetic
    behind it, so a 46MB near-miss reads as a near-miss.

    A2b: the RS term was missing here. `_fit_with_cache` charges it (A2) and the
    explorer's `offload_pct` follows the fit (A3), but this table did not — so a
    row could read "fits" (`over_mb` <= 0) while the fit it describes offloaded a
    layer. Same term, same multiplier, one source: `recurrent_state_mb` times
    `LAUNCH_PARALLEL`, exactly as the fit computes it. Zero for a dense model and
    for a hybrid whose geometry is unknown (the fit says `rs=unknown` there).

    A2d-budget: `rs_unknown` rides BESIDE the charge. The fit's explain line has
    said `rs=unknown(est N MB)` since A2d-gap (e29396f), but this row — the
    arithmetic behind the Models page's "OVER by / headroom" line — still
    reported a bare number for the same shape, and a number with no provenance
    reads as a measured one. The flag never replaces the charge: dropping the
    term would free VRAM llama.cpp is about to allocate, which is the launch OOM
    the fit exists to prevent. A boolean beside `rs_mb`, not a string in it, so
    every existing numeric consumer of the row is untouched (nothing indexes the
    row positionally; the row is JSON over the API and a new key is additive).

    A2d-kv: `kv_unknown` rides beside `kv_mb` for the same reason, for the KV
    cache this time — a missing `attention.head_count_kv` or a partial
    sliding-window geometry produced a confident 0 here with nothing to say it
    was an absence of evidence. Same shape: additive boolean, charge unchanged.
    """
    k = k or spec.cache_type_policy.k
    v = v or spec.cache_type_policy.v
    kv_mb = (ctx * kv_bytes_per_token(spec, k, v)
             + swa_kv_bytes(spec, k, v, ctx)) / 2**20
    file_mb = gguf.bytes / 2**20
    rs_mb = recurrent_state_mb(spec) * LAUNCH_PARALLEL
    return {"file_mb": round(file_mb), "mmproj_mb": round(mm_mb),
            "kv_mb": round(kv_mb), "kv_unknown": kv_geometry_unknown(spec),
            "rs_mb": round(rs_mb),
            "rs_unknown": recurrent_state_unknown(spec),
            "budget_mb": round(usable_vram),
            "over_mb": round(file_mb + mm_mb + kv_mb + rs_mb - usable_vram),
            "ctx": ctx, "kv_type": k}


def recommended_quant(quants: list[dict]) -> str | None:
    """Best QUALITY that still runs at GPU speed. `quants` are largest-first, so
    quality descends down the list; prefer a quant that fits on the GPU (or only
    lightly offloads) over a bigger one that spills to RAM and crawls."""
    fast = [q for q in quants
            if q["fit"].get("ok") and q["fit"].get("speed") in ("gpu", "light")]
    if fast:
        return fast[0]["quant"]      # largest fast-enough = best quality @ speed
    fits = [q for q in quants if q["fit"].get("ok")]
    return fits[-1]["quant"] if fits else None   # else the least-offloaded


def _calculate(profile: HardwareProfile, registry: Registry,
               use_case: str, backend: str | None = None) -> RunPlan | None:
    explain: list[str] = []
    pool = [m for m in registry.models.values() if use_case in m.use_cases] or \
        list(registry.models.values())
    if use_case == "coding":
        tooled = [m for m in pool if "tools" in m.capabilities]
        if tooled:
            explain.append("coding: restricting to tools-capable models "
                           f"({', '.join(m.slug for m in tooled)})")
            pool = tooled
        else:
            explain.append("coding: WARNING - no tools-capable model available; "
                           "agent tool calling will not work")

    def total_b(m: ModelSpec) -> float:
        # capability proxy = largest gguf size. n_layers was WRONG here: an 8B
        # with 36 layers outranked the 35B MoE (regression caught 2026-07-16)
        return max((g.bytes for g in m.ggufs), default=0)

    def _on_disk(g: GgufFile) -> bool:
        from .runtime import rigma_home
        try:
            return (rigma_home() / "models" / g.file).exists()
        except Exception:
            return False

    for spec in sorted(pool, key=total_b, reverse=True):
        # LOCALITY BEFORE QUALITY. Live 2026-07-20: `up -y` on a model whose
        # 14.4GB quant sat on disk silently began downloading the 26GB quant,
        # because registry order (largest first) was the only order. Rule: any
        # on-disk quant that fits beats any remote one; remote quants are only
        # considered when nothing local fits. Cold-starting a fresh machine
        # still works — the local list is empty and the order degenerates to
        # the old one.
        local = [g for g in spec.ggufs if _on_disk(g)]
        ordered = local + [g for g in spec.ggufs if g not in local]
        if local:
            explain.append(f"{spec.slug}: preferring on-disk quant(s) "
                           f"{', '.join(g.quant for g in local)}")
        for gguf in ordered:  # on-disk first, then registry order
            ctx = min(CTX_DEFAULT.get(use_case, 16384), spec.native_ctx)
            floor = _ctx_floor(spec)
            while ctx >= floor:
                # The backend decides which cache rungs can actually be FUSED,
                # so it has to be known before the fit, not after it.
                be = _backend(profile, backend)
                flags = fit_gguf(spec, gguf, profile, ctx, explain, backend=be)
                if flags:
                    flags = _grow_ctx(spec, gguf, profile, flags, explain,
                                      backend=be)
                    return RunPlan(model_slug=spec.slug, gguf=gguf,
                                   backend=be,
                                   flags=flags,
                                   origin="calculator", explain=explain)
                ctx = _next_ctx_rung(ctx, floor)
    return None


def fallback_plans(plan: RunPlan, registry: Registry,
                   profile: HardwareProfile) -> list[RunPlan]:
    out: list[RunPlan] = []
    spec = registry.models.get(plan.model_slug)
    if spec is not None:
        smaller = [g for g in spec.ggufs if g.bytes < plan.gguf.bytes]
        for gguf in smaller:  # registry order: largest first
            explain = [f"fallback: {plan.gguf.quant} failed to launch"]
            ctx = min(plan.flags.ctx, spec.native_ctx or plan.flags.ctx)
            flags = None
            floor = _ctx_floor(spec)
            while ctx >= floor and flags is None:
                flags = fit_gguf(spec, gguf, profile, ctx, explain,
                                 backend=plan.backend)
                if flags is None:
                    ctx = _next_ctx_rung(ctx, floor)
            if flags is not None:
                out.append(_apply_calibration(RunPlan(
                    model_slug=spec.slug, gguf=gguf, backend=plan.backend,
                    flags=flags, origin="fallback", explain=explain), profile))
    have_ggufs = [m for m in registry.models.values() if m.ggufs]
    if have_ggufs:
        floor_spec = min(have_ggufs, key=lambda m: m.ggufs[-1].bytes)
        if (floor_spec.slug, floor_spec.ggufs[-1].quant) != (plan.model_slug,
                                                             plan.gguf.quant):
            out.append(RunPlan(
                model_slug=floor_spec.slug, gguf=floor_spec.ggufs[-1],
                backend="cpu",
                flags=ComboFlags(ctx=_ctx_floor(floor_spec), ngl=0),
                origin="fallback:floor",
                explain=["fallback floor: smallest model on CPU"]))
    return out


# How far the machine's usable budget may fall below what a combo was VERIFIED
# at before the combo stops being trustworthy. Live readings move — the
# desktop's own VRAM drifts by gigabytes and free RAM by more — so an exact
# comparison would flap a verified combo off on a busy afternoon. A material
# shortfall is the case the check exists for.
_COMBO_BUDGET_SLACK = 0.10


def _combo_rejection(combo, spec: ModelSpec, gguf: GgufFile,
                     profile: HardwareProfile) -> str:
    """("" | why the curated combo must not be used as-is).

    AUDIT F23: the combo path returned its stored flags with NO fit check at
    all — `_budgets` was never called and the combo's own declared `budget`
    field was read nowhere in the codebase. That is the DEFAULT path for exactly
    the hardware the README's verified table advertises, so on a pressured
    desktop `rigma sweep` benchmarked an over-budget config and could crown a
    winner picked by paging noise.
    """
    usable_vram, usable_ram = _budgets(profile, ubatch=launch_ubatch(spec))
    if combo.budget is not None:
        if usable_vram < combo.budget.vram_mb * (1 - _COMBO_BUDGET_SLACK):
            return (f"verified at {combo.budget.vram_mb}MB usable VRAM; this "
                    f"machine has {usable_vram:.0f}MB")
        if usable_ram < combo.budget.ram_mb * (1 - _COMBO_BUDGET_SLACK):
            return (f"verified at {combo.budget.ram_mb}MB usable RAM; this "
                    f"machine has {usable_ram:.0f}MB")
    # Even with no declared budget the PLACEMENT has to still exist: the combo's
    # own ctx may no longer be placeable at all on this machine right now.
    if fit_gguf(spec, gguf, profile, combo.flags.ctx, [],
                backend=combo.backend) is None:
        return (f"its ctx {combo.flags.ctx} can no longer be placed in "
                f"{usable_vram:.0f}MB usable VRAM")
    return ""


def _tier_note(rel: str, profile: HardwareProfile) -> list[str]:
    """Say when the matched combo is keyed to a different RAM tier.

    AUDIT F15-4: `find_combo` falls back to the nearest lower RAM tier, so the
    reference box (31.4 GB, tier 32) now gets the verified ram-16 combo. A silent
    fallback would hide that the numbers came from a machine with less RAM.
    """
    m = re.search(r"/ram-(\d+)/", rel)
    if m is None:
        return []
    keyed = int(m.group(1))
    if keyed == profile.ram_tier_gb:
        return []
    return [f"note: this combo is keyed to the ram-{keyed} tier; this machine's "
            f"RAM tier is ram-{profile.ram_tier_gb}, so the nearest lower tier "
            "was used"]


def resolve(profile: HardwareProfile, registry: Registry,
            use_case: str = "general", model_override: str | None = None,
            backend_override: str | None = None) -> RunPlan:
    if not registry.models:
        raise ResolveError("registry has no models")
    gpu = profile.primary_gpu
    rejected: list[str] = []
    # A curated combo pins its own backend, so honouring an explicit request
    # means skipping the combo path — otherwise asking for ROCm would silently
    # return a Vulkan combo and the UI would report a backend it never ran.
    if gpu and model_override is None and not backend_override:
        hit = registry.find_combo(gpu.vendor, gpu.slug, round(gpu.vram_mb / 1024),
                                  profile.ram_tier_gb, use_case)
        if hit:
            combo, rel = hit
            kind = "class" if rel.startswith("_class/") else "combo"
            # AUDIT F24: an unguarded subscript and a bare `next()` meant a
            # registry snapshot that dropped a model or a quant killed
            # up/plan/sweep with a raw KeyError/StopIteration, where every other
            # failure in this module raises ResolveError — and cli.py catches
            # only that. The registry validates this in CI, but CI runs after
            # the push and codeload serves master regardless.
            spec = registry.models.get(combo.model)
            gguf = next((g for g in (spec.ggufs if spec else [])
                         if g.quant == combo.quant), None)
            if spec is None or gguf is None:
                raise ResolveError(
                    f"registry {kind} '{rel}' names "
                    + (f"an unknown model '{combo.model}'" if spec is None
                       else f"the quant '{combo.quant}', which "
                            f"'{combo.model}' does not list")
                    + " — run `rigma update` (your cached registry and the "
                      "packaged one disagree)")
            why = _combo_rejection(combo, spec, gguf, profile)
            if not why:
                # w13b2 NIT 2: a registry combo is hand-authored, and its `env`
                # used to be copied straight onto the plan — so a combo carrying
                # `LLAMA_ATTN_ROT_DISABLE` (or a lowercase variant) reached a
                # plain launch, the silent quality regression C11 closed for the
                # calibration merge. Route the copy through the SAME gate, so a
                # hand-edited combo cannot reopen the path. No in-tree combo
                # carries a lever today; this is the read-path guard.
                combo_flags, dropped_levers = _without_quality_env_levers(
                    combo.flags.model_dump())
                plan = RunPlan(
                    model_slug=combo.model, gguf=gguf, backend=combo.backend,
                    flags=(ComboFlags.model_validate(combo_flags)
                           if dropped_levers else combo.flags),
                    origin=f"{kind}:{rel}",
                    explain=([f"registry match: {rel}"]
                             + _tier_note(rel, profile) + combo.sources))
                if dropped_levers:
                    plan.explain.append(
                        f"registry {kind} dropped quality-degrading env lever(s) "
                        + ", ".join(dropped_levers) + ": the engine turns the "
                        "quality-preserving default ON for a reason "
                        "(see bench._QUALITY_ENV_LEVERS)")
                # A model's stored launch defaults apply on the combo path too:
                # `rigma up` with no --model lands here, and a stored ubatch/ngl
                # that only worked through the calculator would be a default
                # that applies to some launches and not others.
                return _apply_calibration(
                    _with_launch_defaults(plan, spec, profile), profile)
            rejected = [f"registry {kind} '{rel}' NOT used: {why}",
                        "falling back to the fit calculator"]
    if model_override:
        if model_override not in registry.models:
            raise ResolveError(
                f"unknown model: {model_override} — run `rigma update` "
                f"(your cached registry may predate it)")
        registry = Registry(registry.gpus,
                            {model_override: registry.models[model_override]},
                            registry.combos)
    plan = _calculate(profile, registry, use_case, backend_override)
    if plan:
        if rejected:
            plan.explain = rejected + list(plan.explain)
        plan = _with_launch_defaults(plan, registry.models.get(plan.model_slug),
                                     profile)
        return _apply_calibration(plan, profile)
    # absolute floor: smallest model, smallest quant, CPU
    have_ggufs = [m for m in registry.models.values() if m.ggufs]
    if not have_ggufs:
        raise ResolveError("no model in the registry has a gguf to run")
    spec = min(have_ggufs, key=lambda m: m.ggufs[-1].bytes)
    plan = RunPlan(
        model_slug=spec.slug, gguf=spec.ggufs[-1], backend="cpu",
        flags=ComboFlags(ctx=_ctx_floor(spec), ngl=0), origin="calculator",
        explain=rejected + ["floor: nothing larger fits"])
    return _apply_calibration(_with_launch_defaults(plan, spec, profile),
                              profile)
