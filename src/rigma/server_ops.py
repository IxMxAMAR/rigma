from __future__ import annotations

import json
import logging
import os
import platform
import time

import psutil

from .atomicio import atomic_write_json
from .models import CACHE_BYTES
from .runtime import rigma_home
from . import engine_compat

_log = logging.getLogger(__name__)


def ram_snapshot() -> dict:
    m = psutil.virtual_memory()
    return {"ram_free_mb": int(m.available / 2**20),
            "ram_total_mb": int(m.total / 2**20)}


def engine_version(backend: str = "") -> str:
    """Which engine build is actually installed — measured, not the manifest claim.

    R3-ENG-6. This used to return the MANIFEST's version string, which is a claim
    about what was downloaded rather than a fact about what is on disk. That broke
    calibration in a way nothing could detect: `bench.save_calibration` stamps every
    entry with this value, and `bench.calibration_stale` compares a stored entry
    against this same value — so an engine change compared the manifest to itself and
    could never invalidate anything. On the owner's machine the pinned directory
    `~/.rigma/engines/b9867/rocm/` held a THIRD-PARTY FORK, build 10709, and every
    calibration measured on it was labelled `b9867`.

    Now the binary is asked. `engine_build.cached_build` memoises on (path, size,
    nanosecond mtime), so this is one process spawn per binary per process rather
    than per call, and replacing a binary at the same path invalidates it.

    Falls back to the manifest string when no binary can be found or run: the honest
    answer is then "we do not know", and the manifest is at least the intended value.
    A backend of "" checks every installed backend and returns the first that answers,
    which is what a caller with no plan context wants.
    """
    from . import engine_build, runtime
    try:
        man = runtime._engines_manifest()
        want = str(man.get("version", ""))
        assets = man.get("assets") or {}
        backends = [backend] if backend else [
            b for b in ("vulkan", "rocm", "cuda", "cpu")
            if f"windows/{b}" in assets or f"linux/{b}" in assets]
        name = "llama-server.exe" if os.name == "nt" else "llama-server"
        for b in backends:
            root = runtime.rigma_home() / "engines" / want / b
            if not root.exists():
                continue
            exe = root / name
            if not exe.exists():
                exe = next(root.rglob(name), None)
            if exe is None:
                continue
            got = engine_build.cached_build(exe)
            if got.ok:
                # The real build, so a calibration can go stale when it changes and a
                # fork can be told apart from the pin.
                return got.identity
        return want
    except Exception:
        return ""


def expected_tg(model: str, quant: str, backend: str) -> float | None:
    """Calibrated decode speed for the running combo, if bench ever ran.

    The number lives under "measured" — that is the shape `bench.save_calibration`
    writes, and it is the only writer of this file. This read used a flat
    `["tg_tps"]` that no writer has ever produced, so it returned None for every
    real calibration on disk and the engine-room verdict was permanently
    "unknown". The unit test passed throughout, because it hand-wrote the flat
    shape instead of calling the writer.

    The flat branch is kept for entries written before the nesting existed;
    without it, fixing the read would silently retire every old calibration.
    """
    try:
        cal = json.loads((rigma_home() / "calibration.json")
                         .read_text(encoding="utf-8"))
        entry = cal[f"{model}:{quant}:{backend}"]
        got = entry.get("measured", entry).get("tg_tps")
        if got is None:
            return None
        # Speed on a fixed model+quant+engine is not a constant: it depends on
        # whether the weights fit in VRAM, and on Windows that depends on what
        # ELSE holds VRAM. Measured 2026-08-21, the same combo ran at 9.95 and
        # 37.59 tok/s purely because the desktop's footprint changed by 2.8GB.
        # An expectation carried across that is worse than no expectation.
        from .bench import calibration_stale
        from .probe import gpu_used_mb
        if calibration_stale(entry, gpu_used_mb()) is not None:
            return None
        return float(got)
    except Exception:
        return None


def verdict(last_tg: float | None, exp: float | None) -> str:
    if last_tg is None or exp is None:
        return "unknown"
    return "degraded" if last_tg < 0.6 * exp else "healthy"


def log_path():
    """The newest engine log, or None when no engine has ever logged."""
    logs = sorted((rigma_home() / "logs").glob("server-*.log"),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    return logs[0] if logs else None


def log_tail(lines: int = 200) -> str:
    """The last `lines` lines of the newest engine log (see log_tail_bounded)."""
    return log_tail_bounded(lines)[0]


# A generous per-line allowance for the byte bound. The old reader pulled the
# WHOLE file into memory and sliced it, which on a server that had been up for
# days meant hundreds of megabytes through the event loop for a 200-line panel
# (IMP-11). The READ is now bounded; `lines` only bounds the display.
_TAIL_LINE_BYTES = 512


def log_tail_bounded(lines: int = 200) -> tuple[str, bool]:
    """(tail, truncated) for the newest engine log, without reading it whole."""
    from .runs import read_tail_bytes
    p = log_path()
    if p is None:
        return "", False
    n = max(10, min(int(lines), 1000))
    text, truncated = read_tail_bytes(p, max(64 * 1024, n * _TAIL_LINE_BYTES))
    return "\n".join(text.splitlines()[-n:]), truncated


# The whole-log reader is bounded by BYTES, not lines: a chatty session can
# produce a million lines, and nothing here needs more than the launch banner.
_LOG_MAX_BYTES = 8_000_000


def log_text(max_bytes: int = _LOG_MAX_BYTES) -> str:
    """The WHOLE newest engine log (bounded by bytes).

    `log_tail` hard-caps at 1000 lines, and the load-time warnings
    `engine_log.findings` exists for land in the first few hundred lines of a
    launch — so on any session whose log has grown past the cap the feature
    reports nothing at all (AUDIT F51).

    Raises OSError when there is no readable log, so a caller can tell "nothing
    found" from "could not look": the endpoint used to answer `{"findings": []}`
    for both, which reads as "the engine is healthy"."""
    p = log_path()
    if p is None:
        raise FileNotFoundError("no engine log yet")
    size = p.stat().st_size
    with p.open("rb") as fh:
        if size > max_bytes:
            fh.seek(size - max_bytes)
        raw = fh.read()
    return raw.decode("utf-8", errors="replace")


def _model_on_disk(gguf) -> bool:
    return (rigma_home() / "models" / gguf.file).exists()


def _registered_engine_for(gguf, backend: str):
    """A registered engine that can load this model, or None.

    `engine_registry` could already describe, verify and select a non-pinned engine, but
    nothing in the launch path ever called `select` — so a model that only a registered
    fork can load still went to the pin and died with `invalid ggml type 142. should be in
    [0, 42)`, which is exactly the failure registration was added to prevent. The owner
    installed PrismML's fork and registered it, and `rigma up` still picked the pinned
    b9867 build.

    The decision is made on the model's ACTUAL tensor types, read from its header, so this
    can only ever prefer a build that demonstrably accepts this file. A model the pin can
    load is unaffected: `select` only returns an engine whose declared types cover every
    type in the file, and a registered engine with unknown capabilities is not chosen here
    (it is offered by `rigma engine-runtimes`, where a human can act on it).

    Never raises. Any failure means "no opinion", and the pin is used as before.
    """
    try:
        from . import engine_registry
        from .gguf_meta import read_tensor_index
        path = rigma_home() / "models" / gguf.file
        if not path.exists():
            return None
        idx = read_tensor_index(path)
        if not idx.types_complete or not idx.type_counts:
            return None
        engine, _reason = engine_registry.select(list(idx.type_counts), backend)
        if engine is None or not engine.exe.exists():
            return None
        return engine
    except Exception:
        return None


def engine_binary_for(gguf, backend: str, os_name: str):
    """The `llama-server` to launch for `gguf` on `backend`: a registered engine that can
    load it, else the pin. Returns `(exe, record)`.

    THE ONE SEAM. Rigma launches an engine from three places — `up` (which has its own
    fallback ladder), `sweep`, and `perform_switch` (the UI's path) — and each one
    previously called `runtime.ensure_engine` directly. A registered engine therefore had
    to be wired into all three or it was wired into none: fixing only `perform_switch` left
    `rigma up --model <pq2_0 model>` still launching the pinned b9867 build, which refuses
    the file with `invalid ggml type 142. should be in [0, 42)` and then silently falls back
    to SmolLM2. That is exactly what happened after the first version of this fix.

    `record` is the description that goes into `state["engine_binary"]`, so which binary
    served a launch is answerable afterwards rather than only from the process table.

    `record["is_prism_fork"]` is the build's fork identity, from the SAME decision that
    chose it (`engine_compat.engine_is_prism_fork`). Callers that are about to build an
    argv must use `engine_binary_for_plan`, which freezes it onto the plan; a caller
    that uses this function directly gets no frozen identity and therefore no
    fork-only flags.
    """
    from . import runtime
    custom = _registered_engine_for(gguf, backend)
    if custom is not None:
        return custom.exe, {"kind": "registered", "name": custom.name,
                            "path": str(custom.exe), "source": custom.source,
                            "is_prism_fork": engine_compat.engine_is_prism_fork(custom)}
    exe = runtime.ensure_engine(backend, os_name)
    return exe, {"kind": "pinned", "name": f"{os_name}/{backend}",
                 "path": str(exe), "source": "", "is_prism_fork": False}


def engine_binary_for_plan(plan, os_name: str):
    """`engine_binary_for`, plus THE FREEZE of the fork identity onto `plan`.

    This is the function the launch paths call. It picks the binary from the model's
    tensor types and immediately records whether that binary is the PrismML fork on
    `plan.engine_is_prism_fork`, so `RunPlan.server_args` reads a decision instead of
    re-asking one. The two must be the same decision: `_registered_engine_for` needs
    the model on disk, and `ensure_model` — which runs AFTER this on a first launch —
    is what puts it there. Re-deriving at argv-build time therefore named the fork for
    a model that was not downloaded when the PINNED binary was chosen, and the pinned
    mainline build was handed the fork-only `--reasoning-effort`; llama-server exits
    in argparse on an unknown argument, so the first launch of any model died.

    Returns `(exe, record)` exactly like `engine_binary_for`.
    """
    exe, record = engine_binary_for(plan.gguf, plan.backend, os_name)
    plan.engine_is_prism_fork = bool(record.get("is_prism_fork"))
    return exe, record


def _calib_marker_path():
    return rigma_home() / "calibrating.json"


def read_calib_marker() -> dict | None:
    """First-load tuning progress, if a calibration is running right now. Stale
    markers (a crash mid-tune) are ignored after 20 min so the UI never sticks
    on 'optimizing' forever."""
    p = _calib_marker_path()
    try:
        import time
        if time.time() - p.stat().st_mtime > 1200:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_calib_marker(model: str, step: str) -> None:
    p = _calib_marker_path()
    # R3-STORE-10: fixed temp name -> unique temp + retried replace.
    atomic_write_json(p, {"model": model, "step": step})


def _clear_calib_marker() -> None:
    try:
        _calib_marker_path().unlink()
    except FileNotFoundError:
        pass


def _free_current(profile, state: dict, reg):
    """A copy of `profile` with the CURRENTLY-running model's RAM footprint
    added back — perform_switch/ctx-change kill that engine before launching,
    so fitting against live free RAM is wrong. Without this, a ctx change on
    the running model probes against its own occupied RAM and reports an
    absurdly small ceiling (live repro 2026-07-18: 35B ctx change said 'tops
    out at 8,192' while it was running fine at 32K).

    VRAM is credited the same way now that the budget accounts for what the
    desktop actually holds (2026-08-21). Without it the outgoing engine's own
    13GB would count as "someone else's", and a ctx change on the running model
    would plan against a card that looks entirely full."""
    # R3-CTX-1: a record that cannot name the model is not proof that nothing is
    # running. A hard kill leaves the engine holding the card with no record, and
    # a UI-only `up` used to write an empty one on top of a live engine. Either
    # way this returned the profile UNCHANGED, so the engine's own VRAM counted
    # as someone else's, the budget collapsed to what is left on an apparently
    # full card, and a ctx change was refused with "tops out around 8,192" while
    # the engine was running at 65,536. Ask the engine, which knows.
    #
    # This runs BEFORE the `not state` early return on purpose: an empty record
    # is the case where it matters most, and the first version of this fallback
    # sat after it, so `_free_current(prof, {}, reg)` still credited nothing.
    # The repro caught that, not the reasoning.
    if not state or not reg.models.get(state.get("model", "")):
        live = ""
        try:
            from . import orphan
            live = orphan.running_gguf_file(
                int((state or {}).get("public_port") or 11500))
        except Exception:
            live = ""
        if live:
            for _slug, _spec in reg.models.items():
                if any(g.file == live for g in (_spec.ggufs or [])):
                    state = {**(state or {}), "model": _slug, "gguf": live,
                             "unloaded": False}
                    break
    if not state:
        return profile
    spec = reg.models.get(state.get("model", ""))
    if spec is None or not spec.ggufs:
        return profile
    # The file the engine ACTUALLY holds. `state["gguf"]` records it since
    # 2026-08-21; the largest-gguf guess below is the fallback for state written
    # before that. The guess was harmless while a model listed a handful of
    # files and became dangerous the moment one listed 103: this model's repo
    # includes a 50GB BF16, so crediting the largest handed back three times the
    # card's capacity, made the GPU look empty, and put the planner straight
    # back to assuming VRAM that does not exist.
    running = next((g for g in spec.ggufs if g.file == state.get("gguf")), None)
    freed_mb = ((running.bytes if running is not None
                 else max(g.bytes for g in spec.ggufs)) / 2**20)
    if spec.mmproj and not state.get("no_vision"):
        freed_mb += spec.mmproj.bytes / 2**20
    update = {"ram_free_mb": profile.ram_free_mb + int(freed_mb)}
    # The engine we are about to kill still holds its VRAM. Credit it back, or
    # the measured "desktop" footprint includes our own model and the plan
    # shrinks to fit a card that is about to be freed. Never below zero, and
    # never below what an unloaded desktop would read.
    if profile.vram_used_mb is not None and not state.get("unloaded"):
        # VRAM must never be over-credited the way RAM safely can: the whole
        # point of measuring it is to stop planning against memory the card
        # does not have. Cap the credit at what is actually held.
        update["vram_used_mb"] = max(0.0, profile.vram_used_mb
                                     - min(freed_mb, profile.vram_used_mb))
    return profile.model_copy(update=update)


def _measured_placement(rp, ctx: int) -> dict:
    """Weight placement that was MEASURED at this context, if there is one.

    `fit_gguf` recomputes ngl / n_cpu_moe from a VRAM model that is deliberately
    conservative — the right answer for a context nobody has tried, and the
    wrong one for a context somebody has. Qwen3.8-27B was pinned at ngl 63 after
    measuring 33.24 t/s there and relaunched at ngl 59, because the calculator
    disagreed with the machine. The machine wins: a number that came from a
    stopwatch beats a number that came from arithmetic about the same thing.

    Only on an EXACT context match. Placement measured at 32K says nothing about
    256K — the KV cache is most of what moved — so a near miss falls back to the
    calculator rather than guessing.

    AUDIT 06R3-1: the two values are TYPED here, before they leave. The caller
    merges them with `ComboFlags.model_copy(update=...)`, which by design does
    not validate, so a string in the user-editable calibration.json reached
    `RunPlan.server_args` and killed the launch with `TypeError: '>' not
    supported between instances of 'str' and 'int'` — the same bypass AUDIT
    06-4 closed in `resolve._apply_calibration`, in the sibling merge site it
    did not cover. JSON has no integer type, so a whole-number float is a
    legitimate way for a row to read back and is coerced; anything else is a
    corrupt row, and a corrupt row is dropped rather than half-applied.
    """
    try:
        from .bench import load_calibration
        entry = load_calibration().get(
            f"{rp.model_slug}:{rp.gguf.quant}:{rp.backend}") or {}
        if int(entry.get("ctx") or 0) != int(ctx):
            return {}
        flags = entry.get("flags") or {}
        out = {}
        for key in ("ngl", "n_cpu_moe"):
            if key not in flags:
                continue
            value = flags[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                return {}
            if isinstance(value, float):
                if not value.is_integer():
                    return {}
                value = int(value)
            if value < 0:
                return {}
            out[key] = value
        return out
    except Exception:
        return {}          # a missing or corrupt calibration is not an error


def launch_fit_spec(spec, flags, *, vision: bool, ctx: int = 0):
    """The spec to run the fit against, plus whether it differs from `spec`.

    AUDIT F18: docs/audit-2026-09-04-full.md — the plan and the launch have to
    agree about what sits on the card. `resolve.with_launch_overheads` folds
    both disagreements into the one mmproj slot `fit_gguf` already treats as
    non-offloadable memory: the projector comes OUT when vision is off (717-888
    MiB on the two shipped vision models, enough to cost two layers of GPU
    residency), and the speculative draft cache goes IN when speculation is on
    (~565 MiB at 32K, previously budgeted nowhere because spec_type was applied
    after the fit).

    The second return value says whether that changed the number. It is the
    guard on the extra fit: when nothing moved, the resolver's own arithmetic
    already stands and re-running it would only substitute the calculator's
    answer for a curated combo's.
    """
    from .resolve import with_launch_overheads
    ctx = int(ctx or flags.ctx)
    fit = with_launch_overheads(spec, vision=vision, ctx=ctx,
                                spec_type=flags.spec_type,
                                n_max=flags.spec_n_max)
    was = spec.mmproj.bytes if spec.mmproj else 0
    now = fit.mmproj.bytes if fit.mmproj else 0
    return fit, was != now


def _resolve_for(slug: str, state: dict, registry, profile,
                 backend: str | None = None):
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import resolve
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
    p = _free_current(p, state, reg)   # count the outgoing engine as freed
    return resolve(p, reg, use_case=state.get("use_case", "general"),
                   model_override=slug, backend_override=backend), reg, p


def plan_placement(plan) -> dict:
    """The device-side placement a plan will actually apply, for state.json.

    DR2-1-res. `resolve._spilled` decides what fraction of a model's weights
    stays in system RAM from exactly two plan fields: `ngl` (dense) and
    `n_cpu_moe` (MoE expert offload). Recording them at launch is what lets
    `planned_vram_mb` rebuild a DEVICE-SIDE prediction later — the plan object
    itself is gone once the engine is up, and state.json previously carried no
    placement at all, so the reader could only ASSUME "fully resident".

    Recorded as a dict, not two loose keys, so an old record (no `placement`)
    and a fully-resident record (`n_cpu_moe` 0) stay distinguishable: absent
    must read as unknown, never as the confident zero.
    """
    f = plan.flags
    return {"ngl": int(f.ngl), "n_cpu_moe": int(f.n_cpu_moe)}


def recorded_placement(state: dict) -> dict | None:
    """The device-side placement the record's launch used, or None if unknown.

    Tolerant reader (DR2-1-res): a record written before `placement` existed,
    an adopted/orphan engine, a hand-edited file, or a partially-written dict
    all return None. None means UNKNOWN, never "fully resident" — the caller
    must not fall back to `ngl=99 / n_cpu_moe=0`, which is the assumption that
    made the VRAM axis unusable for the owner's MoE (deep-review-3.md). Both
    keys are required; a placement missing one is not a placement.
    """
    if not isinstance(state, dict):
        return None
    p = state.get("placement")
    if not isinstance(p, dict):
        return None
    try:
        return {"ngl": int(p["ngl"]), "n_cpu_moe": int(p["n_cpu_moe"])}
    except (KeyError, TypeError, ValueError):
        return None


def planned_vram_mb(state: dict, registry=None) -> float | None:
    """The running plan's own weights+KV prediction, in MiB, or None.

    A17d. `engine_log.compare_plan` can only judge the VRAM axis against the
    PLAN'S OWN prediction for the same ctx / cache / slots — `memtruth.planned_mb`,
    the exact function `verify_plan` compares the fit oracle against — and the
    live `/api/server/findings` route passed nothing, so the axis was permanently
    `not_comparable`. This rebuilds the plan from the state the launch wrote and
    hands it to that same function; it is not a second estimate.

    DR2-1-res: the prediction is DEVICE-SIDE. `memtruth.planned_mb`'s weight
    term is the whole GGUF file, which is the device figure only when the engine
    put the whole file on the device; for a dense spill (`ngl < n_layers`) or an
    expert offload (`n_cpu_moe > 0`) it overstates the device figure by the
    RAM-resident weights. The placement the launch recorded (`placement` in
    state.json, via `recorded_placement`) is what `resolve._spilled` needs to
    scale that term, so the number returned here already excludes the weights
    the plan itself left in system RAM. It is therefore comparable to
    `compare_plan`'s device-buffer `actual` without the DR2-1 suppression.

    Returns None — never a number — whenever an input is missing: no model, no
    quant, no ctx, an unknown model/quant, an invalid cache type, OR no recorded
    placement. A whole-file prediction with no recorded placement would be a
    confident "fully resident" assumption (GUIDANCE 5); "cannot compute" must
    stay None and let the axis say `not_comparable`.

    DR2-2: ONE registry, and no re-parse per poll. The spec is resolved once from
    the registry the caller handed (or one `Registry.load()` when it handed
    none) and passed into `memtruth.planned_mb`, so the weight term and the KV
    term can no longer come from two different registries — and no hidden second
    load happens inside `planned_mb`. The whole prediction is then memoised for a
    short TTL (below), because `/api/server` polls every 5 s from every open tab
    and the live server is built with `registry=None`, so without the memo each
    poll paid a full registry parse.
    """
    state = state if isinstance(state, dict) else {}
    key = _plan_cache_key(state, registry)
    now = time.monotonic()
    if (_plan_cache["key"] == key
            and now - _plan_cache["at"] < _PLAN_CACHE_TTL):
        return _plan_cache["mb"]
    mb = _planned_vram_mb(state, registry)
    # Hold the registry reference too: `id()` is only unique among LIVE objects,
    # and a cached id whose object was collected could otherwise be reused.
    _plan_cache.update(key=key, at=now, mb=mb, registry=registry)
    return mb


# DR2-2: the prediction for a RUNNING plan is a pure function of the running
# model's spec, its ctx and its cache types — none of which moves while the
# engine is up — so it is memoised rather than rebuilt on every poll. Measured
# on this machine before the fix: `Registry.load()` 6.3 ms and
# `planned_vram_mb(state)` (two loads, one hidden inside `planned_mb`) 12.8 ms.
# After the fix a cache miss is one load and a hit is ~0.05 ms. Keyed on the
# RIGMA_HOME as well, so two apps in one process (tests) cannot read each
# other's registry; the caller's registry object is held, so its id cannot be
# recycled. A registry that changes on disk is picked up within the TTL.
_PLAN_CACHE_TTL = 20.0
_plan_cache: dict = {"key": None, "at": 0.0, "mb": None, "registry": None}


def _plan_cache_key(state: dict, registry) -> tuple:
    """A total cache key: never raises, so a hand-edited state cannot 500 the poll."""
    try:
        from .runtime import rigma_home
        home = str(rigma_home())
    except Exception:
        home = ""
    try:
        ctx = int(state.get("ctx") or 0)
    except (TypeError, ValueError):
        ctx = str(state.get("ctx"))
    # DR2-1-res: the placement is part of the prediction (it scales the weight
    # term), so it must be part of the key — otherwise two states that differ
    # only in ngl / n_cpu_moe would serve each other's number.
    p = recorded_placement(state)
    placement_key = None if p is None else (p["ngl"], p["n_cpu_moe"])
    return (home, None if registry is None else id(registry),
            str(state.get("model") or ""), str(state.get("quant") or ""), ctx,
            str(state.get("kv_cache") or ""), str(state.get("backend") or ""),
            placement_key)


def _planned_vram_mb(state: dict, registry) -> float | None:
    """The uncached body of `planned_vram_mb` (see it for the contract)."""
    from . import memtruth
    from .models import ComboFlags, RunPlan
    from .registry import Registry
    from .resolve import _spilled
    slug = str(state.get("model") or "")
    quant = str(state.get("quant") or "")
    ctx = int(state.get("ctx") or 0)
    if not slug or not quant or ctx <= 0:
        return None
    # DR2-1-res: no recorded placement means the plan-side basis is unknown. The
    # whole-file weight term would then be a confident "fully resident" guess,
    # so refuse a prediction rather than hand back one the axis would trust.
    placement = recorded_placement(state)
    if placement is None:
        return None
    try:
        reg = registry if registry is not None else Registry.load()
        spec = reg.models.get(slug)
        if spec is None:
            return None
        gguf = next((g for g in spec.ggufs if g.quant == quant), None)
        if gguf is None:
            return None
        k = str(state.get("kv_cache") or "") or "f16"
        flags = ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=k,
                           ngl=placement["ngl"],
                           n_cpu_moe=placement["n_cpu_moe"])
        plan = RunPlan(model_slug=slug, gguf=gguf,
                       backend=str(state.get("backend") or "unknown"),
                       flags=flags, origin="state")
        # DR2-2: pass the spec resolved from THIS registry, so the KV geometry
        # cannot come from a different (process-global) one.
        whole_plus_kv = float(memtruth.planned_mb(plan, spec))
        # DR2-1-res: `planned_mb` charges the WHOLE file; the plan's recorded
        # placement says what share it left in system RAM (`_spilled`, the one
        # implementation the page and the fit already share). Subtract that share
        # so the figure is the DEVICE-side one `compare_plan` measures against.
        whole_weights_mb = gguf.bytes / 2**20
        return whole_plus_kv - whole_weights_mb * _spilled(spec, flags)
    except Exception:
        # A state written by a hand edit, or a registry that predates the model,
        # is "no prediction", never an error on a read route.
        _log.debug("planned_vram_mb: no plan prediction from state",
                   exc_info=True)
        return None


def switch_options(state: dict, registry=None, profile=None) -> list[dict]:
    """Alternative plans limited to models already on disk (no downloads).

    Resolves each model against ONLY its on-disk quants — the resolver may
    prefer a quant that was never downloaded (e.g. under RAM pressure), and
    filtering on that preference hid genuinely usable local models."""
    from .probe import probe_hardware
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    p = profile if profile is not None else probe_hardware(reg.gpus)
    out = []
    for slug in sorted(reg.models):
        if slug == state.get("model"):
            continue
        spec = reg.models[slug]
        on_disk = [g for g in spec.ggufs if _model_on_disk(g)]
        if not on_disk:
            continue
        trimmed = Registry(reg.gpus,
                           {**reg.models,
                            slug: spec.model_copy(update={"ggufs": on_disk})},
                           reg.combos, reg.use_cases)
        try:
            rp, _, _ = _resolve_for(slug, state, trimmed, p)
        except Exception:
            continue
        if rp.model_slug != slug or not _model_on_disk(rp.gguf):
            continue
        reason = (f"{max(1, rp.flags.ctx // 1024)}K context, "
                  f"{rp.gguf.quant} on disk")
        if rp.backend == "cpu":
            reason += " — CPU fallback, will be slow (free RAM for GPU)"
        out.append({"model": rp.model_slug, "quant": rp.gguf.quant,
                    "ctx": rp.flags.ctx, "backend": rp.backend,
                    "reason": reason})
    out.sort(key=lambda o: -o["ctx"])
    return out


# AUDIT 06R3-3: this used to be a THIRD, narrower cache-type vocabulary
# ("f16", "q8_0", "q5_1", "q4_0"). AUDIT 06-3 added bf16/q4_1/q5_0 to
# models.CACHE_BYTES so the fit math prices them and quant_quality publishes a
# reference loss figure for each — but every HTTP boundary kept rejecting them
# with `400 kv must be one of f16, q8_0, q5_1, q4_0`, so the Models-page
# explorer could not ask for a cache type the UI itself compares. Derived from
# the one table rather than restated, so the vocabularies cannot drift again.
# test_phase0_contracts asserts the equality in both directions.
KV_CACHE_TYPES = tuple(CACHE_BYTES)


def vram_snapshot(registry=None) -> dict | None:
    """Card capacity, what the desktop holds, and what is left for a model.

    Exists because the alternative is invisible. llama.cpp reports every layer
    as "on GPU" whenever the driver accepted the allocation, and on Windows the
    driver accepts allocations it intends to page — so a model can be 29% in
    system RAM with nothing anywhere saying so (measured 2026-08-21).
    """
    from .probe import probe_hardware
    from .registry import Registry
    from .resolve import COMPUTE_BUFFER_MB, VRAM_RESERVE_MB, _budgets
    from . import state as st
    reg = registry if registry is not None else Registry.load()
    prof = probe_hardware(reg.gpus)
    total = sum(g.vram_mb for g in prof.gpus)
    if not total:
        return None
    # Credit back the model we are already running. The adapter counter reports
    # everything held on the card, which INCLUDES our own engine — so with a
    # 15.7GB model loaded the panel said "other apps hold 15.7 GB of VRAM,
    # leaving 0.1 GB for the model" and advised closing a browser (owner,
    # 2026-08-25). The resolver has credited this back since 2026-08-21; this
    # read never did, so the number the user saw and the number rigma planned
    # against disagreed. "Other apps" has to mean other apps.
    s = st.read_state() or {}
    if s.get("model") and not s.get("unloaded"):
        prof = _free_current(prof, s, reg)
    usable, _ = _budgets(prof)
    floor = VRAM_RESERVE_MB[prof.os]
    desktop = prof.vram_used_mb
    return {
        "total_mb": total,
        # None when unmeasurable — the UI must not render "0 MB held" then
        "desktop_mb": None if desktop is None else round(desktop),
        "usable_mb": round(usable),
        "assumed_mb": floor + COMPUTE_BUFFER_MB,
        # only worth telling the user about when it is worse than assumed
        "pressured": desktop is not None and desktop > floor,
    }


def available_backends(registry=None) -> list[dict]:
    """Backends this GPU can run, with whether the engine build is on disk.

    The gpu table has always listed more than one for AMD and NVIDIA cards —
    the resolver simply took the first. Surfacing the rest is what makes the
    choice real, and `ready` is what stops the UI offering ROCm without saying
    it costs a ~1.2GB download first.
    """
    import platform

    from .probe import probe_hardware
    from .registry import Registry
    from .runtime import _engines_manifest, rigma_home
    reg = registry if registry is not None else Registry.load()
    gpu = probe_hardware(reg.gpus).primary_gpu
    names = list(gpu.backends) if gpu and gpu.backends else []
    if "cpu" not in names:
        names.append("cpu")          # always available, always last resort
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    try:
        man = _engines_manifest()
    except Exception:
        return [{"name": n, "ready": False, "buildable": False} for n in names]
    root = rigma_home() / "engines" / man["version"]
    return [{"name": n,
             # a build we have no pinned asset for cannot be offered at all
             "buildable": f"{os_name}/{n}" in man.get("assets", {}),
             "ready": (root / n / ".ready").exists()}
            for n in names]


def available_engine_runtimes(registry=None) -> list[dict]:
    """Which ENGINE RUNTIME this machine can run: llama.cpp, and vLLM.

    ADDITIVE, and deliberately a separate function from `available_backends`
    above. That one answers a different question — which llama.cpp COMPUTE
    backend (vulkan/cuda/rocm/cpu) this GPU can drive — and the UI already
    renders its rows. Widening it to carry engine runtimes would change a
    contract two call sites depend on and would put "vllm" in a list the
    backend picker iterates.

    The whole body lives in `engines.py` so there is one definition of what
    "available" means; this is the seam the CLI (and, next, the UI) reads.
    Never raises: an unreadable manifest is a verdict, not a traceback.
    """
    from . import engines
    return [a.as_dict() for a in engines.engine_runtimes(registry)]


def perform_switch(model: str, registry=None, profile=None,
                   ctx: int | None = None, force_calibrate: bool = False,
                   kv: str | None = None, vision: bool | None = None,
                   quant: str | None = None,
                   backend: str | None = None) -> dict:
    """Stop the running engine and launch `model` in its place; with `ctx`,
    relaunch (same model allowed) at a requested context size; with `kv`,
    force the KV-cache quantisation (f16/q8_0/q5_1/q4_0). Growing the cache
    (q8_0 -> f16) may not fit — the launch failure path reports it honestly
    and leaves the UI manageable.

    `vision=False` runs a vision model text-only. The projector is loaded
    alongside the weights and cannot be offloaded, so it costs VRAM for the
    whole session whether or not an image is ever sent — 888MB on
    Qwen3.8-27B, which on a 16GB card is 4x the context window. Choosing to
    drop it is the single largest context lever such a model has.

    Raises RuntimeError with a user-facing message on any failure; a failure
    after the old engine died clears state (one requested plan, one honest
    result — the fallback ladder stays a CLI behavior)."""
    from . import runtime
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if kv is not None and kv not in KV_CACHE_TYPES:
        raise RuntimeError(f"kv must be one of {', '.join(KV_CACHE_TYPES)}")
    # `quant` joins ctx/kv as a reason to relaunch the SAME model: swapping
    # between two downloaded quants is the whole point of asking for one.
    # `backend` joins ctx/kv/quant as a reason to relaunch the SAME model —
    # switching Vulkan<->ROCm is exactly a same-model relaunch.
    if (model == s.get("model") and not s.get("unloaded") and ctx is None
            and kv is None and quant is None and backend is None
            and not force_calibrate):
        raise RuntimeError(f"{model} is already running")
    from .registry import Registry
    reg_full = registry if registry is not None else Registry.load()
    spec_full = reg_full.models.get(model)
    if spec_full is None:
        raise RuntimeError(f"unknown model: {model} — run: rigma update")
    # The model's own configuration fills in anything the caller did not ask
    # for. Precedence is explicit request > model default > resolver: a stored
    # default that could not be overridden would make changing context once
    # impossible on a model that pins it.
    launch = getattr(spec_full, "launch", None)
    if launch is not None:
        d = launch.as_overrides()
        if ctx is None and "ctx" in d:
            ctx = d["ctx"]
        if kv is None and "kv" in d:
            kv = d["kv"]
        if quant is None and "quant" in d:
            quant = d["quant"]
        if vision is None and "vision" in d:
            vision = d["vision"]
        if backend is None and "backend" in d:
            backend = d["backend"]
    on_disk = [g for g in spec_full.ggufs if _model_on_disk(g)]
    if not on_disk:
        raise RuntimeError(
            f"{model} is not downloaded — run: rigma up --model {model}")
    # resolve against on-disk quants only — the resolver may prefer a quant
    # that was never downloaded (live repro 2026-07-17: switch-back refused
    # while a perfectly usable quant sat on disk)
    # An explicit quant narrows the choice to exactly one file. Without this the
    # resolver picked, so a user with two quants on disk could not ask for the
    # smaller one — the trade you make when you want more context (owner,
    # 2026-08-19). Matched leniently: the label shown in the UI carries the
    # quanter's decoration, e.g. "Q3_K_M (RVN)".
    if quant:
        want = str(quant).strip().lower()
        picked = [g for g in on_disk if g.quant.lower() == want]
        if not picked:
            picked = [g for g in on_disk if want in g.quant.lower()
                      or g.quant.lower() in want]
        if not picked:
            have = ", ".join(sorted(g.quant for g in on_disk)) or "none"
            raise RuntimeError(
                f"{model} has no downloaded quant matching {quant!r} — "
                f"on disk: {have}")
        on_disk = picked[:1]
    trimmed = Registry(reg_full.gpus,
                       {**reg_full.models,
                        model: spec_full.model_copy(update={"ggufs": on_disk})},
                       reg_full.combos, reg_full.use_cases)
    rp, _, p = _resolve_for(model, s, trimmed, profile, backend)
    if rp.model_slug != model or not _model_on_disk(rp.gguf):
        raise RuntimeError(f"{model} does not fit this machine right now")
    # AUDIT F18: docs/audit-2026-09-04-full.md — speculation and vision are
    # settled BEFORE the fit. Both change what the card will actually hold, and
    # resolving them 30 lines after the arithmetic is what made this call charge
    # itself for a projector it then omitted from the command line.
    # Speculation only when the FILE carries the draft head: asking llama.cpp
    # for draft-mtp against a gguf without the tensors resets the Vulkan driver
    # rather than erroring (see ComboFlags.spec_type).
    if launch is not None and launch.is_set("spec_type"):
        from .hangar import file_has_mtp
        if launch.spec_type != "draft-mtp" or file_has_mtp(rp.gguf):
            rp.flags = rp.flags.model_copy(update={
                "spec_type": launch.spec_type,
                "spec_n_max": launch.spec_n_max or 1})
    # `vision` is remembered across relaunches: a ctx change must not silently
    # switch vision back on and eat the VRAM the user just freed.
    if vision is None:
        vision = not bool((st.read_state() or {}).get("no_vision"))
    notice = ""
    if ctx is not None:
        # honest relaunch at a requested context: real fit math, not hope.
        # rp.flags.ctx is the calculator's grow-to-fit maximum for this quant.
        from .resolve import fit_for_launch, step_down_notice
        want = max(2048, min(int(ctx), spec_full.native_ctx))
        # The requested cache type — an explicit `kv`, or the model's stored
        # launch default — is a CEILING, fitted in BEFORE the placement is
        # chosen. Applying it AFTER the fit (the old order, below) let a stored
        # `kv: q8_0` re-impose a cache the fit had just rejected: on the owner's
        # 16GB card, ctx 262144 + q8_0 is 15,576MB against a 14,954MB budget, so
        # the resolver's fully-resident q5_1 was overwritten by q8_0 and
        # `-ngl 99` paged 622MB to system RAM with no error printed (WDDM).
        # `backend` matters too: on ROCm/CUDA a q5_1 cache has no fused
        # flash-attention kernel, and `-fa on` would run attention on the CPU
        # without saying so.
        flags, stepped = fit_for_launch(
            spec_full, rp.gguf, p, want, kv=kv or "", vision=vision,
            spec_type=rp.flags.spec_type, n_max=rp.flags.spec_n_max,
            backend=rp.backend, explain=rp.explain)
        if flags is None:
            raise RuntimeError(
                f"ctx {want:,} doesn't fit — {model} ({rp.gguf.quant}) tops "
                f"out around {rp.flags.ctx:,} on this machine")
        update = {
            "ctx": flags.ctx, "n_cpu_moe": flags.n_cpu_moe, "ngl": flags.ngl,
            "cache_type_k": flags.cache_type_k,
            "cache_type_v": flags.cache_type_v}
        # ...except where the placement was MEASURED at this exact context.
        update.update(_measured_placement(rp, want))
        rp.flags = rp.flags.model_copy(update=update)
        if stepped:
            notice = step_down_notice(stepped, rp.backend, flags.cache_type_k,
                                      want)
            rp.explain.append(notice)
    else:
        # No ctx asked for, so the plan is the resolver's — which priced the
        # full spec. Re-place the weights only when the resident overhead has
        # actually moved; the freed projector memory is worth GPU layers, and
        # an unbudgeted draft cache is worth an offload nobody planned. A
        # requested cache type is a reason to re-fit on its own, because the
        # resolver priced the spec's policy, not the request.
        fit_spec, differs = launch_fit_spec(spec_full, rp.flags, vision=vision)
        if differs or kv is not None:
            from .resolve import fit_for_launch, step_down_notice
            got, stepped = fit_for_launch(
                spec_full, rp.gguf, p, rp.flags.ctx, kv=kv or "", vision=vision,
                spec_type=rp.flags.spec_type, n_max=rp.flags.spec_n_max,
                backend=rp.backend, explain=rp.explain)
            if got is not None:
                update = {"ngl": got.ngl, "n_cpu_moe": got.n_cpu_moe,
                          "cache_type_k": got.cache_type_k,
                          "cache_type_v": got.cache_type_v}
                # AUDIT F17/F18: docs/audit-2026-09-04-full.md — same rule as the
                # ctx branch above. `resolve` may already have written a measured
                # placement into these two keys; the calculator must not overwrite
                # a stopwatch. Reached on the ordinary path — a text-only vision
                # model that idle-unloads and reloads comes through here — and
                # that is the 37.59 -> 9.95 tok/s collapse the audit measured.
                update.update(_measured_placement(rp, rp.flags.ctx))
                rp.flags = rp.flags.model_copy(update=update)
                if stepped:
                    notice = step_down_notice(stepped, rp.backend,
                                              got.cache_type_k, rp.flags.ctx)
                    rp.explain.append(notice)
    # NOTE: the requested `kv` is deliberately NOT re-applied here. It was
    # fitted in above; forcing it afterwards is the bug this ordering removes.
    # vision projector: attach it if it's on disk, otherwise run text-only
    # rather than refusing — a vision model still works for text, and the user
    # can download the projector separately to turn vision on
    mm = getattr(reg_full.models.get(model), "mmproj", None)
    extra = (["--mmproj", str(rigma_home() / "models" / mm.file)]
             if vision and mm is not None and _model_on_disk(mm) else None)
    # Some ggufs ship a chat template that llama.cpp cannot use — Apriel-1.6's
    # decensored build self-assigns `{%- set messages = messages ... -%}`, which
    # minja evaluates as unbounded recursion and the process dies with
    # STATUS_STACK_BUFFER_OVERRUN before it ever allocates a context. A repaired
    # template dropped next to the model overrides the embedded one.
    tmpl = rigma_home() / "templates" / f"{model}.jinja"
    if tmpl.is_file():
        extra = (extra or []) + ["--chat-template-file", str(tmpl)]
    os_name = {"Windows": "windows", "Linux": "linux",
               "Darwin": "darwin"}[platform.system()]
    model_path = rigma_home() / "models" / rp.gguf.file
    # R3-ENG-3, the missing half: a registered engine that can load THIS model beats the
    # pin. Capability is decided by the file's own tensor types, not by a version guess,
    # and falls back to the pin whenever nothing registered fits — so a machine that never
    # registered anything behaves exactly as before.
    #
    # `_engine_binary` is kept out of `state["engine"]`: that field is the engine RUNTIME
    # (llamacpp/vllm), a string `bench`/`hwid` compare for equality when attributing
    # calibration, so a description there would corrupt provenance. It is handed to
    # `write_state` at the end of the launch instead, because that call rebuilds the record
    # from its arguments and would drop anything set on `s` here.
    #
    # C3: `engine_binary_for_plan` also freezes the chosen binary's fork identity onto
    # `rp`, so the argv this switch launches carries a fork-only flag only when the
    # binary that will actually run is the fork.
    exe, _engine_binary = engine_binary_for_plan(rp, os_name)
    port = int(s["public_port"]) - 1
    st.kill_recorded(s, "engine_pid")   # AUDIT F08-1: identity-checked
    if not _await_port_free(port):      # Windows TIME_WAIT grace
        # R3-SRV-1: this used to be a no-op on Windows (see the docstring), so a
        # port held by something we did not kill was never noticed here. The
        # auto-calibration below spends minutes; failing now is better than
        # failing after it.
        raise RuntimeError(
            f"port {port + 1} is still in use after waiting — something other "
            f"than the engine Rigma stopped is holding it. Close whatever is "
            f"using that port, or launch on another one.")
    # First load of a never-seen model+quant: with the old engine already dead,
    # VRAM is free — auto-tune the hardware-specific toggles ONCE, then launch
    # the winner. Cached forever after. Skipped for ctx-relaunches (a deliberate
    # reconfigure, not a fresh load), CPU, and when disabled.
    from .bench import auto_calibrate, is_calibrated
    if (ctx is None and rp.backend != "cpu"
            and os.environ.get("RIGMA_AUTO_CALIBRATE", "1") != "0"
            and (force_calibrate
                 or not is_calibrated(rp.model_slug, rp.gguf.quant, rp.backend))):
        _write_calib_marker(rp.model_slug, "starting")
        try:
            rp = auto_calibrate(rp, exe, model_path, port=port, extra_args=extra,
                                progress=lambda lbl:
                                _write_calib_marker(rp.model_slug, lbl))
        except Exception:
            pass   # tuning is best-effort; fall through to a normal launch
        finally:
            _clear_calib_marker()
        if not _await_port_free(port):   # last trial engine released the port
            # R3-SRV-1: say WHICH port and why, here, rather than letting the
            # launch fail below with a message about the engine. The old code
            # returned silently whether or not the port came free.
            raise RuntimeError(
                f"port {port + 1} is still in use after waiting — something other "
                f"than the engine Rigma stopped is holding it. Close whatever is "
                f"using that port, or launch on another one.")
    try:
        sp = runtime.launch_server(exe, rp, model_path,
                                   port=int(s["public_port"]) - 1,
                                   extra_args=extra)
    except Exception:
        # old engine is gone but the UI is still up — record an unloaded
        # state (not clear) so the UI stays manageable and can retry a load
        # AUDIT F22: docs/audit-2026-09-04-full.md — merge, never rebuild. The
        # whole-record form dropped kv_fp here, orphaning a prompt cache that
        # costs four minutes of prefill to rebuild, plus no_vision and kv_cache.
        st.update_state(engine_pid=-1,
                        ui_pid=int(s.get("ui_pid", os.getpid())),
                        unloaded=True)
        raise
    # Bring back the prompt cache if one was saved under EXACTLY this
    # configuration. A 120K window costs about four minutes of prefill to
    # rebuild and a couple of seconds to read off disk.
    #
    # A8 (docs/review/findings-r3 row 7): the result used to be discarded —
    # `kvcache.restore(...)` under `except Exception: pass`. That `except` never
    # saw the ordinary failure, because `restore` does not raise when the engine
    # refuses a slot load: it RETURNS `(False, reason)`. So the reason was lost
    # and the user paid the four minutes with nothing said and no trace to
    # diagnose. Both halves are fixed here: the reason is logged, and it is
    # surfaced on the launch output (`notice`, the channel `step_down_notice`
    # already uses, which `/api/server/switch` returns as JSON) and in the plan's
    # own explanation.
    #
    # `kv_fp` is still written when the restore fails, deliberately, and that is
    # a correction to the finding's proposed remedy. `kv_fp` is not a claim that
    # a restore happened: it is the key this engine's cache is SAVED under at
    # unload (`perform_unload`) and the name prefix snapshots are filed under
    # (`serve._prefix_ctx`). Withholding it would skip the unload save, leaving
    # an unreadable kv-<fp>.bin in place forever — every later launch would fail
    # the same restore, write no fingerprint, and skip the save again, which is
    # precisely the repeated silent re-prefill the finding describes. Keeping
    # the key is what lets the unload overwrite the bad file with the live slot.
    # Nothing anywhere reads a recorded `kv_fp` to decide that the slot is warm
    # and skip prefill, so the recorded value cannot cause the cold start; the
    # silence could.
    from . import kvcache
    kv_fp = kvcache.launch_fingerprint(rp, exe)
    kv_err: str | None = None
    try:
        _restored, kv_err = kvcache.restore(int(s["public_port"]) - 1,
                                           runtime.rigma_home() / "sessions",
                                           kv_fp)
    except Exception as e:          # restore() reports, it does not raise
        kv_err = str(e)[:200]
    if kv_err:
        # "about four minutes" is the measurement in kvcache's module docstring
        # (120K window, ~560 t/s prefill on this machine), not a new claim.
        cold = (f"the saved prompt cache for this exact configuration would not "
                f"load ({kv_err}) — this conversation will be re-prefilled from "
                f"zero, about four minutes on a 120K window. The cache is "
                f"rewritten when the model is next unloaded.")
        _log.warning("kv-cache restore failed for %s [%s]: %s — re-prefilling "
                     "from zero", rp.model_slug, kv_fp, kv_err)
        rp.explain.append(cold)
        notice = f"{notice}\n{cold}" if notice else cold
    st.write_state(rp.model_slug, rp.gguf.quant, int(s["public_port"]),
                   engine_pid=sp.proc.pid,
                   ui_pid=int(s.get("ui_pid", os.getpid())),
                   backend=rp.backend, use_case=s.get("use_case", "general"),
                   ctx=rp.flags.ctx, kv_cache=rp.flags.cache_type_k or "",
                   no_vision=not vision, gguf=rp.gguf.file, kv_fp=kv_fp,
        # Carried through so `rigma status` can say which binary is serving. Set above
        # from the selection that actually ran, so it cannot drift from the exe used.
        engine_binary=_engine_binary,
        # DR2-1-res: the placement THIS launch applied, so the VRAM axis can make
        # the plan's prediction device-side instead of assuming "fully resident".
        # `rp` is the final plan (post auto-calibration), so this is what ran.
        placement=plan_placement(rp))
    out = st.read_state() or {}
    if notice:
        # Transient, in the response only: a switch that stepped the cache down
        # must say so where the caller can see it, without writing a key that
        # every later read would have to know to clear.
        out = {**out, "notice": notice}
    return out


def _await_port_free(port: int, tries: int = 10, delay: float = 0.3) -> None:
    """After killing the old engine, its port lingers briefly on Windows; wait
    for it to free before relaunching to avoid a bind crash.

    R3-SRV-1: this set SO_REUSEADDR on the probe socket, which defeats the probe
    it is used for on Windows. There the flag means "allow binding a port another
    socket is already bound to" (the BSD meaning is the narrower "skip TIME_WAIT"),
    so the bind SUCCEEDED while the old engine still held the port and the wait
    returned immediately — exactly the case it exists to catch. The flag is for a
    listener that wants to rebind quickly, not for a test of whether a port is
    free; without it the bind fails while the port is held and succeeds once it is
    released, which is the question being asked.

    It also used to return silently after the last attempt, so a port still held
    by something Rigma did not kill produced a launch that failed later with a
    message about the engine rather than about the port. It still does not raise
    — the wait is a courtesy, not a promise — but it now returns whether the port
    came free, so the caller can say so instead of letting the next failure be
    misattributed.
    """
    import socket
    import time
    for attempt in range(tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sk:
            try:
                sk.bind(("127.0.0.1", port))
                return True
            except OSError:
                if attempt < tries - 1:
                    time.sleep(delay)
    return False


def perform_unload() -> dict:
    """Stop the engine to free VRAM/RAM; the UI and state stay up so the
    model can be reloaded (or another one launched) with one click."""
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if s.get("unloaded"):
        raise RuntimeError("engine is already unloaded")
    # Write the conversation's KV cache out BEFORE the process dies. Everything
    # llama-server caches lives in that process; unloading to free the card is
    # otherwise paid for with a full re-prefill on the way back.
    if s.get("kv_fp"):
        from . import kvcache, runtime
        sessions = runtime.rigma_home() / "sessions"
        try:
            kvcache.save(int(s["public_port"]) - 1, sessions, s["kv_fp"],
                         meta={k: s.get(k) for k in ("model", "quant", "gguf",
                                                     "backend", "ctx")})
            kvcache.prune(sessions)
        except Exception:
            pass      # never let a cache write block freeing the GPU
    st.kill_recorded(s, "engine_pid")   # AUDIT F08-1: identity-checked
    # AUDIT F22: docs/audit-2026-09-04-full.md — an unload changes exactly two
    # things: the engine is gone, and the record says so. Everything else is the
    # configuration the user chose and must survive. Rebuilding the record from
    # write_state's defaults reverted no_vision to False, so the next load
    # re-attached the projector and ate the VRAM the unload was for.
    return st.update_state(engine_pid=-1,
                           ui_pid=int(s.get("ui_pid", os.getpid())),
                           unloaded=True)


def perform_load(registry=None, profile=None) -> dict:
    """Relaunch the model recorded in an unloaded state."""
    from . import state as st
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    if not s.get("unloaded"):
        raise RuntimeError(f"{s['model']} is already loaded")
    return perform_switch(s["model"], registry, profile)


def perform_recalibrate(registry=None, profile=None) -> dict:
    """Forget the running model's tune and re-optimize it now (unload -> quick
    sweep -> launch the fresh winner). For when a noisy measurement crowned a
    config that's actually slower — the sweep always includes baseline, so a
    re-tune can only match or beat plain defaults."""
    from . import state as st
    from .bench import clear_calibration
    s = st.read_state()
    if s is None:
        raise RuntimeError("not running")
    clear_calibration(s["model"], s["quant"], s.get("backend", "unknown"))
    return perform_switch(s["model"], registry, profile, force_calibrate=True)
