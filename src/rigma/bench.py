from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

from . import hwid
from .models import ComboFlags, RunPlan
from .runtime import launch_server, rigma_home


class BenchResult(BaseModel):
    pp_tps: float
    tg_tps: float
    prompt_tokens: int
    gen_tokens: int


def run_bench(port: int, prompt_tokens: int = 2048, gen_tokens: int = 128) -> BenchResult:
    filler = "The quick brown fox jumps over the lazy dog. " * (prompt_tokens // 8)
    r = httpx.post(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        json={"messages": [{"role": "user",
                            "content": filler + "\nSummarize in one sentence."}],
              "max_tokens": gen_tokens},
        timeout=1800)
    r.raise_for_status()
    t = r.json().get("timings")
    # AUDIT F08-7: "the engine answered but did not report timings" used to be
    # collapsed into "the engine measured 0 tokens/s", with ok=True set from "no
    # exception". A build/proxy that omits timings then made every sweep row read
    # 0.0 while looking successful — and `crowned_row` could crown one, writing
    # its flags and `calibrated: true` so the model was tuned forever on no data.
    if not t:
        raise RuntimeError("engine returned no timings — cannot measure speed")
    pp = float(t.get("prompt_per_second", 0.0))
    tg = float(t.get("predicted_per_second", 0.0))
    if pp == 0.0 and tg == 0.0:
        raise RuntimeError("engine reported 0 tokens/s for both prefill and "
                           "generation — no usable measurement")
    return BenchResult(pp_tps=pp, tg_tps=tg,
                       prompt_tokens=prompt_tokens, gen_tokens=gen_tokens)


def _capabilities(slug: str) -> tuple:
    try:
        from .registry import Registry
        spec = Registry.load().models.get(slug)
        if spec is not None:
            return tuple(spec.capabilities)
    except Exception:
        pass
    return ()


def _sweepable_caps(plan) -> tuple:
    """Capabilities the sweep may act on, with `mtp` decided by the FILE.

    A sweep runs real llama-server launches, so a capability that the model
    advertises but this quant does not carry is not a wasted row — it is a
    driver reset mid-sweep. The gguf being benched is on disk by definition,
    so the tensor table can always be consulted."""
    caps = set(_capabilities(plan.model_slug))
    caps.discard("mtp")
    try:
        from .hangar import file_has_mtp
        if file_has_mtp(plan.gguf) is True:
            caps.add("mtp")
    except Exception:
        pass          # unverifiable stays out: absence of proof is not proof
    return tuple(caps)


def _tools_capable(slug: str) -> bool:
    """Whether the model declares the `tools` capability. Unknown → True:
    assume tools and protect quality rather than risk degrading it."""
    try:
        from .registry import Registry
        spec = Registry.load().models.get(slug)
        if spec is not None:
            return "tools" in spec.capabilities
    except Exception:
        pass
    return True


def calibration_path() -> Path:
    return rigma_home() / "calibration.json"


def load_calibration() -> dict:
    try:
        return json.loads(calibration_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def current_identity(backend: str = "", gpu=None) -> hwid.HardwareIdentity:
    """The hardware this process is running on, for keying a calibration.

    Never raises: an identity that cannot be read is an EMPTY one, which degrades
    to the old model:quant:backend behaviour rather than losing the calibration
    entirely. A probe failure must not make the tool refuse to measure.

    Reads the raw enumeration directly rather than going through
    `probe_hardware()`, which needs the registry's GPU table. The table only picks
    which BACKENDS a card advertises, and the backend is passed in here already —
    so the table would add a registry dependency to a cache key and change nothing
    about the identity itself.
    """
    try:
        if gpu is None:
            from .probe import enumerate_vulkan
            rows = enumerate_vulkan()
            gpu = rows[0] if rows else None
        if gpu is None:
            return hwid.HardwareIdentity(backend=backend or "cpu")
        if isinstance(gpu, dict):
            return hwid.identity_from_gpu(backend or "vulkan", gpu)
        return hwid.identity_from_gpu(backend or "vulkan", {
            "vendor_id": getattr(gpu, "vendor_id", None),
            "device_id": getattr(gpu, "device_id", None),
            "device_uuid": getattr(gpu, "device_uuid", ""),
            "driver_version": getattr(gpu, "driver_version", ""),
            "name": getattr(gpu, "name", ""),
        })
    except Exception:
        return hwid.HardwareIdentity(backend=backend or "")


def _identity_cache_key(backend: str) -> hwid.HardwareIdentity:
    """Cached per backend: `probe_hardware()` is a subprocess away from slow, and
    the calibration key is built on several hot paths.

    An EMPTY identity is deliberately not cached. A transient probe failure would
    otherwise be frozen for the life of the process, and every calibration key
    built afterwards would silently lose its hardware component — the exact
    silent-degradation this module exists to prevent.
    """
    got = _IDENT_CACHE.get(backend)
    if got is not None:
        return got
    ident = current_identity(backend)
    if ident.vendor_id or ident.uuid:
        _IDENT_CACHE[backend] = ident
    return ident


_IDENT_CACHE: dict[str, hwid.HardwareIdentity] = {}


def calibration_key(model: str, quant: str, backend: str) -> str:
    """The current calibration key for a model+quant+backend."""
    return hwid.calibration_key(model, quant, _identity_cache_key(backend))


def legacy_key(model: str, quant: str, backend: str) -> str:
    """The key used before R3-CAL-1: model:quant:backend, no hardware.

    Kept because entries written under it are still valid ON THE MACHINE THAT
    WROTE THEM — discarding every existing calibration on upgrade would make the
    tool worse for the person upgrading, which is not what fixing a cache key is
    supposed to do.
    """
    return f"{model}:{quant}:{backend}"


def calibration_entry(cal: dict, model: str, quant: str,
                      backend: str) -> tuple[str, dict]:
    """The entry for a model+quant+backend, and the key it was found under.

    Tries the identity key, then the legacy key. Returns ("", {}) on a miss, which
    callers already treat as "not calibrated".
    """
    for k in (calibration_key(model, quant, backend), legacy_key(model, quant, backend)):
        if k in cal:
            return k, cal[k]
    return "", {}


def prune_calibration(cal: dict, keep_per_identity: int = 1) -> dict:
    """Keep only the newest `keep_per_identity` entries for each hardware identity.

    A single machine has 1-4 identities (iGPU + dGPU, plus a backend each), so this
    is about not letting the file grow without bound across GPU swaps and driver
    experiments. Entries with no recorded hardware are kept — they are from before
    identity existed and cannot be attributed to an identity to prune.
    """
    by_ident: dict[str, list[tuple[str, str]]] = {}
    keep: set[str] = set()
    for k, entry in cal.items():
        ident = (entry.get("hardware") or {}).get("id")
        if not ident:
            keep.add(k)
            continue
        by_ident.setdefault(ident, []).append((entry.get("date") or "", k))
    for rows in by_ident.values():
        rows.sort(reverse=True)          # newest date first, then key
        keep.update(k for _, k in rows[:keep_per_identity])
    return {k: v for k, v in cal.items() if k in keep}


def save_calibration(key: str, measured: dict, flags: dict | None = None,
                     calibrated: bool = False, ctx: int = 0,
                     identity: hwid.HardwareIdentity | None = None) -> None:
    cal = load_calibration()
    entry = cal.get(key, {})
    entry["measured"] = measured
    if flags is not None:
        entry["flags"] = flags
    if calibrated:
        entry["calibrated"] = True   # one-time first-load tune has run
    # What the numbers were measured ON. An entry used to carry a day-granularity
    # date and nothing else, so there was no way to tell that a calibration
    # predated an engine bump or was measured at a different context — it simply
    # kept being applied. `schema` marks entries that carry this; anything
    # without it is from before and is read leniently.
    entry["schema"] = 3
    entry["engine"] = _engine_version()
    # R3-CAL-1: WHICH CARD. Without it a 3090 silently inherits a 4090's number,
    # which is the failure mode that is invisible rather than loud. The identity is
    # recorded as a field as well as being part of the key so a stale entry can
    # explain itself instead of just being absent.
    if identity is not None:
        entry["hardware"] = identity.as_dict()
    # What the desktop was holding when this was measured. Without it there is
    # nothing in the entry to distinguish 9.95 tok/s from 37.59 for the same
    # model on the same engine (measured 2026-08-21).
    try:
        from .probe import gpu_used_mb
        used = gpu_used_mb()
        if used is not None:
            entry["vram_used_mb"] = round(used)
    except Exception:
        pass
    if ctx:
        entry["ctx"] = ctx
    entry["date"] = datetime.date.today().isoformat()
    cal[key] = entry
    cal = prune_calibration(cal)
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(cal, indent=2), encoding="utf-8")


# How much the desktop's VRAM footprint may drift before a calibration stops
# meaning anything. Measured 2026-08-21: 2,828MB of drift changed the same model
# on the same engine from 9.95 to 37.59 tok/s, because Windows silently paged a
# third of the weights to system RAM. A few hundred MB is ordinary desktop
# noise; anything more changes what the number means.
VRAM_DRIFT_TOLERANCE_MB = 700


def calibration_stale(entry: dict, vram_used_mb: float | None,
                      engine: str = "", ctx: int = 0,
                      identity: hwid.HardwareIdentity | None = None) -> str | None:
    """Why this calibration should not be trusted, or None if it should.

    Speed on a fixed model+quant+engine is not a constant: it depends on whether
    the weights actually fit in VRAM, and on Windows that depends on what ELSE
    is holding VRAM. A calibration taken while a browser held 4GB describes a
    machine that no longer exists once the browser closes.

    R3-CAL-1 adds two more ways to be stale, and they are NOT the same:
      * a different CARD (hard) — the number is about other hardware entirely;
      * the same card under a new DRIVER (soft) — real, but a re-measure rather
        than a discard. A driver update made non-FA PP 5% and FA 15% faster.
    Both are reported here as reasons not to trust the number, which is the one
    thing callers do with them.
    """
    if not entry:
        return None
    if identity is not None:
        hard = hwid.hard_mismatch(entry, identity)
        if hard:
            return hard
        soft = hwid.soft_reasons(entry, identity, engine=engine, ctx=ctx)
        if soft:
            return "; ".join(soft)
        # Identity already covered engine/ctx/driver, so fall through only to the
        # VRAM check rather than repeating the same comparisons below.
        was = entry.get("vram_used_mb")
        if was is None or vram_used_mb is None:
            return None
        drift = abs(float(vram_used_mb) - float(was))
        if drift > VRAM_DRIFT_TOLERANCE_MB:
            return (f"measured with {was:,.0f} MiB of VRAM held by other apps, "
                    f"now {vram_used_mb:,.0f} MiB — a {drift:,.0f} MiB shift changes "
                    "whether the model fits on the GPU at all")
        return None
    if engine and entry.get("engine") and entry["engine"] != engine:
        return f"measured on engine {entry['engine']}, now on {engine}"
    if ctx and entry.get("ctx") and entry["ctx"] != ctx:
        return f"measured at ctx {entry['ctx']:,}, now {ctx:,}"
    was = entry.get("vram_used_mb")
    # Entries written before this was recorded, and machines where it cannot be
    # read, are read leniently: no reading is not a reading of zero, and
    # invalidating every old entry at once helps nobody.
    if was is None or vram_used_mb is None:
        return None
    drift = abs(float(vram_used_mb) - float(was))
    if drift > VRAM_DRIFT_TOLERANCE_MB:
        return (f"measured with {was:,.0f} MiB of VRAM held by other apps, "
                f"now {vram_used_mb:,.0f} MiB — a {drift:,.0f} MiB shift changes "
                "whether the model fits on the GPU at all")
    return None


def is_calibrated(model: str, quant: str, backend: str) -> bool:
    """True once a model+quant+backend has been auto-tuned on THIS HARDWARE —
    so first-load calibration runs exactly once, never on every load.

    R3-CAL-1: the lookup is identity-aware, so a calibration measured on another
    GPU does not count as calibrated here. The legacy key is still honoured for
    the machine that wrote it, which is why this is not simply a key change.
    """
    _, entry = calibration_entry(load_calibration(), model, quant, backend)
    if not entry:
        return False
    # A calibrated flag from different hardware is not a calibration for this one:
    # re-tune rather than serve a number measured elsewhere.
    if hwid.hard_mismatch(entry, _identity_cache_key(backend)):
        return False
    return bool(entry.get("calibrated"))


def clear_calibration(model: str, quant: str, backend: str) -> bool:
    """Forget one model's tune so it re-optimizes on next load (or falls back to
    the safe defaults). Returns True if there was an entry to clear.

    Clears BOTH the identity key and the legacy key: "forget this tune" should mean
    forgotten, and leaving a legacy entry behind would make the next lookup find it
    and report the model as still calibrated.
    """
    cal = load_calibration()
    hit = [k for k in (calibration_key(model, quant, backend),
                       legacy_key(model, quant, backend)) if cal.pop(k, None) is not None]
    if not hit:
        return False
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(cal, indent=2), encoding="utf-8")
    return True


def reset_all_calibration() -> int:
    """Wipe every stored tune. Returns how many were cleared."""
    n = len(load_calibration())
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text("{}", encoding="utf-8")
    return n


SPEC_CROWN_MARGIN = 1.15   # a speculation config must beat the best
                           # non-spec row by >=15% to be crowned: the short
                           # predictable-text bench flatters draft acceptance,
                           # so a narrow bench win is a real-world loss


def rows_log_path() -> Path:
    return rigma_home() / "logs" / "bench-rows.jsonl"


def _log_row(entry: dict) -> None:
    """One measurement, appended. Separated so a test can make it fail."""
    p = rows_log_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, default=str) + "\n")


def _log_rows(plan, rows: list[dict], best: dict | None) -> None:
    """Keep every measurement a sweep made, not only the one it crowned.

    A sweep loads a real engine per config and benches it — the most expensive
    numbers Rigma produces. Only the winner's two floats reached
    calibration.json; every losing row was returned, printed once, and dropped.
    And because calibration is only written when the winner has flags, an
    explicit sweep where the BASELINE won recorded nothing at all, not even the
    baseline speed it had just spent several model loads measuring.

    Append-only, structure-only, and wrapped: this is bookkeeping, and a sweep
    that lost its log is still a sweep. Same shape as serve._shape_log.
    """
    try:
        won = (best or {}).get("label")
        stamp = datetime.date.today().isoformat()
        for r in rows:
            _log_row({"date": stamp, "model": plan.model_slug,
                      "quant": plan.gguf.quant, "backend": plan.backend,
                      "ctx": plan.flags.ctx, "engine": _engine_version(),
                      "label": r.get("label"), "flags": r.get("flags") or {},
                      "tg_tps": r.get("tg_tps"), "pp_tps": r.get("pp_tps"),
                      "ok": bool(r.get("ok")), "error": r.get("error", ""),
                      "crowned": r.get("label") == won})
    except Exception:
        pass          # a sweep that lost its log is still a sweep


def _engine_version() -> str:
    """Which llama.cpp build produced these numbers. A calibration measured on
    one engine is not evidence about another, and nothing recorded this."""
    try:
        from .server_ops import engine_version
        return engine_version()
    except Exception:
        return ""


def crowned_row(rows: list[dict]) -> dict | None:
    """The config a sweep actually crowns. ONE rule, two readers.

    `run_sweep` saves the winner; `rigma sweep` prints it. They used to decide
    separately, and the CLI's version had no margin gate and required non-empty
    flags — so it could announce "winner: spec-mtp-2 … saved to calibration"
    while calibration.json held the baseline.

    The margin gate: real-world draft acceptance is LOWER than on the bench's
    predictable filler (live 2026-07-21: crowned at 44.7 t/s, ran at 36.8). A
    speculation config must beat the best non-spec row decisively or the
    non-spec row is crowned instead.

    Sorts a copy, so the answer does not depend on whether the caller sorted.
    """
    ok = sorted((r for r in rows if r.get("ok")),
                key=lambda r: r.get("tg_tps", 0.0), reverse=True)
    best = next(iter(ok), None)
    if best is not None and (best.get("flags") or {}).get("spec_type"):
        plain = next((r for r in ok
                      if not (r.get("flags") or {}).get("spec_type")), None)
        if plain is not None and \
                best["tg_tps"] < plain["tg_tps"] * SPEC_CROWN_MARGIN:
            best = plain
    return best


def sweep_configs(base: ComboFlags, moe: bool,
                  caps: tuple | list = ()) -> list[tuple[str, dict]]:
    """Flag-override sets to A/B on this machine. Baseline first; each entry is
    a partial ComboFlags update. Axes come from the RDNA4 findings: FA gates the
    fast KV path, symmetric KV precision, prefill batch, Vulkan coopmat, and
    (MoE only) graphics-queue + offload depth."""
    cfgs: list[tuple[str, dict]] = [("baseline", {})]
    cfgs.append(("fa-off", {"flash_attn": "off"}))
    cfgs.append(("kv-q8", {"cache_type_k": "q8_0", "cache_type_v": "q8_0"}))
    cfgs.append(("kv-q4", {"cache_type_k": "q4_0", "cache_type_v": "q4_0"}))
    cfgs.append(("batch-big", {"batch": 16384, "ubatch": 2048}))
    cfgs.append(("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}}))
    if moe:
        cfgs.append(("gfxqueue-on", {"env": {"GGML_VK_ALLOW_GRAPHICS_QUEUE": "1"}}))
        if base.n_cpu_moe > 0:
            cfgs.append(("moe-less-offload",
                         {"n_cpu_moe": max(0, base.n_cpu_moe - 1)}))
    # speculation trials: exhaustive sweep ONLY, and gated by
    # SPEC_CROWN_MARGIN in run_sweep — the short bench flatters acceptance.
    # `caps` here is the FILE's answer (see run_sweep), not the model's: MTP is
    # dropped or kept per quant, and trialling draft-mtp against a quant without
    # the tensors resets the Vulkan driver mid-sweep.
    if "mtp" in (caps or ()):
        cfgs.append(("spec-mtp-2", {"spec_type": "draft-mtp", "spec_n_max": 2}))
        cfgs.append(("spec-mtp-4", {"spec_type": "draft-mtp", "spec_n_max": 4}))
    return cfgs


def quick_configs(base: ComboFlags, moe: bool,
                  caps: tuple | list = ()) -> list[tuple[str, dict]]:
    """The short first-load set: only the toggles that genuinely can't be
    defaulted and are worth a per-machine measurement. The rest are already
    applied automatically by the resolver. Full exhaustive set = sweep_configs
    (via `rigma sweep`)."""
    cfgs: list[tuple[str, dict]] = [("baseline", {})]
    cfgs.append(("fa-off", {"flash_attn": "off"}))
    cfgs.append(("coopmat-off", {"env": {"GGML_VK_DISABLE_COOPMAT": "1"}}))
    if moe:
        cfgs.append(("gfxqueue-on", {"env": {"GGML_VK_ALLOW_GRAPHICS_QUEUE": "1"}}))
    # NOTE: spec-mtp trials deliberately NOT in the auto first-load set.
    # Live lesson 2026-07-21: the 96-token bench summarises highly
    # predictable filler, which inflates MTP draft acceptance — draft-mtp
    # measured 44.7 t/s and was crowned, then real varied chat ran at 36.8
    # (rejected drafts are pure overhead). Speculation lives in the explicit
    # exhaustive sweep, where the margin gate below applies.
    return cfgs


def run_sweep(plan: RunPlan, exe, model_path, port: int = 11601,
              prompt_tokens: int = 2048, gen_tokens: int = 96,
              progress=None, configs=None, extra_args=None,
              mark_calibrated: bool = False) -> list[dict]:
    """Launch `plan` under each config on `port`, bench it, and persist the best
    tg/s config to calibration (which resolve() then applies automatically). The
    caller guarantees `port` is free (scratch port, or mid-switch with the old
    engine already killed) — this never touches a live server."""
    is_moe = plan.flags.n_cpu_moe > 0
    if configs is None:
        configs = sweep_configs(plan.flags, is_moe,
                                caps=_sweepable_caps(plan))
    # Never let the sweep crown q4_0 KV on a tools-capable model: llama.cpp's
    # own function-calling docs warn extreme KV quantization significantly
    # degrades tool calling, and the sweep scores tokens/sec only — it would
    # trade a silent quality regression for a speed win. (Mirrors the
    # registry's DeltaNet q8_0 cache policy.)
    if _tools_capable(plan.model_slug):
        configs = [(label, o) for label, o in configs
                   if o.get("cache_type_k") != "q4_0"
                   and o.get("cache_type_v") != "q4_0"]
    rows: list[dict] = []
    for label, override in configs:
        flags = plan.flags.model_copy(update=override)
        trial = plan.model_copy(update={"flags": flags})
        if progress:
            progress(label)
        try:
            srv = launch_server(exe, trial, model_path, port=port, timeout=300.0,
                                extra_args=extra_args)
        except Exception as e:  # a config that OOMs/crashes is a valid "loss"
            rows.append({"label": label, "flags": override, "tg_tps": 0.0,
                         "pp_tps": 0.0, "ok": False, "error": str(e)[:200]})
            continue
        try:
            res = run_bench(port, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens)
            rows.append({"label": label, "flags": override, "tg_tps": res.tg_tps,
                         "pp_tps": res.pp_tps, "ok": True})
        except Exception as e:  # loaded but wouldn't serve — count as a loss
            rows.append({"label": label, "flags": override, "tg_tps": 0.0,
                         "pp_tps": 0.0, "ok": False, "error": str(e)[:200]})
        finally:
            srv.stop()
    rows.sort(key=lambda r: r["tg_tps"], reverse=True)
    best = crowned_row(rows)
    _log_rows(plan, rows, best)
    if best is not None and (best["flags"] or mark_calibrated):
        key = calibration_key(plan.model_slug, plan.gguf.quant, plan.backend)
        save_calibration(key, {"tg_tps": best["tg_tps"], "pp_tps": best["pp_tps"]},
                         flags=best["flags"], calibrated=mark_calibrated,
                         ctx=plan.flags.ctx,
                         identity=_identity_cache_key(plan.backend))
    return rows


def auto_calibrate(plan: RunPlan, exe, model_path, port: int = 11601,
                   extra_args=None, progress=None) -> RunPlan:
    """One-time, first-load tune: if this model+quant+backend has never been
    calibrated on this machine, A/B the quick config set on `port` and return
    the plan with the winning flags applied. Cached forever after (subsequent
    loads return instantly). No-op on CPU or when already calibrated."""
    # Read under whichever key has the entry (identity first, legacy second) but
    # always WRITE and re-read under the identity key: `run_sweep` persists to the
    # identity key, so reading flags back under a legacy key would miss the tune
    # that was just measured and silently apply nothing.
    key = calibration_key(plan.model_slug, plan.gguf.quant, plan.backend)
    ident = _identity_cache_key(plan.backend)
    _, entry = calibration_entry(load_calibration(), plan.model_slug,
                                plan.gguf.quant, plan.backend)
    # R3-CAL-1: an entry measured on other hardware is not a calibration for this
    # one. Ignore it (rather than erroring) and re-measure.
    if entry and hwid.hard_mismatch(entry, ident):
        entry = {}

    def _apply(p: RunPlan) -> RunPlan:
        flags = load_calibration().get(key, {}).get("flags") or {}
        if not flags:
            return p
        return p.model_copy(update={
            "flags": p.flags.model_copy(update=flags),
            "origin": p.origin if p.origin.endswith("+calibrated")
            else p.origin + "+calibrated"})

    if entry.get("calibrated"):
        # Adopt a legacy entry onto the identity key so the next lookup is a
        # direct hit and the stale key can be pruned away.
        if key not in load_calibration():
            save_calibration(key, entry.get("measured", {}),
                             flags=entry.get("flags"), calibrated=True,
                             ctx=entry.get("ctx", 0), identity=ident)
        return _apply(plan)
    if plan.backend == "cpu":
        return plan   # nothing worth measuring on CPU
    run_sweep(plan, exe, model_path, port=port,
              configs=quick_configs(plan.flags, plan.flags.n_cpu_moe > 0,
                                    caps=_capabilities(plan.model_slug)),
              extra_args=extra_args, progress=progress, mark_calibrated=True)
    return _apply(plan)


def verdict(result: BenchResult, expected: dict | None) -> str:
    if not expected or "tg_tps" not in expected:
        return "no expectation recorded for this combo"
    floor = expected["tg_tps"][0]
    if result.tg_tps >= floor:
        return f"OK (within/above expected range, floor {floor} t/s)"
    return (f"BELOW expected floor ({floor} t/s) — combo may need tuning "
            f"on this machine")
