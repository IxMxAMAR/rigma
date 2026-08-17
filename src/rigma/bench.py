from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

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
    t = r.json().get("timings", {})
    return BenchResult(pp_tps=float(t.get("prompt_per_second", 0.0)),
                       tg_tps=float(t.get("predicted_per_second", 0.0)),
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


def save_calibration(key: str, measured: dict, flags: dict | None = None,
                     calibrated: bool = False, ctx: int = 0) -> None:
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
    entry["schema"] = 2
    entry["engine"] = _engine_version()
    if ctx:
        entry["ctx"] = ctx
    entry["date"] = datetime.date.today().isoformat()
    cal[key] = entry
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(cal, indent=2), encoding="utf-8")


def is_calibrated(model: str, quant: str, backend: str) -> bool:
    """True once a model+quant+backend has been auto-tuned on this machine —
    so first-load calibration runs exactly once, never on every load."""
    return bool(load_calibration().get(f"{model}:{quant}:{backend}", {})
                .get("calibrated"))


def clear_calibration(model: str, quant: str, backend: str) -> bool:
    """Forget one model's tune so it re-optimizes on next load (or falls back to
    the safe defaults). Returns True if there was an entry to clear."""
    cal = load_calibration()
    if cal.pop(f"{model}:{quant}:{backend}", None) is None:
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
        key = f"{plan.model_slug}:{plan.gguf.quant}:{plan.backend}"
        save_calibration(key, {"tg_tps": best["tg_tps"], "pp_tps": best["pp_tps"]},
                         flags=best["flags"], calibrated=mark_calibrated,
                         ctx=plan.flags.ctx)
    return rows


def auto_calibrate(plan: RunPlan, exe, model_path, port: int = 11601,
                   extra_args=None, progress=None) -> RunPlan:
    """One-time, first-load tune: if this model+quant+backend has never been
    calibrated on this machine, A/B the quick config set on `port` and return
    the plan with the winning flags applied. Cached forever after (subsequent
    loads return instantly). No-op on CPU or when already calibrated."""
    key = f"{plan.model_slug}:{plan.gguf.quant}:{plan.backend}"
    entry = load_calibration().get(key, {})

    def _apply(p: RunPlan) -> RunPlan:
        flags = load_calibration().get(key, {}).get("flags") or {}
        if not flags:
            return p
        return p.model_copy(update={
            "flags": p.flags.model_copy(update=flags),
            "origin": p.origin if p.origin.endswith("+calibrated")
            else p.origin + "+calibrated"})

    if entry.get("calibrated"):
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
