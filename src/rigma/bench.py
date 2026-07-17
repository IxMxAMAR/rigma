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


def calibration_path() -> Path:
    return rigma_home() / "calibration.json"


def load_calibration() -> dict:
    try:
        return json.loads(calibration_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_calibration(key: str, measured: dict, flags: dict | None = None) -> None:
    cal = load_calibration()
    entry = cal.get(key, {})
    entry["measured"] = measured
    if flags is not None:
        entry["flags"] = flags
    entry["date"] = datetime.date.today().isoformat()
    cal[key] = entry
    calibration_path().parent.mkdir(parents=True, exist_ok=True)
    calibration_path().write_text(json.dumps(cal, indent=2), encoding="utf-8")


def sweep_configs(base: ComboFlags, moe: bool) -> list[tuple[str, dict]]:
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
    return cfgs


def run_sweep(plan: RunPlan, exe, model_path, port: int = 11601,
              prompt_tokens: int = 2048, gen_tokens: int = 96,
              progress=None) -> list[dict]:
    """Launch `plan` under each sweep config on a SCRATCH port, bench it, and
    persist the best tg/s config to calibration (which resolve() then applies
    automatically). Never touches the live server on 11499/11500."""
    is_moe = plan.flags.n_cpu_moe > 0
    rows: list[dict] = []
    for label, override in sweep_configs(plan.flags, is_moe):
        flags = plan.flags.model_copy(update=override)
        trial = plan.model_copy(update={"flags": flags})
        if progress:
            progress(label)
        try:
            srv = launch_server(exe, trial, model_path, port=port, timeout=300.0)
        except Exception as e:  # a config that OOMs/crashes is a valid "loss"
            rows.append({"label": label, "flags": override, "tg_tps": 0.0,
                         "pp_tps": 0.0, "ok": False, "error": str(e)[:200]})
            continue
        try:
            res = run_bench(port, prompt_tokens=prompt_tokens, gen_tokens=gen_tokens)
            rows.append({"label": label, "flags": override, "tg_tps": res.tg_tps,
                         "pp_tps": res.pp_tps, "ok": True})
        finally:
            srv.stop()
    rows.sort(key=lambda r: r["tg_tps"], reverse=True)
    best = next((r for r in rows if r["ok"]), None)
    if best and best["flags"]:
        key = f"{plan.model_slug}:{plan.gguf.quant}:{plan.backend}"
        save_calibration(key, {"tg_tps": best["tg_tps"], "pp_tps": best["pp_tps"]},
                         flags=best["flags"])
    return rows


def verdict(result: BenchResult, expected: dict | None) -> str:
    if not expected or "tg_tps" not in expected:
        return "no expectation recorded for this combo"
    floor = expected["tg_tps"][0]
    if result.tg_tps >= floor:
        return f"OK (within/above expected range, floor {floor} t/s)"
    return (f"BELOW expected floor ({floor} t/s) — combo may need tuning "
            f"on this machine")
