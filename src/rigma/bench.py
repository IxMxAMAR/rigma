from __future__ import annotations

import datetime
import json
from pathlib import Path

import httpx
from pydantic import BaseModel

from .runtime import rigma_home


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


def verdict(result: BenchResult, expected: dict | None) -> str:
    if not expected or "tg_tps" not in expected:
        return "no expectation recorded for this combo"
    floor = expected["tg_tps"][0]
    if result.tg_tps >= floor:
        return f"OK (within/above expected range, floor {floor} t/s)"
    return (f"BELOW expected floor ({floor} t/s) — combo may need tuning "
            f"on this machine")
