import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from rigma.bench import (
    BenchResult,
    load_calibration,
    run_bench,
    save_calibration,
    verdict,
)


@pytest.fixture
def oai_server():
    fake = Path(__file__).parent / "fake_oai_server.py"
    proc = subprocess.Popen([sys.executable, str(fake), "--port", "11598"])
    for _ in range(50):
        try:
            if httpx.get("http://127.0.0.1:11598/health", timeout=1).status_code == 200:
                break
        except Exception:
            time.sleep(0.1)
    yield 11598
    proc.terminate()


def test_run_bench_reads_timings(oai_server):
    r = run_bench(oai_server, prompt_tokens=2048, gen_tokens=128)
    assert r.pp_tps == 650.0 and r.tg_tps == 55.5
    assert r.prompt_tokens == 2048 and r.gen_tokens == 128


def test_calibration_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    save_calibration("m:q:vulkan", {"tg_tps": 57.1}, flags={"n_cpu_moe": 8})
    cal = load_calibration()
    assert cal["m:q:vulkan"]["measured"]["tg_tps"] == 57.1
    assert cal["m:q:vulkan"]["flags"]["n_cpu_moe"] == 8
    assert cal["m:q:vulkan"]["date"]


def test_verdict():
    r = BenchResult(pp_tps=650, tg_tps=55.5, prompt_tokens=2048, gen_tokens=128)
    assert "OK" in verdict(r, {"tg_tps": [40, 60]})
    assert "BELOW" in verdict(r, {"tg_tps": [70, 90]})
    assert "no expectation" in verdict(r, None)
