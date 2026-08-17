import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest

from rigma.bench import (
    BenchResult,
    crowned_row,
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


# --- one crowning rule, used by both the sweep and the CLI --------------------
def _row(label, tg, flags=None, ok=True):
    return {"label": label, "flags": flags or {}, "tg_tps": tg,
            "pp_tps": 600.0, "ok": ok}


def test_crowned_row_demotes_a_narrow_speculation_win():
    """A spec config must beat the best non-spec row by SPEC_CROWN_MARGIN.
    The short bench runs predictable filler, which flatters draft acceptance:
    live 2026-07-21 crowned 44.7 t/s and then ran at 36.8."""
    rows = [_row("spec-mtp-2", 44.0, {"spec_type": "draft-mtp"}),
            _row("baseline", 40.0)]
    assert crowned_row(rows)["label"] == "baseline"   # 44.0 < 40.0 * 1.15


def test_crowned_row_keeps_a_decisive_speculation_win():
    rows = [_row("spec-mtp-4", 50.0, {"spec_type": "draft-mtp"}),
            _row("baseline", 40.0)]
    assert crowned_row(rows)["label"] == "spec-mtp-4"


def test_crowned_row_does_not_depend_on_input_order():
    """run_sweep sorts in place before crowning; the CLI reads the returned
    list. Sorting inside the rule means the two cannot disagree."""
    rows = [_row("baseline", 40.0), _row("fa-off", 47.0),
            _row("spec-mtp-2", 44.0, {"spec_type": "draft-mtp"})]
    assert crowned_row(rows)["label"] == "fa-off"


def test_crowned_row_ignores_failed_configs():
    rows = [_row("kv-q4", 99.0, {"cache_type_k": "q4_0"}, ok=False),
            _row("baseline", 40.0)]
    assert crowned_row(rows)["label"] == "baseline"


def test_crowned_row_is_none_when_everything_failed():
    assert crowned_row([_row("baseline", 0.0, ok=False)]) is None
    assert crowned_row([]) is None


def test_the_sweep_and_the_cli_report_the_same_winner():
    """The CLI used to recompute the winner with its own rule and no margin
    gate, so `rigma sweep` could print "winner: spec-mtp-2 ... saved to
    calibration" while calibration.json held the baseline."""
    rows = [_row("spec-mtp-2", 44.0, {"spec_type": "draft-mtp"}),
            _row("baseline", 40.0)]
    crowned = crowned_row(rows)
    cli_winner = next((r for r in rows if r["ok"] and r["flags"]), None)
    assert crowned is not None
    assert cli_winner is not None and cli_winner["label"] != crowned["label"], \
        "this test is worthless if the old and new rules already agree"
