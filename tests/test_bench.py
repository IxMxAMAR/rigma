import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
from typer.testing import CliRunner

import rigma.bench as bench
import rigma.cli as cli
import rigma.runtime as runtime
from rigma.bench import (
    BenchResult,
    crowned_row,
    load_calibration,
    run_bench,
    save_calibration,
    verdict,
)
from rigma.models import ComboFlags, GgufFile, RunPlan


def _free_port() -> int:
    """An ephemeral loopback port the OS just told us was free.

    REC-1: this fixture used to hard-code 11598, and it is the ONLY server in
    the suite that binds a literal port — every other one passes 0 and lets the
    kernel choose. So two concurrent `pytest tests` runs collided here: the
    second run's child died on `address already in use`, the parent's /health
    poll was then answered by the FIRST run's server (so the fixture reported
    "ready" and the tests silently measured another process), and when the
    first run tore its server down the second blocked on an established socket
    to a server that had gone away — the frozen-suite / leaked-children failure
    recorded as REC-1. Binding an ephemeral port removes the shared resource.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def oai_server():
    fake = Path(__file__).parent / "fake_oai_server.py"
    # Retry on a fresh port: the child can still lose the race to another
    # process between our probe and its bind. What must NOT happen is falling
    # through to someone else's server, so a dead child is treated as "try
    # again", never as "ready".
    for _attempt in range(5):
        port = _free_port()
        proc = subprocess.Popen([sys.executable, str(fake), "--port", str(port)])
        ready = False
        for _ in range(50):
            if proc.poll() is not None:
                break  # the child could not bind — pick another port
            try:
                if httpx.get(f"http://127.0.0.1:{port}/health",
                             timeout=1).status_code == 200:
                    ready = True
                    break
            except Exception:
                time.sleep(0.1)
        if ready:
            try:
                yield port
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
    # the loop used to fall through silently, so a server that never came
    # up surfaced as a confusing connection error inside whichever test
    # happened to run first (AUDIT F60)
    pytest.fail("fake engine never answered /health on any free port")


def test_run_bench_reads_timings(oai_server):
    r = run_bench(oai_server, prompt_tokens=2048, gen_tokens=128)
    assert r.pp_tps == 650.0 and r.tg_tps == 55.5
    assert r.prompt_tokens == 2048 and r.gen_tokens == 128


class _Resp:
    """A minimal stand-in for the httpx response run_bench reads."""

    def __init__(self, body):
        self._body = body

    def raise_for_status(self):
        pass

    def json(self):
        return self._body


def test_run_bench_rejects_a_response_without_timings(monkeypatch):
    """AUDIT F08-7: "the engine answered but reported no timings" used to be
    recorded as a successful 0.0 t/s row, which could then be crowned and
    written to calibration as if it had been measured."""
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp({"choices": []}))
    with pytest.raises(RuntimeError, match="timings"):
        run_bench(11500, prompt_tokens=16, gen_tokens=8)


def test_run_bench_rejects_zero_rates(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(
        {"timings": {"prompt_per_second": 0.0, "predicted_per_second": 0.0}}))
    with pytest.raises(RuntimeError):
        run_bench(11500, prompt_tokens=16, gen_tokens=8)


def test_run_bench_still_accepts_real_timings(monkeypatch):
    monkeypatch.setattr(httpx, "post", lambda *a, **k: _Resp(
        {"timings": {"prompt_per_second": 650.0, "predicted_per_second": 55.5}}))
    r = run_bench(11500, prompt_tokens=16, gen_tokens=8)
    assert r.pp_tps == 650.0 and r.tg_tps == 55.5


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


def test_the_sweep_and_the_cli_report_the_same_winner(monkeypatch, tmp_path):
    """The CLI used to recompute the winner with its own rule and no margin
    gate, so `rigma sweep` could print "winner: spec-mtp-2 ... saved to
    calibration" while calibration.json held the baseline.

    Both REAL paths are driven here on the same fake rows: `bench.run_sweep`
    and the `rigma sweep` command through `typer.testing.CliRunner`. The
    fixture makes the old rule (first row with flags) and `crowned_row`
    disagree — spec-mtp-2 at 44.0 t/s is flagged and fastest, but it fails the
    margin gate against fa-off at 41.0 (44.0 < 41.0 * 1.15), so fa-off is the
    one crowned. Reinstating the old CLI rule makes this test fail.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192),
                   origin="calculator")
    configs = [("spec-mtp-2", {"spec_type": "draft-mtp", "spec_n_max": 2}),
               ("fa-off", {"flash_attn": "off"}),
               ("baseline", {})]

    class _Srv:
        def stop(self):
            pass

    measured = {"tg": 40.0}

    def _launch(exe, trial, model_path, **kw):
        if trial.flags.spec_type == "draft-mtp":
            measured["tg"] = 44.0
        elif trial.flags.flash_attn == "off":
            measured["tg"] = 41.0
        else:
            measured["tg"] = 40.0
        return _Srv()

    monkeypatch.setattr(bench, "launch_server", _launch)
    monkeypatch.setattr(bench, "run_bench",
                        lambda port, **kw: BenchResult(
                            pp_tps=600.0, tg_tps=measured["tg"],
                            prompt_tokens=8, gen_tokens=8))
    monkeypatch.setattr(bench, "sweep_configs",
                        lambda base, moe, caps=(): configs)

    # path 1: the sweep the resolver/CLI share
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    crowned = crowned_row(rows)
    assert crowned is not None and crowned["label"] == "fa-off"

    # path 2: the shipped CLI command, on the same fake rows
    monkeypatch.setattr(cli.Registry, "load", lambda: None)
    monkeypatch.setattr(cli, "_profile", lambda reg: None)
    monkeypatch.setattr(cli, "resolve",
                        lambda p, reg, use_case="general", model_override=None:
                        plan)
    monkeypatch.setattr(runtime, "ensure_engine", lambda backend, os_name: "exe")
    monkeypatch.setattr(runtime, "ensure_model", lambda gguf: "m.gguf")

    res = CliRunner().invoke(cli.app, ["sweep"])
    assert res.exit_code == 0, res.output
    # the real run_sweep ran inside the CLI, not a reimplementation
    assert "trying spec-mtp-2 ..." in res.output
    assert f"winner: {crowned['label']}" in res.output
    assert "winner: spec-mtp-2" not in res.output
