"""R3-VLLM-4: vLLM was specified, documented and diagnosable — but UNREACHABLE.

`engines.detect_engine_runtime` existed and was correct, `vllm_argv` built the
command line, `launch_vllm_server` polled `/health`, and `rigma engine-runtimes`
reported the verdict honestly. What was missing was any way for a user to CHOOSE:
nothing called `detect_engine_runtime`, there was no `--engine` flag, and no state
field recorded which engine was running. So the objective's "selectable engine
backend" was one wiring pass short of true, and `engine-runtimes` was a report about
a road with no on-ramp.

These tests pin the wiring, not the arithmetic (test_engines.py owns that).
"""
import json

import pytest
from typer.testing import CliRunner

import rigma.cli as cli
from rigma import engines, state as st

runner = CliRunner()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


# --- the flag exists and validates -----------------------------------------

def test_engine_is_a_documented_option_on_up():
    """Discoverability is the fix. If `--engine` is not in `up --help`, a user
    cannot find the only way to select an engine runtime."""
    res = runner.invoke(cli.app, ["up", "--help"])
    assert res.exit_code == 0
    assert "--engine" in res.output


def test_an_unknown_engine_name_is_refused_by_name(home):
    """A typo must not silently run llama.cpp — that is the failure mode the whole
    finding is about. It must name the valid values."""
    res = runner.invoke(cli.app, ["up", "--engine", "tensorrt", "--dry-run"])
    assert res.exit_code != 0
    out = res.output
    assert "tensorrt" in out
    assert "llamacpp" in out and "vllm" in out


def test_engine_names_are_case_insensitive(home, monkeypatch):
    """`--engine VLLM` is what a user types."""
    monkeypatch.setattr(engines, "vllm_availability",
                        lambda: engines.EngineAvailability(
                            engines.VLLM, False, "unsupported-os", "no windows"))
    res = runner.invoke(cli.app, ["up", "--engine", "VLLM", "--dry-run"])
    # it must be RECOGNISED (so not the unknown-name error) and then refused for
    # the real reason, which dry-run reports without launching anything
    assert "tensorrt" not in res.output
    assert "vllm" in res.output.lower()


# --- the honest refusal -----------------------------------------------------

def test_requesting_an_unavailable_vllm_refuses_instead_of_falling_back(
        home, monkeypatch):
    """THE CORE BEHAVIOUR. `detect_engine_runtime` returns llama.cpp with the
    reason recorded, which is right for a STORED preference — but when a user
    types `--engine vllm` on this command line they asked for vLLM NOW. Silently
    running llama.cpp and printing a working UI is how someone concludes vLLM
    works. It must refuse, name the reason, and exit non-zero."""
    monkeypatch.setattr(engines, "vllm_availability",
                        lambda: engines.EngineAvailability(
                            engines.VLLM, False, "unsupported-os",
                            "vLLM does not support Windows natively"))
    res = runner.invoke(cli.app, ["up", "--engine", "vllm", "--dry-run"])
    assert res.exit_code != 0, res.output
    assert "does not support Windows" in res.output
    # and it must NOT claim to have started anything
    assert "chat UI" not in res.output


def test_the_refusal_says_what_to_do_instead(home, monkeypatch):
    """A refusal that does not say how to proceed is a dead end. The engine
    verdict carries the WSL2/Linux route; the refusal must surface it."""
    monkeypatch.setattr(engines, "vllm_availability",
                        lambda: engines.EngineAvailability(
                            engines.VLLM, False, "unsupported-os",
                            "vLLM does not support Windows natively"))
    res = runner.invoke(cli.app, ["up", "--engine", "vllm", "--dry-run"])
    assert "llamacpp" in res.output.lower(), (
        "the refusal must name the engine that does work")


def test_requesting_llamacpp_explicitly_is_fine(home):
    """The flag must not break the default path."""
    res = runner.invoke(cli.app, ["up", "--engine", "llamacpp", "--dry-run"])
    assert res.exit_code == 0, res.output


def test_no_engine_flag_is_unchanged_behaviour(home):
    """Every existing invocation must behave exactly as before."""
    res = runner.invoke(cli.app, ["up", "--dry-run"])
    assert res.exit_code == 0, res.output


# --- the decision function itself ------------------------------------------

def test_detect_returns_llamacpp_when_nothing_is_requested():
    d = engines.detect_engine_runtime(None)
    assert d.runtime == engines.LLAMACPP and d.requested == ""


def test_detect_falls_back_with_the_reason_when_vllm_cannot_run(monkeypatch):
    """The STORED-preference behaviour, which the refusal above deliberately does
    not use. Both are needed; this pins the difference so nobody 'simplifies' one
    into the other."""
    monkeypatch.setattr(engines, "vllm_availability",
                        lambda: engines.EngineAvailability(
                            engines.VLLM, False, "unsupported-os", "no windows"))
    d = engines.detect_engine_runtime("vllm")
    assert d.runtime == engines.LLAMACPP
    assert d.requested == "vllm"
    assert "cannot run it" in d.reason and "no windows" in d.reason


def test_detect_selects_vllm_when_it_can_run(monkeypatch):
    monkeypatch.setattr(engines, "vllm_availability",
                        lambda: engines.EngineAvailability(
                            engines.VLLM, True, "runnable", "ready"))
    d = engines.detect_engine_runtime("vllm")
    assert d.runtime == engines.VLLM and d.availability.available


def test_detect_rejects_an_unknown_name():
    with pytest.raises(ValueError, match="tensorrt"):
        engines.detect_engine_runtime("tensorrt")


# --- the state field --------------------------------------------------------

def test_state_records_which_engine_is_running(home):
    """Without this the UI and `rigma status` cannot tell a vLLM launch from a
    llama.cpp one, and would show llama.cpp's compute backend for both."""
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=1,
                   backend="vulkan", engine="vllm")
    s = st.read_state()
    assert s["engine"] == "vllm"
    # and the two axes stay distinct
    assert s["backend"] == "vulkan"


def test_state_does_not_invent_an_engine_it_did_not_launch(home):
    """THE BACKWARD-COMPATIBILITY HALF, and the reason the default is None.

    `write_state` is called by paths that launch no engine at all (UI-only
    `rigma up`, `perform_unload`). A default of "llamacpp" would make every record
    they write assert llama.cpp — and, worse, would OVERWRITE the engine a vLLM
    launch had recorded. Absent means "not recorded"; a launcher that knows says
    so.
    """
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=1, backend="vulkan")
    assert st.read_state().get("engine") in (None, ""), (
        "an engine was recorded by a path that never launched one")


def test_a_recorded_engine_survives_a_rewrite_that_does_not_name_it(home):
    """The overwrite case, stated directly: `update_state` merges, and a launch
    path that names the engine must have it stick."""
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=1, backend="vllm",
                   engine="vllm")
    assert st.read_state()["engine"] == "vllm"
    st.update_state(ctx=8192)          # a change that has no opinion on engine
    assert st.read_state()["engine"] == "vllm", (
        "an unrelated edit dropped the engine")


# --- the API surface the UI reads -------------------------------------------

def test_api_server_reports_engine_runtimes(home, monkeypatch):
    """The UI needs the same list the CLI has, or the engine picker cannot exist.

    `server_running()` identity-checks both pids against their recorded creation
    times, so the record names THIS process — otherwise the route answers
    {"error": "not running"} and the test would pass for the wrong reason.
    """
    import os

    from fastapi.testclient import TestClient

    from rigma import serve
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vulkan")
    app = serve.build_app(upstream_port=11500)
    c = TestClient(app)
    body = c.get("/api/server").json()
    assert "engine_runtimes" in body, (
        "the UI has no way to list engine runtimes")
    names = {r["engine"] for r in body["engine_runtimes"]}
    assert {"llamacpp", "vllm"} <= names, names
    # The RUNNING engine is reported separately from the list of what could — and
    # this record never recorded one, so it must NOT be invented.
    assert body.get("engine") in (None, ""), (
        "the route asserted an engine the record does not have")


def test_api_reports_a_recorded_engine(home):
    """And when a launch DID name the engine, the route reports it."""
    import os

    from fastapi.testclient import TestClient

    from rigma import serve
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vllm", engine="vllm")
    c = TestClient(serve.build_app(upstream_port=11500))
    assert c.get("/api/server").json()["engine"] == "vllm"


def test_engine_runtimes_json_is_machine_readable(home):
    """`rigma engine-runtimes --json` is the scripting seam; it must be valid JSON
    with the fields a caller needs, not a pretty-printed blob."""
    res = runner.invoke(cli.app, ["engine-runtimes", "--json"])
    assert res.exit_code == 0, res.output
    rows = json.loads(res.output)
    assert isinstance(rows, list) and rows
    for r in rows:
        assert {"engine", "available", "reason"} <= set(r), r


# --- `rigma status` names the engine ----------------------------------------

def test_status_names_vllm_instead_of_printing_empty_brackets(home):
    """A vLLM launch has no quant, so the old line read `running: Qwen/Qwen3-8B ()
    up 3 min` — empty brackets where the explaining fact belongs."""
    import os
    st.write_state("Qwen/Qwen3-8B", "", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="vllm", engine="vllm")
    res = runner.invoke(cli.app, ["status"])
    assert res.exit_code == 0, res.output
    assert "vLLM" in res.output
    assert "()" not in res.output, res.output


def test_status_is_unchanged_for_a_llamacpp_launch(home):
    import os
    st.write_state("m", "Q4_K_M", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="vulkan", engine="llamacpp")
    res = runner.invoke(cli.app, ["status"])
    assert "m (Q4_K_M)" in res.output, res.output


def test_status_does_not_invent_an_engine(home):
    """A record with no engine and no quant must not claim either."""
    import os
    st.write_state("m", "", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vulkan")
    res = runner.invoke(cli.app, ["status"])
    assert "()" not in res.output, res.output
    assert "vLLM" not in res.output, res.output
