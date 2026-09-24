import os

import pytest
from fastapi.testclient import TestClient

from rigma import state
from rigma.serve import build_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def test_server_info_404_when_not_running(client):
    assert client.get("/api/server").status_code == 404


def test_server_info_fields_and_verdict(tmp_path, monkeypatch, client):
    state.write_state("m", "q", 18500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), backend="vulkan",
                      use_case="creative", ctx=4096)
    # built with the real writer on purpose: a hand-written fixture here is how
    # expected_tg stayed broken while its test stayed green (see test_server_ops)
    from rigma import bench
    bench.save_calibration("m:q:vulkan", {"tg_tps": 50.0, "pp_tps": 600.0})
    info = client.get("/api/server").json()
    assert info["model"] == "m" and info["ctx"] == 4096
    assert info["use_case"] == "creative"
    assert info["ram_free_mb"] > 0 and info["ram_total_mb"] > 0
    assert info["expected_tg"] == 50.0
    assert info["verdict"] == "unknown"  # no turn telemetry yet
    assert info["last_tg"] is None


def test_server_log_route(tmp_path, monkeypatch, client):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server-1.log").write_text("alpha\nbeta", encoding="utf-8")
    r = client.get("/api/server/log?lines=50")
    assert r.status_code == 200 and r.text.endswith("beta")
    assert r.headers["cache-control"] == "no-store"


def test_switch_route_validation(client):
    assert client.post("/api/server/switch", json={}).status_code == 400
    r = client.post("/api/server/switch", json={"model": "x"})
    assert r.status_code == 502 and "not running" in r.json()["error"]


def test_switch_options_404_when_not_running(client):
    assert client.get("/api/server/switch-options").status_code == 404


# --- F51: the engine's load-time decisions must be reachable ------------------
def test_findings_reads_the_whole_log_not_a_1000_line_tail(tmp_path, monkeypatch,
                                                           client):
    """The warnings `engine_log.findings` exists for land in the FIRST few
    hundred lines of a launch. The endpoint called `log_tail(1000)`, and
    log_tail hard-caps at 1000 lines — so on any session whose log had grown
    past that the feature reported nothing at all (AUDIT F51)."""
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server-1.log").write_text(
        "W srv load_model: cache_reuse is not supported by this context, "
        "it will be disabled\n" + ("filler line\n" * 5000),
        encoding="utf-8")
    r = client.get("/api/server/findings")
    assert r.status_code == 200
    ids = [f["id"] for f in r.json()["findings"]]
    assert "cache_reuse_disabled" in ids, r.json()


def test_findings_distinguishes_cannot_look_from_found_nothing(
        tmp_path, monkeypatch, client):
    """A read failure returned `{"findings": []}`, which is indistinguishable
    from "the engine is healthy" (AUDIT F51)."""
    r = client.get("/api/server/findings")     # no logs directory at all
    assert r.status_code == 503
    assert "cannot read the engine log" in r.json()["error"]
    assert r.json()["findings"] == []


def test_server_info_carries_the_findings_too(tmp_path, monkeypatch, client):
    """Folded into the response the Engine panel already polls, so they arrive
    with everything else (AUDIT F51)."""
    state.write_state("m", "q", 18500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), backend="vulkan",
                      use_case="creative", ctx=4096)
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "server-1.log").write_text(
        "W srv load_model: cache_reuse is not supported by this context\n",
        encoding="utf-8")
    info = client.get("/api/server").json()
    assert "cache_reuse_disabled" in [f["id"] for f in info["engine_findings"]]
    assert "engine_findings_error" not in info


def test_server_info_reports_an_unreadable_log_without_failing(
        tmp_path, monkeypatch, client):
    state.write_state("m", "q", 18500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), backend="vulkan",
                      use_case="creative", ctx=4096)
    info = client.get("/api/server").json()     # no log to read
    assert info["engine_findings"] == []
    assert "cannot read the engine log" in info["engine_findings_error"]
