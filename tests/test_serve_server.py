import json
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
    (tmp_path / "calibration.json").write_text(json.dumps(
        {"m:q:vulkan": {"tg_tps": 50.0}}), encoding="utf-8")
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
