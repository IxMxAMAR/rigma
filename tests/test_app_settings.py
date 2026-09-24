"""IMP-5: the idle auto-unload timeout is visible and configurable.

Rigma unloads the model after idle time to free VRAM, but the timeout was an
env var read once at startup, so it was invisible and could not be changed
from the product. It is now a stored setting with an off switch, echoed by
/api/server, and the poller re-reads it so a change applies live.
"""
import os

import pytest
from fastapi.testclient import TestClient

from rigma import app_settings, state
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


@pytest.fixture
def client(home):
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def test_default_is_off(home):
    """The safe direction, and the pre-setting default: never unload unless
    asked. An unload out from under a slow prefill looks like a bug."""
    assert app_settings.load()["idle_unload_minutes"] == 0.0
    assert app_settings.idle_unload_minutes() == 0.0


def test_save_and_reload(home):
    app_settings.save({"idle_unload_minutes": 15})
    assert app_settings.load()["idle_unload_minutes"] == 15.0
    assert app_settings.idle_unload_minutes() == 15.0
    # the write is atomic: no temp file is left behind
    assert list(home.glob("settings.json*")) == [home / "settings.json"]


def test_a_corrupt_file_falls_back_to_defaults(home):
    (home / "settings.json").write_text("{not json", encoding="utf-8")
    assert app_settings.load() == app_settings.DEFAULTS
    # a wrong-typed value is ignored rather than crashing the server
    (home / "settings.json").write_text(
        '{"idle_unload_minutes": "soon"}', encoding="utf-8")
    assert app_settings.load()["idle_unload_minutes"] == 0.0


@pytest.mark.parametrize("bad", [
    {"idle_unload_minutes": -1},
    {"idle_unload_minutes": "soon"},
    {"idle_unload_minutes": app_settings.MAX_IDLE_MINUTES + 1},
    {"nonsense": 1},
    {},
])
def test_validate_refuses_bad_patches(home, bad):
    clean, err = app_settings.validate(bad)
    assert err and clean == {}
    with pytest.raises(ValueError):
        app_settings.save(bad)


def test_the_env_var_still_overrides(home, monkeypatch):
    """Builds and launchers before this setting used RIGMA_KEEP_ALIVE_MIN."""
    app_settings.save({"idle_unload_minutes": 15})
    monkeypatch.setenv("RIGMA_KEEP_ALIVE_MIN", "3")
    assert app_settings.idle_unload_minutes() == 3.0


def test_settings_endpoints_round_trip(client):
    got = client.get("/api/settings").json()
    assert got["settings"]["idle_unload_minutes"] == 0.0
    assert got["idle_unload_minutes"] == 0.0
    assert got["env_override"] is False

    r = client.post("/api/settings", json={"idle_unload_minutes": 20})
    assert r.status_code == 200
    assert r.json()["settings"]["idle_unload_minutes"] == 20.0
    assert client.get("/api/settings").json()["idle_unload_minutes"] == 20.0

    # the off switch
    assert client.post("/api/settings",
                       json={"idle_unload_minutes": 0}).status_code == 200
    assert client.get("/api/settings").json()["idle_unload_minutes"] == 0.0

    r = client.post("/api/settings", json={"idle_unload_minutes": -5})
    assert r.status_code == 400 and "idle_unload_minutes" in r.json()["error"]


def test_server_info_echoes_the_current_value(client):
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid())
    info = client.get("/api/server").json()
    assert info["idle_unload_minutes"] == 0.0 and info["idle_unload"] is False
    client.post("/api/settings", json={"idle_unload_minutes": 30})
    info = client.get("/api/server").json()
    assert info["idle_unload_minutes"] == 30.0 and info["idle_unload"] is True
