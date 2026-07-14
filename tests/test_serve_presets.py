import pytest
from fastapi.testclient import TestClient

from rigma.models import UseCase
from rigma.registry import Registry
from rigma.serve import build_app


def _fake_reg(**prompts):
    return Registry([], {}, {}, {k: UseCase(name=k, system_prompt=v)
                                 for k, v in prompts.items()})


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt="D",
                                registry=_fake_reg(general="G")))


def test_preset_crud_cycle(client):
    lst = client.get("/api/presets").json()
    assert lst[0]["id"] == "usecase:general" and lst[0]["builtin"]
    p = client.post("/api/presets",
                    json={"name": "mine", "system_prompt": "M",
                          "params": {"temperature": 1.1}}).json()
    assert p["builtin"] is False
    upd = client.post(f"/api/presets/{p['id']}",
                      json={"name": "renamed", "id": "EVIL"}).json()
    assert upd["name"] == "renamed" and upd["id"] == p["id"]
    assert client.delete(f"/api/presets/{p['id']}").status_code == 200
    assert client.delete(f"/api/presets/{p['id']}").status_code == 404


def test_builtin_presets_are_immutable(client):
    r = client.post("/api/presets/usecase:general", json={"name": "hax"})
    assert r.status_code == 403 and r.json() == {"error": "builtin preset"}
    assert client.delete("/api/presets/usecase:general").status_code == 403


def test_update_missing_preset_404(client):
    assert client.post("/api/presets/nope", json={}).status_code == 404
