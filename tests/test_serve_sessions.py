import pytest
from fastapi.testclient import TestClient

from rigma import sessions
from rigma.serve import build_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt="DEFAULT"))


def test_session_crud_cycle(client):
    s = client.post("/api/sessions", json={"title": "t"}).json()
    assert s["title"] == "t" and s["messages"] == []
    assert client.get("/api/sessions").json()[0]["id"] == s["id"]
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert got == s
    upd = client.post(f"/api/sessions/{s['id']}",
                      json={"system_prompt": "be brief", "use_rag": True,
                            "id": "EVIL"}).json()
    assert upd["system_prompt"] == "be brief" and upd["use_rag"] is True
    assert upd["id"] == s["id"]  # immutable fields ignored
    assert client.delete(f"/api/sessions/{s['id']}").status_code == 200
    assert client.get(f"/api/sessions/{s['id']}").status_code == 404


def test_update_truncates_messages(client):
    s = client.post("/api/sessions", json={}).json()
    sess = sessions.load(s["id"])
    sess["messages"] = [{"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"}]
    sessions.save(sess)
    upd = client.post(f"/api/sessions/{s['id']}",
                      json={"messages": [{"role": "user", "content": "a"}]}).json()
    assert len(upd["messages"]) == 1


def test_missing_session_is_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope", json={}).status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404
