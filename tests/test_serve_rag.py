from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def test_rag_status_not_running(client):
    r = client.get("/api/rag/status").json()
    assert r["running"] is False and r["sources"] == [] and r["indexing"] is False


def test_rag_add_source_rejects_bad_path(client):
    r = client.post("/api/rag/sources", json={"path": "Z:/definitely/not/here"})
    assert r.status_code == 400


def test_rag_add_source_indexes_in_background(client, tmp_path):
    import time
    docs = tmp_path / "docs"
    docs.mkdir()
    # keep the patch active until the background task finishes, or the task
    # would call the REAL rag.ingest after the context manager exits
    with patch("rigma.rag.ingest", return_value="ok") as ing:
        r = client.post("/api/rag/sources", json={"path": str(docs)})
        assert r.status_code == 202
        assert str(docs) in r.json()["sources"][0]
        deadline = time.time() + 5
        while time.time() < deadline:  # each status GET spins the app's loop
            if not client.get("/api/rag/status").json()["indexing"]:
                break
            time.sleep(0.05)
    assert ing.called
    assert client.get("/api/rag/status").json()["error"] == ""


def test_grounded_chat_streams_from_the_engine(client, oai_upstream,
                                               tmp_path, monkeypatch):
    # THE rework (owner 2026-07-21): "use in this chat" used to swap the whole
    # pipeline for a single-shot sidecar Q&A box — last message only, no
    # history, no tools, no follow-ups. Grounded search now means a NORMAL
    # conversation where the sidecar is up and the model can call
    # search_my_documents mid-chat.
    from rigma.serve import build_app
    c2 = TestClient(build_app(upstream_port=oai_upstream.port,
                              default_prompt=""))
    s = c2.post("/api/sessions", json={}).json()
    c2.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", return_value={}) as ens,          patch("rigma.rag.recorded_sidecar_port", return_value=8899):
        r = c2.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "Hel" in r.text and "[DONE]" in r.text      # engine streamed it
    assert ens.called, "grounding must bring the sidecar up"
    body = oai_upstream.last()
    names = [t_["function"]["name"] for t_ in body.get("tools") or []]
    assert "search_my_documents" in names


def test_grounded_nudge_is_in_the_system_prompt(client, oai_upstream):
    from rigma.serve import build_app
    c2 = TestClient(build_app(upstream_port=oai_upstream.port,
                              default_prompt="be helpful"))
    s = c2.post("/api/sessions", json={}).json()
    c2.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", return_value={}),          patch("rigma.rag.recorded_sidecar_port", return_value=8899):
        c2.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    body = oai_upstream.last()
    sys_msg = body["messages"][0]
    assert sys_msg["role"] == "system"
    assert "search_my_documents" in sys_msg["content"],         "a weak model needs telling that grounding is on"


def test_grounded_sidecar_failure_degrades_to_plain_chat(client, oai_upstream):
    # grounding must NEVER kill the conversation — a dead sidecar means the
    # model just answers without documents
    from rigma.serve import build_app
    c2 = TestClient(build_app(upstream_port=oai_upstream.port,
                              default_prompt=""))
    s = c2.post("/api/sessions", json={}).json()
    c2.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", side_effect=RuntimeError("down")):
        r = c2.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "Hel" in r.text and "[DONE]" in r.text


def test_grounded_continue_is_allowed_now(client, oai_upstream):
    # the old pipeline 400'd continue for grounded chats; a grounded chat is a
    # real conversation now, so continue works like anywhere else
    from rigma.serve import build_app
    c2 = TestClient(build_app(upstream_port=oai_upstream.port,
                              default_prompt=""))
    s = c2.post("/api/sessions", json={}).json()
    c2.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    c2.post(f"/api/sessions/{s['id']}",
            json={"messages": [{"role": "user", "content": "q"},
                               {"role": "assistant", "content": "a"}]})
    with patch("rigma.rag.ensure_sidecar", return_value={}):
        r = c2.post(f"/api/sessions/{s['id']}/chat",
                    json={"message": None, "continue": True})
    assert r.status_code == 200


def test_remove_source(client, tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    with patch("rigma.rag.ingest", return_value="ok"):
        client.post("/api/rag/sources", json={"path": str(docs)})
    assert client.get("/api/rag/status").json()["sources"]
    r = client.request("DELETE", "/api/rag/sources",
                       json={"path": str(docs)})
    assert r.status_code == 200
    assert client.get("/api/rag/status").json()["sources"] == []
