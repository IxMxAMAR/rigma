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


def test_rag_chat_turn_grounded_with_citations(client):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    answer = {"answer": "Grounded.", "citations": [{"source": "a.md"}],
              "abstained": False}
    with patch("rigma.rag.ensure_sidecar", return_value={}), \
         patch("rigma.rag.ask", return_value=answer):
        r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "Grounded." in r.text and "event: citations" in r.text
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert got["messages"][-1] == {"role": "assistant", "content": "Grounded."}


def test_rag_chat_turn_abstained_prefix(client):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    answer = {"answer": "", "citations": [], "abstained": True}
    with patch("rigma.rag.ensure_sidecar", return_value={}), \
         patch("rigma.rag.ask", return_value=answer):
        r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "abstained" in r.text


def test_rag_chat_turn_sidecar_failure_is_error_event(client):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", side_effect=RuntimeError("boom")):
        r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "event: error" in r.text and "[DONE]" in r.text


def test_rag_chat_turn_empty_non_abstained_answer_is_error_event(client):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    answer = {"answer": "", "citations": [], "abstained": False}
    with patch("rigma.rag.ensure_sidecar", return_value={}), \
         patch("rigma.rag.ask", return_value=answer):
        r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "event: error" in r.text and "[DONE]" in r.text
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user"]


def test_rag_chat_turn_malformed_reply_is_error_event(client):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", return_value={}), \
         patch("rigma.rag.ask", return_value=None):
        r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
    assert "event: error" in r.text and "[DONE]" in r.text
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user"]
