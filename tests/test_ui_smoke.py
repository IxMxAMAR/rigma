"""The exact request sequence app.js performs, against a fake upstream."""
from unittest.mock import patch

from fastapi.testclient import TestClient

from rigma.serve import build_app


def test_full_ui_conversation_flow(tmp_path, monkeypatch, oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt="D"))

    # boot: status (404 - no state in hermetic home), session list, assets
    assert c.get("/api/status").status_code == 404
    assert c.get("/api/sessions").json() == []
    assert c.get("/").status_code == 200
    for asset in ("style.css", "md.js", "app.js"):
        assert c.get(f"/ui/{asset}").status_code == 200

    # new chat -> first message -> title set, both roles persisted
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "write a story"})
    assert "[DONE]" in r.text
    s = c.get(f"/api/sessions/{s['id']}").json()
    assert s["title"] == "write a story" and len(s["messages"]) == 2

    # regenerate: truncate assistant server-side, null-message turn
    c.post(f"/api/sessions/{s['id']}", json={"messages": s["messages"][:1]})
    c.post(f"/api/sessions/{s['id']}/chat", json={"message": None})
    s = c.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in s["messages"]] == ["user", "assistant"]

    # rename + RAG toggle + grounded turn
    assert c.post(f"/api/sessions/{s['id']}",
                  json={"title": "story"}).json()["title"] == "story"
    c.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    grounded = {"answer": "From your docs.", "citations": ["a.md"],
                "abstained": False}
    with patch("rigma.rag.ensure_sidecar", return_value={}), \
         patch("rigma.rag.ask", return_value=grounded):
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "cite it"})
    assert "From your docs." in r.text and "event: citations" in r.text

    # rail summary reflects everything
    lst = c.get("/api/sessions").json()
    assert lst[0]["title"] == "story" and lst[0]["use_rag"] is True
    assert lst[0]["message_count"] == 4
