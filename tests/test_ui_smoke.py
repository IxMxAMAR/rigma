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
    for asset in ("style.css", "md.js", "store.js", "app.js"):
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

    # rename + grounded-chat toggle. Grounding no longer swaps the pipeline
    # for a sidecar Q&A box (owner rework 2026-07-21) — the ENGINE still
    # answers, with the sidecar up and search_my_documents advertised.
    assert c.post(f"/api/sessions/{s['id']}",
                  json={"title": "story"}).json()["title"] == "story"
    c.post(f"/api/sessions/{s['id']}", json={"use_rag": True})
    with patch("rigma.rag.ensure_sidecar", return_value={}) as ens, \
         patch("rigma.rag.recorded_sidecar_port", return_value=8899):
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "cite it"})
    assert "Hel" in r.text and "[DONE]" in r.text
    assert ens.called

    # rail summary reflects everything
    lst = c.get("/api/sessions").json()
    assert lst[0]["title"] == "story" and lst[0]["use_rag"] is True
    assert lst[0]["message_count"] == 4


def test_preset_settings_flow(tmp_path, monkeypatch, oai_upstream):
    """The request sequence the Phase-2 UI will perform for presets/settings."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt="D"))

    p = c.post("/api/presets", json={"name": "Noir", "system_prompt": "NOIR",
                                     "params": {"temperature": 1.2}}).json()
    assert [x["id"] for x in c.get("/api/presets").json()].count(p["id"]) == 1

    s = c.post("/api/sessions", json={}).json()
    c.post(f"/api/sessions/{s['id']}",
           json={"preset_id": p["id"], "notes": "Ember the dragon",
                 "params": {"temperature": 0.5}})
    r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "write"})
    assert "[DONE]" in r.text
    sent = oai_upstream.last()
    assert sent["temperature"] == 0.5
    # system context is coalesced into ONE leading system message (a second
    # system block 400s strict chat templates), so notes ride along with it
    assert sent["messages"][0]["role"] == "system"
    assert sent["messages"][0]["content"].startswith("NOIR")
    assert "Ember the dragon" in sent["messages"][0]["content"]
    assert sum(1 for m in sent["messages"] if m["role"] == "system") == 1
    assert sent["messages"][1] == {"role": "user", "content": "write"}


def test_v2_ui_is_served(tmp_path, monkeypatch):
    # the parallel /v2 route: built assets ship inside the wheel; the legacy
    # UI at / must remain untouched while v2 grows (UI-REWORK-PLAN.md)
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1))
    r = c.get("/v2")
    assert r.status_code in (200, 503)
    if r.status_code == 200:                    # dist committed
        assert "<div id=\"root\">" in r.text
        import re
        m = re.search(r'assets/(index-[\w-]+\.js)', r.text)
        assert m, "index.html must reference the hashed bundle"
        a = c.get(f"/v2/assets/{m.group(1)}")
        assert a.status_code == 200
        assert "immutable" in a.headers.get("cache-control", "")
    # traversal dies
    assert c.get("/v2/assets/..%2f..%2fserve.py").status_code == 404
    # legacy root unaffected
    assert c.get("/").status_code == 200
