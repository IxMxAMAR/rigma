"""Workflow methods: one-click activity setups (owner request 2026-07-21)."""
import os

import pytest
from fastapi.testclient import TestClient

from rigma import methods, sessions
from rigma import state as st
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_catalog_is_complete_and_ui_ready():
    cat = methods.catalog()
    ids = {m["id"] for m in cat}
    assert {"coding", "book", "roleplay", "research", "tutor",
            "organize"} <= ids
    for m in cat:
        assert m["name"] and m["tagline"] and m["guide"], m["id"]
        a = m["apply"]
        assert a["system_prompt"] and a["notes_template"], m["id"]
        assert "temperature" in a["params"], m["id"]
        assert a["effort"] in ("", "off", "auto", "on"), m["id"]
        # every param must survive the whitelist, or apply would 500
        checked = sessions.validate_params(a["params"])
        assert checked, m["id"]


def test_apply_sets_the_whole_bundle():
    s = sessions.create("test")
    out = methods.apply_to_session(s, "book")
    assert out is not None
    assert "novelist" in out["system_prompt"]
    assert out["params"]["temperature"] == 0.85
    assert out["effort"] == "auto"
    assert out["method"] == "book"
    assert "STORY BIBLE" in out["notes"]


def test_apply_never_overwrites_user_notes():
    s = sessions.create("test")
    s["notes"] = "My precious worldbuilding."
    out = methods.apply_to_session(s, "book")
    assert out["notes"] == "My precious worldbuilding."


def test_roleplay_turns_code_off_but_keeps_tools():
    s = sessions.create("test")
    out = methods.apply_to_session(s, "roleplay")
    assert out["use_tools"] is True and out["allow_code"] is False
    assert out["effort"] == "off"


def test_unknown_method_is_none():
    assert methods.apply_to_session(sessions.create("t"), "yoga") is None


def test_endpoints_roundtrip():
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=1))
    cat = client.get("/api/methods").json()["methods"]
    assert any(m["id"] == "coding" for m in cat)
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/method", json={"id": "coding"})
    assert r.status_code == 200
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["method"] == "coding"
    assert "engineer" in got["system_prompt"]
    assert got["params"]["temperature"] == 0.6
    # bad ids 404
    assert client.post(f"/api/sessions/{sid}/method",
                       json={"id": "nope"}).status_code == 404
    assert client.post("/api/sessions/zzz/method",
                       json={"id": "coding"}).status_code == 404


def test_finish_chapter_macro_replaces_the_ritual(monkeypatch):
    """The hardcoded book_next_chapter ritual is gone; the same move is now
    an ordinary macro made of aux-prompt -> note -> new_chat."""
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())

    async def fake_aux(prompt, max_tokens=120):
        assert "STORY SO FAR" in prompt
        return "Ananya reached the tank at dawn."

    import rigma.methods_api as mapi
    real_register = mapi.register

    def patched(app, **kw):
        return real_register(app, **{**kw, "aux_complete": fake_aux})
    monkeypatch.setattr(mapi, "register", patched)

    client = TestClient(build_app(upstream_port=1))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "book"})
    client.post(f"/api/sessions/{sid}", json={
        "title": "Chapter 3",
        "messages": [{"role": "user", "content": "write ch3"},
                     {"role": "assistant", "content": "the chapter prose"}]})
    with client.stream("POST", f"/api/sessions/{sid}/macro",
                       json={"macro_id": "finish_chapter",
                             "confirm": "run"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "macro_done" in body
    old = client.get(f"/api/sessions/{sid}").json()
    assert "tank at dawn" in old["notes"]
    # and the next chapter chat exists, carrying the bible and the numbering
    import re as _re
    nid = _re.search(r'"new_session_id":\s*"([^"]+)"', body).group(1)
    nxt = client.get(f"/api/sessions/{nid}").json()
    assert nxt["method"] == "book" and "tank at dawn" in nxt["notes"]
    assert nxt["title"] == "Chapter 4"


def test_the_ritual_endpoint_is_gone():
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=1))
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert client.post(f"/api/sessions/{sid}/ritual").status_code == 404


def test_no_method_declares_a_ritual_any_more():
    assert all("ritual" not in m for m in methods.catalog())
