"""Workflow methods: one-click activity setups (owner request 2026-07-21)."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

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


class _AuxUpstream(BaseHTTPRequestHandler):
    """Answers every non-streaming completion with a fixed bible line."""
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        payload = json.dumps({"choices": [{"message": {
            "content": "Ananya reached the tank at dawn."}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *a):
        pass


@pytest.fixture
def aux_upstream():
    srv = HTTPServer(("127.0.0.1", 0), _AuxUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_book_ritual_updates_bible_and_spawns_next_chapter(aux_upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=aux_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "book"})
    client.post(f"/api/sessions/{sid}", json={
        "title": "Chapter 3",
        "messages": [{"role": "user", "content": "write ch3"},
                     {"role": "assistant",
                      "content": "The chapter prose goes here." * 20}]})
    r = client.post(f"/api/sessions/{sid}/ritual")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["title"] == "Chapter 4"
    assert "tank at dawn" in d["bible_entry"]
    # old chat's bible grew
    old = client.get(f"/api/sessions/{sid}").json()
    assert "tank at dawn" in old["notes"]
    # the new chat carries method + bible and its title survives auto-titling
    new = client.get(f"/api/sessions/{d['new_session_id']}").json()
    assert new["method"] == "book"
    assert "tank at dawn" in new["notes"]
    assert new["title"] == "Chapter 4"


def test_ritual_404s_without_one(aux_upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=aux_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "coding"})
    assert client.post(f"/api/sessions/{sid}/ritual").status_code == 404
