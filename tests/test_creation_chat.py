"""Clicking "create method" opens a chat that can only build one (spec §6)."""
import os
import re

import pytest
from fastapi.testclient import TestClient

from rigma import method_drafts, methods, sessions
from rigma import state as st
from rigma.methods_api import BUILDER_PROMPT
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    return tmp_path


@pytest.fixture
def client():
    return TestClient(build_app(upstream_port=1))


def test_the_builder_prompt_stays_short_and_decisive():
    """Pinned, not stylistic. A 9-rule doctrine measurably sent this owner's
    35B into 15.7K-char deliberation spirals with no reply at all; the
    rewritten 5-rule version produced 870 chars, a reply AND the tool call.
    If someone grows this prompt later, this test should stop them."""
    assert len(BUILDER_PROMPT) < 400, len(BUILDER_PROMPT)
    sentences = [s for s in re.split(r"[.!?]", BUILDER_PROMPT) if s.strip()]
    assert len(sentences) <= 5, sentences
    # either/or framings are deliberation traps
    assert " vs " not in BUILDER_PROMPT and " versus " not in BUILDER_PROMPT


def test_creating_a_draft_returns_a_bound_session(client):
    d = client.post("/api/methods/draft", json={}).json()
    assert d["draft_id"] and d["session_id"]
    s = sessions.load(d["session_id"])
    assert s["method_draft_id"] == d["draft_id"]
    assert s["system_prompt"] == BUILDER_PROMPT


def test_the_greeting_is_a_stored_assistant_message(client):
    """Pre-seeded, not generated: instant, deterministic, and it cannot
    ramble — while still reading as the model's voice in the transcript."""
    d = client.post("/api/methods/draft", json={}).json()
    s = sessions.load(d["session_id"])
    assert s["messages"], "the creation chat opens empty"
    first = s["messages"][0]
    assert first["role"] == "assistant"
    assert first["content"].strip()


def test_the_draft_session_forces_a_tool_call(client):
    """one_action is what turns on serve.py's existing force_call path, so
    the creation chat inherits tool_choice:'required' AND its per-model
    HTTP-400 fallback rather than re-implementing either."""
    d = client.post("/api/methods/draft", json={}).json()
    assert sessions.load(d["session_id"])["one_action"] is True


def test_a_normal_session_is_not_a_draft_session(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    assert sessions.load(sid)["method_draft_id"] == ""


def test_the_draft_can_be_read_back(client):
    d = client.post("/api/methods/draft", json={}).json()
    got = client.get(f"/api/methods/draft/{d['draft_id']}")
    assert got.status_code == 200
    assert got.json()["id"] == d["draft_id"]
    assert client.get("/api/methods/draft/nope").status_code == 404


def test_promote_over_http_saves_the_method(client):
    d = client.post("/api/methods/draft", json={}).json()
    doc = method_drafts.load(d["draft_id"])
    doc["name"] = "Haiku writing"
    doc["tagline"] = "17 syllables"
    doc["apply"]["system_prompt"] = "You write haiku."
    method_drafts.save(doc)
    r = client.post(f"/api/methods/draft/{d['draft_id']}/promote")
    assert r.status_code == 200, r.text
    assert methods.get(r.json()["id"])["name"] == "Haiku writing"
    assert method_drafts.load(d["draft_id"]) is None


def test_promote_reports_what_is_missing(client):
    d = client.post("/api/methods/draft", json={}).json()
    r = client.post(f"/api/methods/draft/{d['draft_id']}/promote")
    assert r.status_code == 400
    assert r.json()["errors"]
    assert method_drafts.load(d["draft_id"]) is not None   # draft survives


def test_a_draft_never_appears_in_the_catalog(client):
    client.post("/api/methods/draft", json={})
    ids = {m["id"] for m in client.get("/api/methods").json()["methods"]}
    assert not any(i.startswith("draft_") for i in ids)
