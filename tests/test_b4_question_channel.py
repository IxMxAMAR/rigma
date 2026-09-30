"""B4: mcode's `ask_user` reaches the UI channel, and its answer comes back.

`serve.py` passed `on_permission` to `drive_turn_acp` but never `on_question`, so
every ACP `elicitation/create` was answered `decline` and the agent silently took
its non-interactive fallback — the question was never shown to anyone.

The handler is the SAME handshake as the permission one, deliberately: one
`_approvals` slot per session, the same `approval/asked` event, the same
`POST /api/sessions/{sid}/approval` route. These tests drive a real chat turn whose
adapter asks one question, answer it through that route, and assert the answer the
handler returned reaches the turn's output.

No model and no mcode process: the ACP driver is replaced by a fake that calls the
handler it is given, which is the seam under test.
"""
from __future__ import annotations

import json
import os
import threading
import time

from fastapi.testclient import TestClient

from rigma import harness, harness_mcode, serve, sessions
from rigma import state as st

DUMMY_PORT = 11499


def _app_and_session(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: True)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    real_read = st.read_state
    monkeypatch.setattr(st, "read_state",
                        lambda: {**(real_read() or {}),
                                 "public_port": 11500, "ctx": 32768})
    s = sessions.create(title="t")
    s["harness"] = "mcode"
    s["mcode_transport"] = "acp"
    sessions.save(s)
    return s


def _approvals_of(client):
    """The `_approvals` slot dict, reached through the route that reads it.

    It is a closure inside `build_app`, so it is not an attribute; the route's
    free variables are the only handle — the same technique
    tests/test_r3_prompt_queue.py uses for the prompt queue."""
    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        names = fn.__code__.co_freevars
        if "_approvals" in names:
            return fn.__closure__[names.index("_approvals")].cell_contents
    raise AssertionError("could not reach the _approvals slot")


def _serve_source() -> str:
    with open(serve.__file__, encoding="utf-8") as f:
        return f.read()


def test_the_question_handler_is_wired_and_uses_the_permission_channel():
    """The handler must be PASSED (it was not before), and it must reuse the
    permission round-trip rather than growing a second one."""
    src = _serve_source()
    assert "on_question=_answer_question" in src, (
        "drive_turn_acp is still not given an on_question handler")
    assert 'event="approval/asked"' in src, (
        "the question must be published on the channel the UI already reads")
    assert "def _answer_question(" in src
    # ... and the same response route, not a second one.
    assert "/api/sessions/{sid}/approval" in src


def test_a_question_reaches_the_approval_channel_and_its_answer_is_returned(
        monkeypatch, tmp_path):
    """The whole round trip: the adapter asks, the question is published on the
    approval channel, the route carries the answer back, and the handler returns
    it to the adapter."""
    s = _app_and_session(monkeypatch, tmp_path)

    def _fake_acp(prompt, **kw):
        answer = kw["on_question"]("elicitation/create", {
            "message": "Which directory?",
            "requestedSchema": {"type": "object",
                                "properties": {"path": {"type": "string"}}}})
        yield harness.TurnEvent("text", text=f"answer={json.dumps(answer)}")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        approvals = _approvals_of(c)
        out = {}

        def _turn():
            out["r"] = c.post(f"/api/sessions/{s['id']}/chat",
                              json={"message": "go"})

        t = threading.Thread(target=_turn, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and s["id"] not in approvals:
            time.sleep(0.01)
        assert s["id"] in approvals, (
            "the question never reached the approval channel the UI reads")
        slot = approvals[s["id"]]
        assert slot["kind"] == "question", slot
        r = c.post(f"/api/sessions/{s['id']}/approval",
                   json={"requestId": slot["requestId"],
                         "answer": {"path": "C:/work"}})
        assert r.status_code == 200, r.text
        t.join(15)
        assert not t.is_alive(), "the turn never finished"

    body = out["r"].text
    assert "Which directory?" in body, body
    assert "C:/work" in body, body


def test_a_question_nobody_answers_is_declined_and_does_not_hang(
        monkeypatch, tmp_path):
    """No client renders elicitation yet (OD-8), so the wait is bounded and the
    handler returns None. `drive_turn_acp` turns that into
    `answer_elicitation(accepted=False)` — a REAL decline mcode can fall back
    from, not a request left waiting forever."""
    s = _app_and_session(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "QUESTION_WAIT_SECS", 0.05)

    def _fake_acp(prompt, **kw):
        answer = kw["on_question"]("elicitation/create", {"message": "Which?"})
        yield harness.TurnEvent("text", text=f"answer={answer}")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "go"})

    assert r.status_code == 200, r.text
    assert "answer=None" in r.text, r.text
