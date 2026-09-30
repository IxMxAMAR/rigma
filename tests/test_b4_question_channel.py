"""B4: mcode's `ask_user` reaches the UI channel, and its answer comes back.

`serve.py` passed `on_permission` to `drive_turn_acp` but never `on_question`, so
every ACP `elicitation/create` was answered `decline` and the agent silently took
its non-interactive fallback — the question was never shown to anyone.

The handler is the SAME handshake as the permission one, deliberately: one slot
per question in `_questions`, keyed by request id, the same `approval/asked`
event, the same `POST /api/sessions/{sid}/approval` route. These tests drive a
real chat turn whose adapter asks one question, answer it through that route,
and assert the answer the handler returned reaches the turn's output.

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


def _questions_of(client):
    """The `_questions` slot dict, reached through the route that reads it.

    It is a closure inside `build_app`, so it is not an attribute; the route's
    free variables are the only handle — the same technique
    tests/test_r3_prompt_queue.py uses for the prompt queue. OD12-n2: question
    slots are keyed by REQUEST ID here, not by session."""
    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        names = fn.__code__.co_freevars
        if "_questions" in names:
            return fn.__closure__[names.index("_questions")].cell_contents
    raise AssertionError("could not reach the _questions slot")


def _slot_registry(client):
    """Either registry, so a probe can poll before the fix exists too.

    On the pre-OD12-n2 code only `_approvals` is a free variable of the route;
    after it, `_questions` is. Prefer the question registry."""
    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        names = fn.__code__.co_freevars
        for reg in ("_questions", "_approvals"):
            if reg in names:
                return fn.__closure__[names.index(reg)].cell_contents
    raise AssertionError("could not reach an approval slot registry")


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
        questions = _questions_of(c)
        out = {}

        def _turn():
            out["r"] = c.post(f"/api/sessions/{s['id']}/chat",
                              json={"message": "go"})

        t = threading.Thread(target=_turn, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and not questions:
            time.sleep(0.01)
        assert questions, (
            "the question never reached the approval channel the UI reads")
        slot = next(iter(questions.values()))
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


# --- OD-12: the decision is published, expiry included -----------------------


def _approval_events(body: str) -> list[dict]:
    """The `{event, data}` approval payloads the chat stream carried, in order.

    Read off the SSE body the browser reads rather than from a server closure: the
    wire is the contract the UI folds, so these assert on the wire itself."""
    out: list[dict] = []
    for block in body.split("\n\n"):
        name = ""
        data = ""
        for line in block.splitlines():
            if line.startswith("event: "):
                name = line[len("event: "):].strip()
            elif line.startswith("data: "):
                data += line[len("data: "):]
        if name != "approval" or not data:
            continue
        try:
            out.append(json.loads(data))
        except json.JSONDecodeError:
            continue
    return out


def _asked(body: str) -> list[dict]:
    return [d for d in _approval_events(body)
            if d.get("event") == "approval/asked"]


def _decided(body: str) -> list[dict]:
    return [d for d in _approval_events(body)
            if d.get("event") == "approval/decided"]


def test_a_question_that_times_out_says_expired_with_the_same_id(
        monkeypatch, tmp_path):
    """OD-12. On expiry the handler returns a real decline, and it must ALSO say so
    on the channel the ask went out on. Without the decided event the row stays
    `awaiting`, the form stays clickable, and the next click hits the route's 409
    and surfaces as an error — the server knew it declined and said nothing."""
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
    asked = _asked(r.text)
    decided = _decided(r.text)
    assert len(asked) == 1, asked
    assert len(decided) == 1, decided
    assert decided[0]["data"]["decision"] == "expired", decided[0]
    assert decided[0]["data"]["id"] == asked[0]["data"]["id"], (asked, decided)


def test_a_question_answered_inside_the_window_is_answered_not_expired(
        monkeypatch, tmp_path):
    """The other half of OD-12: an answer that beats the clock IS the decision, and
    the trail must say so. It must not also say `expired` — a pair would claim the
    server took the answer and let the question lapse."""
    s = _app_and_session(monkeypatch, tmp_path)

    def _fake_acp(prompt, **kw):
        answer = kw["on_question"]("elicitation/create", {
            "message": "Which directory?",
            "requestedSchema": {"type": "object",
                                "properties": {"path": {"type": "string"}}}})
        yield harness.TurnEvent("text", text=f"answer={json.dumps(answer)}")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        questions = _questions_of(c)
        out: dict = {}

        def _turn():
            out["r"] = c.post(f"/api/sessions/{s['id']}/chat",
                              json={"message": "go"})

        t = threading.Thread(target=_turn, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and not questions:
            time.sleep(0.01)
        assert questions, "the question never reached the channel"
        slot = next(iter(questions.values()))
        r = c.post(f"/api/sessions/{s['id']}/approval",
                   json={"requestId": slot["requestId"],
                         "answer": {"path": "C:/work"}})
        assert r.status_code == 200, r.text
        t.join(15)
        assert not t.is_alive(), "the turn never finished"

    body = out["r"].text
    assert "C:/work" in body, body
    decided = _decided(body)
    assert len(decided) == 1, decided
    assert decided[0]["data"]["decision"] == "answered", decided[0]
    assert decided[0]["data"]["id"] == slot["requestId"], decided


def test_the_expiry_fires_once_and_a_late_click_cannot_add_a_decision(
        monkeypatch, tmp_path):
    """OD-12's own scenario. By the time the row is stale the server has already
    said `expired` exactly once, and the click that follows is refused (409) rather
    than accepted — one ask never ends with two decisions."""
    s = _app_and_session(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "QUESTION_WAIT_SECS", 0.05)

    def _fake_acp(prompt, **kw):
        kw["on_question"]("elicitation/create", {"message": "Which?"})
        yield harness.TurnEvent("text", text="done")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "go"})
        asked = _asked(r.text)
        assert len(asked) == 1, asked
        # The turn is over, so the slot is gone: this is the late click.
        late = c.post(f"/api/sessions/{s['id']}/approval",
                      json={"requestId": asked[0]["data"]["id"],
                            "answer": {"path": "C:/work"}})

    assert late.status_code == 409, late.text
    decided = _decided(r.text)
    assert len(decided) == 1, decided
    assert decided[0]["data"]["decision"] == "expired", decided[0]


def test_an_answer_cannot_win_after_the_expiry_has_been_claimed(
        monkeypatch, tmp_path):
    """The lock's whole purpose. Once the handler has claimed the expiry it has
    already returned a decline and published `expired`; the route must refuse an
    answer that lands after that rather than accept one the server has denied — the
    same request cannot be both answered and expired."""
    s = _app_and_session(monkeypatch, tmp_path)
    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        questions = _questions_of(c)
        questions["q-claimed"] = {
            "requestId": "q-claimed", "answer": None, "kind": "question",
            "event": threading.Event(), "lock": threading.Lock(),
            "expired": True, "sid": s["id"],
        }
        r = c.post(f"/api/sessions/{s['id']}/approval",
                   json={"requestId": "q-claimed", "answer": {"path": "C:/work"}})
        assert r.status_code == 409, r.text
        assert not questions["q-claimed"]["event"].is_set(), (
            "an expired question must not be woken by a late answer")


def test_two_concurrent_questions_each_keep_their_own_slot(
        monkeypatch, tmp_path):
    """OD12-n2. Two questions in flight for ONE session must each be answerable
    by their OWN request id.

    Before the fix one `_approvals` slot per session let the second question
    overwrite the first: the first answer was refused (409, "no longer the one
    being waited on") and the first question expired although the user had
    answered it. Worse, the first handler's `finally` popped the SECOND
    question's slot, so a second question whose window was still open could no
    longer be answered at all."""
    s = _app_and_session(monkeypatch, tmp_path)
    monkeypatch.setattr(serve, "QUESTION_WAIT_SECS", 2.0)
    real_urandom = os.urandom
    minted = [b"\x01\x01\x01\x01", b"\x02\x02\x02\x02"]

    def _mint(n):
        if n == 4 and minted:
            return minted.pop(0)
        return real_urandom(n)

    monkeypatch.setattr(serve.os, "urandom", _mint)

    def _fake_acp(prompt, **kw):
        out: dict = {}

        def ask(tag):
            out[tag] = kw["on_question"]("elicitation/create",
                                         {"message": f"Q{tag}"})

        t1 = threading.Thread(target=ask, args=("1",), daemon=True)
        t2 = threading.Thread(target=ask, args=("2",), daemon=True)
        t1.start()
        time.sleep(0.3)
        t2.start()
        t1.join()
        t2.join()
        yield harness.TurnEvent("text", text=json.dumps(out))

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        registry = _slot_registry(c)
        out: dict = {}

        def _turn():
            out["r"] = c.post(f"/api/sessions/{s['id']}/chat",
                              json={"message": "go"})

        t = threading.Thread(target=_turn, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and not any(
                v.get("requestId") == "q-02020202" for v in registry.values()):
            time.sleep(0.01)
        assert any(v.get("requestId") == "q-02020202"
                   for v in registry.values()), "the second question never asked"

        first = c.post(f"/api/sessions/{s['id']}/approval",
                       json={"requestId": "q-01010101",
                             "answer": {"path": "FIRST"}})
        second = c.post(f"/api/sessions/{s['id']}/approval",
                        json={"requestId": "q-02020202",
                              "answer": {"path": "SECOND"}})
        assert first.status_code == 200, first.text
        assert second.status_code == 200, second.text
        t.join(15)
        assert not t.is_alive(), "the turn never finished"

    body = out["r"].text
    assert "FIRST" in body, body
    assert "SECOND" in body, body
    decided = {d["data"]["id"]: d["data"]["decision"] for d in _decided(body)}
    assert decided == {"q-01010101": "answered",
                       "q-02020202": "answered"}, decided


def test_an_answer_for_another_session_is_refused_and_does_not_reach_the_turn(
        monkeypatch, tmp_path):
    """OD12N2-n1. Keying question slots by request id alone dropped the
    per-session scoping the old `_approvals` slot had for free: an answer for
    session A posted through session B's route was accepted (200) and landed on
    A's question. The request id is 32 random bits shown only on A's own stream,
    so this is a dropped check rather than a misdirected answer — but it is the
    check the `_approvals` comment says the session key exists for, and the base
    refused it 409.

    The same-session answer must be unchanged: still 200, still delivered."""
    s = _app_and_session(monkeypatch, tmp_path)
    other = sessions.create(title="other")
    other["harness"] = "mcode"
    other["mcode_transport"] = "acp"
    sessions.save(other)
    monkeypatch.setattr(serve, "QUESTION_WAIT_SECS", 5.0)

    def _fake_acp(prompt, **kw):
        answer = kw["on_question"]("elicitation/create", {
            "message": "Which directory?",
            "requestedSchema": {"type": "object",
                                "properties": {"path": {"type": "string"}}}})
        yield harness.TurnEvent("text", text=f"answer={json.dumps(answer)}")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        questions = _questions_of(c)
        out: dict = {}

        def _turn():
            out["r"] = c.post(f"/api/sessions/{s['id']}/chat",
                              json={"message": "go"})

        t = threading.Thread(target=_turn, daemon=True)
        t.start()
        deadline = time.time() + 10
        while time.time() < deadline and not questions:
            time.sleep(0.01)
        assert questions, "the question never reached the channel"
        slot = next(iter(questions.values()))

        hijack = c.post(f"/api/sessions/{other['id']}/approval",
                        json={"requestId": slot["requestId"],
                              "answer": {"path": "HIJACK"}})
        assert hijack.status_code == 409, hijack.text
        # The owning session's answer is untouched by the refusal above.
        legit = c.post(f"/api/sessions/{s['id']}/approval",
                       json={"requestId": slot["requestId"],
                             "answer": {"path": "LEGIT"}})
        assert legit.status_code == 200, legit.text
        t.join(15)
        assert not t.is_alive(), "the turn never finished"

    body = out["r"].text
    assert "LEGIT" in body, body
    assert "HIJACK" not in body, (
        "the cross-session answer reached the turn's question")
    decided = _decided(body)
    assert len(decided) == 1, decided
    assert decided[0]["data"]["decision"] == "answered", decided[0]
    assert decided[0]["data"]["id"] == slot["requestId"], decided


def test_a_slot_does_not_survive_a_raising_ask_publish(monkeypatch, tmp_path):
    """OD12N2-n2. The slot is registered and then the `approval/asked` event is
    published. A raising publish (a closed loop at shutdown) used to leave the
    slot behind — and unlike the per-session `_approvals`, `_questions` is never
    overwritten, so the leak is permanent and a later answer with that id would
    be accepted for a question that no longer exists.

    The publish is made to raise here (the loop the app is running on is the one
    that raises), which is exactly the window between registration and the
    guarded wait."""
    s = _app_and_session(monkeypatch, tmp_path)

    def _fake_acp(prompt, **kw):
        kw["on_question"]("elicitation/create", {"message": "Which?"})
        yield harness.TurnEvent("text", text="unreachable")

    monkeypatch.setattr(harness_mcode, "drive_turn_acp", _fake_acp)

    import asyncio
    real_cst = asyncio.BaseEventLoop.call_soon_threadsafe

    def _raising_cst(self, callback, *args):
        ev = args[0] if args else None
        if (getattr(ev, "event", "") == "approval/asked"
                and (getattr(ev, "data", None) or {}).get("kind") == "question"):
            raise RuntimeError("event loop is closed")
        return real_cst(self, callback, *args)

    monkeypatch.setattr(asyncio.BaseEventLoop, "call_soon_threadsafe",
                        _raising_cst)

    with TestClient(serve.build_app(upstream_port=DUMMY_PORT)) as c:
        questions = _questions_of(c)
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "go"})
        assert r.status_code == 200, r.text
        assert questions == {}, (
            "the slot leaked: a question whose ask never published is still "
            "registered and could be answered")

    # The publish failure is reported, not swallowed into a silent decline.
    assert "event loop is closed" in r.text, r.text
