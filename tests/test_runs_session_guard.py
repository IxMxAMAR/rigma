"""AUDIT 03-5: a normal chat turn must not run against a session an active
autonomous run is driving.

A run reaches `_llm_turn` through `_drain_turn` directly, so it never joins
`_streaming`; `chat_turn` never looked at `session["run_id"]`. Typing into the
run's chat therefore started a second agent loop on the same transcript. The
turn is now refused with 409 while the run is running/paused, and the run's
chat is hidden from the rail until the run ends.

The run state is stubbed with `runs.create` (status running, no loop): the
guard is about the run's STATUS, not about a live task, and no engine is
contacted.
"""
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions
from rigma import state as st


def _client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    c = TestClient(serve.build_app(upstream_port=1))
    c.__enter__()
    return c


def _stub_run(sess):
    """A run owning `sess`, exactly as start_run links them (status running)."""
    rid = runs.create("mission", sess["id"])["id"]
    sess["run_id"] = rid
    sessions.save(sess)
    return rid


def test_chat_turn_is_refused_while_a_run_drives_the_session(
        tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("run chat")
        _stub_run(sess)                                    # running, no loop

        r = c.post(f"/api/sessions/{sess['id']}/chat",
                   json={"message": "hi from the chat"})
        assert r.status_code == 409, r.text
        assert "autonomous run" in r.json()["error"]
        # the message was not appended to the run's transcript
        assert sessions.load(sess["id"])["messages"] == []
    finally:
        c.__exit__(None, None, None)


def test_the_rails_hides_the_run_session_while_the_run_is_live(
        tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("run chat")
        rid = _stub_run(sess)
        ids = [s["id"] for s in c.get("/api/sessions").json()]
        assert sess["id"] not in ids, "the run's chat is a writable rail row"

        # a terminal run no longer owns the chat: it comes back to the rail
        runs.set_status(runs.load(rid), "stopped", "test teardown")
        ids = [s["id"] for s in c.get("/api/sessions").json()]
        assert sess["id"] in ids
    finally:
        c.__exit__(None, None, None)


def test_a_paused_run_still_owns_the_session(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("run chat")
        rid = _stub_run(sess)
        r = runs.load(rid)
        r["paused"] = True
        runs.save(r)
        assert c.post(f"/api/sessions/{sess['id']}/chat",
                      json={"message": "hi"}).status_code == 409
    finally:
        c.__exit__(None, None, None)


def test_a_plain_chat_is_not_blocked_by_the_run_guard(tmp_path, monkeypatch):
    """The guard must key on a LIVE run, not on the presence of run_id."""
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("ordinary chat")
        # empty session + no message -> the route's own 400, i.e. the guard let
        # it through (409 would mean it did not)
        r = c.post(f"/api/sessions/{sess['id']}/chat", json={})
        assert r.status_code == 400, r.text
    finally:
        c.__exit__(None, None, None)


def test_a_finished_runs_session_is_not_blocked(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("old run chat")
        rid = _stub_run(sess)
        runs.set_status(runs.load(rid), "done", "task_complete (verified)")
        r = c.post(f"/api/sessions/{sess['id']}/chat", json={})
        assert r.status_code == 400, r.text      # guard passed, route's own 400
    finally:
        c.__exit__(None, None, None)
