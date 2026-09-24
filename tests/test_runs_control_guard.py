"""AUDIT 03-6: pause/resume/inject must 409 unless a driver will service them.

All three wrote the field and returned the run without consulting the status
or `_run_tasks`: resume on a `done` run answered 200, inject answered
{"queued": true} for guidance nothing would ever consume, and pause left
`paused: true` on a terminal run for the next restart to clear.

The run state is stubbed with `runs.create` — no engine, no loop.
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


def _post(c, rid, path, body=None):
    return (c.post(f"/api/runs/{rid}/{path}", json=body) if body is not None
            else c.post(f"/api/runs/{rid}/{path}"))


def test_control_endpoints_refuse_a_terminal_run(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("done chat")
        rid = runs.create("m", sess["id"])["id"]
        runs.set_status(runs.load(rid), "done", "task_complete (verified)")

        for path, body in (("pause", None), ("resume", None),
                           ("inject", {"message": "hello?"})):
            r = _post(c, rid, path, body)
            assert r.status_code == 409, (path, r.status_code, r.text)
            assert "restart it first" in r.json()["error"]

        r = runs.load(rid)
        assert r["paused"] is False, "pause wrote paused onto a terminal run"
        assert not r.get("steer_queue"), "inject queued into a dead run"
    finally:
        c.__exit__(None, None, None)


def test_control_endpoints_refuse_a_run_with_no_driver(tmp_path, monkeypatch):
    """Status `running` is not enough: a boot-orphaned run has no task."""
    c = _client(tmp_path, monkeypatch)
    try:
        sess = sessions.create("orphan chat")
        rid = runs.create("m", sess["id"])["id"]        # running, no loop

        for path, body in (("pause", None), ("resume", None),
                           ("inject", {"message": "hi"})):
            r = _post(c, rid, path, body)
            assert r.status_code == 409, (path, r.status_code, r.text)
            assert "no driver" in r.json()["error"]
        assert runs.load(rid)["paused"] is False
    finally:
        c.__exit__(None, None, None)


def test_control_endpoints_still_404_an_unknown_run(tmp_path, monkeypatch):
    c = _client(tmp_path, monkeypatch)
    try:
        assert _post(c, "nope", "pause").status_code == 404
        assert _post(c, "nope", "inject", {"message": "x"}).status_code == 404
    finally:
        c.__exit__(None, None, None)
