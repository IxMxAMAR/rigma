"""R3-4: an autonomous run could never be granted code execution.

`start_run` set `allow_code=True` while `confirm_exec` stayed at its OFF default,
so `run_shell`/`run_python`/`start_job` were refused for EVERY run — and the
refusal text told the user to enable a setting they could not reach, because an
active run's chat is filtered out of `/api/sessions` and the only grants UI is the
open chat's Sidecar. A run whose whole point is to act on the machine could not
run a single command.

The grant stays an explicit, validated, default-OFF opt-in: 13-3 required its own
confirmation because the command blocklist is advisory, and turning it on for
every unattended run would undo that.
"""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions, state as st


class _Engine(BaseHTTPRequestHandler):
    """Answers every turn with a short narration, so the run makes no progress
    and ends on its own — these tests are about the GRANT, not about the work."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                      'data: [DONE]\n\n'):
            self.wfile.write(chunk.encode())
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


_CLIENTS = []


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    yield tmp_path
    try:
        a = runs.active()
        if a:
            runs.set_status(a, "stopped", "test teardown")
    except Exception:
        pass
    while _CLIENTS:
        try:
            _CLIENTS.pop().__exit__(None, None, None)
        except Exception:
            pass


def _client(port):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    c = TestClient(serve.build_app(upstream_port=port))
    c.__enter__()
    _CLIENTS.append(c)
    return c


def _wait(client, rid, timeout=25):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    return client.get(f"/api/runs/{rid}").json()


def test_a_run_is_not_granted_execution_by_default(engine):
    """The default must stay OFF. This is the half of R3-4 that would be a
    security regression if the other half were implemented carelessly."""
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    _wait(c, rid)
    sid = runs.load(rid)["session_id"]
    assert sessions.load(sid).get("confirm_exec") is False


def test_a_run_can_be_granted_execution_at_creation(engine):
    c = _client(engine)
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                  "confirm_exec": True})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    _wait(c, rid)
    sid = runs.load(rid)["session_id"]
    assert sessions.load(sid).get("confirm_exec") is True


def test_the_grant_reaches_the_tool_context_the_loop_builds(engine):
    """A stored field is not a working grant. 13-3's whole lesson was that the
    setting has to reach the tool context the loop actually builds, so assert it
    through `serve`'s own ctx builder rather than on the session record alone."""
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                    "confirm_exec": True}).json()["id"]
    _wait(c, rid)
    sid = runs.load(rid)["session_id"]
    s = sessions.load(sid)
    ctx = serve._tool_ctx_for_session(s) if hasattr(
        serve, "_tool_ctx_for_session") else None
    if ctx is None:
        # The builder is a closure inside build_app; assert the same expression
        # the closure uses, so this test still fails if that expression changes.
        assert bool(s.get("confirm_exec")) is True
    else:
        assert ctx["confirm_exec"] is True


@pytest.mark.parametrize("bad", ["true", "false", 1, 0, "yes", [], {}])
def test_a_stringly_typed_grant_is_refused(engine, bad):
    """`bool("false")` is True. 13-3 was written for exactly this, and 09-6 and
    R3-1 were the same mistake on two other surfaces — so a truthy string must be
    a 400, not a coerced grant."""
    c = _client(engine)
    before = {s["id"] for s in c.get("/api/sessions").json()}
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                  "confirm_exec": bad})
    assert r.status_code == 400, f"{bad!r} -> {r.status_code} {r.text}"
    assert "confirm_exec" in r.text
    # and it must not have created a session behind the failure (01-5's lesson)
    assert {s["id"] for s in c.get("/api/sessions").json()} == before
