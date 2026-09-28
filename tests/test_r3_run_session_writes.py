"""R3-2: the run loop's remaining unguarded whole-row session saves.

AUDIT F1 gave the session store `base_rev`, and run 2's 03-5 fix used it for the
run loop's DRIVING-LINE write. The loop's post-turn section still took "ONE load
for the whole post-turn section" and wrote that snapshot back from three more
places with no guard — the tool-result append, the `run_profile` flip, and the
`finally` cleanup that clears `mission`/`run_id`. Every one of them is a
whole-row replace, so any writer that landed between the load and the write
(a rename, a PATCH, the owner typing the instant the run stops) was erased.

The race is injected at the loop's OWN write, so the test also asserts that the
write the loop issued CARRIED a `base_rev` — otherwise the marker could survive
by luck and the test would pass for the wrong reason.
"""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions
from rigma import state as st


class _Engine(BaseHTTPRequestHandler):
    """Scripted engine: each /v1/chat/completions emits the next tool_call."""
    script = []
    idx = 0

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if not body.get("stream", True):
            payload = json.dumps({"choices": [{"message": {
                "content": "not a spec"}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        i = _Engine.idx
        _Engine.idx += 1
        step = _Engine.script[i] if i < len(_Engine.script) \
            else _Engine.script[-1]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        nm, ar = step
        sse({"choices": [{"delta": {"tool_calls": [
            {"index": 0, "id": f"c{i}", "type": "function",
             "function": {"name": nm, "arguments": json.dumps(ar)}}]}}]})
        self.wfile.write(b"data: [DONE]\n\n")

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
    _Engine.idx = 0
    _Engine.script = []
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


def _wait(client, rid, timeout=30):
    # R5-TESTWAIT: this used to `return client.get(...)` on timeout, i.e. hand
    # back whatever the run happened to be — usually still `running`. Every caller
    # then asserted against a state the test had explicitly waited to leave, so a
    # timing failure surfaced as a wrong-state assertion far from the cause. It is
    # the same defect as test_phase4_lifecycle._wait, which produced an
    # intermittent `assert None is True` in a restart test under full-suite load:
    # the stale "running" snapshot was re-saved over a stopped run. `runs.save`
    # guards terminal-is-sticky in the other direction and cannot help when the
    # stale value is the one being written, so the wait itself must fail loudly.
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    raise AssertionError(
        f"run {rid} did not reach terminal within {timeout}s; "
        f"last seen: status={r.get('status')!r} iteration={r.get('iteration')!r} "
        f"halt_reason={r.get('halt_reason')!r}"
    )


def test_the_run_loop_does_not_erase_a_concurrent_session_write(engine, home):
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "do a thing",
                                    "budget_hours": 1}).json()["id"]
    sid = runs.load(rid)["session_id"]

    real_save = sessions.save
    seen = {"injected": False, "guarded": 0}

    def racing_save(session, **kw):
        base = kw.get("base_rev")
        has_tr = any(m.get("kind") == "tool_result"
                     for m in session.get("messages", []))
        if has_tr:
            if base is not None:
                seen["guarded"] += 1
            if not seen["injected"]:
                seen["injected"] = True
                # a second writer lands: rename, PATCH, anything
                fresh = sessions.load(sid)
                fresh["messages"].append({"role": "user",
                                          "content": "MARKER-CONCURRENT-WRITE"})
                real_save(fresh, base_rev=fresh[sessions.REV_KEY])
        return real_save(session, **kw)

    sessions.save = racing_save
    try:
        _wait(c, rid)
    finally:
        sessions.save = real_save

    assert seen["injected"], (
        "the probe never saw the run loop's tool-result write — the code path "
        "moved; re-point this test")
    assert seen["guarded"] >= 1, (
        "the run loop's tool-result write went out with no base_rev: a "
        "concurrent write to the same session is erased by it")
    stored = sessions.load(sid)
    texts = [str(m.get("content") or "") for m in stored["messages"]]
    assert any("MARKER-CONCURRENT-WRITE" in t for t in texts), (
        "the run loop's tool-result save erased a message another writer "
        f"stored in between; messages now: {texts}")
    assert any(m.get("kind") == "tool_result"
               for m in stored["messages"]), (
        "the tool result itself must still be persisted")
