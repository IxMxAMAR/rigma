"""A18: the run loop's FIRST load must survive a transient unreadable run.json.

`runs.load` swallows every exception and returns None, so the first load in
`_run_loop` — a bare `_runs.load(run_id)` followed by `run["session_id"]` —
raised TypeError on ONE transient read, killed the driver task before its first
iteration, and left the run `running` with its slot claimed: pause and inject
answered 409 "run has no driver", restart 409 "run is running", a new run 409
"a run is already active". Only Stop cleared it.

R3-RUN-5 gave the INNER loop exactly this guard (`_load_run_for_loop` retries,
then the slot is released with a terminal status); the first load never got it.
Found by the full suite: tests/test_phase4_lifecycle.py::
test_restart_reattaches_and_finishes failed under suite concurrency with the
`run["session_id"]` line in the traceback and passed in isolation.

Both halves are exercised here: a one-shot failure is retried and the loop keeps
driving, and a state that never becomes readable releases the slot with a
terminal status and a reason instead of stranding it.
"""
import inspect
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve
from rigma import state as st


class _Engine(BaseHTTPRequestHandler):
    """Scripted engine: each streaming /v1/chat/completions emits the next
    tool_call; non-streaming calls (the mission compiler) answer with junk."""
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


def _fail_the_loops_load_once(seen):
    """A `runs.load` that returns None exactly once, and only for the call the
    RUN LOOP makes — not for `runs.save`'s sticky-status probe or for the GET
    route, which would make the failure something other than the loop's first
    load. `_run_loop` calls `_runs.load` directly before the fix and through
    `_load_run_for_loop` after it, so both callers count."""
    real = runs.load

    def flaky(rid):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if frame is not None else ""
        if not seen["failed"] and caller in ("_run_loop", "_load_run_for_loop"):
            seen["failed"] = True
            seen["caller"] = caller
            return None
        return real(rid)

    return flaky


def _wait_for(c, rid, until, timeout=10.0):
    """Poll the run until `until(r)`, or fail with the last state seen."""
    end = time.monotonic() + timeout
    r: dict = {}
    while time.monotonic() < end:
        r = c.get(f"/api/runs/{rid}").json()
        if until(r):
            return r
        time.sleep(0.05)
    raise AssertionError(f"run {rid} never satisfied the wait; last seen {r}")


def test_the_first_load_retries_a_transient_read(engine, home):
    """ONE transient unreadable run.json must not kill the driver task."""
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"})]
    c = _client(engine)
    seen = {"failed": False, "caller": ""}
    real = runs.load
    runs.load = _fail_the_loops_load_once(seen)
    try:
        rid = c.post("/api/runs", json={"mission": "small job",
                                        "budget_hours": 1}).json()["id"]
        r = _wait_for(c, rid, lambda r: r.get("iteration", 0) >= 1)
    finally:
        runs.load = real
    assert seen["failed"], ("the probe never failed the loop's first load — the "
                            "code path moved; re-point this test")
    assert r.get("iteration", 0) >= 1, (
        "the run loop died on its first load: one transient unreadable run.json "
        f"left it with no driver; last seen {r}")
    # …and a driver is genuinely there to service a control request, not just a
    # run that happens to read as `running`.
    assert c.post(f"/api/runs/{rid}/pause").status_code == 200


def test_an_unreadable_first_load_releases_the_slot(engine, home):
    """When the retries are exhausted there is no run dict to hand `set_status`
    and `runs.active()` cannot see the run either (it loads the same file), so
    the slot has to be released by id. Left claimed, every later run 409s."""
    c = _client(engine)
    real = runs.load
    runs.load = lambda _rid: None       # the state never becomes readable
    try:
        rid = c.post("/api/runs", json={"mission": "small job",
                                        "budget_hours": 1}).json()["id"]
        end = time.monotonic() + 10
        doc = None
        while time.monotonic() < end:
            try:
                doc = json.loads((runs.run_dir(rid) / "run.json")
                                 .read_text(encoding="utf-8"))
            except Exception:
                doc = None
            if doc and doc.get("status") in runs.TERMINAL:
                break
            time.sleep(0.05)
    finally:
        runs.load = real
    assert doc is not None, "the run was never written at all"
    assert doc.get("id") == rid
    assert doc.get("status") in runs.TERMINAL, (
        f"an unreadable run.json left the run non-terminal: {doc}")
    assert doc.get("status") == "interrupted"
    assert doc.get("halt_reason"), "the release must say WHY it happened"
    assert doc.get("stop_reason") == "interrupted"
    assert runs.active() is None, (
        "the run slot stayed claimed — every later POST /api/runs answers 409 "
        '"a run is already active"')
