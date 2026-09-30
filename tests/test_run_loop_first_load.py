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
import logging
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


# ---------------------------------------------------------------------------
# A18c: a READABLE but UNUSABLE run.json is the same defect class.
#
# A18 only guarded `runs.load` returning None. But `runs.load` is a bare
# `json.loads`: `{}`, `[]` and `{"session_id": null}` are all VALID JSON, parse
# fine, and then die at `run["session_id"]` — the loop task gone, the run
# `running` on disk, the slot still claimed. `_run_is_drivable` closes it.
# ---------------------------------------------------------------------------

# Every shape that parses but cannot be driven. `{"id": rid, "session_id": ""}`
# is the nastiest: it HAS an id, so it reaches the session_id dereference.
_BAD_RECORDS = [
    {},
    [],
    {"session_id": None},
    {"session_id": ""},
    {"id": "placeholder", "session_id": ""},
]


def _terminal_doc(rid):
    """The run.json as a dict, or None while it is absent/torn."""
    try:
        return json.loads((runs.run_dir(rid) / "run.json").read_text(
            encoding="utf-8"))
    except Exception:
        return None


@pytest.mark.parametrize("bad", _BAD_RECORDS)
def test_load_run_for_loop_rejects_an_undrivable_record(home, bad):
    """The inner loop's loader must not hand the loop a record it will crash
    on. `(None, False)` is what makes the caller release the slot."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    (runs.run_dir(rid) / "run.json").write_text(
        json.dumps(bad), encoding="utf-8")

    got = serve._load_run_for_loop(runs, rid)

    assert got == (None, False), (
        f"a readable-but-unusable run.json ({bad!r}) was reported as drivable: "
        f"{got!r} — the loop would die at run['session_id']")


def test_load_run_for_loop_still_accepts_a_real_record(home):
    """The guard must not reject the healthy case."""
    run = runs.create("mission", "sess-1")

    got, readable = serve._load_run_for_loop(runs, run["id"])

    assert readable is True
    assert got["session_id"] == "sess-1"


@pytest.mark.parametrize("bad", _BAD_RECORDS)
def test_release_handles_an_undrivable_record(home, bad):
    """`_release_unreadable_run` must write a terminal status for a record that
    is not a dict (or has no usable id), not crash on `run.get` / `run['id']`."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    record = ({**bad, "id": rid} if isinstance(bad, dict) and "id" not in bad
              else bad)
    (runs.run_dir(rid) / "run.json").write_text(
        json.dumps(record), encoding="utf-8")

    serve._release_unreadable_run(runs, rid, "run state could not be read")

    doc = _terminal_doc(rid)
    assert doc is not None, "the release never wrote a record"
    assert doc.get("id") == rid
    assert doc.get("status") == "interrupted", (
        f"a usable-looking-but-undrivable run.json left the run non-terminal: "
        f"{doc}")
    assert doc.get("halt_reason"), "the release must say WHY it happened"
    assert doc.get("stop_reason") == "interrupted"
    assert runs.active() is None, "the run slot stayed claimed"


def test_an_empty_object_run_json_releases_the_loop_slot(engine, home):
    """The exact reported case, end to end: a run whose run.json is `{}` while
    the loop is loading it must end terminal, not `running` forever."""
    c = _client(engine)
    real = runs.load

    def empty_for_the_loop(rid):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if frame is not None else ""
        if caller in ("_run_loop", "_load_run_for_loop"):
            return {}
        return real(rid)

    runs.load = empty_for_the_loop
    try:
        rid = c.post("/api/runs", json={"mission": "small job",
                                        "budget_hours": 1}).json()["id"]
        end = time.monotonic() + 10
        doc = None
        while time.monotonic() < end:
            doc = _terminal_doc(rid)
            if doc and doc.get("status") in runs.TERMINAL:
                break
            time.sleep(0.05)
    finally:
        runs.load = real
    assert doc is not None and doc.get("id") == rid
    assert doc.get("status") == "interrupted", (
        f"run.json = {{}} wedged the run: {doc}")
    assert doc.get("halt_reason")
    assert runs.active() is None


# ---------------------------------------------------------------------------
# A18d: three bounded robustness nits on the release path.
# ---------------------------------------------------------------------------


def test_a_release_preserves_the_unreadable_bytes(home):
    """The recoverability claim, pinned: whatever was in run.json is copied
    aside before the minimal terminal record replaces it."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    (runs.run_dir(rid) / "run.json").write_text("{ not json", encoding="utf-8")

    serve._release_unreadable_run(runs, rid, "why")

    backups = list(runs.run_dir(rid).glob("run.json.unreadable-*"))
    assert len(backups) == 1, backups
    assert backups[0].read_text(encoding="utf-8") == "{ not json"


def test_a_second_release_in_the_same_second_keeps_its_own_backup(
        home, monkeypatch):
    """A18d(2): the old name was second-granular and guarded by
    `not dst.exists()`, so a second release in the same second kept only the
    first run's bytes. Frozen time makes the collision deterministic."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    rj = runs.run_dir(rid) / "run.json"
    monkeypatch.setattr(serve.time, "time", lambda: 1_700_000_000.0)

    rj.write_text("{}", encoding="utf-8")
    serve._release_unreadable_run(runs, rid, "first")
    rj.write_text('{"broken": 1}', encoding="utf-8")
    serve._release_unreadable_run(runs, rid, "second")

    backups = sorted(p.read_text(encoding="utf-8")
                     for p in runs.run_dir(rid).glob("run.json.unreadable-*"))
    assert backups == ['{"broken": 1}', "{}"], (
        f"a second release in the same second lost its own bytes: {backups}")


def test_a_failed_backup_copy_is_reported_not_silent(home, monkeypatch, caplog):
    """A18d(1): a transient copy failure must not silently overwrite the
    original with no backup and no trace. The copy is retried, then logged; the
    terminal release still lands (the slot is what the user is waiting on)."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    (runs.run_dir(rid) / "run.json").write_text("{}", encoding="utf-8")

    def boom(*a, **k):
        raise PermissionError(32, "sharing violation")

    monkeypatch.setattr(serve.shutil, "copy2", boom)
    with caplog.at_level(logging.WARNING, logger="rigma.serve"):
        serve._release_unreadable_run(runs, rid, "run state could not be read")

    doc = _terminal_doc(rid)
    assert doc is not None and doc.get("status") == "interrupted", doc
    assert not list(runs.run_dir(rid).glob("run.json.unreadable-*"))
    assert any("no backup" in r.getMessage() for r in caplog.records), (
        [r.getMessage() for r in caplog.records])


def test_an_unwritable_run_json_logs_the_residual_loudly(
        home, monkeypatch, caplog):
    """A18d(3): when run.json is unreadable AND unwritable the terminal status
    cannot land, so the file keeps saying `running` even though the slot is
    released. That residual must be named at ERROR, not hidden behind a generic
    'could not write'."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    (runs.run_dir(rid) / "run.json").write_text("{}", encoding="utf-8")

    def boom(*a, **k):
        raise PermissionError(32, "locked")

    monkeypatch.setattr(runs, "set_status", boom)
    with caplog.at_level(logging.ERROR, logger="rigma.serve"):
        serve._release_unreadable_run(runs, rid, "run state could not be read")

    assert runs.active() is None, "the slot must be released regardless"
    assert any("still says 'running'" in r.getMessage()
               for r in caplog.records), (
        [r.getMessage() for r in caplog.records])
