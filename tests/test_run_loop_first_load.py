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
import pathlib
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions
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
# DR1: the release must not destroy a record that was only LOCKED.
#
# `runs.load` returns None for both "the bytes could not be read" and "the bytes
# were read and do not parse". The release treated both as unusable and wrote a
# 4-field stub `{"id","status","halt_reason","stop_reason"}` over whatever was
# there. A transient sharing violation against the atomic replace clears within
# milliseconds, and `set_status` -> `atomic_write_text` retries the replace for
# ~1 s — so the stub landed over a GOOD file: `session_id`, `mission`, `spec`,
# `deadline` and every counter gone, and `restart_run` answering 409 "the run's
# chat session was deleted" from then on, permanently.
#
# The test above pins only status and slot over a valid file, which is exactly
# how the destructive overwrite got through. These two assert the behaviour the
# owner needs: the record survives, and the run can still be RESTARTED.
# ---------------------------------------------------------------------------


def test_a_transiently_unreadable_run_json_survives_the_release(
        engine, home):
    """The owner's need, end to end: after a run's state was unreadable long
    enough to release the slot, `restart` must still reattach — which it can
    only do if `session_id` (and the mission) are still on disk."""
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"})]
    c = _client(engine)
    sid = c.post("/api/sessions", json={}).json()["id"]
    run = runs.create("keep my mission", sid)
    rid = run["id"]

    real = runs.load
    runs.load = lambda _rid: None       # unreadable at the release
    try:
        serve._release_unreadable_run(runs, rid, "run state could not be read")
    finally:
        runs.load = real

    doc = json.loads((runs.run_dir(rid) / "run.json").read_text(
        encoding="utf-8"))
    assert doc.get("status") == "interrupted", doc
    assert doc.get("session_id") == sid, (
        "the release replaced a GOOD run.json with a 4-field stub — "
        f"session_id is gone: {doc}")
    assert doc.get("mission") == "keep my mission", doc
    assert doc.get("deadline"), doc
    assert runs.active() is None, "the slot stayed claimed"

    # …and the consequence the stub caused: a permanent 409.
    resp = c.post(f"/api/runs/{rid}/restart")
    assert resp.status_code == 200, (
        "restart could not reattach after the release: "
        f"{resp.status_code} {resp.text}")
    assert resp.json().get("restarted") is True


def test_a_failed_run_json_read_leaves_the_file_alone(home, monkeypatch):
    """DR1, the transient-lock case: the READ raised, so the bytes on disk are
    not evidence of anything. They must be left byte-identical — no stub, and no
    backup either, because there is nothing wrong with them."""
    run = runs.create("mission", "sess-1")
    rid = run["id"]
    rj = runs.run_dir(rid) / "run.json"
    before = rj.read_bytes()

    real_load, real_read = runs.load, pathlib.Path.read_text
    runs.load = lambda _rid: None

    def _locked(self, *a, **k):
        if self == rj:
            raise PermissionError(32, "The process cannot access the file")
        return real_read(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", _locked)
    try:
        serve._release_unreadable_run(runs, rid, "run state could not be read")
    finally:
        runs.load = real_load

    assert rj.read_bytes() == before, (
        "a run.json whose READ failed was overwritten anyway — the bytes may "
        "have been a perfectly good record")
    assert not list(runs.run_dir(rid).glob("run.json.unreadable-*")), (
        "there was nothing wrong with the bytes, so there is nothing to back up")
    # the pointer was readable and pointed at THIS run, so the slot is released
    assert runs.active() is None


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


# ---------------------------------------------------------------------------
# DR8: the first-load `return` sat BEFORE the try/finally, and an unreadable
# active.json was treated as this run's claim.
# ---------------------------------------------------------------------------


def test_a_released_first_load_still_runs_the_loops_cleanup(engine, home):
    """DR8: a run released for an unreadable state skipped every piece of
    end-of-run cleanup, because its `return` was before the `try/finally`. The
    owner sees two of the consequences: the chat keeps `mission`/`run_id` (so it
    is never auto-titled again and compacts in masking style), and progress.md
    never gets its "RUN INTERRUPTED" line.

    The lock is transient in the way the guard exists for: it is gone by the
    time the release re-reads, so the run IS released and its record IS
    recovered — the cleanup then has a session to clear."""
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "small job",
                                    "budget_hours": 1}).json()["id"]
    sid = runs.load(rid)["session_id"]

    real = runs.load
    seen = {"n": 0}

    def flaky(rid_):
        frame = inspect.currentframe().f_back
        caller = frame.f_code.co_name if frame is not None else ""
        # exactly the loop's three retries fail; the release's own re-read and
        # the finally's re-read see the file again
        if caller == "_load_run_for_loop" and seen["n"] < 3:
            seen["n"] += 1
            return None
        return real(rid_)

    runs.load = flaky
    try:
        _wait_for(c, rid, lambda r: r.get("status") in runs.TERMINAL)
        end = time.monotonic() + 10
        while time.monotonic() < end:
            if not (sessions.load(sid) or {}).get("mission"):
                break
            time.sleep(0.05)
    finally:
        runs.load = real

    assert seen["n"] == 3, ("the probe never failed the loop's first load — the "
                            "code path moved; re-point this test")
    sess = sessions.load(sid)
    assert not sess.get("mission"), (
        "the chat kept the run's mission — the first-load return skipped the "
        "loop's finally, so it is never auto-titled again")
    assert not sess.get("run_id"), sess.get("run_id")
    log = (runs.run_dir(rid) / "progress.md").read_text(encoding="utf-8")
    assert "RUN INTERRUPTED" in log, (
        "the run ended with no RUN INTERRUPTED line in progress.md: " + log)


def test_an_unreadable_pointer_does_not_clear_another_runs_claim(home):
    """DR8: `runs.active()` returns None while a pointer is unreadable, so
    `POST /api/runs` can already have started run B in that window. Run A's
    release used to read the pointer as `{"id": A}` whenever the read failed and
    clear it — dropping B's claim and leaving B's driver running with no slot."""
    a = runs.create("A", "sess-a")
    b = runs.create("B", "sess-b")        # active.json now points at B
    ap = runs._active_path()
    assert json.loads(ap.read_text(encoding="utf-8"))["id"] == b["id"]

    real_load, real_read = runs.load, pathlib.Path.read_text

    def _locked(self, *a_, **k):
        if self == ap:
            raise PermissionError(32, "The process cannot access the file")
        return real_read(self, *a_, **k)

    runs.load = lambda rid: None if rid == a["id"] else real_load(rid)
    pathlib.Path.read_text = _locked
    try:
        serve._release_unreadable_run(runs, a["id"], "run state could not be read")
    finally:
        runs.load = real_load
        pathlib.Path.read_text = real_read

    assert ap.exists(), (
        "run A's release DELETED active.json while it was unreadable — that is "
        "run B's claim")
    assert json.loads(real_read(ap, encoding="utf-8"))["id"] == b["id"]


# ---------------------------------------------------------------------------
# DR1-residual: a run the boot reaper could not see.
#
# When the READ failed and the pointer was readable (the common sub-case), the
# release clears active.json and leaves run.json saying `running`. The boot
# reaper only reconciled `_runs.active()`, so this run was INVISIBLE to it:
# `restart_run` answered 409 "run is running" across reboots, and
# pause/resume/inject 409 "run has no driver" — a dead end only Stop cleared.
#
# The reaper now sweeps every run DIRECTORY, not just the pointer. It reads
# through `runs.load`, so a file that still cannot be READ is left exactly as it
# is — DR1's guarantee is preserved: reconcile means an honest terminal status,
# never a blind stub.
# ---------------------------------------------------------------------------

def test_the_boot_reaper_reconciles_a_run_the_pointer_cannot_see(home):
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()                    # what the read-failed release does

    assert runs.active() is None
    assert runs.load(rid)["status"] == "running"
    assert "running" not in runs.RESTARTABLE

    serve._reconcile_orphaned_runs(runs)

    doc = runs.load(rid)
    assert doc["status"] == "interrupted", doc
    assert doc["status"] in runs.RESTARTABLE, (
        "the reaper wrote a status restart_run still refuses")
    assert doc["halt_reason"], "the reaper must say WHY it reconciled"


def test_the_boot_reaper_never_overwrites_a_run_json_it_cannot_read(
        home, monkeypatch):
    """DR1's core guarantee, re-pinned on the new sweep: a run.json whose read
    failed may be a perfectly good record a transient lock hid, so it is left
    byte-identical — no stub, no terminal status."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()
    rj = runs.run_dir(rid) / "run.json"
    before = rj.read_bytes()

    real_read = pathlib.Path.read_text

    def _locked(self, *a, **k):
        if self == rj:
            raise PermissionError(32, "The process cannot access the file")
        return real_read(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", _locked)

    serve._reconcile_orphaned_runs(runs)

    assert rj.read_bytes() == before, (
        "the sweep overwrote a run.json whose READ failed")


def test_the_boot_reaper_never_touches_a_run_a_live_task_is_driving(home):
    """The sweep must not interrupt a run that is actually being driven."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    serve._reconcile_orphaned_runs(runs, driven_ids={rid})

    assert runs.load(rid)["status"] == "running"


def test_a_read_failed_release_is_reaped_at_the_next_boot(engine, home):
    """DR1-residual, end to end: after a read-failed release, a simulated boot
    reconciles the run and `restart_run` no longer answers 409."""
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"})]
    c = _client(engine)
    sid = c.post("/api/sessions", json={}).json()["id"]
    run = runs.create("keep my mission", sid)
    rid = run["id"]
    rj = runs.run_dir(rid) / "run.json"

    # The READ fails (a lock), so the release cannot write a terminal status:
    # run.json keeps saying `running` and the pointer is cleared.
    real_load, real_read = runs.load, pathlib.Path.read_text
    runs.load = lambda _rid: None

    def _locked(self, *a, **k):
        if self == rj:
            raise PermissionError(32, "The process cannot access the file")
        return real_read(self, *a, **k)

    pathlib.Path.read_text = _locked
    try:
        serve._release_unreadable_run(runs, rid, "run state could not be read")
    finally:
        runs.load = real_load
        pathlib.Path.read_text = real_read

    # the verifier's state: the reaper's precondition (`active()` is not None)
    # is gone, and the record says running, which restart_run refuses.
    assert runs.active() is None
    assert runs.load(rid)["status"] == "running"
    assert "running" not in runs.RESTARTABLE

    # ...and a boot — a fresh app's lifespan — reconciles it.
    c2 = _client(engine)

    assert runs.load(rid)["status"] == "interrupted", runs.load(rid)
    resp = c2.post(f"/api/runs/{rid}/restart")
    assert resp.status_code == 200, (
        "restart could not reattach after a read-failed release: "
        f"{resp.status_code} {resp.text}")
    assert resp.json().get("restarted") is True


# ---------------------------------------------------------------------------
# DR3-4: the sweep must not reconcile a run ANOTHER LIVE PROCESS is driving.
#
# `driven_ids` only knows THIS process's tasks, and two Rigma processes can
# share one RIGMA_HOME — the CLI's own "free it or pass a different --port"
# invites a second instance. Forcing such a run terminal would then propagate
# to its real driver through `save`'s sticky-terminal rule.
#
# The check is `runs.driver_is_live_elsewhere`, which reads the stamp every
# non-terminal `save` writes (pid + create time, so a recycled pid cannot make
# a dead driver look alive).
#
# DR4-2: these tests used to fabricate `os.getpid() + 1` and monkeypatch
# `state._is_recorded_process` to a lambda that IGNORED its arguments, so the
# pid+create-time mechanism they are named for never ran — rename the writer's
# key or swap the call's arguments and DR3-4 died with a green suite. They now
# drive the real thing: a real foreign child process for the pid, its real
# create time for the stamp, and no stub over the identity call.
# ---------------------------------------------------------------------------

def _stamp_driver(rid, pid, started):
    """Write a driver stamp the way a WRITER in another process would.

    `started` must be the REAL create time of `pid` (from `state._create_time`),
    never a fabricated constant: the point of these tests is to exercise the
    create-time comparison, not to satisfy it with a number that matches
    nothing."""
    doc = runs.load(rid)
    doc["driver_pid"] = pid
    doc["driver_started_at"] = started
    (runs.run_dir(rid) / "run.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8")
    return doc


def _foreign_sleeper():
    """A REAL process that is neither this one nor driving anything.

    A plain interpreter asleep in `time.sleep` — no engine, no model, no port
    (§0.1). The caller owns it and reaps it in a `finally`."""
    return subprocess.Popen([sys.executable, "-c",
                             "import time; time.sleep(30)"])


def _reap(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:        # pragma: no cover - defensive
        proc.kill()
        proc.wait(timeout=10)


def test_the_boot_reaper_skips_a_run_another_live_process_drives(home):
    """The REAL identity check decides this, against a REAL process.

    `_is_recorded_process` is NOT monkeypatched: the sweep reaches it with the
    child's true pid and create time, so an argument-order slip at
    `runs.driver_is_live_elsewhere`'s call site fails here instead of passing."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    from rigma import state
    proc = _foreign_sleeper()
    try:
        started = state._create_time(proc.pid)
        assert started > 0.0, (
            "could not read the foreign child's create time, so this test "
            "cannot exercise the mechanism it exists for")
        _stamp_driver(rid, proc.pid, started)

        # The reader half, executed for real — this is exactly the call the old
        # arg-ignoring lambda replaced.
        assert runs.driver_is_live_elsewhere(runs.load(rid)) is True

        serve._reconcile_orphaned_runs(runs)

        assert runs.load(rid)["status"] == "running", (
            "the sweep forced a run another live process was driving to terminal")
    finally:
        _reap(proc)


def test_the_boot_reaper_reconciles_a_live_pid_with_a_different_create_time(home):
    """DR42-n1: the create-time COMPARISON itself, on the driver path.

    A live pid whose recorded create time DIFFERS is a recycled number — some
    other process owns it now — so the run is NOT being driven elsewhere and
    must be reconciled, not skipped. The stamp is a REAL foreign child's pid
    with a deliberately different time (never a fabricated identity), so the
    branch that compares the two times is what decides this. Replace the
    comparison in `state._recorded_identity_matches` with
    `psutil.Process(pid).create_time(); return True` and this test fails while
    every other DR3-4/DR4-2 test stays green."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    from rigma import state
    proc = _foreign_sleeper()
    try:
        real = state._create_time(proc.pid)
        assert real > 0.0, (
            "could not read the foreign child's create time, so this test "
            "cannot exercise the comparison it exists for")
        # The pid IS live, but this stamp is NOT its create time: the recorded
        # identity and the live process disagree.
        _stamp_driver(rid, proc.pid, real - 10_000.0)

        assert runs.driver_is_live_elsewhere(runs.load(rid)) is False, (
            "a live pid with a DIFFERENT create time was read as the recorded "
            "driver, so a recycled pid still makes a dead driver look alive")

        serve._reconcile_orphaned_runs(runs)

        assert runs.load(rid)["status"] == "interrupted", (
            "the sweep skipped a run whose recorded driver identity does not "
            "match the live process that now owns the pid")
    finally:
        _reap(proc)


def test_the_boot_reaper_still_reaps_a_run_whose_driver_is_gone(home):
    """The other direction, on a REAL dead process: the same stamp once the
    child is gone is an orphan like any other — DR1-residual stays closed."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    from rigma import state
    proc = _foreign_sleeper()
    started = state._create_time(proc.pid)
    assert started > 0.0
    _stamp_driver(rid, proc.pid, started)
    _reap(proc)

    assert runs.driver_is_live_elsewhere(runs.load(rid)) is False

    serve._reconcile_orphaned_runs(runs)

    assert runs.load(rid)["status"] == "interrupted"


def test_a_run_this_process_stamped_is_still_reaped(home):
    """Our OWN stamp is never "live elsewhere": a `running` record this process
    wrote and then stopped driving is exactly the orphan the sweep is for, so
    stamping must not make DR1-residual unreachable.

    DR4-2, the WRITER half: this reads the stamp a REAL `runs.save` wrote and
    checks it through the production helper, so a renamed writer key or a stamp
    of 0.0 is caught here (it used to be asserted only for `driver_pid`)."""
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    run["iteration"] = 1
    runs.save(run)                          # the real writer path
    doc = runs.load(rid)

    from rigma import state
    assert doc.get("driver_pid") == os.getpid()
    assert doc.get("driver_started_at", 0.0) > 0.0, (
        "runs.save wrote no real create time, so the identity check has nothing "
        "to compare a recycled pid against")
    assert state._is_recorded_process(
        doc["driver_pid"], doc["driver_started_at"]) is True, (
        "the stamp does not identify a live process through the production helper")

    serve._reconcile_orphaned_runs(runs)

    assert runs.load(rid)["status"] == "interrupted"


# ---------------------------------------------------------------------------
# DR4-3: an UNKNOWN driver identity must read as "not live elsewhere", so the
# run IS reconciled. `_driver_stamp` writes 0.0 when it cannot read its own
# create time, and reading that as "live" left active.json wedged — the
# start_run 409 the boot sweep exists to clear. The process-KILLING path keeps
# the opposite, conservative default (see tests/test_state.py).
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("drop", [False, True], ids=["zero", "missing-key"])
def test_an_unknown_driver_identity_does_not_wedge_the_run(home, drop):
    run = runs.create("m", "sess")
    rid = run["id"]
    runs.clear_active()

    # A foreign pid that is not this process. Whether it is alive is irrelevant:
    # the record carries no identity to check, which is the case under test.
    doc = runs.load(rid)
    doc["driver_pid"] = os.getpid() + 1
    if drop:
        doc.pop("driver_started_at", None)
    else:
        doc["driver_started_at"] = 0.0
    (runs.run_dir(rid) / "run.json").write_text(
        json.dumps(doc, indent=2), encoding="utf-8")

    assert runs.driver_is_live_elsewhere(runs.load(rid)) is False, (
        "an unknown driver identity was read as 'live elsewhere', so the run "
        "can never be reconciled")
    serve._reconcile_orphaned_runs(runs)

    assert runs.load(rid)["status"] == "interrupted"

