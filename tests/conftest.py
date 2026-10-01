"""Shared fixtures."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _no_hang_on_run_ui(monkeypatch):
    """serve.run_ui blocks forever serving the UI; a test that reaches it
    unmocked would hang the ENTIRE suite (and squat on port 11500). Fail fast
    instead. Tests that intend to exercise `up`'s serving path patch run_ui
    themselves — their patch runs after this one and wins."""
    import rigma.serve as serve

    def _boom(*a, **k):
        raise RuntimeError("serve.run_ui reached in a test without mocking it "
                           "— `rigma up` was invoked in a way that serves")
    monkeypatch.setattr(serve, "run_ui", _boom, raising=False)


@pytest.fixture(autouse=True)
def _isolate_user_custom_models(monkeypatch, tmp_path_factory):
    """The user's real ~/.rigma/custom installs must never leak into tests
    (live repro 2026-07-17: installing SmolLM2 flipped test_floor_never_fails).
    Reads RIGMA_HOME at call time so tests that set their own home keep the
    normal layout."""
    import rigma.registry as registry
    empty = tmp_path_factory.mktemp("no-custom")
    monkeypatch.setattr(
        registry, "_custom_dir",
        lambda: (Path(os.environ["RIGMA_HOME"]) / "custom" / "models")
        if os.environ.get("RIGMA_HOME") else empty)


class _OpenAIUpstream(BaseHTTPRequestHandler):
    """Streams 'Hel'+'lo' as OpenAI chat chunks; records the last request body."""
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _OpenAIUpstream.last_body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for tok in ("Hel", "lo"):
            chunk = {"choices": [{"delta": {"content": tok}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def oai_upstream():
    _OpenAIUpstream.last_body = None
    srv = HTTPServer(("127.0.0.1", 0), _OpenAIUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield SimpleNamespace(port=srv.server_address[1],
                          last=lambda: _OpenAIUpstream.last_body)
    srv.shutdown()


@pytest.fixture(autouse=True, scope="session")
def _tests_run_against_the_source_tree():
    """A stale site-packages copy of rigma once shadowed the editable install
    (2026-07-21) and every subsequent test run silently validated the WRONG
    code. If imports ever resolve outside this repo, fail everything loudly."""
    import rigma
    src = str(Path(__file__).resolve().parent.parent / "src")
    got = str(Path(rigma.__file__).resolve())
    if not got.startswith(src):
        pytest.exit(f"rigma imports from {got}, not the source tree {src} — "
                    "run `pip install -e .` and remove the shadowing copy",
                    returncode=3)


@pytest.fixture(autouse=True, scope="session")
def _never_touch_the_real_rigma_home(tmp_path_factory):
    """514 zombie runs landed in the user's real ~/.rigma because one test
    module created runs without redirecting RIGMA_HOME (2026-07-21). Default
    the whole session to a throwaway home; tests that monkeypatch their own
    RIGMA_HOME still override this per-test."""
    os.environ["RIGMA_HOME"] = str(tmp_path_factory.mktemp("rigma-home"))


# --- one FULL suite at a time (REC-1 / OD-16) --------------------------------
#
# WHAT WENT WRONG. Twice, two full `pytest` runs overlapped and both froze —
# not slow, *stopped*, 0 CPU over 25 s — each blocked on an established loopback
# socket, with 72 orphaned `fake_acp_server.py` processes left behind. The shared
# resource was a fixed TCP port (`tests/test_bench.py` bound 11598 literally; it
# was the only literal bind in the suite) and that is fixed at da50a67, but
# nothing then *stopped* two full runs from starting, and this program's own
# "at most three pytest processes" rule permits it.
#
# SO: a run that collects the WHOLE suite takes an exclusive lock, and a second
# full run exits immediately with a sentence that names the failure and the way
# out. A run of specific FILES is deliberately not locked — verifiers run
# different files concurrently all the time, and blocking that would break the
# workflow this guard exists to protect.
#
# The lock lives in the system temp dir, keyed by the tests directory, so it is
# per-checkout and never dirties the repo. Its owner is a pid PLUS the pid's
# create time, for the same reason `state._is_recorded_process` records both: a
# recycled pid must not make a dead owner look alive and wedge every later run.
#
# WHICH RUNS ARE FULL is decided from pytest's OWN parsed positional list
# (`config.args`), NOT by hand-parsing argv. Probed with a throwaway
# `pytest_configure` plugin on pytest 9.1.1: for `pytest -m "not hardware"` — how
# the full suite is actually invoked — `invocation_params.args` is
# `('-m', 'not hardware')`, so the marker's VALUE arrives as a bare token that
# looks positional, while `config.args` is `[<rootdir>]`. Filtering `-`-prefixed
# tokens out of the raw argv therefore leaves `['not hardware']`, which is not
# the tests dir, and the full run is misclassified as targeted — the guard would
# never lock the very run it exists for. `config.args` held no options for any
# form probed (no path, `tests`, `tests/`, the absolute tests dir, a named file,
# each with `-m "not hardware"`, `-q`, `-o addopts=`, `-k`, `--lf`).


def _tests_dir() -> Path:
    return Path(__file__).resolve().parent


def _full_suite_run(config, tests_dir=None) -> bool:
    """Does this invocation collect the whole suite (as opposed to named files)?

    True when no path was given — pytest's parsed `config.args` is then
    `[<rootdir>]` — or when a given path IS the tests directory (or the repo
    root, which contains it). A path that names a file, or a subdirectory of
    `tests`, is a targeted run and is never locked.

    `tests_dir` is injectable so the end-to-end test can point the same rule at
    a throwaway checkout instead of the real one.
    """
    tests_dir = Path(tests_dir).resolve() if tests_dir else _tests_dir()
    root_dir = tests_dir.parent
    try:
        given = list(config.args or ())
    except AttributeError:
        given = []
    if not given:
        # No parsed positionals at all: pytest would collect the rootdir.
        return True
    for a in given:
        try:
            resolved = Path(str(a)).resolve()
        except OSError:
            continue
        if resolved == tests_dir or resolved == root_dir:
            return True
    return False


def _lock_path_for(tests_dir) -> Path:
    """A per-checkout lock file in the temp dir, not in the repo."""
    import hashlib
    import tempfile
    key = hashlib.sha1(str(Path(tests_dir).resolve()).encode("utf-8")).hexdigest()[:12]
    return Path(tempfile.gettempdir()) / f"rigma-full-suite-{key}.lock"


def _lock_path() -> Path:
    return _lock_path_for(_tests_dir())


def _owner_is_alive(pid: int, started_at) -> bool:
    """Is the recorded owner still the process that took the lock?

    A pid ALONE is not enough: pids are recycled, and a live process that merely
    inherited the number must not make a dead owner look alive — the same
    distinction `state._is_recorded_process` draws for the engine pid.

    The safe direction for a LOCK is the one the termination path takes, not the
    driver path: an identity that cannot be READ must not make a live owner's
    lock stealable. Stealing a live owner's lock is precisely the failure the
    lock exists to prevent — two concurrent FULL suites, REC-1's shared fixed
    port and 72 orphaned fake_acp_server processes. Refusing to steal from a
    LIVE pid at worst makes the next run wait for that process to exit (its own
    finalizer then releases the lock); it cannot wedge forever, because a pid
    that is actually gone fails the psutil call below and is taken over.

    `started_at == 0.0` is not hypothetical: `_acquire_suite_lock` writes exactly
    that when psutil cannot read its OWN create time, so treating it as dead made
    the lock written after a psutil failure immediately stealable. SUITELOCK-n1.
    """
    if pid <= 0:
        return False
    try:
        import psutil
        owner_created = float(psutil.Process(pid).create_time())
    except Exception:
        # The pid is gone (or unreadable): no live owner to protect, so a
        # leftover lock from a killed/crashed run is taken over.
        return False
    try:
        recorded = float(started_at)
    except (TypeError, ValueError):
        # A record with no usable create time at all (missing/None/garbage) is
        # not an identity we can attribute; take it over, as before.
        return False
    if recorded <= 0.0:
        # 0.0 means "the owner could not read its own create time", not "the
        # owner is gone". The pid IS live, so do not steal its lock.
        return True
    return abs(owner_created - recorded) < 1.0


def _acquire_suite_lock(path: Path):
    """Take `path`, or report the live holder.

    Returns `(path, None)` on success and `(None, holder)` when another live
    full suite owns it. A lock whose owner is gone, whose owner is a recycled
    pid, or whose contents cannot be read is removed and taken over — a power
    cut must not leave the suite unstartable.
    """
    for _ in range(2):
        try:
            fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            holder = None
            try:
                holder = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                holder = None
            if holder and _owner_is_alive(int(holder.get("pid") or 0),
                                          holder.get("started_at")):
                return None, holder
            try:
                path.unlink()
            except OSError:
                return None, holder
            continue
        import psutil
        try:
            started = float(psutil.Process(os.getpid()).create_time())
        except Exception:
            started = 0.0
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump({"pid": os.getpid(), "started_at": started}, f)
        return path, None
    return None, None


def _release_suite_lock(path: Path) -> None:
    """Drop the lock — but only while it is still OURS.

    If our record was taken over while we ran (an unreadable record, a failed
    identity check), the new owner's file must not be deleted out from under
    it. A lock we cannot read is left alone; a dead owner's leftover is taken
    over by the next run's `_acquire_suite_lock` anyway."""
    try:
        holder = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return
    if int(holder.get("pid") or 0) != os.getpid():
        return
    try:
        path.unlink()
    except OSError:
        pass


_SUITE_LOCK_RETURNCODE = 4
_SUITE_LOCK_REFUSAL = (
    "another FULL test suite is already running{who}. Two full runs at once is "
    "the failure recorded as REC-1: they shared a fixed port and froze, leaving "
    "72 orphaned fake_acp_server processes. Wait for it to finish, or run just "
    "the files you need — targeted runs are not locked."
)


def _lock_refusal_message(holder) -> str:
    who = ""
    if holder:
        who = (f" (held by pid {holder.get('pid')}, "
               f"started {holder.get('started_at')})")
    return _SUITE_LOCK_REFUSAL.format(who=who)


def _enter_suite_lock(config, tests_dir=None):
    """Return the lock this run must hold, or `None` for a targeted run.

    Raises `pytest.exit` (with the REC-1 sentence) when a second FULL run finds
    a live owner. Extracted from the fixture so the end-to-end test can drive
    the exact same decision, message and lock against a throwaway checkout.
    """
    if not _full_suite_run(config, tests_dir):
        return None
    path, holder = _acquire_suite_lock(
        _lock_path_for(tests_dir) if tests_dir else _lock_path())
    if path is None:
        pytest.exit(_lock_refusal_message(holder),
                    returncode=_SUITE_LOCK_RETURNCODE)
    return path


@pytest.fixture(autouse=True, scope="session")
def _one_full_suite_at_a_time(request):
    """Refuse to start a second FULL suite; let targeted runs through."""
    path = _enter_suite_lock(request.config)
    try:
        yield
    finally:
        if path is not None:
            _release_suite_lock(path)
