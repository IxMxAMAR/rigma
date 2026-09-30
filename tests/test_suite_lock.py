"""REC-1 / OD-16: a second FULL suite must refuse to start, with a sentence.

Two overlapping full runs froze twice (0 CPU over 25 s, 72 orphaned
`fake_acp_server.py` processes). The shared resource was a fixed port and is
fixed at 43e63de, but nothing stopped a second full run from starting. These
tests pin the guard's decision rules and the lock's takeover rules, plus one
end-to-end proof that launches real pytest processes against a throwaway
checkout.

The state these tests deliberately DO put the code in — because a naive lock
would wedge the suite forever there — is a lock file whose owner is **gone**,
one whose owner pid is **alive but is a different process** (a recycled pid),
and one whose contents are **unreadable**. All must be taken over, not honoured.
"""
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

import psutil
import pytest

import conftest

TESTS = Path(conftest.__file__).resolve().parent
ROOT = TESTS.parent


def _config(*parsed, invocation=None):
    """A fake Config carrying pytest's parsed positionals (`config.args`) and the
    raw argv (`invocation_params.args`), as the real object does. When
    `invocation` is omitted it mirrors the common case where argv is exactly the
    parsed positionals; pass it explicitly for invocations whose argv carries an
    option VALUE that looks positional (`-m "not hardware"`). The raw argv is
    what the WIP hand-parsed; the parsed list is what the fixed classifier uses.
    """
    if invocation is None:
        invocation = parsed
    return SimpleNamespace(
        args=list(parsed),
        invocation_params=SimpleNamespace(args=list(invocation)),
    )


# --- which runs are "full" ---------------------------------------------------

def test_no_path_with_the_marker_is_a_full_run():
    """`pytest -m "not hardware"` — the REAL full-suite invocation. Its raw argv
    carries the marker's VALUE as a bare token ('not hardware') while pytest's
    parsed `config.args` is `[<rootdir>]`. Hand-parsing argv (the WIP) sees 'not
    hardware', finds no path equal to the tests dir, and returns False — so the
    guard would never lock the very run it exists for. This test fails on the
    un-fixed WIP and passes on the `config.args` classifier."""
    cfg = _config(str(ROOT), invocation=["-m", "not hardware"])
    assert conftest._full_suite_run(cfg) is True


def test_no_path_with_quiet_is_a_full_run():
    assert conftest._full_suite_run(_config(str(ROOT), invocation=["-q"])) is True


def test_no_path_with_o_addopts_is_a_full_run():
    assert conftest._full_suite_run(
        _config(str(ROOT), invocation=["-o", "addopts=", "-q"])) is True


def test_empty_parsed_args_is_a_full_run():
    assert conftest._full_suite_run(_config()) is True


def test_the_tests_directory_is_a_full_run():
    assert conftest._full_suite_run(_config("tests")) is True


def test_the_tests_directory_with_a_slash_is_a_full_run():
    assert conftest._full_suite_run(_config("tests/")) is True


def test_the_absolute_tests_directory_is_a_full_run():
    assert conftest._full_suite_run(_config(str(TESTS))) is True


def test_a_named_file_is_not_a_full_run():
    """Verifiers run different files concurrently; blocking that would break
    the workflow this guard exists to protect."""
    assert conftest._full_suite_run(_config("tests/test_bench.py")) is False


def test_several_named_files_are_not_a_full_run():
    assert conftest._full_suite_run(
        _config("tests/test_bench_sweep.py",
                "tests/test_run_loop_first_load.py")) is False


def test_marker_followed_by_a_named_file_is_not_a_full_run():
    cfg = _config("tests/test_suite_lock.py",
                  invocation=["-m", "not hardware", "tests/test_suite_lock.py"])
    assert conftest._full_suite_run(cfg) is False


def test_a_subdirectory_of_tests_is_not_a_full_run():
    assert conftest._full_suite_run(_config("tests/subdir")) is False


def test_options_are_not_mistaken_for_paths():
    """`config.args` carries positionals only, whatever the argv said."""
    cfg = _config("tests", invocation=["-o", "addopts=", "-q",
                                       "-m", "not hardware", "tests"])
    assert conftest._full_suite_run(cfg) is True


# --- taking, honouring and taking over the lock ------------------------------

def test_a_fresh_lock_is_taken_and_records_this_process(tmp_path):
    path, holder = conftest._acquire_suite_lock(tmp_path / "s.lock")
    assert holder is None and path is not None
    rec = json.loads(path.read_text(encoding="utf-8"))
    assert rec["pid"] == os.getpid()
    assert conftest._owner_is_alive(rec["pid"], rec["started_at"]) is True


def test_a_second_full_run_is_refused_while_the_first_holds_it(tmp_path):
    lock = tmp_path / "s.lock"
    first, _ = conftest._acquire_suite_lock(lock)
    assert first is not None
    second, holder = conftest._acquire_suite_lock(lock)
    assert second is None
    assert holder and holder["pid"] == os.getpid()
    assert first.exists(), "the loser must not remove the winner's lock"


def test_a_dead_owners_lock_is_taken_over(tmp_path):
    """The state a naive lock wedges on: a run that was killed (a power cut, a
    closed terminal) leaves its file behind, and every later suite must still
    start."""
    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    dead.wait()
    assert dead.pid != os.getpid()
    assert conftest._owner_is_alive(dead.pid, 12345.0) is False
    lock = tmp_path / "s.lock"
    lock.write_text(json.dumps({"pid": dead.pid, "started_at": 12345.0}),
                    encoding="utf-8")
    path, holder = conftest._acquire_suite_lock(lock)
    assert holder is None and path is not None
    assert json.loads(path.read_text(encoding="utf-8"))["pid"] == os.getpid()


def test_a_recycled_pid_does_not_wedge_the_suite(tmp_path):
    """A LIVE pid whose create time does not match the record is some other
    process that inherited the number — the same distinction
    `state._is_recorded_process` draws for the engine pid."""
    lock = tmp_path / "s.lock"
    lock.write_text(
        json.dumps({"pid": os.getpid(),
                    "started_at": psutil.Process(os.getpid()).create_time() - 500.0}),
        encoding="utf-8")
    path, holder = conftest._acquire_suite_lock(lock)
    assert holder is None and path is not None


def test_an_unreadable_lock_is_taken_over(tmp_path):
    """A half-written or truncated file must not make the suite unstartable."""
    lock = tmp_path / "s.lock"
    lock.write_text("{not json", encoding="utf-8")
    path, holder = conftest._acquire_suite_lock(lock)
    assert holder is None and path is not None


def test_a_truncated_json_lock_is_taken_over(tmp_path):
    lock = tmp_path / "s.lock"
    lock.write_text('{"pid": 123, "started_at":', encoding="utf-8")
    path, holder = conftest._acquire_suite_lock(lock)
    assert holder is None and path is not None


def test_releasing_lets_the_next_run_start(tmp_path):
    lock = tmp_path / "s.lock"
    path, _ = conftest._acquire_suite_lock(lock)
    conftest._release_suite_lock(path)
    again, holder = conftest._acquire_suite_lock(lock)
    assert holder is None and again is not None


def test_release_does_not_delete_a_lock_taken_over_by_someone_else(tmp_path):
    """If our record was taken over while we ran, the new owner's file must not
    be deleted out from under it."""
    lock = tmp_path / "s.lock"
    path, _ = conftest._acquire_suite_lock(lock)
    lock.write_text(json.dumps({"pid": os.getpid() + 1, "started_at": 1.0}),
                    encoding="utf-8")
    conftest._release_suite_lock(path)
    assert lock.exists()


def test_the_lock_lives_outside_the_repo():
    """A lock file in the working tree would show up in `git status` and could
    be committed."""
    p = conftest._lock_path()
    assert p.parent == Path(tempfile.gettempdir()), p
    assert TESTS not in p.parents
    assert ROOT not in p.parents
    assert p.name.startswith("rigma-full-suite-")


# --- end to end: two real pytest processes ------------------------------------

_SYNTHETIC_CONFTEST = '''
import importlib.util
from pathlib import Path

import pytest

_REAL_PATH = {real_path!r}
_spec = importlib.util.spec_from_file_location("_real_suite_conftest", _REAL_PATH)
_real = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_real)

_TESTS = Path(__file__).resolve().parent


@pytest.fixture(autouse=True, scope="session")
def _one_full_suite_at_a_time(request):
    path = _real._enter_suite_lock(request.config, _TESTS)
    try:
        yield
    finally:
        if path is not None:
            _real._release_suite_lock(path)
'''

_HOLDER_TEST = """
import time
from pathlib import Path


def test_hold():
    here = Path(__file__).resolve().parent
    (here / "holding").write_text("1", encoding="utf-8")
    deadline = time.time() + 30
    while not (here / "release").exists():
        if time.time() > deadline:
            raise AssertionError("holder was never released")
        time.sleep(0.05)
"""


def _run(cmd, cwd, env, timeout=120):
    return subprocess.run(
        cmd, cwd=str(cwd), env=env, timeout=timeout,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace")


def test_end_to_end_second_full_run_is_refused_but_targeted_is_not(tmp_path):
    """A real full run holds the lock in a subprocess; a second full run exits
    non-zero with the REC-1 sentence, a named-file run still passes, and after
    the owner is KILLED a later full run takes the dead lock over and passes."""
    root = tmp_path / "checkout"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (tests / "conftest.py").write_text(
        _SYNTHETIC_CONFTEST.format(real_path=str(Path(conftest.__file__).resolve())),
        encoding="utf-8")
    (tests / "test_hold.py").write_text(_HOLDER_TEST, encoding="utf-8")
    (tests / "test_target.py").write_text("def test_ok():\n    assert True\n",
                                          encoding="utf-8")

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    # `-m "not hardware"` mirrors the REAL full-suite invocation; `-p
    # no:cacheprovider`'s value is a second bare token that looks positional to a
    # hand-parser. The fixed classifier reads them from `config.args`, where
    # neither appears.
    cmd = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider",
           "-m", "not hardware"]

    lock = conftest._lock_path_for(tests)
    holder = subprocess.Popen(cmd, cwd=str(root), env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding="utf-8", errors="replace")
    try:
        deadline = time.time() + 90
        while not (tests / "holding").exists():
            if holder.poll() is not None:
                raise AssertionError(
                    "holder exited before taking the lock:\n" + holder.stdout.read())
            if time.time() > deadline:
                raise AssertionError("holder never took the lock")
            time.sleep(0.05)
        assert lock.exists(), "the full run must have taken the lock"

        second = _run(cmd, root, env)
        assert second.returncode == conftest._SUITE_LOCK_RETURNCODE, second.stdout
        assert "REC-1" in second.stdout, second.stdout
        assert "targeted runs are not locked" in second.stdout, second.stdout

        targeted = _run(cmd + ["tests/test_target.py"], root, env)
        assert targeted.returncode == 0, targeted.stdout
        assert "1 passed" in targeted.stdout, targeted.stdout

        # Crash path: kill the owner so its `finally` never runs. A DEAD owner's
        # leftover lock must be taken over by the next full run, not wedge it.
        holder.kill()
        holder.wait(timeout=30)
        assert lock.exists(), "a killed owner leaves its lock file behind"
        (tests / "release").write_text("1", encoding="utf-8")
        revived = _run(cmd, root, env)
        assert revived.returncode == 0, revived.stdout
        assert "2 passed" in revived.stdout, revived.stdout
        assert not lock.exists(), "the revived run must take over and release"
    finally:
        (tests / "release").write_text("1", encoding="utf-8")
        if holder.poll() is None:
            try:
                holder.wait(timeout=30)
            except subprocess.TimeoutExpired:
                holder.kill()
                holder.wait(timeout=10)


def test_targeted_run_is_never_locked_even_by_a_held_lock(tmp_path, monkeypatch):
    """While a live lock is held for this tests dir, a full config is refused
    (pytest.exit) but a named-file config returns None and never raises — the
    distinction that keeps verifiers running concurrently."""
    lock = tmp_path / "held.lock"
    monkeypatch.setattr(conftest, "_lock_path", lambda: lock)
    path, _ = conftest._acquire_suite_lock(lock)
    assert path is not None
    try:
        targeted = _config(str(TESTS / "test_suite_lock.py"))
        assert conftest._enter_suite_lock(targeted) is None
        with pytest.raises(pytest.exit.Exception):
            conftest._enter_suite_lock(_config(str(ROOT)))
    finally:
        conftest._release_suite_lock(path)
