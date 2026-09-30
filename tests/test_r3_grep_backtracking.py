"""DR6: grep's content `pattern` must not be able to stop the server.

R3-16 removed the glob ReDoS, but the CONTENT pattern stayed a raw
model-supplied regex run with `rx.search(line)` and no bound. `re` cannot be
interrupted: MEASURED on this host, `(a+)+$` against `"a"*n + "b"` takes 0.04 s
at n=20, 0.70 s at n=24 and 3.28 s at n=26 — a 4x step per two characters, so
n=40 is about 15 hours. A SIGINT raised from another thread does NOT stop the
match, so no in-process deadline can work, and no line cap can bound it either:
a cap small enough to make `2**cap` survivable (about 20) would hide everything
past the 20th character, while a cap large enough to be useful still allows
`2**cap`. The search therefore runs in a child interpreter that the parent can
kill at a wall-clock deadline.

These tests use a slow-but-FINITE catastrophic pattern and a budget
monkeypatched small enough to trip it, so nothing here can hang the suite: the
BEFORE state of each is bounded by the ~3.3 s the old in-process search took.
"""
from __future__ import annotations

import time

from rigma import tools


def _ws(tmp_path):
    return {"workspace": str(tmp_path)}


def _slow_line(n: int = 26) -> str:
    """A run that makes `(a+)+$` backtrack exponentially but FINITELY."""
    return "a" * n + "b"


def _tiny_budget(monkeypatch, seconds: float) -> None:
    """A budget below the pathological time and far above child startup.

    `raising=False` so the test still RUNS on the pre-DR6 code (where the
    constants do not exist) instead of erroring on the monkeypatch — that is
    what makes "fails before" an honest assertion failure.
    """
    monkeypatch.setattr(tools, "_GREP_REGEX_BUDGET", seconds, raising=False)
    monkeypatch.setattr(tools, "_GREP_BYTES_PER_SEC", 1 << 40, raising=False)


def test_a_catastrophic_pattern_is_stopped_by_the_wall_clock(tmp_path, monkeypatch):
    """The owner's outcome: a bad pattern comes back as an error, not a hang."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "slow.txt").write_text(_slow_line() + "\n", encoding="utf-8")
    _tiny_budget(monkeypatch, 1.5)

    started = time.monotonic()
    out = tools._grep({"pattern": "(a+)+$"}, _ws(tmp_path))
    elapsed = time.monotonic() - started

    assert out.startswith("error:"), out
    assert "did not finish" in out, out
    assert "catastrophic backtracking" in out, out
    # The point of the fix is a BOUND. Before it this call ran the full ~3.3 s
    # and answered "no matches"; after it the budget decides.
    assert elapsed < 20.0, elapsed


def test_a_stuck_search_child_is_killed_not_abandoned(tmp_path, monkeypatch):
    """Bounding the wait is not enough: the child burning the CPU must die.

    `_grep_search_bounded` hands the timed-out child to `tools._kill_tree`. If it
    only stopped reading, the catastrophic match would keep running in the
    background and the "bound" would be a lie.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "slow.txt").write_text(_slow_line() + "\n", encoding="utf-8")
    _tiny_budget(monkeypatch, 1.0)

    killed = []
    real = tools._kill_tree
    monkeypatch.setattr(
        tools, "_kill_tree",
        lambda pid, proc=None, **k: killed.append(pid) or real(pid, proc))

    out = tools._grep({"pattern": "(a+)+$"}, _ws(tmp_path))

    assert killed, "the stuck search child was left running"
    assert out.startswith("error:"), out


def test_a_tree_kill_that_fails_still_takes_the_worker_itself(tmp_path, monkeypatch):
    """A tree kill that cannot be confirmed must not abandon the worker.

    The worker spawns nothing of its own, so `Popen.kill()` is enough for it; if
    the kill were skipped the catastrophic match would keep running and the
    wall-clock "bound" would be a lie.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "slow.txt").write_text(_slow_line() + "\n", encoding="utf-8")
    _tiny_budget(monkeypatch, 1.0)
    monkeypatch.setattr(tools, "_kill_tree", lambda *a, **k: False)

    killed = []
    real_popen = tools.subprocess.Popen

    class _Spy:
        def __init__(self, *a, **k):
            self._p = real_popen(*a, **k)

        def __getattr__(self, name):
            return getattr(self._p, name)

        def kill(self):
            killed.append("kill")
            self._p.kill()

    monkeypatch.setattr(tools.subprocess, "Popen", _Spy)

    out = tools._grep({"pattern": "(a+)+$"}, _ws(tmp_path))

    assert killed == ["kill"], killed
    assert out.startswith("error:"), out


def test_a_search_that_cannot_start_reports_instead_of_running_unbounded(
        tmp_path, monkeypatch):
    """The bound must not silently degrade to the unbounded in-process search.

    If the child cannot be started, running `rx.search` here anyway would
    reintroduce exactly the hang DR6 is about. The honest answer is an error.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "a.txt").write_text("needle\n", encoding="utf-8")

    def no_spawn(*a, **k):
        raise OSError("no fork")

    monkeypatch.setattr(tools.subprocess, "Popen", no_spawn)

    out = tools._grep({"pattern": "needle"}, _ws(tmp_path))
    assert out.startswith("error: could not start the bounded search"), out
