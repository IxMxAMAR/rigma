"""The DSH adapter, tested without a DSH install and without a network.

Two things are worth pinning here. First, the generated patch: a patch row
replaces a whole config row, so a missing `apiKeyEnv` or a protocol left at the
Anthropic default is a turn that fails on the wire for a reason nobody can see
from Rigma's output. Second, the subprocess contract: a malformed line, a dead
runner, a non-zero exit and a hung turn must all come back as events, never as an
exception — the caller is a tool.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from rigma import harness_dsh
from rigma import tools


@pytest.fixture(autouse=True)
def _fresh_runtime_state():
    """Reset the adapter's per-chat history around every test in this file.

    `_spoke` (R4-SESS-1) is MODULE state, by design: it has to remember, for the
    life of the server process, that a chat has already spoken, because that is
    what makes a lost runtime detectable at all. In a test session that leaks
    across tests — the second test's "first" turn on the same chat id looks like a
    returning one and gains a context-lost notice that has nothing to do with what
    that test is checking.

    Reset on BOTH sides: before, so a test never inherits another's history; after,
    so nothing this file does leaks into the rest of the suite.
    """
    harness_dsh._spoke.clear()
    yield
    harness_dsh._spoke.clear()


def _fake_home(tmp_path):
    """A directory that looks enough like a checkout for the path checks."""
    home = tmp_path / "dsh"
    (home / "python" / "sdk" / "src").mkdir(parents=True, exist_ok=True)
    bin_dir = home / "python" / "sdk-runtime" / "node_modules" / ".bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "dsh.CMD").write_text("@echo off\n", encoding="utf-8")
    return home


def _fake_runner(tmp_path, body: str) -> list[str]:
    """A stand-in for `python -m rigma._dsh_runner` that prints canned NDJSON."""
    script = tmp_path / "fake_runner.py"
    script.write_text(textwrap.dedent(body).lstrip(), encoding="utf-8")
    return [sys.executable, str(script)]


def _drive(monkeypatch, argv, home, **kwargs):
    monkeypatch.setattr(harness_dsh, "_runner_argv", lambda: argv)
    kwargs.setdefault("base_url", "http://127.0.0.1:11500/v1")
    kwargs.setdefault("model", "local-model")
    kwargs.setdefault("prompt", "hello")
    kwargs.setdefault("dsh_home", str(home))
    return list(harness_dsh.drive_turn(**kwargs))


def test_patch_restates_the_keys_a_replaced_row_would_lose(tmp_path):
    path = harness_dsh.patch_file(32768, 4096, tmp_path)
    raw = path.read_text(encoding="utf-8")
    assert path.name.endswith(".yaml")
    assert "protocol: chat-completions" in raw
    assert "apiKeyEnv: DEEPSEEK_API_KEY" in raw
    assert "defaultContextWindow: 32768" in raw
    assert "maxTokens: 4096" in raw
    try:
        import yaml
    except ImportError:
        return  # no YAML parser here; the raw-text assertions above still hold
    doc = yaml.safe_load(raw)
    assert isinstance(doc, list) and len(doc) == 1
    assert doc[0]["id"] == "llm-deepseek"
    config = doc[0]["config"]
    assert config["protocol"] == "chat-completions"
    assert config["apiKeyEnv"] == "DEEPSEEK_API_KEY"
    assert config["defaultContextWindow"] == 32768
    assert config["maxTokens"] == 4096


def test_a_missing_checkout_is_not_available(monkeypatch, tmp_path):
    monkeypatch.setenv("RIGMA_DSH_HOME", str(tmp_path / "not-here"))
    assert harness_dsh.home() is None
    assert harness_dsh.sdk_src() is None
    assert harness_dsh.dsh_bin() is None
    assert harness_dsh.available() is False


def test_the_env_override_names_the_checkout(monkeypatch, tmp_path):
    home = _fake_home(tmp_path)
    monkeypatch.setenv("RIGMA_DSH_HOME", str(home))
    assert harness_dsh.home() == home
    assert harness_dsh.sdk_src() == home / "python" / "sdk" / "src"
    assert harness_dsh.dsh_bin().name.lower() == "dsh.cmd"
    assert harness_dsh.available() is True


def test_the_runner_entry_point_is_the_package_module():
    assert harness_dsh._runner_argv() == [sys.executable, "-m", "rigma._dsh_runner"]


def test_drive_turn_translates_the_stream_and_skips_a_malformed_line(
    monkeypatch, tmp_path
):
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import json, sys
        rows = [
            {"type": "notice", "text": "session.status working"},
            {"type": "thinking", "text": "weighing options"},
            {"type": "tool", "name": "read_file", "args": {"path": "a.py"}},
            {"type": "tool_result", "name": "read_file", "text": "1: x = 1"},
            {"type": "text", "text": "partial answer"},
            "{ this is not json at all",
            {"type": "done", "text": "final answer", "finish_reason": "stop"},
        ]
        for row in rows:
            sys.stdout.write(row if isinstance(row, str) else json.dumps(row))
            sys.stdout.write("\\n")
        sys.stdout.flush()
        """,
    )
    events = _drive(monkeypatch, argv, home)

    assert [e.kind for e in events] == [
        "notice",
        "thinking",
        "tool",
        "tool_result",
        "text",
        "text",
    ]
    assert events[0].text.startswith("session.status")
    assert events[2].name == "read_file" and events[2].args == {"path": "a.py"}
    assert events[3].name == "read_file" and events[3].ok is True
    assert events[-1].text == "final answer"
    assert not [e for e in events if e.kind == "error"]


# A recorded runner stream. `test_harness_dsh_live.py` is the only end-to-end
# proof of the seam and it is skipif'd without a DSH checkout, so CI never saw
# the event shapes the runner actually emits. This replay is unconditional.
GOLDEN = Path(__file__).parent / "golden" / "dsh_turn.ndjson"


def test_a_recorded_turn_replays_through_the_translation(monkeypatch, tmp_path):
    """Replay the committed NDJSON stream of one DSH turn.

    The fixture is synthesised from the documented runner event shapes (never a
    real session), but it is the CI-visible guard: if `_event_for` stops
    translating `thinking`/`tool`/`tool_result`, or the final `done` stops
    surfacing, this fails on a machine with no DSH checkout.
    """
    home = _fake_home(tmp_path)
    script = tmp_path / "replay_runner.py"
    script.write_text(
        "import pathlib, sys\n"
        f"sys.stdout.write(pathlib.Path({str(GOLDEN)!r}).read_text("
        "encoding='utf-8'))\n"
        "sys.stdout.flush()\n",
        encoding="utf-8",
    )
    events = _drive(monkeypatch, [sys.executable, str(script)], home)

    assert [e.kind for e in events] == [
        "notice", "thinking", "tool", "tool_result", "text", "text"]
    assert events[2].name == "list_directory"
    assert events[2].args == {"path": "."}
    assert events[3].name == "list_directory" and events[3].ok is True
    assert events[-1].text == "There are two Python files in the workspace."
    assert not [e for e in events if e.kind == "error"]


def test_a_nonzero_exit_without_done_is_one_error_naming_stderr(monkeypatch, tmp_path):
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import sys
        sys.stdout.write('{"type": "text", "text": "starting"}\\n')
        sys.stdout.flush()
        sys.stderr.write("boom: could not reach the model server\\n")
        sys.stderr.flush()
        raise SystemExit(3)
        """,
    )
    events = _drive(monkeypatch, argv, home)

    assert [e.kind for e in events] == ["text", "error"]
    assert "3" in events[-1].text
    assert "boom: could not reach the model server" in events[-1].text


def test_a_hung_turn_times_out_and_is_killed(monkeypatch, tmp_path):
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import sys, time
        sys.stdout.write('{"type": "notice", "text": "working"}\\n')
        sys.stdout.flush()
        time.sleep(120)
        """,
    )
    started = time.monotonic()
    events = _drive(monkeypatch, argv, home, timeout=1.5)
    elapsed = time.monotonic() - started

    assert [e.kind for e in events] == ["notice", "error"]
    assert "timed out" in events[-1].text
    assert elapsed < 30, "the sleeping child must not hold the turn open"


def test_an_empty_done_is_never_silence(monkeypatch, tmp_path):
    """A failed model call arrives as `done` with no text. Passing that through
    as silence would make a dead endpoint look like a turn with nothing to say.

    Two turns rather than two `done` rows in one: one job is one turn now, so a
    second `done` on the same stream belongs to the NEXT turn, not this one.
    """
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import json, sys
        for line in sys.stdin:
            if not line.strip():
                continue
            job = json.loads(line)
            reason = "error" if "fail" in str(job.get("prompt")) else "stop"
            sys.stdout.write(json.dumps(
                {"type": "done", "text": "", "finish_reason": reason}) + "\\n")
            sys.stdout.flush()
        """,
    )
    failed = _drive(monkeypatch, argv, home, prompt="fail please")
    stopped = _drive(monkeypatch, argv, home, prompt="all good")

    assert [e.kind for e in failed] == ["error"]
    assert "model call failed" in failed[0].text
    assert [e.kind for e in stopped] == ["notice"]
    assert "stop" in stopped[0].text


def test_a_runner_that_cannot_start_is_an_error_event_not_a_raise(
    monkeypatch, tmp_path
):
    home = _fake_home(tmp_path)
    argv = [str(tmp_path / "no-such-interpreter" / "python.exe")]
    events = _drive(monkeypatch, argv, home, timeout=10)
    assert [e.kind for e in events] == ["error"]
    assert events[0].text


def test_a_missing_checkout_under_an_explicit_home_is_an_error_event(
    monkeypatch, tmp_path
):
    argv = _fake_runner(tmp_path, "raise SystemExit(0)\n")
    events = _drive(monkeypatch, argv, tmp_path / "not-here")
    assert [e.kind for e in events] == ["error"]
    assert "not found" in events[0].text


@pytest.mark.skipif(
    not harness_dsh.available(), reason="no DSH checkout on this machine"
)
def test_the_real_checkout_resolves_to_the_sdk_source_and_cli():
    assert harness_dsh.home().is_dir()
    assert harness_dsh.sdk_src().is_dir()
    assert harness_dsh.dsh_bin().is_file()
    assert harness_dsh.available() is True


# -- the seam's contract, and what driving the real CLI found ------------------


def test_every_adapter_takes_the_arguments_the_server_actually_passes():
    """The contract, checked against the ONE caller that matters.

    `serve.py` calls every adapter with the same keyword set. An adapter that
    cannot accept one of them raises TypeError — and the caller CATCHES it and
    reports a failed turn, so the arm looks broken rather than mismatched. That
    is precisely how the DSH adapter shipped unable to run a single turn through
    the product: it had no `state` parameter, worked when driven directly, and
    every unit test called it directly.

    The keyword set is READ OUT OF serve.py, not retyped here. A hand-written
    copy is the same class of bug one level up: it only ever covers the names
    somebody remembered to write down, so a NEW keyword added to the call site
    is invisible to the guard that exists to notice exactly that. Parsed with
    `ast` because the call is multi-line and keyword-only.
    """
    import ast
    import inspect
    from pathlib import Path

    from rigma import harness as seam

    serve_py = (Path(__file__).resolve().parents[1] / "src" / "rigma"
                / "serve.py")
    tree = ast.parse(serve_py.read_text(encoding="utf-8"), filename=str(serve_py))
    passed: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        name = fn.attr if isinstance(fn, ast.Attribute) else ""
        if name == "drive_turn":
            passed |= {k.arg for k in node.keywords if k.arg}
    # if the call site ever stops naming these, this guard has lost its subject
    assert {"state", "cancel", "permission"} <= passed, sorted(passed)

    checked = 0
    for name in ("dsh", "mcode"):
        mod = seam.adapter(name)
        assert mod is not None, f"{name} should have an adapter"
        # `adapter()` returns the MODULE; the server calls `adapter.drive_turn`.
        params = inspect.signature(mod.drive_turn).parameters
        missing = sorted(k for k in passed if k not in params)
        assert not missing, f"{name}.drive_turn cannot accept {missing} from serve.py"
        checked += 1
    assert checked == 2


@pytest.fixture(autouse=True)
def _clean_pool():
    """The runner pool is module-level and deliberately long-lived, so no test
    may inherit one. Every entry is a live Node child."""
    harness_dsh._reap_all()
    yield
    harness_dsh._reap_all()


_ECHO_PID = """
    import json, os, sys
    for line in sys.stdin:
        if not line.strip():
            continue
        sys.stdout.write(json.dumps(
            {"type": "done", "text": str(os.getpid()),
             "finish_reason": "completed"}) + "\\n")
        sys.stdout.flush()
    """


def test_the_runner_outlives_a_turn_so_a_session_can_be_continued(
        monkeypatch, tmp_path):
    """Continuity is a property of the PROCESS, not of a session id.

    Measured: a second `run()` on one harness continues the session and carries
    the history, while a second PROCESS handed the same id dies with
    `JsonRpcError: session "..." already exists`. So the same chat MUST get the
    same runner — this is the whole mechanism, and the pid is the proof.
    """
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _ECHO_PID)

    first = _drive(monkeypatch, argv, home, session_id="chat-1")
    second = _drive(monkeypatch, argv, home, session_id="chat-1")

    assert first[0].text == second[0].text, "same chat must reuse the same runner"


def test_a_different_chat_gets_its_own_runner(monkeypatch, tmp_path):
    """Two conversations must not share one agent — that would put one chat's
    history in front of the other's model."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _ECHO_PID)

    one = _drive(monkeypatch, argv, home, session_id="chat-1")
    two = _drive(monkeypatch, argv, home, session_id="chat-2")

    assert one[0].text != two[0].text


def test_a_changed_model_does_not_reuse_the_runner(monkeypatch, tmp_path):
    """A runner is built for one endpoint and model. Reusing it after either
    changed would quietly answer from the wrong one."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _ECHO_PID)

    one = _drive(monkeypatch, argv, home, session_id="chat-1", model="a")
    two = _drive(monkeypatch, argv, home, session_id="chat-1", model="b")

    assert one[0].text != two[0].text


def test_a_turn_that_does_not_finish_is_dropped_from_the_pool(monkeypatch, tmp_path):
    """A killed child or a session mid-prompt is not a runtime to write into
    again. Dropping it means the next turn starts clean instead of inheriting
    the wreckage."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import sys
        sys.stdout.write('{"type": "notice", "text": "working"}\\n')
        sys.stdout.flush()
        sys.exit(3)
        """,
    )
    events = _drive(monkeypatch, argv, home, session_id="chat-x", timeout=10)

    assert events[-1].kind == "error"
    assert "chat-x" not in harness_dsh._pool


def test_a_finished_turn_stays_in_the_pool(monkeypatch, tmp_path):
    """The other half of the same rule, so the drop above is not just 'always
    drop' — which would pass its test and break continuity entirely."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _ECHO_PID)
    _drive(monkeypatch, argv, home, session_id="chat-y")
    assert "chat-y" in harness_dsh._pool


def test_the_pool_is_bounded(monkeypatch, tmp_path):
    """It is a long-lived server process, so an unbounded pool is an unbounded
    number of Node children."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _ECHO_PID)
    for i in range(harness_dsh._POOL_MAX + 2):
        _drive(monkeypatch, argv, home, session_id=f"chat-{i}")
    assert len(harness_dsh._pool) <= harness_dsh._POOL_MAX


def test_a_stop_is_a_notice_not_a_failure(monkeypatch, tmp_path):
    """The person who pressed stop does not need to be told their own action
    failed. Reporting it as an error puts a red failure in the transcript for
    the one outcome they chose — measured against the real CLI: a cancel at
    0.25s returned at 0.34s, and said `error` before this was fixed."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import sys, time
        sys.stdout.write('{"type": "notice", "text": "working"}\\n')
        sys.stdout.flush()
        time.sleep(120)
        """,
    )
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    events = _drive(monkeypatch, argv, home, timeout=60, cancel=cancel)

    assert [e.kind for e in events] == ["notice", "notice"]
    assert events[-1].text == "stopped"
    assert not any(e.kind == "error" for e in events)


def test_a_hard_stop_kills_the_tree_not_just_the_runner(monkeypatch):
    """A timeout must not leave the Node agent (and its subagents) running.

    `proc.kill()` is TerminateProcess on the direct child only, and the child is
    `python -m rigma._dsh_runner` while the dsh CLI is a Node GRANDCHILD. Killing
    the runner means its `finally: live.close()` never runs, so the agent keeps
    the model server busy and holds VRAM. The cancel watcher already uses
    `kill_tree` for exactly this reason; the timeout path went through
    `proc.kill()` (09-1)."""
    calls = []

    class FakeProc:
        pid = 999999
        stdin = stdout = stderr = None

        def poll(self):
            return None

        def kill(self):
            calls.append("proc.kill")

        def terminate(self):
            calls.append("proc.terminate")

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(harness_dsh._harness, "kill_tree",
                        lambda proc: calls.append("kill_tree"))
    run = harness_dsh._Run(proc=FakeProc(), hard=True)
    harness_dsh._stop(run)

    assert calls == ["kill_tree"], calls


def test_a_timeout_is_still_an_error_not_a_stop(monkeypatch, tmp_path):
    """The two must not be confused: a cancel is the user's choice, a timeout is
    a fault. If a timeout were reported as a stop, a hung turn would look like a
    turn the user ended."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import sys, time
        sys.stdout.write('{"type": "notice", "text": "working"}\\n')
        sys.stdout.flush()
        time.sleep(120)
        """,
    )
    events = _drive(monkeypatch, argv, home, timeout=1.5, cancel=threading.Event())

    assert events[-1].kind == "error"
    assert "timed out" in events[-1].text


def _tree_proc(*, wait_ok: bool = True):
    """A process stub for `kill_tree`: never spawned, never left behind."""
    class FakeProc:
        pid = 4242
        stdin = stdout = stderr = None

        def kill(self):
            pass

        def terminate(self):
            pass

        def wait(self, timeout=None):
            if not wait_ok:
                raise subprocess.TimeoutExpired("fake", timeout)
            return 0

        def poll(self):
            return None     # `_stop` must take the kill path

    return FakeProc()


def _exited_proc():
    """A runner stub that has ALREADY exited: `_stop` must not assume the tree
    went with it (DR4). Never spawned, never signalled."""
    class FakeProc:
        pid = 4242
        stdin = stdout = stderr = None

        def kill(self):
            pass

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return 0

        def poll(self):
            return 0        # the runner is gone; the grandchild may not be

    return FakeProc()


def _posix_kill_tree(monkeypatch):
    """Put `tools._kill_tree` on its real POSIX branch with no real process.

    The os primitives are stubbed, so `killpg` is recorded, never sent. This is
    what makes the two tests below exercise the branch `kill_tree` actually
    takes on Linux/macOS rather than a `_kill_tree` stub. `raising=False` adds
    `os.getpgid`/`os.killpg`, which do not exist on a Windows host.
    """
    monkeypatch.setattr(tools.sys, "platform", "linux")
    monkeypatch.setattr(tools.os, "getpgid", lambda pid: 9000 + pid, raising=False)
    monkeypatch.setattr(tools.os, "killpg", lambda pgid, sig: None, raising=False)
    monkeypatch.setattr(tools.os, "kill", lambda pid, sig: None)


def test_a_posix_stop_still_reaches_the_group_after_the_runner_exited(monkeypatch):
    """DR4: a runner that already exited does not mean its tree is gone.

    The process Rigma holds is `python -m rigma._dsh_runner`; the dsh CLI is a
    Node GRANDCHILD in the runner's detached group. `_stop` recorded
    `tree_killed = True` "by definition" when the runner had exited, although
    the grandchild may still be alive holding VRAM. On POSIX the group is
    reachable through the reaped leader's pid, so the tree kill must still be
    attempted. `tools._kill_tree` is stubbed, so this asserts the attempt, not a
    real killpg (UNVERIFIED on this Windows host).
    """
    killed = []
    monkeypatch.setattr(harness_dsh._harness, "_DETACH_CHILDREN", True)
    monkeypatch.setattr(tools, "_kill_tree",
                        lambda pid, proc=None, **k: killed.append(pid) or True)

    run = harness_dsh._Run(proc=_exited_proc(), hard=True)
    harness_dsh._stop(run)

    assert killed == [4242], killed
    assert run.tree_killed and run.tree_killed.confirmed


def test_a_windows_stop_does_not_taskkill_an_exited_runners_pid(monkeypatch):
    """On Windows there is no group to reach once the runner is reaped, and
    `taskkill /T` on a reaped pid is the pid-reuse hazard DR9 records. The
    assumption that the tree died with the runner stays there."""
    killed = []
    monkeypatch.setattr(harness_dsh._harness, "_DETACH_CHILDREN", False)
    monkeypatch.setattr(tools, "_kill_tree",
                        lambda pid, proc=None, **k: killed.append(pid) or True)

    run = harness_dsh._Run(proc=_exited_proc(), hard=True)
    harness_dsh._stop(run)

    assert killed == [], killed
    assert run.tree_killed is True


def test_a_failed_tree_kill_is_reported_not_assumed(monkeypatch):
    """`kill_tree` reports what it could CONFIRM, not what it attempted.

    `_kill_tree` is the only thing that reaches the Node grandchild, and it can
    come back unconfirmed — `taskkill /T` is refused here, and the Python runner
    dying says nothing about the grandchild it left. `proc.kill()` still ends the
    turn either way, so the message used to say the agent was killed while it was
    in fact still running against the model server, holding VRAM. The user's only
    signal is that message.
    """
    monkeypatch.setattr(tools, "_kill_tree", lambda *a, **k: False)
    result = harness_dsh._harness.kill_tree(_tree_proc())
    assert result.ok is False
    assert result.attempted is True

    # and the turn that timed out says so, rather than claiming a kill
    run = harness_dsh._Run(proc=_tree_proc(), hard=True)
    events = list(harness_dsh._read_events(run, 0.0))
    assert events[-1].kind == "error"
    assert "STILL BE RUNNING" in events[-1].text, events[-1].text
    assert "could not be confirmed dead" in events[-1].text
    assert "and was killed" not in events[-1].text


def test_a_tree_kill_that_worked_says_nothing_extra(monkeypatch):
    """The warning must not become noise on the path that works."""
    monkeypatch.setattr(tools, "_kill_tree", lambda *a, **k: True)
    result = harness_dsh._harness.kill_tree(_tree_proc())
    assert result.ok is True
    assert result.confirmed is True

    run = harness_dsh._Run(proc=_tree_proc(), hard=True)
    events = list(harness_dsh._read_events(run, 0.0))
    assert events[-1].kind == "error"
    assert "and was killed" in events[-1].text
    assert "STILL BE RUNNING" not in events[-1].text
    assert "could not be confirmed dead" not in events[-1].text


def test_a_posix_tree_kill_reports_the_death_it_confirmed(monkeypatch):
    """POSIX + already-dead → ok True.

    The old body hardcoded `ok = False` off Windows, so a tree that HAD died was
    reported as unconfirmed on every Linux/macOS timeout. This runs the real
    `tools._kill_tree` POSIX branch (killpg then poll) with its os primitives
    stubbed, so the branch under test is the one POSIX takes.
    """
    _posix_kill_tree(monkeypatch)

    result = harness_dsh._harness.kill_tree(_tree_proc(wait_ok=True))

    assert result.ok is True
    assert result.attempted is True
    assert result.confirmed is True


def test_a_posix_tree_kill_reports_a_death_it_could_not_confirm(monkeypatch):
    """POSIX + still-alive-after-kill → ok False and attempted True."""
    _posix_kill_tree(monkeypatch)

    result = harness_dsh._harness.kill_tree(_tree_proc(wait_ok=False))

    assert result.ok is False
    assert result.attempted is True
    assert result.confirmed is False


def test_the_tree_kill_is_delegated_exactly_once(monkeypatch):
    """One delegation is the contract (AUDIT F35).

    An earlier revision ran `taskkill` here AND inside `tools._kill_tree`, so the
    kill happened twice. `kill_tree` must not spawn a killer of its own.
    """
    calls = []
    monkeypatch.setattr(tools, "_kill_tree",
                        lambda pid, proc=None, **kw: calls.append(pid) or True)
    monkeypatch.setattr(subprocess, "run",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("kill_tree spawned its own killer")))

    result = harness_dsh._harness.kill_tree(_tree_proc())

    assert calls == [4242], calls
    assert result.ok is True


def test_a_confirmed_dead_tree_is_not_reported_as_a_failure(monkeypatch):
    """The timeout message must not claim the agent may still be running when the
    kill was confirmed.

    The old `kill_tree` returned `ok = False` off Windows unconditionally, so a
    Linux/macOS timeout claimed the agent might still be holding VRAM even when
    the process was already gone. The message is the only signal the owner gets.
    """
    monkeypatch.setattr(tools, "_kill_tree", lambda *a, **k: True)

    run = harness_dsh._Run(proc=_tree_proc(), hard=True)
    events = list(harness_dsh._read_events(run, 0.0))

    assert events[-1].kind == "error"
    assert "and was killed" in events[-1].text, events[-1].text
    assert "could not be confirmed dead" not in events[-1].text
    assert "STILL BE RUNNING" not in events[-1].text


def test_a_dsh_child_is_detached_on_posix_only(monkeypatch):
    """The pooled runner must be in its OWN process group on POSIX.

    `kill_tree` reaches a tree with `killpg` there, and a runner left in RIGMA's
    group would make a stop take the server down with it. Windows is unchanged:
    `_detached_kwargs` adds nothing there. The platform decision is the module
    constant, flipped here so no global `os.name` patch is needed.
    """
    captured = []

    class _Stop(Exception):
        pass

    def spy(argv, **kw):
        captured.append(kw)
        raise _Stop

    monkeypatch.setattr(harness_dsh.subprocess, "Popen", spy)

    monkeypatch.setattr(harness_dsh._harness, "_DETACH_CHILDREN", True)
    with pytest.raises(_Stop):
        harness_dsh._spawn("pkey", ("k",), {})
    assert captured[-1].get("start_new_session") is True, captured[-1]

    monkeypatch.setattr(harness_dsh._harness, "_DETACH_CHILDREN", False)
    with pytest.raises(_Stop):
        harness_dsh._spawn("pkey", ("k",), {})
    assert "start_new_session" not in captured[-1], captured[-1]


# --- R4-SESS-1: a lost agent context must not be silent ----------------------
#
# DSH keeps a conversation inside the PROCESS that created it; the SDK has no way
# to reopen a session it did not create. So a fresh runtime for a chat that has
# already spoken means the agent has lost everything it knew — the pool evicted
# it, an earlier turn failed, or Rigma restarted. Nothing said so, and the
# transcript still looked continuous, which is the part that is actually wrong:
# the user reads a reply as a continuation of a conversation the model can no
# longer see.

_RUNNER_BODY = """
import json, sys
for line in sys.stdin:
    if not line.strip():
        continue
    print(json.dumps({"type": "text", "text": "ok"}), flush=True)
    print(json.dumps({"type": "done", "text": "ok"}), flush=True)
"""


def _cleanup() -> None:
    """Kill every pooled runner and forget the per-chat history.

    Both, because the notice is driven by `_spoke` as well as the pool: leaving
    `_spoke` populated across tests would make a later test's FIRST turn look
    like a returning one.
    """
    harness_dsh._reap_all()
    harness_dsh._spoke.clear()


def _first_turn(monkeypatch, tmp_path):
    home = _fake_home(tmp_path)
    argv = _fake_runner(tmp_path, _RUNNER_BODY)
    first = _drive(monkeypatch, argv, home, session_id="chat-1")
    return home, argv, first


def test_the_first_turn_does_not_claim_context_was_lost(monkeypatch, tmp_path):
    """A chat's FIRST turn also spawns a fresh runtime. Warning there would be a
    false alarm, which is worse than the silence this replaces."""
    _cleanup()
    try:
        _home, _argv, first = _first_turn(monkeypatch, tmp_path)
        assert [e for e in first if e.kind == "notice" and "lost" in e.text] == []
    finally:
        _cleanup()


def test_a_second_turn_on_a_live_runtime_does_not_claim_loss(monkeypatch, tmp_path):
    """The normal case — the pool still holds the process — must stay quiet."""
    _cleanup()
    try:
        home, argv, _first = _first_turn(monkeypatch, tmp_path)
        second = _drive(monkeypatch, argv, home, session_id="chat-1")
        assert [e for e in second if e.kind == "notice" and "lost" in e.text] == []
    finally:
        _cleanup()


def test_a_lost_runtime_is_reported_on_the_next_turn(monkeypatch, tmp_path):
    """The case the whole thing exists for: the process is gone, so the agent's
    context is gone, and the turn says so."""
    _cleanup()
    try:
        home, argv, _first = _first_turn(monkeypatch, tmp_path)
        # Evict the runtime, the way the pool, a failed turn or a restart would.
        harness_dsh._drop("chat-1")
        second = _drive(monkeypatch, argv, home, session_id="chat-1")
        notices = [e for e in second if e.kind == "notice" and "lost" in e.text]
        assert len(notices) == 1, [e.kind for e in second]
        # It must say what was lost AND that Rigma's own transcript is intact,
        # because the second half is what stops it reading as data loss.
        assert "DSH" in notices[0].text
        assert "transcript" in notices[0].text
    finally:
        _cleanup()


def test_a_different_chat_is_not_told_it_lost_context(monkeypatch, tmp_path):
    """Tracking is PER CHAT. A second chat's first turn has lost nothing, and
    telling it otherwise would train the user to ignore the notice."""
    _cleanup()
    try:
        home, argv, _first = _first_turn(monkeypatch, tmp_path)
        harness_dsh._drop("chat-1")
        other = _drive(monkeypatch, argv, home, session_id="chat-2")
        assert [e for e in other if e.kind == "notice" and "lost" in e.text] == []
    finally:
        _cleanup()
