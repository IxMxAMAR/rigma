"""The DSH adapter, tested without a DSH install and without a network.

Two things are worth pinning here. First, the generated patch: a patch row
replaces a whole config row, so a missing `apiKeyEnv` or a protocol left at the
Anthropic default is a turn that fails on the wire for a reason nobody can see
from Rigma's output. Second, the subprocess contract: a malformed line, a dead
runner, a non-zero exit and a hung turn must all come back as events, never as an
exception — the caller is a tool.
"""
from __future__ import annotations

import sys
import textwrap
import threading
import time

import pytest

from rigma import harness_dsh


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
    as silence would make a dead endpoint look like a turn with nothing to say."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import json, sys
        rows = [
            {"type": "done", "text": "", "finish_reason": "error"},
            {"type": "done", "text": "", "finish_reason": "stop"},
        ]
        for row in rows:
            sys.stdout.write(json.dumps(row) + "\\n")
        sys.stdout.flush()
        """,
    )
    events = _drive(monkeypatch, argv, home)

    assert [e.kind for e in events] == ["error", "notice"]
    assert "model call failed" in events[0].text
    assert "stop" in events[1].text


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
    """
    import inspect

    from rigma import harness as seam

    passed = {
        "base_url": "http://127.0.0.1:1/v1", "model": "m", "prompt": "p",
        "system_prompt": "s", "session_id": "sid", "cwd": ".",
        "max_tokens": 1, "context_window": 2, "state": {}, "cancel": None,
        "permission": "full",
    }
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


def test_a_turn_does_not_hand_back_a_session_it_cannot_resume(monkeypatch, tmp_path):
    """Storing a handle the next turn cannot use turns turn 2 into a hard
    failure — measured, `JsonRpcError: session "..." already exists`. A fresh
    turn that answers beats a resuming turn that dies."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import json, sys
        sys.stdout.write(json.dumps(
            {"type": "done", "text": "hi", "finish_reason": "completed",
             "session_id": "session-abc123"}) + "\\n")
        sys.stdout.flush()
        """,
    )
    hstate = {"session_id": ""}
    events = _drive(monkeypatch, argv, home, state=hstate)

    assert [e.kind for e in events] == ["text"]
    assert hstate["session_id"] == "", "no resume, so no handle to store"


def test_a_handle_is_used_and_stored_once_the_sdk_can_resume(monkeypatch, tmp_path):
    """The other half: with CAN_RESUME on, the handle round-trips. This is the
    one-line flip the constant exists for, so it is pinned rather than assumed."""
    home = _fake_home(tmp_path)
    argv = _fake_runner(
        tmp_path,
        """
        import json, sys
        job = json.loads(sys.stdin.readline())
        sys.stdout.write(json.dumps(
            {"type": "done", "text": job.get("session_id") or "none",
             "finish_reason": "completed", "session_id": "session-abc123"}) + "\\n")
        sys.stdout.flush()
        """,
    )
    monkeypatch.setattr(harness_dsh, "CAN_RESUME", True)
    hstate = {"session_id": "session-previous"}
    events = _drive(monkeypatch, argv, home, state=hstate)

    # the id was READ from state and passed to the runner...
    assert events[0].text == "session-previous"
    # ...and the new one was written back
    assert hstate["session_id"] == "session-abc123"


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

