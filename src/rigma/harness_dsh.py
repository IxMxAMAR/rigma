"""Hand one chat turn to the DeepSeek Harness, across a process boundary.

WHY A SUBPROCESS AND NOT AN IMPORT.

DSH owns its own agent loop, its own tools and its own system prompt. It is a
Node runtime reached through a pure-Python SDK that is NOT installed in Rigma's
venv. Importing it here would put a second agent loop — and a Node child it can
wedge on — inside the process that is streaming a chat. So Rigma only ever
spawns `python -m rigma._dsh_runner` with the checkout's SDK source prepended to
that child's PYTHONPATH, and speaks NDJSON to it. Rigma's venv stays untouched,
and a hung DSH is a hung child — killable — rather than a hung server.

WHAT THE PARENT SEES is a batch turn with live progress: `notice` events arrive
while DSH works, then one final `text`. The SDK has no streaming text callback
and no cancel API — `on_notification` is the only live hook, and closing the
child is the only interrupt. Nothing here pretends otherwise.

NOTHING HERE KNOWS ABOUT SSE, either. The transport is one JSON object per line
on the child's stdout, so a malformed line is a skipped line and never a crash.
"""
from __future__ import annotations

import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

# The seam's event vocabulary, shared with every other adapter. Re-exported
# rather than redefined so a driver and the turn loop cannot drift apart — and
# so `harness_dsh.TurnEvent` keeps naming the same type.
from .harness import TurnEvent
from . import harness as _harness

# The checkout this machine actually has. The env override exists because a
# checkout is a user's directory, not a fact about Rigma.
_DEFAULT_HOME = r"C:\AI\deepseek-harness"
_PATCH_NAME = "rigma-dsh-patch.yaml"
_SDK_REL = ("python", "sdk", "src")
_CLI_REL = ("python", "sdk-runtime", "node_modules", ".bin", "dsh.CMD")


def home() -> Path | None:
    """The DSH checkout, or None when there is not one here."""
    raw = (os.environ.get("RIGMA_DSH_HOME") or "").strip() or _DEFAULT_HOME
    path = Path(raw)
    return path if path.is_dir() else None


def sdk_src() -> Path | None:
    """The directory that has to be on the child's PYTHONPATH, or None."""
    root = home()
    path = root.joinpath(*_SDK_REL) if root else None
    return path if path is not None and path.is_dir() else None


def dsh_bin() -> Path | None:
    """The runnable CLI. Passing this bypasses the separate runtime-bin package."""
    root = home()
    path = root.joinpath(*_CLI_REL) if root else None
    return path if path is not None and path.is_file() else None


def available() -> bool:
    """Whether a turn can be handed over at all: source AND CLI, or neither."""
    return sdk_src() is not None and dsh_bin() is not None


def data_home() -> Path:
    """Where DSH keeps its OWN sessions and profile state.

    NOT the checkout. DSH writes session logs under `dsh_home`, so pointing it at
    the source tree does two bad things: it litters an install Rigma does not own,
    and it makes sessions collide across runs — a turn whose id already exists is
    RESUMED rather than run, so the model is never called and the turn comes back
    empty. Measured 2026-09-21: two live tests passed alone and failed in the full
    suite for exactly that reason. Rigma-owned, beside Rigma's own state.
    """
    from . import runtime
    path = runtime.rigma_home() / "dsh"
    path.mkdir(parents=True, exist_ok=True)
    return path


def patch_file(context_window: int, max_tokens: int, tmpdir) -> Path:
    """Write the one-row YAML patch that makes DSH talk to Rigma's server.

    A patch row REPLACES that row's whole config, so `apiKeyEnv` is restated:
    dropping it would silently unset the key the child was handed. The protocol
    is restated because the default is `messages` (Anthropic), which would POST
    `/v1/messages` at an OpenAI-compatible server, and the stock window/token
    defaults (1000000 / 256000) are what a 32K local server rejects outright.
    """
    path = Path(tmpdir) / _PATCH_NAME
    path.write_text(
        "# Generated per turn. A patch row REPLACES the whole row config, so\n"
        "# every key this turn depends on is restated here.\n"
        "- id: llm-deepseek\n"
        "  config:\n"
        "    protocol: chat-completions\n"
        "    apiKeyEnv: DEEPSEEK_API_KEY\n"
        f"    defaultContextWindow: {int(context_window)}\n"
        f"    maxTokens: {int(max_tokens)}\n",
        encoding="utf-8",
    )
    return path


def _runner_argv() -> list[str]:
    """The child command.

    A seam, and the one place the entry point is named: tests point this at a
    script that prints canned NDJSON, so the translation is testable without a
    DSH install and without a network.
    """
    return [sys.executable, "-m", "rigma._dsh_runner"]


@dataclass
class _Run:
    """State the reader loop and the cleanup path both need to see."""

    proc: subprocess.Popen | None = None
    hard: bool = False  # timed out: kill, do not ask nicely
    done: bool = False
    stderr: deque = field(default_factory=lambda: deque(maxlen=400))
    lines: queue.Queue = field(default_factory=queue.Queue)


def _tail(state: _Run) -> str:
    """The last of the child's stderr, so a failure is diagnosable from the event."""
    return "".join(state.stderr).strip()[-2000:]


def _event_for(payload: dict) -> TurnEvent | None:
    """Translate one runner event, or None when there is nothing to show."""
    kind = str(payload.get("type") or "").strip().lower()
    if kind in ("text", "thinking", "notice"):
        text = str(payload.get("text") or "")
        if not text and kind != "notice":
            return None  # an empty text event is noise, not progress
        return TurnEvent(kind=kind, text=text)
    if kind == "tool":
        args = payload.get("args")
        return TurnEvent(
            kind="tool",
            name=str(payload.get("name") or ""),
            args=args if isinstance(args, dict) else {},
        )
    if kind == "tool_result":
        return TurnEvent(
            kind="tool_result",
            text=str(payload.get("text") or ""),
            name=str(payload.get("name") or ""),
            ok=bool(payload.get("ok", True)),
        )
    if kind == "error":
        return TurnEvent(kind="error", text=str(payload.get("text") or "DSH error"))
    return None  # unknown kinds are skipped, so the runner can grow new ones


def _start_readers(state: _Run) -> None:
    """Drain both pipes on daemon threads.

    Windows has no selectable pipes, so a blocking read on the parent's thread
    would make the timeout unenforceable. Threads keep the deadline honest.
    """
    proc = state.proc
    assert proc is not None

    def pump_lines() -> None:
        try:
            if proc.stdout is not None:
                for line in proc.stdout:
                    state.lines.put(line)
        except Exception:
            pass
        finally:
            state.lines.put(None)  # EOF marker

    def pump_stderr() -> None:
        try:
            if proc.stderr is not None:
                for line in proc.stderr:
                    state.stderr.append(line)
        except Exception:
            pass

    threading.Thread(target=pump_lines, daemon=True).start()
    threading.Thread(target=pump_stderr, daemon=True).start()


def _exit_code(state: _Run) -> int | None:
    proc = state.proc
    if proc is None:
        return None
    try:
        return proc.wait(timeout=10)
    except Exception:
        return None


def _read_events(state: _Run, timeout: float) -> Iterator[TurnEvent]:
    """Turn the child's stdout into events, enforcing the turn deadline."""
    deadline = time.monotonic() + max(0.1, float(timeout))
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            state.hard = True
            yield TurnEvent(
                kind="error",
                text=(
                    f"DSH turn timed out after {timeout:g}s and was killed. "
                    f"stderr tail: {_tail(state)}"
                ),
            )
            return
        try:
            line = state.lines.get(timeout=min(remaining, 1.0))
        except queue.Empty:
            continue
        if line is None:
            break
        try:
            payload = json.loads(line)
        except (ValueError, TypeError):
            continue  # a malformed line is skipped; diagnostics live on stderr
        if not isinstance(payload, dict):
            continue
        if str(payload.get("type") or "").strip().lower() == "done":
            state.done = True
            text = str(payload.get("text") or "")
            reason = str(payload.get("finish_reason") or "")
            if text:
                yield TurnEvent(kind="text", text=text)  # the turn's final answer
            elif reason == "error":
                # DSH reports a failed model call as an empty done, not as an
                # exception. Passing that through as silence would make a dead
                # endpoint look like a turn that simply had nothing to say.
                yield TurnEvent(
                    kind="error",
                    text="DSH turn ended in error with no reply text "
                    "(the model call failed — check the endpoint and model id)",
                )
            else:
                yield TurnEvent(
                    kind="notice",
                    text=f"DSH finished with reason {reason or 'unknown'} and no reply text",
                )
            continue
        event = _event_for(payload)
        if event is not None:
            yield event

    if state.done:
        return
    code = _exit_code(state)
    yield TurnEvent(
        kind="error",
        text=(
            f"DSH runner exited with code {code} without finishing the turn. "
            f"stderr tail: {_tail(state)}"
        ),
    )


def _stop(state: _Run) -> None:
    """Leave no child behind, however the turn ended."""
    proc = state.proc
    if proc is None:
        return
    try:
        if state.hard:
            if proc.poll() is None:
                proc.kill()
        elif proc.poll() is None:
            # Not a kill: the runner's own `finally` closes the harness, and that
            # is what reaps the Node child. Terminating first would orphan it.
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    for stream in (proc.stdin, proc.stdout, proc.stderr):
        if stream is not None:
            try:
                stream.close()
            except Exception:
                pass


def drive_turn(
    *,
    base_url: str,
    model: str,
    prompt: str,
    system_prompt: str = "",
    session_id: str = "",
    cwd: str = "",
    max_tokens: int = 4096,
    context_window: int = 32768,
    timeout: float = 1800.0,
    dsh_home: str | None = None,
    cancel: threading.Event | None = None,
) -> Iterator[TurnEvent]:
    """Run one turn on DSH and yield what happened.

    Never raises: a failure arrives as an `error` event, because the caller is a
    tool and a tool that raises teaches the model nothing it can act on.

    `cancel` kills the child process, which is the same lever the timeout uses.
    The SDK is BATCH — it reports a whole turn rather than streaming it — so
    there is no per-event boundary at which a cancel could be noticed, and
    killing the process is the only stop that arrives promptly. NOT VERIFIED
    END TO END: DSH is not installed on the machine this was written on, so the
    path is written to the same shape as the mcode one and exercised only by
    unit tests with a stand-in child.
    """
    state = _Run()
    tmpdir = ""
    try:
        root = Path(dsh_home) if dsh_home else home()
        src = root.joinpath(*_SDK_REL) if root else None
        cli = root.joinpath(*_CLI_REL) if root else None
        if src is None or not src.is_dir():
            yield TurnEvent(kind="error", text=f"DSH SDK source not found under {root}")
            return
        if cli is None or not cli.is_file():
            yield TurnEvent(kind="error", text=f"DSH CLI not found under {root}")
            return

        tmpdir = tempfile.mkdtemp(prefix="rigma-dsh-")
        job = {
            "base_url": base_url,
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "session_id": session_id,
            "cwd": cwd,
            "max_tokens": int(max_tokens),
            "context_window": int(context_window),
            "dsh_home": str(root),
            "data_home": str(data_home()),
            "patch_path": str(patch_file(context_window, max_tokens, tmpdir)),
        }
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")

        state.proc = subprocess.Popen(
            _runner_argv(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env=env,
        )
        _start_readers(state)
        if cancel is not None:
            def _watch_cancel() -> None:
                """Same shape as mcode's: wait on the event, poll only to end.

                The TREE, not the runner: the SDK drives a Node child, and
                killing the Python runner alone would leave that child running
                with the pipes open — the read loop would never see EOF, which
                turns a stop into a hang.
                """
                while not cancel.wait(0.25):
                    if state.proc is None or state.proc.poll() is not None:
                        return
                _harness.kill_tree(state.proc)
            threading.Thread(target=_watch_cancel, daemon=True).start()
        try:
            assert state.proc.stdin is not None
            state.proc.stdin.write(json.dumps(job) + "\n")
            state.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass  # the child died on startup; the exit path below reports why
        yield from _read_events(state, timeout)
    except Exception as exc:
        yield TurnEvent(
            kind="error", text=f"DSH turn could not start: {type(exc).__name__}: {exc}"
        )
    finally:
        _stop(state)
        if tmpdir:
            shutil.rmtree(tmpdir, ignore_errors=True)
