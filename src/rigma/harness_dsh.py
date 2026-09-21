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

import atexit
import json
import os
import queue
import subprocess
import sys
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


# Deliberately EMPTY. DSH has never been driven end to end from this machine —
# the module docstring says so — so there is no build this adapter was measured
# against, and inventing a number would make `conformance` report agreement with
# something that was never checked. Empty means "unknown", and `conformance`
# says unknown rather than implying fine.
VERIFIED = "0.1.6-alpha.2"

# WHY CONTINUITY IS A PROCESS, NOT A SESSION ID — read before changing `_pool`.
#
# Measured 2026-09-21 against 0.1.6-alpha.2: `RunResult.session_id` exists, but a
# session can only be continued by the PROCESS that created it. A second `run()`
# on one harness resumes correctly — the follow-up request carried the first
# turn's history, verified from the wire — while a second PROCESS handed that id
# dies with `JsonRpcError: session "..." already exists`. The cause is upstream:
# the SDK's server resolves a session from an IN-MEMORY map and only on a miss
# calls `agents.create`, which refuses an id already on disk
# (`packages/sdk/server/src/server.ts`, `getOrCreateSession`). The TypeScript
# layer has a real resume and the CLI exposes it as `--session-id`, but
# `DeepSeekHarnessConfig` has no field for it.
#
# So the handle is NOT passed through `state`: storing one would guarantee that
# every turn after the first fails, which is strictly worse than starting fresh,
# because a turn that dies teaches the model nothing. Continuity comes from the
# pooled runner outliving a turn instead — see `_pool` below, which is where this
# is actually implemented. Nothing reads this comment; it is here so the next
# person does not "fix" continuity by putting the id back in `state`.


def backend_version() -> str:
    """The version that would MOVE under Rigma, or "" when it cannot say.

    The CLI's own `--version` first: measured 2026-09-21 against 0.1.6-alpha.2, it
    prints a stable semver that only changes on a release, which is exactly what
    drift detection needs. (The earlier claim that a checkout has no version
    command was wrong — `dsh.CMD --version` answers.)

    `git describe` only as a fallback, because it is the wrong SHAPE for this:
    measured, it returns `dsh-v0.1.5-rc.2-1687-gddefc45fbc-dirty` — the commit
    count moves on every commit and `-dirty` is permanent on a working checkout,
    so a pinned value could never match it and every turn would report drift.
    Useful when there is no CLI to ask, which is the only reason it is still here.

    Never raises — this runs from a menu.
    """
    root = home()
    if root is None:
        return ""
    cli = root.joinpath(*_CLI_REL)
    if cli.is_file():
        try:
            out = subprocess.run([str(cli), "--version"], capture_output=True,
                                 text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "describe", "--tags", "--always", "--dirty"],
            capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


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
    hstate: dict | None = None          # the backend-owned handle, in and out
    cancel: threading.Event | None = None


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
            # The turn is OVER. Return rather than continue: the runtime stays up
            # for the next turn, so waiting for more lines here would spin until
            # the deadline and report a timeout on a turn that had already
            # answered — which is exactly what a pooled runner did before this.
            return
        event = _event_for(payload)
        if event is not None:
            yield event

    if state.cancel is not None and state.cancel.is_set():
        # A stop is a NOTICE, not a failure. The person who pressed stop does not
        # need to be told their own action failed, and reporting it as an error
        # would put a red failure in the transcript for the one outcome they
        # chose. Same shape as mcode's, for the same reason.
        yield TurnEvent(kind="notice", text="stopped")
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


# -- keeping a runtime alive between turns ------------------------------------
#
# A DSH session can only be continued by the PROCESS that created it. Measured
# 2026-09-21 against 0.1.6-alpha.2: a second `run()` on one harness resumes and
# the follow-up request carries the first turn's history, while a second PROCESS
# handed the same id dies with `JsonRpcError: session "..." already exists`.
# So the runner is NOT spawned per turn — it is pooled per Rigma chat session and
# fed one job per turn, which is what makes continuity exist at all.
#
# Bounded, and every entry is killed at interpreter exit: a leaked entry is a
# leaked Node child, and this is a long-lived server process.

_POOL_MAX = 4
_pool: dict[str, "_Live"] = {}
_pool_lock = threading.Lock()


class _Live:
    """A runner process, its reader threads, and the config it was built for."""

    def __init__(self, proc: subprocess.Popen, run: "_Run", key: tuple) -> None:
        self.proc = proc
        self.run = run
        self.key = key
        # One runtime speaks one NDJSON stream. Two turns writing into it at once
        # would interleave jobs and replies into a stream neither could parse, so
        # the second waits. Rigma turns a session one at a time, so this should
        # never contend — it is here because the failure it prevents is a
        # corrupted stream, which is close to undiagnosable from the outside.
        self.turn_lock = threading.Lock()


def _kill(live: "_Live") -> None:
    live.run.hard = True      # kill the tree; do not wait for a graceful exit
    _stop(live.run)


def _reap_all() -> None:
    with _pool_lock:
        lives = list(_pool.values())
        _pool.clear()
    for live in lives:
        _kill(live)


atexit.register(_reap_all)


def _drop(pkey: str) -> None:
    with _pool_lock:
        live = _pool.pop(pkey, None)
    if live is not None:
        _kill(live)


def _take(pkey: str, key: tuple) -> "_Live | None":
    """The pooled runner for this chat, if it is alive AND still right for us."""
    with _pool_lock:
        live = _pool.get(pkey)
    if live is None:
        return None
    if live.key != key or live.proc.poll() is not None:
        _drop(pkey)          # wrong endpoint, or dead: not reusable
        return None
    return live


def _spawn(pkey: str, key: tuple, env: dict) -> "_Live":
    run = _Run()
    run.proc = subprocess.Popen(
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
    _start_readers(run)
    live = _Live(run.proc, run, key)
    evicted: list[_Live] = []
    with _pool_lock:
        while len(_pool) >= _POOL_MAX:
            victim = next(iter(_pool), None)   # oldest first
            if victim is None:
                break
            evicted.append(_pool.pop(victim))
        _pool[pkey] = live
    for old in evicted:
        _kill(old)                 # killed outside the lock, never held across it
    return live


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
    state: dict | None = None,
    cancel: threading.Event | None = None,
    permission: str = "full",
) -> Iterator[TurnEvent]:
    """Run one turn on DSH and yield what happened.

    Never raises: a failure arrives as an `error` event, because the caller is a
    tool and a tool that raises teaches the model nothing it can act on.

    `state`, when given, is read for the DSH session to CONTINUE and written back
    with the one this turn produced. It is the BACKEND's handle and not Rigma's:
    DSH's own ids look like `session-<32 hex>`, so Rigma's chat id means nothing
    to it. This parameter was MISSING until 2026-09-21, which made every turn
    through the web seam raise `TypeError: unexpected keyword argument 'state'`
    and come back as a failed turn — the adapter worked when driven directly and
    was broken in the product, which is the gap that driving it end to end found.

    `permission` is ACCEPTED AND IGNORED. DSH's confinement is its own bundle's
    business — `sdk-minimal` pins the mode — and mapping a Rigma setting onto it
    would be Rigma choosing the agent's policy, which is the line this seam does
    not cross. The argument exists so the contract is one shape for every
    adapter rather than one shape per adapter.

    `cancel` kills the child process, which is the same lever the timeout uses.
    The SDK is BATCH — it reports a whole turn rather than streaming it — so
    there is no per-event boundary at which a cancel could be noticed, and
    killing the process is the only stop that arrives promptly. Verified against
    the real CLI 0.1.6-alpha.2: a cancel at 0.25s returned at 0.35s.
    """
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

        # Everything that would make a pooled runtime WRONG for this turn. The
        # patch is deliberately absent: it is per-runtime now, not per-turn, and
        # the runner generates and keeps its own.
        key = (str(root), str(data_home()), str(base_url), str(model),
               str(cwd), int(context_window), int(max_tokens))
        # Keyed by RIGMA's chat session, because that is the conversation and the
        # only name stable across turns. The DSH handle is not known until the
        # first turn answers, and it is the RUNTIME that owns it, not this side —
        # so continuity comes from reusing the process, not from passing an id.
        pkey = str(session_id or "").strip() or f"cwd:{cwd}"
        job = {
            "base_url": base_url,
            "model": model,
            "prompt": prompt,
            "system_prompt": system_prompt,
            "cwd": cwd,
            "max_tokens": int(max_tokens),
            "context_window": int(context_window),
            "dsh_home": str(root),
            "data_home": str(data_home()),
            "patch_path": "",       # the runner owns the patch for its lifetime
        }
        env = os.environ.copy()
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = str(src) + (os.pathsep + existing if existing else "")

        live = _take(pkey, key) or _spawn(pkey, key, env)
        run = live.run
        run.done = False            # per-turn: the reader stops at this turn's
        run.hard = False            # `done`, not at the runtime's first one
        run.cancel = cancel         # so a stop is reported as a stop

        # A watcher PER TURN, and it must end with the turn. The runtime is
        # shared, so a watcher that outlived its turn would leave a thread per
        # turn waiting on an event that will never fire again.
        stop_watch = threading.Event()
        if cancel is not None:
            def _watch_cancel() -> None:
                """The TREE, not the runner: the SDK drives a Node child, and
                killing the Python runner alone leaves that child holding the
                pipes open — the read loop would never see EOF, which turns a
                stop into a hang."""
                while not cancel.wait(0.25):
                    if stop_watch.is_set() or live.proc.poll() is not None:
                        return
                if not stop_watch.is_set():
                    _harness.kill_tree(live.proc)
            threading.Thread(target=_watch_cancel, daemon=True).start()

        live.turn_lock.acquire()
        try:
            assert live.proc.stdin is not None
            live.proc.stdin.write(json.dumps(job) + "\n")
            live.proc.stdin.flush()     # one job per LINE; the runtime stays up
        except (BrokenPipeError, OSError):
            pass  # the child died on startup; the exit path below reports why
        try:
            yield from _read_events(run, timeout)
        finally:
            stop_watch.set()
            # A turn that did not finish leaves the runtime in a state we cannot
            # vouch for — a killed child, or a session mid-prompt. Drop it, so
            # the next turn starts clean instead of writing into the wreckage.
            # `hard` catches the timeout, which sets it before yielding its error
            # and would otherwise leave a live child with no turn to serve.
            if not run.done or run.hard:
                _drop(pkey)
            live.turn_lock.release()
    except Exception as exc:
        yield TurnEvent(
            kind="error", text=f"DSH turn could not start: {type(exc).__name__}: {exc}"
        )
