"""Run exactly one DeepSeek Harness turn, in a process Rigma is allowed to kill.

WHY THIS IS A SEPARATE MODULE AND A SEPARATE PROCESS.

Rigma's process must never import `deepseek_harness`. The SDK is not installed in
Rigma's venv, its runtime is a Node CLI, and a wedged turn must not be able to
wedge the server that is streaming a chat. So the import happens HERE, in a child
whose PYTHONPATH names the checkout's `python/sdk/src` for the length of one turn,
and whose entire contract with the parent is one JSON object per stdout line.

The protocol is deliberately dumb: one job object in on stdin, NDJSON out, no
shared state. Diagnostics go to stderr — including anything the SDK decides to
print, because stdout is reserved for events and a stray `print()` would corrupt
the stream. `close()` runs in `finally` so the Node child is reaped even when the
turn failed.
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

_CLI_REL = ("python", "sdk-runtime", "node_modules", ".bin", "dsh.CMD")

# Captured at import: after main() redirects sys.stdout, this is the only handle
# that still points at the event stream.
_STDOUT = sys.stdout


def _emit(obj: dict) -> None:
    """One JSON object per line, and nothing else, ever."""
    _STDOUT.write(json.dumps(obj, default=str) + "\n")
    _STDOUT.flush()


# What a person watching a turn actually wants to see. DSH emits its whole
# internal lifecycle as notifications — measured 2026-09-21, a one-line turn
# produced fifteen of them (agent/inbox/spliced, step/start, request/header,
# session/title, delivery-accepted…) and forwarding every one buried the reply
# in noise on a progress line. Tool activity and failures are the signal.
_NOTICE_WORTHY = ("tool", "error", "fail", "denied", "retry", "timeout")


def _notice_text(notification) -> str | None:
    """A one-line summary worth showing, or None when it is internal chatter.

    The parent renders these live, so a whole payload would be both noise and
    potentially enormous — only the method and the inner event type are used.
    """
    method = str(getattr(notification, "method", "") or "notification")
    detail = ""
    payload = getattr(notification, "payload", None)
    if isinstance(payload, dict):
        event = payload.get("event")
        if isinstance(event, dict):
            detail = str(event.get("type") or "")
        elif payload.get("status"):
            detail = str(payload.get("status"))
    blob = f"{method} {detail}".lower()
    if not any(word in blob for word in _NOTICE_WORTHY):
        return None
    return f"{method} {detail}".strip()[:200]


def _cli_for(home: str) -> Path | None:
    path = Path(home).joinpath(*_CLI_REL) if home else None
    return path if path is not None and path.is_file() else None


class _Live:
    """One SDK runtime, kept alive so the NEXT turn can continue its session.

    Continuity is a property of the PROCESS, not of the session id. Measured
    2026-09-21 against 0.1.6-alpha.2: a second `run()` on the same
    `DeepSeekHarness` continues the session correctly — the follow-up request
    carried the first turn's history — while a second PROCESS handed the same id
    dies with `JsonRpcError: session "..." already exists`. So a runner that is
    spawned per turn can never resume, and this one is not.
    """

    def __init__(self) -> None:
        self.harness = None
        self.key: tuple | None = None
        self.session_id = ""
        self.scratch = ""

    def close(self) -> None:
        if self.harness is not None:
            try:
                self.harness.close()
            except Exception:
                pass
            self.harness = None
        if self.scratch:
            shutil.rmtree(self.scratch, ignore_errors=True)
            self.scratch = ""


def _run_turn(job: dict, live: _Live) -> int:
    """Run one turn on the live runtime and emit the outcome. Never raises.

    Returns 0 when a `done` was emitted. Anything else means this runtime is not
    trustworthy for a second turn, and `main` exits so the parent spawns a fresh
    one rather than writing into a broken session.
    """
    try:
        from deepseek_harness import DeepSeekHarness, DeepSeekHarnessConfig
    except Exception as exc:
        _emit(
            {
                "type": "error",
                "text": (
                    "cannot import deepseek_harness — is the SDK source on "
                    f"PYTHONPATH? ({type(exc).__name__}: {exc})"
                ),
            }
        )
        return 1

    try:
        prompt = str(job.get("prompt") or "")
        system_prompt = str(job.get("system_prompt") or "").strip()
        if system_prompt:
            # DSH owns its system prompt, so a Rigma session's prompt can only
            # reach it as a delimited preamble on the input.
            prompt = f"{system_prompt}\n\n---\n\n{prompt}"

        home = str(job.get("dsh_home") or "")       # the CHECKOUT: CLI + SDK
        cli = _cli_for(home)
        if cli is None:
            _emit({"type": "error", "text": f"no dsh CLI under {home!r}"})
            return 1
        # DSH's own session/config dir, which is NOT the checkout. Its session
        # logs live here, so sharing this with a source tree makes one turn
        # resume another and return empty without ever calling the model.
        data_home = str(job.get("data_home") or "") or home

        max_tokens = int(job.get("max_tokens") or 4096)
        context_window = int(job.get("context_window") or 32768)
        cwd = str(job.get("cwd") or "")
        if not cwd.strip() or not Path(cwd).is_dir():
            # Stable for the LIFE of this runtime, not per turn: DSH keys its
            # session directories off the cwd, so a cwd that moved between turns
            # would put the session somewhere the next turn cannot find.
            if not live.scratch:
                live.scratch = tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            cwd = live.scratch

        # Everything that would make the running runtime WRONG for this job. The
        # patch path is deliberately NOT part of it: it is per-turn by nature,
        # and the patch below is regenerated here instead, once per runtime.
        key = (home, data_home, str(job.get("base_url") or ""),
               str(job.get("model") or ""), cwd, context_window, max_tokens)
        if live.harness is not None and live.key != key:
            live.close()            # a different endpoint or model: rebuild

        if live.harness is None:
            if not live.scratch:
                live.scratch = tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            # The parent's patch may already be gone, or may belong to a turn
            # that is over; this runtime generates the one it will keep using.
            patch = str(job.get("patch_path") or "")
            if not patch or not Path(patch).is_file():
                from rigma.harness_dsh import patch_file

                patch = str(patch_file(context_window, max_tokens, live.scratch))
            config = DeepSeekHarnessConfig(
                profile="sdk-minimal",
                dsh_bin=str(cli),
                dsh_home=data_home,
                base_url=str(job.get("base_url") or ""),
                api_key="local",
                model=str(job.get("model") or ""),
                max_tokens=max_tokens,
                cwd=cwd,
                runtime_cwd=cwd,
                patches=(patch,),
            )
            live.harness = DeepSeekHarness(config)
            live.harness.start()
            live.key = key

        def on_notification(notification) -> None:
            try:
                line = _notice_text(notification)
                if line:
                    _emit({"type": "notice", "text": line})
            except Exception:
                pass  # a progress line must never break the turn

        # The session this runtime already owns, which is what makes the second
        # and later turns continue instead of starting the agent from nothing.
        result = live.harness.run(
            prompt,
            session_id=live.session_id or None,
            on_notification=on_notification,
        )
        live.session_id = str(getattr(result, "session_id", "") or "") or live.session_id
        _emit(
            {
                "type": "done",
                "text": str(getattr(result, "final_response", "") or ""),
                "finish_reason": str(getattr(result, "finish_reason", "") or ""),
                # Reported for the record and for the cross-process case, where
                # it is what the NEXT process would need if the SDK ever learns
                # to open a session it did not create.
                "session_id": live.session_id,
            }
        )
        return 0
    except Exception as exc:
        _emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"[:2000]})
        return 1


def main() -> int:
    # From here on, anything the SDK prints lands on stderr: stdout carries
    # events, and a stray print would be an unparseable line to the parent.
    sys.stdout = sys.stderr
    live = _Live()
    code = 0
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                job = json.loads(line)
                if not isinstance(job, dict):
                    raise ValueError("job must be a JSON object")
            except Exception as exc:
                _emit({"type": "error", "text": f"bad job on stdin: {exc}"})
                return 1
            code = _run_turn(job, live)
            if code != 0:
                # This runtime cannot be trusted for another turn. Exit, so the
                # parent spawns a fresh one instead of writing into a session
                # that just failed.
                return code
    finally:
        live.close()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
