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


def _run_turn(job: dict) -> int:
    """Import the SDK, run the turn, emit the outcome. Never raises."""
    harness = None
    scratch = ""
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
        cwd = str(job.get("cwd") or "")
        if not cwd.strip() or not Path(cwd).is_dir():
            scratch = tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            cwd = scratch

        patch = str(job.get("patch_path") or "")
        if not patch or not Path(patch).is_file():
            # The parent normally writes it; regenerate rather than run with the
            # stock Anthropic protocol, which an OpenAI server cannot answer.
            from rigma.harness_dsh import patch_file

            scratch = scratch or tempfile.mkdtemp(prefix="rigma-dsh-cwd-")
            patch = str(
                patch_file(
                    int(job.get("context_window") or 32768), max_tokens, scratch
                )
            )

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
        harness = DeepSeekHarness(config)
        harness.start()

        def on_notification(notification) -> None:
            try:
                line = _notice_text(notification)
                if line:
                    _emit({"type": "notice", "text": line})
            except Exception:
                pass  # a progress line must never break the turn

        session_id = str(job.get("session_id") or "")
        result = harness.run(
            prompt,
            session_id=session_id or None,
            on_notification=on_notification,
        )
        _emit(
            {
                "type": "done",
                "text": str(getattr(result, "final_response", "") or ""),
                "finish_reason": str(getattr(result, "finish_reason", "") or ""),
            }
        )
        return 0
    except Exception as exc:
        _emit({"type": "error", "text": f"{type(exc).__name__}: {exc}"[:2000]})
        return 1
    finally:
        if harness is not None:
            try:
                harness.close()
            except Exception:
                pass
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)


def main() -> int:
    raw = sys.stdin.read()
    # From here on, anything the SDK prints lands on stderr: stdout carries
    # events, and a stray print would be an unparseable line to the parent.
    sys.stdout = sys.stderr
    try:
        job = json.loads(raw or "{}")
        if not isinstance(job, dict):
            raise ValueError("job must be a JSON object")
    except Exception as exc:
        _emit({"type": "error", "text": f"bad job on stdin: {exc}"})
        return 1
    return _run_turn(job)


if __name__ == "__main__":
    raise SystemExit(main())
