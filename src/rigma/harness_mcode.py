"""Hand one chat turn to MiniMax Code, across a process boundary.

WHY A SUBPROCESS. mcode is a Node CLI with no library API. `exec` is its only
headless entry point, and with `--output-format stream-json` it writes one JSON
object per line. Rigma spawns it, reads that stream and translates it, so no
part of mcode's wire format reaches the chat loop.

WHAT WAS MEASURED, not read. Everything below was observed on 2026-09-21 by
running mcode 0.5.1 against `tests/fake_oai_server.py` — the same fake engine the
DSH adapter is proven with — because the shipped code is a minified bundle and a
contract read out of it is a guess until a real turn agrees:

  * `mcode exec --output-format stream-json --permission full --model
    custom_provider:<name>/<model> "<prompt>"` runs exactly one turn. The model
    reference MUST carry the `custom_provider:` prefix: `--model <name>/<model>`
    is refused with "not available for the configured_provider route".
  * `openai-completions` is a real `--api-format`, and with it mcode POSTs
    `{base_url}/chat/completions` carrying `Authorization: Bearer <the value of
    the --api-key-env variable>`, `stream: true` and 18 tool schemas. That is
    Rigma's own /v1 — the model stays Rigma's.
  * mcode also probes `/v1/responses/input_tokens` before each turn. llama-server
    does not serve it, and a 404 is TOLERATED: verified, the turn completed with
    only the chat-completions request logged. Rigma's proxy needs no new route.
  * A CUSTOM PROVIDER HAS TO BE **SELECTED**, NOT JUST ADDED — and once it is,
    no MiniMax account is involved at all. This was got wrong first, so it is
    written down carefully. Adding a provider without `--use` saves it and
    leaves NO provider active; `exec` then dies with `auth.login_required`,
    "Sign in to MiniMax to use Agent features". That message is about the
    account status of the ACTIVE provider, not about the model — it fires even
    when `--model` names the custom provider explicitly, which is what made it
    look like a hard prerequisite. With `--use` (which tests the endpoint, then
    saves AND selects) the same command runs the turn against the LOCAL model
    with no MiniMax credential anywhere on the machine. Verified 2026-09-21
    both ways, and the upstream README documents this: "BYOK does not require a
    MiniMax login."
  * The cost of that is one throwaway model call: `--use` tests the first model
    before saving, and "a failed connection test saves nothing" — so the
    failure mode is a provider that is not configured, which is a far clearer
    thing to report than an account error.

THE EVENT STREAM. Each line is `{schemaVersion, sequence, timestampMs, runId,
sessionId, turnId, type, ...}`. The types are `exec.started`,
`session.started|session.resumed`, `turn.started`, `item.started|item.updated|
item.completed`, `turn.completed|turn.failed` and `exec.completed`. An `item`
carries `type` in `agent_message` (with `contentDelta`, then `content`),
`reasoning` (the same pair) or `tool_call` (with a `toolCall` object).

WHY THE MEMO. mcode re-emits the SAME item as it progresses — a tool call arrives
as `item.started`, then several `item.updated`, then `item.completed`, all with
one id. Without a per-turn memo a single tool call would be announced four times
and its result twice, which is the wrong-row bug class this project has already
paid for once.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path

from .harness import TurnEvent

# mcode's own id for a provider Rigma adds. The `custom_provider:` prefix is
# part of the id it prints, and the model reference has to spell it out.
PROVIDER = "rigma"
PROVIDER_ID = f"custom_provider:{PROVIDER}"
# The env var mcode reads the key from. Rigma's /v1 does not authenticate —
# llama-server ignores the header — so this is a value that has to EXIST, not
# one that has to be secret.
API_KEY_ENV = "RIGMA_MCODE_KEY"
_API_KEY = "local"

_MARKER = "provider.json"
_SETUP_TIMEOUT = 120.0
# How long past mcode's own --timeout before Rigma stops waiting for it. mcode
# bounds the RUN; this bounds the PROCESS, so a wedged child cannot outlive the
# turn it belongs to.
_KILL_GRACE = 30.0
# Tool-call item status codes seen: 4 on first sight, then 5, then 1 while the
# arguments stream in, then 3 with the result. The numbers are not documented
# and the ORDER was not stable enough to key on, so `output` — which only ever
# appears with the result — is what says "done".
_NO_WINDOW = 0x08000000 if os.name == "nt" else 0


def bin_path() -> str | None:
    """The mcode executable, or None. A CLI, unlike DSH's source checkout."""
    raw = (os.environ.get("RIGMA_MCODE_BIN") or "").strip()
    if raw:
        return raw if Path(raw).exists() else None
    return shutil.which("mcode")


def available() -> bool:
    """Whether a turn can be handed over at all."""
    return bin_path() is not None


def data_home() -> Path:
    """Where mcode keeps its config, its provider list and its sessions.

    Rigma-owned and pointed at with `MINIMAX_DATA_DIR`, so a turn cannot read or
    write the owner's own mcode setup — and so two Rigma installs cannot fight
    over one provider list.
    """
    from . import runtime
    path = runtime.rigma_home() / "mcode"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _env() -> dict:
    return {**os.environ, "MINIMAX_DATA_DIR": str(data_home()),
            API_KEY_ENV: _API_KEY}


def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run a setup command to completion. Returns (code, combined output)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=_env(), timeout=timeout,
                           stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _wanted(base_url: str, model: str, context_window: int,
            max_tokens: int) -> dict:
    return {"base_url": base_url, "model": model,
            "context_limit": int(context_window), "output_limit": int(max_tokens)}


def ensure_provider(exe: str, base_url: str, model: str, context_window: int,
                    max_tokens: int) -> str:
    """Point mcode at Rigma's /v1, once. Returns "" or why it could not.

    `--use` is load-bearing, not a convenience. It is what makes the custom
    provider the ACTIVE one, and mcode's account gate reads the active
    provider's status: a provider that is merely added leaves `exec` refusing
    every turn with "Sign in to MiniMax", even though the model is Rigma's and
    no MiniMax service is involved. `--use` also tests the endpoint first and
    saves NOTHING if that test fails, so a failure here is reported as what it
    is — the provider could not be configured — instead of surfacing later as
    an account error.

    Cached in a marker file rather than re-run every turn: `provider add` is a
    separate Node process, and re-adding a name mcode already has is an error,
    not an update. A changed model means the cached provider is stale, so it is
    removed first — mcode keys a custom provider by name, and the name is ours.
    """
    want = _wanted(base_url, model, context_window, max_tokens)
    marker = data_home() / _MARKER
    try:
        have = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        have = None
    if have == want:
        return ""
    if have is not None:
        _run([exe, "provider", "remove", PROVIDER_ID], _SETUP_TIMEOUT)
    code, out = _run([
        exe, "provider", "add",
        "--name", PROVIDER,
        "--base-url", base_url,
        "--api-format", "openai-completions",
        "--model", model,
        "--api-key-env", API_KEY_ENV,
        "--context-limit", str(int(context_window)),
        "--output-limit", str(int(max_tokens)),
        # see the docstring: this is what clears mcode's account gate
        "--use",
    ], _SETUP_TIMEOUT)
    if code != 0:
        return f"mcode provider add failed ({code}): {out.strip()[:400]}"
    try:
        marker.write_text(json.dumps(want), encoding="utf-8")
    except OSError:
        pass                        # a cache that cannot be written is not fatal
    return ""


def _flatten(out) -> str:
    """A tool result as text. mcode wraps it as `{content:[{type,text}],details}`."""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        parts = out.get("content")
        if isinstance(parts, list):
            text = "\n".join(str(p.get("text", "")) for p in parts
                             if isinstance(p, dict) and p.get("text"))
            if text:
                return text
        return json.dumps(out)[:4000]
    return str(out)[:4000]


def map_event(obj: dict, seen: dict) -> list[TurnEvent]:
    """Translate one mcode stream-json object into Rigma's events.

    Pure but for `seen`, the per-turn memo of item ids — see WHY THE MEMO above.
    An unknown `type` is ignored rather than guessed at: mcode's schema is
    versioned, and a future item kind must not become a wrong row in the
    transcript.
    """
    kind = obj.get("type")
    if kind == "turn.failed":
        err = obj.get("error") or {}
        msg = str(err.get("message") or "the mcode turn failed")
        return [TurnEvent("error", msg)]
    if kind not in ("item.started", "item.updated", "item.completed"):
        return []
    item = obj.get("item") or {}
    iid = str(item.get("id") or "")
    itype = item.get("type")

    if itype in ("agent_message", "reasoning"):
        what = "text" if itype == "agent_message" else "thinking"
        mark = f"{iid}:streamed"
        delta = item.get("contentDelta")
        if delta:
            seen[mark] = True
            return [TurnEvent(what, str(delta))]
        full = item.get("content")
        # A message that never streamed arrives whole on `item.completed`.
        # Emitting both would double the reply.
        if full and not seen.get(mark):
            seen[mark] = True
            return [TurnEvent(what, str(full))]
        return []

    if itype == "tool_call":
        call = item.get("toolCall") or {}
        name = str(call.get("name") or "?")
        args = call.get("input")
        out = call.get("output")
        events: list[TurnEvent] = []
        # mcode announces a call BEFORE its arguments finish streaming, so the
        # first sighting carries no `input`. Announcing then would put an empty
        # argument box in the transcript for every single tool call, so the call
        # is HELD until its arguments exist — or until a result arrives without
        # them, in which case it is emitted just before the result rather than
        # dropped.
        if not seen.get(f"{iid}:call") and (args is not None or out is not None):
            seen[f"{iid}:call"] = True
            events.append(TurnEvent("tool", name=name,
                                    args=args if isinstance(args, dict) else {}))
        if out is not None and not seen.get(f"{iid}:result"):
            seen[f"{iid}:result"] = True
            # mcode does not flag failure in the projected item — a tool that
            # does not exist comes back as status 3 with the reason as TEXT —
            # so `ok` cannot be derived here and the text is what says it.
            events.append(TurnEvent("tool_result", text=_flatten(out), name=name))
        return events
    return []


def drive_turn(*, base_url: str, model: str, prompt: str,
               system_prompt: str = "", session_id: str = "", cwd: str = "",
               max_tokens: int = 4096, context_window: int = 32768,
               timeout: float = 1800.0) -> Iterator[TurnEvent]:
    """Run one mcode turn and yield what happened, in Rigma's vocabulary.

    BLOCKING, and the caller owns the thread — the seam's contract, so a stalled
    mcode stalls a worker rather than the event loop that is streaming a chat.

    `system_prompt` is accepted and IGNORED. mcode owns its own system prompt;
    that is the trade the menu states, and quietly pasting Rigma's into the user
    message would make the transcript describe a turn that did not happen.
    """
    exe = bin_path()
    if exe is None:
        yield TurnEvent("error", "mcode is not on PATH")
        return
    why = ensure_provider(exe, base_url, model, context_window, max_tokens)
    if why:
        yield TurnEvent("error", why)
        return

    argv = [exe, "exec", "--output-format", "stream-json",
            # headless: `smart` needs a TUI to ask, and `off` would disarm the
            # agent's tools entirely
            "--permission", "full",
            "--model", f"{PROVIDER_ID}/{model}",
            "--timeout", f"{int(timeout)}s"]
    if cwd and Path(cwd).is_dir():
        argv += ["--cwd", cwd]
    argv.append(prompt)

    try:
        proc = subprocess.Popen(
            argv, env=_env(), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
            creationflags=_NO_WINDOW)
    except OSError as e:
        yield TurnEvent("error", f"could not start mcode: {e}")
        return

    err_lines: deque = deque(maxlen=40)

    def _drain() -> None:
        """Keep mcode's stderr from filling its pipe and blocking the child.
        The tail is only read if the turn dies without saying why."""
        for ln in proc.stderr:
            err_lines.append(ln.rstrip())

    drain = threading.Thread(target=_drain, daemon=True)
    drain.start()

    killed = threading.Event()

    def _stop() -> None:
        killed.set()
        try:
            proc.kill()
        except OSError:
            pass

    watchdog = threading.Timer(timeout + _KILL_GRACE, _stop)
    watchdog.daemon = True
    watchdog.start()

    seen: dict = {}
    failed = False
    try:
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue            # a partial or non-JSON line is a skipped line
            if not isinstance(obj, dict):
                continue
            for ev in map_event(obj, seen):
                if ev.kind == "error":
                    failed = True
                yield ev
        proc.wait(timeout=10)
    except Exception as e:          # pragma: no cover - defensive
        yield TurnEvent("error", f"mcode stream failed: {e}")
        return
    finally:
        watchdog.cancel()
        if proc.poll() is None:
            _stop()
        try:
            proc.stdout.close()
        except OSError:
            pass

    if killed.is_set():
        yield TurnEvent("error", f"mcode did not finish within "
                                 f"{int(timeout + _KILL_GRACE)}s and was stopped")
        return
    if proc.returncode not in (0, None) and not failed:
        tail = " / ".join(list(err_lines)[-4:])[:400]
        yield TurnEvent("error", f"mcode exited {proc.returncode}: {tail}")
