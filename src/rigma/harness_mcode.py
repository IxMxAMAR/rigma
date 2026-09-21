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
  * A SESSION SURVIVES THE PROCESS. Every event carries a `sessionId`, and
    passing it back as `--session <id>` makes the next run continue that
    conversation: verified 2026-09-21 across three separate processes, the
    second and third reported `session.resumed` with the same id, and the model
    was replayed 3, then 5, then 7 messages. This is the difference between an
    agent and a stateless prompt — mcode's plan, its subagents and its goals all
    live in the session, so a fresh session per turn throws away most of what it
    is good at. Rigma stores the id per backend and hands it back, which is why
    `drive_turn` takes a `state` dict.
  * What mcode sends is a `developer`-role message of ~10.5k characters (its own
    system prompt, headed `# Harness`, `# Core Judgment`, `# Communication &
    Delivery`, `# Environment`), 18 tool schemas, and `max_completion_tokens` /
    `reasoning_effort` rather than `max_tokens` / `temperature`. Rigma's /v1 is
    a byte passthrough, so none of that is rewritten on the way to the engine —
    which is the point. Anything that normalised roles here would silently
    delete the agent's entire instruction set.

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
from . import harness as _harness

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
# (base_url, model, context, output) -> the provider id confirmed IN THIS
# PROCESS. The marker file survives a restart and can therefore be wrong about a
# config that changed underneath it; this cannot, so the ask happens once per
# process rather than once per turn.
_VERIFIED: dict = {}
# How long past mcode's own --timeout before Rigma stops waiting for it. mcode
# bounds the RUN; this bounds the PROCESS, so a wedged child cannot outlive the
# turn it belongs to.
_KILL_GRACE = 30.0
# mcode's documented exit codes. The number alone sends a reader to look it up;
# the number AND the meaning makes a transcript self-explaining.
_EXIT_MEANING = {
    2: "the command was malformed",
    3: "mcode could not use its configuration",
    4: "the run failed",
    6: "the run timed out",
    7: "the run hit a runtime limit",
    70: "mcode hit an internal error",
    130: "the run was cancelled",
}
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


# Rigma's one channel into the agent, and it is the agent's OWN documented one:
# `<dataDir>/AGENTS.md` is the global instruction file, and Rigma owns the data
# dir. This is how an arm can be told about the world it is running in without
# anyone touching its prompt — mcode has NO append/override mechanism for that
# (no flag, no config key, no env var), so a wrapper that wants to contribute
# context has exactly two options: this file, or the user's own project.
#
# WHAT IT MAY SAY. Environment facts the agent cannot discover — above all that
# the model behind its provider is a local one tuned for this machine, not a
# frontier model. What it may NOT do is tell the agent how to work: mcode's
# instruction set is the thing worth borrowing, and adding opinions to it
# dilutes the very benchmark this backend was chosen for.
_AGENTS_MARKER = "<!-- rigma-owned: environment description -->"
_AGENTS_VERSION = 1
_AGENTS_MD = f"""\
{_AGENTS_MARKER}
<!-- version: {_AGENTS_VERSION} — Rigma rewrites this file only while this
     marker is present. Delete the marker, or the file, and Rigma leaves it
     alone from then on. -->

# The environment you are running in

Your provider `{PROVIDER}` is served by Rigma, which runs on the user's own
machine. The model behind it is a local LLM chosen and tuned for that machine's
hardware — not a frontier cloud model. Rigma owns the endpoint, the context
window and the KV policy; you own your tools, your plan and your session.

Two consequences worth having in mind:

- The model is smaller than the ones you may be calibrated for. Small,
  verifiable steps land better than long plans, and re-reading a file is
  cheaper than being wrong about what it says.
- Your workspace is the folder the user opened for this chat. Rigma renders
  your replies and your tool calls into the chat transcript as they happen, so
  the user is watching the work rather than reading a summary of it at the end.

Nothing else here is Rigma's. Your system prompt, your tool roster, your
session and your permission decisions are yours, and Rigma does not rewrite
them. This file is environment description and nothing more.
"""


def ensure_agents_md() -> None:
    """Install Rigma's environment note into the agent's own data dir.

    Written once. Rigma rewrites it only while its marker is still in the file,
    so a user who edits or deletes it is not fought — the file lives in a
    directory Rigma owns, but what it says about the agent's world is worth
    being able to correct by hand.
    """
    path = data_home() / "AGENTS.md"
    try:
        if path.exists():
            head = path.read_text(encoding="utf-8", errors="replace")[:200]
            if _AGENTS_MARKER not in head or f"version: {_AGENTS_VERSION}" in head:
                return
        path.write_text(_AGENTS_MD, encoding="utf-8")
    except OSError:
        pass            # a note that cannot be written is not a failed turn


def _run(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run a setup command to completion. Returns (code, combined output)."""
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=_env(), timeout=timeout,
                           stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return 124, f"timed out after {timeout:.0f}s"
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def _run_out(argv: list[str], timeout: float) -> tuple[int, str]:
    """Run a command whose output is PARSED, not reported: stdout only.

    mcode writes results to stdout and diagnostics to stderr on purpose, so
    mixing them would put a warning into the middle of the JSON.
    """
    try:
        p = subprocess.run(argv, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", env=_env(), timeout=timeout,
                           stdin=subprocess.DEVNULL, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return 124, ""
    return p.returncode, p.stdout or ""


def _providers(exe: str) -> list[dict] | None:
    """mcode's own provider list, or None when mcode would not answer.

    Rigma keeps a marker file so it does not run `provider add` every turn, but
    a marker is RIGMA'S BELIEF about mcode's config, not mcode's config — and
    the two drift. A data dir cleared by hand, a port change, or an upstream
    release that moves where custom providers live all leave Rigma confidently
    skipping a setup step that is no longer done. Asking mcode is the only way
    to know, and `provider list --json` is the asking.
    """
    code, out = _run_out([exe, "provider", "list", "--json"], _SETUP_TIMEOUT)
    if code != 0:
        return None
    start = out.find("{")
    if start < 0:
        return None
    try:
        data = json.loads(out[start:])
    except ValueError:
        return None
    return [p for p in (data.get("providers") or []) if isinstance(p, dict)]


def _ours(provs: list[dict]) -> list[dict]:
    """The providers RIGMA created, by the namespace it asks for."""
    return [p for p in provs
            if str(p.get("providerId") or "") == PROVIDER_ID
            or str(p.get("providerId") or "").startswith(PROVIDER_ID + "-")]


def _active_at(provs: list[dict], base_url: str) -> str:
    """The provider id mcode has ACTIVE at this URL — what `exec --model` must
    name.

    Read back rather than assumed, because `provider add` DEDUPES a name it
    already holds: a second `--name rigma` becomes `rigma-2`. `--use` then
    activates the NEW one while `--model custom_provider:rigma/<model>` keeps
    resolving to the OLD one — a provider Rigma believes it configured and a
    turn that fails against a dead port. Measured 2026-09-21.
    """
    for p in provs:
        if (p.get("active") and p.get("enabled")
                and str(p.get("baseUrl") or "").rstrip("/")
                == base_url.rstrip("/")):
            return str(p.get("providerId") or "")
    return ""


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
    separate Node process. But the marker is only a HINT — once per process,
    mcode is asked whether the thing it describes is still true, because an
    upstream release can move where custom providers live and a marker cannot
    notice.

    Returns the provider ID to name in `--model`, READ BACK from mcode rather
    than assumed. That is not belt-and-braces: `provider add` dedupes a name it
    already holds, so a second `--name rigma` becomes `rigma-2`, `--use`
    activates `rigma-2`, and a `--model custom_provider:rigma/…` — the id Rigma
    used to hardcode — resolves to the FIRST one and fails against whatever port
    that was. Reading the active id back makes the name mcode chose irrelevant.
    """
    want = _wanted(base_url, model, context_window, max_tokens)
    key = (base_url, model, int(context_window), int(max_tokens))
    marker = data_home() / _MARKER
    try:
        have = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        have = None

    if have == want and key in _VERIFIED:
        return _VERIFIED[key], ""

    provs = _providers(exe)
    if provs is None:
        # mcode would not answer. Do NOT re-add on a guess: a transient failure
        # would become a duplicate provider, and the dedupe would then point
        # `--use` at a provider that `--model` does not name.
        if have == want:
            _VERIFIED[key] = PROVIDER_ID
            return PROVIDER_ID, ""
        return "", "mcode would not list its providers"

    live = _active_at(provs, base_url)
    if have == want and live:
        _VERIFIED[key] = live
        return live, ""

    # Clear Rigma's own leftovers before adding. `provider add` DEDUPES a name
    # it already has, and `provider remove` REFUSES without `--yes` — which is
    # how the leftovers accumulated in the first place, since an earlier version
    # of this called remove without it and the removal silently did nothing.
    # Left alone, each change leaks a provider and points `--use` one step
    # further from the id `--model` names.
    for p in _ours(provs):
        _run([exe, "provider", "remove", str(p.get("providerId") or ""),
              "--yes"], _SETUP_TIMEOUT)

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
        return "", f"mcode provider add failed ({code}): {out.strip()[:400]}"

    live = _active_at(_providers(exe) or [], base_url)
    if not live:
        # Saved but not active, or active somewhere else. Either way a turn
        # would run against a provider Rigma did not configure, so say so here
        # rather than letting it surface as a model-not-available error.
        return "", (f"mcode saved a provider but reports nothing active at "
                    f"{base_url}")
    _VERIFIED[key] = live
    try:
        marker.write_text(json.dumps(want), encoding="utf-8")
    except OSError:
        pass                        # a cache that cannot be written is not fatal
    return live, ""


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


def _remember(state: dict | None, obj: dict, *, resumed: bool) -> None:
    """Keep the backend's own handle on this conversation where Rigma can find
    it.

    Written on EVERY session event rather than only the first: a resumed run
    reports the id it was given, and recording it again is how a backend that
    renumbers a session still leaves the right one behind.
    """
    if state is None:
        return
    sid = str(obj.get("sessionId") or "")
    if sid:
        state["session_id"] = sid
    state["resumed"] = resumed


def drive_turn(*, base_url: str, model: str, prompt: str,
               system_prompt: str = "", session_id: str = "", cwd: str = "",
               max_tokens: int = 4096, context_window: int = 32768,
               timeout: float = 1800.0, state: dict | None = None,
               cancel: threading.Event | None = None, permission: str = "full"
               ) -> Iterator[TurnEvent]:
    """Run one mcode turn and yield what happened, in Rigma's vocabulary.

    BLOCKING, and the caller owns the thread — the seam's contract, so a stalled
    mcode stalls a worker rather than the event loop that is streaming a chat.

    `state`, when given, is read for the mcode session to CONTINUE and written
    with the session this turn ran in. See the module docstring: continuity is
    the whole reason to hand a turn to an agent rather than a prompt.

    `system_prompt` is accepted and IGNORED. mcode owns its own system prompt —
    10.5k characters of it, measured — and that is the trade the menu states.
    Quietly pasting Rigma's into the user message would make the transcript
    describe a turn that did not happen, and replacing the agent's instructions
    with Rigma's would throw away the thing worth having.

    `cancel` kills the child. A turn here can run for minutes and spawn
    subagents of its own, so a stop that only takes effect at the next output
    line would not be a stop — mcode goes quiet for long stretches while it
    thinks, and that is the stretch a user wants to interrupt. Killing the
    process also stops its children, which is the part that matters when the
    thing being interrupted is a fleet of subagents.

    Stopping is reported as a NOTICE, not an error: nothing failed. The session
    id was already recorded when the turn started, so the next turn continues
    this conversation rather than starting it over — an interrupt should cost
    the turn, not the thread.
    """
    exe = bin_path()
    if exe is None:
        yield TurnEvent("error", "mcode is not on PATH")
        return
    pid, why = ensure_provider(exe, base_url, model, context_window, max_tokens)
    if why:
        yield TurnEvent("error", why)
        return
    ensure_agents_md()

    resume = str((state or {}).get("session_id") or "").strip()
    # Straight through from the chat's own setting. NOT validated against a
    # list here: the session field is validated at the write, and a second
    # opinion in the adapter could only disagree with it. An unknown value is
    # mcode's to refuse, and it does, by name.
    #
    # `full` is not "no safety" and must not be described as such — a hard,
    # workspace-scoped policy bounds recursive writes and deletes under every
    # mode, measured. See docs/mcode-permission-modes.md.
    argv = [exe, "exec", "--output-format", "stream-json",
            "--permission", str(permission or "full"),
            # the id mcode actually gave us, not the one we asked for
            "--model", f"{pid}/{model}",
            "--timeout", f"{int(timeout)}s"]
    if resume:
        argv += ["--session", resume]
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
    stopped = threading.Event()

    def _stop() -> None:
        killed.set()
        # The TREE, not the shim: mcode is started through a `.cmd`, so the pid
        # Rigma holds is a shell. Killing the shell alone leaves mcode running
        # with the stdout pipe open, and the read loop below would never end.
        _harness.kill_tree(proc)

    def _watch_cancel() -> None:
        """Kill the child the moment the owner asks, not at the next line.

        Waiting on the event rather than checking it inside the read loop is the
        whole point: mcode says nothing while it is thinking, and a silent turn
        is exactly the one worth stopping. The poll alongside it is only so this
        thread ends once the child is gone instead of blocking forever on an
        event nobody will set.
        """
        while not cancel.wait(0.25):
            if proc.poll() is not None:
                return
        stopped.set()
        _stop()

    watchdog = threading.Timer(timeout + _KILL_GRACE, _stop)
    watchdog.daemon = True
    watchdog.start()
    if cancel is not None:
        threading.Thread(target=_watch_cancel, daemon=True).start()

    seen: dict = {}
    failed = False
    said = False
    final_status = ""
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
            kind = obj.get("type")
            if kind in ("session.started", "session.resumed"):
                resumed = kind == "session.resumed"
                _remember(state, obj, resumed=resumed)
                if resumed:
                    # Say it out loud. Continuity the reader cannot see is
                    # indistinguishable from a backend that forgot everything.
                    yield TurnEvent(
                        "notice",
                        text="continuing this chat's MiniMax Code session")
                continue
            if kind == "turn.completed" and state is not None:
                state["usage"] = obj.get("usage") or {}
                continue
            if kind == "exec.completed":
                result = obj.get("result") or {}
                final_status = str(result.get("status") or "")
                if state is not None:
                    state["status"] = final_status
                # The final answer, for a model that never streamed one. mcode
                # reports it either way, and a turn that produced a reply must
                # not come back with an empty transcript.
                out = result.get("output")
                if out and not said:
                    said = True
                    yield TurnEvent("text", str(out))
                continue
            for ev in map_event(obj, seen):
                if ev.kind == "error":
                    failed = True
                elif ev.kind == "text":
                    said = True
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

    # Checked before the timeout branch, because both end in a killed process
    # and only this one is not a failure. Whatever the agent already said stays
    # in the transcript: it was said, and hiding it would make the next turn
    # unintelligible.
    if stopped.is_set():
        yield TurnEvent("notice", text="stopped")
        return
    if killed.is_set():
        yield TurnEvent("error", f"mcode did not finish within "
                                 f"{int(timeout + _KILL_GRACE)}s and was stopped")
        return
    if proc.returncode not in (0, None) and not failed:
        tail = " / ".join(list(err_lines)[-4:])[:400]
        why = _EXIT_MEANING.get(proc.returncode, "")
        yield TurnEvent("error",
                        f"mcode exited {proc.returncode}"
                        + (f" ({why})" if why else "")
                        + (f": {tail}" if tail else ""))
        return
    # A run can end unsuccessfully while the process still exits 0: a step
    # limit or a cancellation is reported in the RESULT, not in the code. The
    # documented advice is to check both, so both are checked.
    if final_status and final_status != "succeeded" and not failed:
        yield TurnEvent("error", f"the run ended as {final_status}")
