"""Hand one chat turn to MiniMax Code, across a process boundary.

WHY A SUBPROCESS. mcode is a Node CLI with no library API. `exec` is its only
headless entry point, and with `--output-format stream-json` it writes one JSON
object per line. Rigma spawns it, reads that stream and translates it, so no
part of mcode's wire format reaches the chat loop.

WHAT WAS MEASURED, not read. Everything below was observed on 2026-09-21 by
running mcode 0.5.1 against `tests/fake_oai_server.py` — the same fake engine the
DSH adapter is proven with — because the shipped code is a minified bundle and a
contract read out of it is a guess until a real turn agrees.

RE-VERIFIED ON 0.5.4 (2026-09-25, `tools/mcode_probe.py`). The event stream,
`provider add --name/--use`, and cross-process `--session` resume were all
re-measured against the installed 0.5.4 CLI and a fake engine, and all three
still hold. Two things changed under us and are worth knowing:

  * `provider add` gained a REQUIRED `--name`, which this adapter already passed
    (it was added for the dedupe problem below), so 0.5.4 needed no change there.
  * mcode now keeps its provider list, sessions and state in a SQLite database
    under `<dataDir>/v2/sqlite/`, not in a JSON file. `provider list --json` is
    therefore still the ONLY honest way to ask what is configured — which is
    what `_providers` already does, and why the marker file is a hint and never
    an authority.
  * The first run in a fresh `MINIMAX_DATA_DIR` is now EXPENSIVE: mcode unpacks
    its builtin skills and agents and writes a ~4.9 MB model catalogue before it
    will answer. That happens inside `ensure_provider`'s timeout, so a cold data
    dir is a slow first turn, not a hang.
  * mcode 0.5.4 sends its ~12 KB instruction block as a `developer`-role
    message. Measured against llama-server b9867 with the prompt cache off, a
    `developer` message costs exactly the same prompt tokens as a `system` one
    (46 vs 46; `assistant` is 68), and is accepted without a warning — llama.cpp
    maps it onto the system role. So the byte passthrough in front of it does
    NOT need to rewrite the role, and must not.

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
import sys
import threading
from collections import deque
from collections.abc import Iterator
from pathlib import Path

# The mcode release this adapter was measured against. Every fact in the module
# docstring — the provider mechanics, the exit codes, the event vocabulary, the
# `--permission` mapping — came from running THIS build, so the number is a fact
# about the CODE and belongs next to it, not in a config a user could edit into
# a lie. `harness.conformance` compares it with what is installed.
#
# It is defined BEFORE the `.harness` imports on purpose: `harness.py` builds its
# BACKENDS table at import time and reads this value through `_verified_of`, so
# when something imports THIS module first the circular import would otherwise
# read a half-initialised module and cache "" as "nobody verified this".
VERIFIED = "0.5.4"

from .harness import TurnEvent          # noqa: E402
from . import harness as _harness       # noqa: E402

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
# A version check is a menu line and a health check, not a turn.
_VERSION_TIMEOUT = 20.0
# (base_url, model, context, output) -> the provider id confirmed IN THIS
# PROCESS. The marker file survives a restart and can therefore be wrong about a
# config that changed underneath it; this cannot, so the ask happens once per
# process rather than once per turn.
_VERIFIED: dict = {}
# How long past mcode's own --timeout before Rigma stops waiting for it. mcode
# bounds the RUN; this bounds the PROCESS, so a wedged child cannot outlive the
# turn it belongs to.
_KILL_GRACE = 30.0
# Whether this process has already told the user its mcode is not the build the
# adapter was measured against. `harness.conformance` can answer that, but it
# only runs from `rigma harness` and the menu's `?check=1` — nothing on the TURN
# path looked, so installing a new mcode and chatting produced a normal-looking
# turn on a build nobody measured. A renamed item type drops tool calls from the
# transcript while the reply still arrives, which is the quiet degradation
# `VERIFIED` exists to make sayable. One subprocess per process, one line in the
# transcript.
_DRIFT_SAID = False
# One JSON event is one line, and mcode puts tool results inside those lines.
# `for line in proc.stdout` held one whole line in memory before parsing and
# json.loads then made a second copy, in the process that also holds every chat
# session. The MCP client got this bound after AUDIT F55; mcode's stdout and
# stderr need the same one (09-4).
_FRAME_MAX = 4_000_000
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


def backend_version(exe: str | None = None) -> str:
    """What the installed CLI says it is, or "" when it will not say.

    Never raises. This runs from a menu and from a health check, and a backend
    that cannot answer its own version is a fact to REPORT — not an exception
    that would take down the thing doing the reporting.
    """
    exe = exe or bin_path()
    if not exe:
        return ""
    try:
        code, out = _run_out([exe, "--version"], _VERSION_TIMEOUT)
    except Exception:
        return ""
    if code != 0:
        return ""
    lines = [ln.strip() for ln in (out or "").splitlines() if ln.strip()]
    return lines[0] if lines else ""


def _drift_notice(exe: str) -> TurnEvent | None:
    """One line, once per process, when the installed build is not `VERIFIED`.

    `drift` is True/False/None for the same reason `harness.conformance` says
    so: a backend that will not answer its version is UNKNOWN, and unknown must
    not be reported as agreement OR as drift. Never raises — this runs inside a
    turn, and a version check must not be able to fail one.
    """
    global _DRIFT_SAID
    if _DRIFT_SAID or not VERIFIED:
        return None
    _DRIFT_SAID = True          # once per process, even if the probe fails
    try:
        have = str(backend_version(exe) or "").strip()
    except Exception:
        return None
    if not have or have == VERIFIED.strip():
        return None
    return TurnEvent("notice", text=(
        f"mcode {have} is not the build this adapter was measured against "
        f"({VERIFIED}). Tool calls can go missing from the transcript while "
        f"the reply still arrives — run `rigma harness` for the detail."))


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


# AUDIT 13-6: mcode is a third-party autonomous agent with its own shell and
# network tools, so `{**os.environ}` handed it HF_TOKEN, GEMINI_API_KEY,
# TAVILY_API_KEY and every other secret the owner had exported — readable by
# `printenv` and by anything mcode spawns. Only what a CLI needs to START and
# find its own caches is passed. A setup that genuinely needs one more name opts
# it in explicitly with RIGMA_HARNESS_ENV_PASSTHROUGH (comma-separated names; a
# trailing `*` passes a whole namespace), which keeps the old behaviour
# reachable without making it the default.
#
# R3-HARN-6: the list moved to `harness.HARNESS_ENV_ALLOWLIST` and this function
# became a thin wrapper. AUDIT 13-6 fixed mcode and left DSH copying the whole
# environment, so the identical leak survived in the sibling adapter for as long
# as the two lists were separate. One list, one function, no second copy to
# forget.
_ENV_ALLOWLIST = _harness.HARNESS_ENV_ALLOWLIST


def _env() -> dict:
    env = _harness.harness_env()
    env["MINIMAX_DATA_DIR"] = str(data_home())
    env[API_KEY_ENV] = _API_KEY
    return env


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


def _mcp_spec(cwd: str) -> dict:
    """How mcode should launch Rigma's MCP server.

    `sys.executable`, never a bare `python`: the server imports Rigma, and the
    interpreter that has Rigma installed is the one already running this.
    """
    from .runtime import rigma_home

    env = {"RIGMA_HOME": str(rigma_home()),
           # Granted HERE, deliberately and visibly. The server is pessimistic by
           # default — `allow_code` is off unless the environment turns it on — so
           # a capability nobody granted is refused rather than assumed. The only
           # roster entry that needs it is `undo_last_change`.
           #
           # This widens which of RIGMA's tools may APPEAR; it does not widen what
           # the arm can do. The arm already writes files with its own tools under
           # its own permission model, and `_ROSTER` is still the real gate.
           "RIGMA_MCP_ALLOW_CODE": "1"}
    if cwd:
        # Passed, not guessed. A tool that silently operated on the wrong
        # directory would be worse than one that refused.
        env["RIGMA_MCP_WORKSPACE"] = str(cwd)
    return {"command": sys.executable,
            "args": ["-m", "rigma.mcp_server"],
            "env": env}


def _same_registration(have, spec) -> bool:
    """Is a stored `rigma` entry the one we would write?

    Compares the COMMAND and ARGS, which are the two things that identify which
    Rigma is being launched, and treats the workspace as free: it is the chat's
    directory and moves from turn to turn, so a mismatch there is normal rather
    than drift.

    `env` is compared on the keys that say WHICH rigma home and WHETHER code
    tools were granted, ignoring the workspace for the same reason. Anything
    else in the stored entry — a key a newer build adds — is left alone: this
    decides whether to rewrite, not what the entry should contain.

    Takes `object` on both sides on purpose. This reads a file a user can edit,
    and `"rigma": "python -m rigma.mcp_server"` is a plausible hand-written
    mistake; it must come back as "not the same" so the entry gets repaired,
    not raise inside a turn.
    """
    if not isinstance(have, dict) or not isinstance(spec, dict):
        return False
    if list(have.get("args") or []) != list(spec.get("args") or []):
        return False
    if str(have.get("command") or "") != str(spec.get("command") or ""):
        return False
    h_env = have.get("env") if isinstance(have.get("env"), dict) else {}
    s_env = spec.get("env") if isinstance(spec.get("env"), dict) else {}
    for key in ("RIGMA_HOME", "RIGMA_MCP_ALLOW_CODE"):
        if str(h_env.get(key) or "") != str(s_env.get(key) or ""):
            return False
    return True


def ensure_mcp(cwd: str = "") -> None:
    """Point the arm at Rigma's MCP server — or take the pointer away.

    This is how Rigma's own tools reach an external agent without touching a
    single message the agent sends to its model: MCP makes Rigma a tool
    PROVIDER, and the arm decides whether to load it and whether to call what it
    offers. Every other route to the same end — normalising roles, prepending a
    prompt, editing the roster, re-running repair on the arm's calls — means
    Rigma editing the arm's conversation.

    Registered ONLY when there is something to offer. An MCP server with an
    empty roster still costs a process launch on every turn, and paying that for
    nothing is worse than the tool appearing a turn later than it could. The
    check runs per turn and is one small file read.

    Written into the data dir Rigma already owns. NOT passed as `--config`,
    which REPLACES mcode's config and would drop the provider settings that make
    the turn work at all. Other servers in the file are preserved — the file is
    Rigma's to manage, not Rigma's to own exclusively.
    """
    path = data_home() / "mcp.json"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raw = {}
    except (OSError, ValueError):
        raw = {}
    servers = raw.get("mcpServers")
    if not isinstance(servers, dict):
        servers = {}

    try:
        # Ask what would ACTUALLY be offered, rather than inferring it from one
        # dependency. The gate used to be `is a RAG sidecar live`, which is only
        # one of four roster entries: with nothing indexed the arm also lost
        # `remember` and `recall`, and now `undo_last_change` — none of which need
        # documents. Two facts that must agree should be one fact.
        from . import mcp_server
        wanted = bool(mcp_server.offered(ws=cwd, code=True))
    except Exception:
        wanted = False

    if wanted:
        # An entry that is PRESENT BUT WRONG is the case that used to be
        # permanent. This only ever assigned when `wanted`, so a registration
        # written by an older build — or one pointing at an interpreter that has
        # since lost Rigma — was left exactly as it was, every turn, forever. The
        # arm then silently had no `remember`, `recall` or `undo_last_change`,
        # and nothing anywhere said so: the file looks configured.
        #
        # The fix is to compare, not to assume. `ensure_mcp` already runs every
        # turn and already reads this file, so the drift check is free.
        #
        # Compared WITHOUT `RIGMA_MCP_WORKSPACE`, which is the chat's workspace
        # and changes legitimately from turn to turn. A repair must not be
        # triggered by a value that is supposed to move.
        spec = _mcp_spec(cwd)
        have = servers.get("rigma")
        if isinstance(have, dict) and _same_registration(have, spec):
            spec = have          # keep the stored workspace; nothing to write
        servers["rigma"] = spec
    else:
        servers.pop("rigma", None)

    if servers:
        raw["mcpServers"] = servers
    else:
        raw.pop("mcpServers", None)

    if not raw and not path.exists():
        # Nothing to say and nowhere to say it. Creating an empty file here
        # would be a side effect with no purpose — and this runs every turn.
        return

    body = json.dumps(raw, indent=2)
    try:
        if path.read_text(encoding="utf-8") == body:
            return              # already exactly right; do not touch the mtime
    except OSError:
        pass
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    except OSError:
        pass            # a registration that cannot be written is not a failed turn


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


# mcode's own vocabulary for the state Rigma could not previously see. These
# are the tool names it uses for goals, todos and subagents; everything below
# reads the STRUCTURED payload of a call rather than flattening it to text.
_GOAL_TOOLS = ("update_goal", "get_goal", "create_goal")
_TODO_TOOLS = ("todowrite",)
_TASK_TOOLS = ("task", "task_append", "task_query", "task_output",
                "task_stop")


def _details(out) -> dict:
    """The structured `details` of a tool result, or {}.

    mcode wraps every tool result as `{tool_name, text, content, details}` and
    `_flatten` keeps only `content` — which is why the goal object, the todo
    array and every task id were being thrown away one line before they were
    read. They are all in `details`.
    """
    if not isinstance(out, dict):
        return {}
    d = out.get("details")
    return d if isinstance(d, dict) else {}


def _state_events(name: str, out) -> list[TurnEvent]:
    """The structured state a tool result carries, in the seam's vocabulary.

    Field names are mcode's own, deliberately NOT renamed. Rigma normalises
    them once, in the UI, where a shape it does not recognise can be shown
    rather than silently dropped — see `chat/goal.ts`. Renaming here would mean
    guessing at a schema this side does not own.
    """
    d = _details(out)
    if not d:
        return []

    if name in _GOAL_TOOLS:
        # `get_goal` with no goal answers `{"goal": null}`, and an error
        # answers `{"error": ...}`. Neither is a goal, and emitting an empty
        # one would blank a panel that was showing something real.
        goal = d.get("goal")
        if not isinstance(goal, dict):
            return []
        # `update_goal` wraps it as {proposal, goal, goalSnapshotPhase}; the
        # goal itself is the part that describes the work.
        return [TurnEvent(kind="state", event="goal", data=goal)]

    if name in _TODO_TOOLS:
        # The argument is the whole list and replaces the previous one; the
        # result echoes it under `details.todos`.
        todos = d.get("todos")
        if not isinstance(todos, list):
            return []
        return [TurnEvent(kind="state", event="todos", data={"todos": todos})]

    if name in _TASK_TOOLS:
        # mcode emits snake_case (`task_id`, `sub_session_id`, `sub_turn_id`);
        # the seam uses camelCase, which is also what the runtime's own
        # normaliser produces. Renaming here is safe because these three names
        # are read by mcode's own parser both ways.
        ids = {
            "taskId": d.get("task_id"),
            "subSessionId": d.get("sub_session_id"),
            "subTurnId": d.get("sub_turn_id"),
        }
        if not any(v for v in ids.values()):
            return []
        return [TurnEvent(kind="state", event="subagent", data={
            **{k: v for k, v in ids.items() if v},
            "name": d.get("agent_name") or d.get("resolved_agent_name"),
            "status": d.get("status"),
        })]

    return []


def _ok_of(call: dict) -> bool | None:
    """Whether the call succeeded, from the wire's own status.

    The adapter used to say this could not be derived: "mcode does not flag
    failure in the projected item — a tool that does not exist comes back as
    status 3 with the reason as TEXT — so `ok` cannot be derived here". The
    status IS on the item: the runtime normalises its numeric enum to
    `started` / `completed` / `failed`. Only `failed` is a failure, and
    anything unrecognised returns None, which the seam renders as UNKNOWN
    rather than as success — the old behaviour claimed success for every
    result including the failures.
    """
    status = str(call.get("status") or "").strip().lower()
    if status == "failed":
        return False
    if status == "completed":
        return True
    return None

def map_event(obj: dict, seen: dict) -> list[TurnEvent]:
    """Translate one mcode stream-json object into Rigma's events.

    Pure but for `seen`, the per-turn memo of item ids — see WHY THE MEMO above.
    An unknown `type` is ignored rather than guessed at: mcode's schema is
    versioned, and a future item kind must not become a wrong row in the
    transcript.
    """
    kind = obj.get("type")
    if kind == "turn.failed":
        # `error` is whatever mcode put there. `.get` on a string or a list
        # raised straight out of the read loop (09-R3-1), so the shape is
        # checked before it is used — the same guard the outer object gets.
        err = obj.get("error")
        msg = str(err.get("message") or "the mcode turn failed") \
            if isinstance(err, dict) else "the mcode turn failed"
        return [TurnEvent("error", msg)]
    if kind not in ("item.started", "item.updated", "item.completed"):
        return []
    item = obj.get("item")
    if not isinstance(item, dict):
        # A malformed nested object must cost one event, not the turn. The
        # outer object was already guarded; this is the level 09-R3-1 was at.
        return []
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
        call = item.get("toolCall")
        if not isinstance(call, dict):
            return []          # see above: one bad event, not a dead turn
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
            events.append(TurnEvent("tool_result", text=_flatten(out),
                                    name=name, ok=_ok_of(call)))
            # What the call DID, in structured form, for the tools whose
            # result carries state rather than prose. Emitted after the result
            # so the transcript reads call-then-outcome-then-state.
            events.extend(_state_events(name, out))
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


def _bounded_lines(stream):
    """Yield `(line, oversize)` from `stream` with a HARD per-line bound.

    `for line in stream` holds one whole line in memory before yielding it, and
    a JSON parse makes a second copy of it. An oversize frame is skipped with
    `oversize=True` after draining past its newline, so framing stays in sync;
    an unterminated one ends the stream rather than spinning the drain (09-4).
    """
    while True:
        line = stream.readline(_FRAME_MAX + 1)
        if not line:
            return
        if "\n" in line or len(line) <= _FRAME_MAX:
            yield line, False
            continue
        found = False
        for _ in range(4):
            tail = stream.readline(_FRAME_MAX + 1)
            if not tail:
                break
            if "\n" in tail:
                found = True
                break
        yield "", True
        if not found:
            return


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
    drift = _drift_notice(exe)
    if drift is not None:
        yield drift
    pid, why = ensure_provider(exe, base_url, model, context_window, max_tokens)
    if why:
        yield TurnEvent("error", why)
        return
    ensure_agents_md()
    ensure_mcp(cwd)

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
        The tail is only read if the turn dies without saying why. The deque
        bounds the COUNT; the line length is bounded here too (09-4)."""
        for ln, oversize in _bounded_lines(proc.stderr):
            if oversize:
                err_lines.append("(stderr line over the frame bound, skipped)")
                continue
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
    saw_end = False
    final_status = ""
    try:
        for line, oversize in _bounded_lines(proc.stdout):
            if oversize:
                yield TurnEvent("notice", text=(
                    f"mcode emitted a frame over {_FRAME_MAX} characters "
                    "and it was skipped"))
                continue
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
            if kind == "turn.completed":
                usage = obj.get("usage") or {}
                if state is not None:
                    state["usage"] = usage
                # Emitted as well as remembered. mcode has always reported
                # this and Rigma has always written it into the adapter state
                # and then read only `session_id` back out, so the turn's
                # token cost was collected and discarded — the one number a
                # reader of a long agent turn most wants. Same `state`/`usage`
                # vocabulary the DSH adapter uses, so one SSE event serves
                # both and the UI needs no second code path.
                if isinstance(usage, dict) and usage:
                    yield TurnEvent(kind="state", event="usage", data=usage)
                continue
            if kind == "exec.completed":
                saw_end = True
                result = obj.get("result")
                if not isinstance(result, dict):
                    result = {}
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
        return
    # An exit 0 with no `exec.completed` is not a finished turn. Without this the
    # generator ended silently and the caller saved an empty assistant reply as a
    # success — the DSH adapter has the equivalent `done` check (09-7).
    if not saw_end and not failed:
        tail = " / ".join(list(err_lines)[-4:])[:400]
        code = proc.returncode
        yield TurnEvent("error", text=(
            f"mcode exited {code if code is not None else '?'} without "
            "completing the turn"
            + (f": {tail}" if tail else "")))
