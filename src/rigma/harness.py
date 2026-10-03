"""Which agent harness runs a session's turns.

WHY A SEAM, AND NOT A REWRITE.

Rigma's own loop is the default and stays the default. It is not a generic agent
loop; it is a loop that compensates for a weak local model — a rescue parser for
tool calls that arrive as prose, JSON argument repair, fuzzy filename recovery,
spill-to-disk for oversized results, artifact verification before it will
believe "done", and one action per turn. An external harness inherits none of
that, and its default composition is sized for a 256K frontier model: DSH's
`web` profile spends roughly 8,500-11,000 tokens on prompt plus tool schemas
before the first turn, against ~3,200 for a focused Rigma Run on a 32K window
(measured 2026-09-20).

So an external harness is an ALTERNATIVE backend behind one interface, never a
replacement. This module is that interface: what a backend is called, how it is
driven, how it is pointed at Rigma's model server, and whether it is installed.

THE MODEL STAYS RIGMA'S. Every backend is pointed at Rigma's own
OpenAI-compatible endpoint, so the tuning work — the combo, the KV policy, the
context size that fits this machine — is unchanged. Only the agent on top of it
differs. That is the whole point of the seam.

WHAT THIS MODULE REFUSES TO DO is pretend. A backend that is known but not yet
implemented raises `HarnessError` naming what is missing, rather than silently
falling back to the native loop. A session that asked for DSH and quietly got
the native loop would be a lie the user cannot see from the output.
"""
from __future__ import annotations

import importlib.util
import os
import shutil
from dataclasses import dataclass
from typing import Callable

NATIVE = "native"
DSH = "dsh"
MCODE = "mcode"


class HarnessError(ValueError):
    """A harness cannot be used: unknown, or present with no turn to hand it.

    The second case is one of `installed`/`runnable` disagreeing — the backend is
    on this machine but has no adapter — which is what `pending` explains. No
    backend Rigma ships is in that state, so the message is written to describe
    the situation rather than to name a backend that is in it.
    """


@dataclass
class TurnEvent:
    """One thing that happened during a turn, in RIGMA's vocabulary.

    The seam's shared language. A driver translates its backend's wire format
    into these and the turn loop renders them, so neither side has to know the
    other's protocol — which is what lets a second backend be added without
    touching the loop that streams a chat.
    """

    kind: str  # "text" | "thinking" | "tool" | "tool_result" | "notice" | "error"
    text: str = ""
    name: str = ""
    args: dict | None = None
    ok: bool = True
    # The backend's OWN event name and payload, when it has one worth keeping.
    #
    # Added for the DSH capability bridge. A goal change, a todo list, a plan
    # mode switch, a subagent's lifecycle and a token count are all STRUCTURED
    # facts with no home in `text` — squashing them into a string would mean
    # re-parsing our own prose in the UI, which is how a field quietly becomes
    # a sentence and stops being data. `event` names what it is (DSH's own
    # session-event type, e.g. "goal/change"), `data` is its payload verbatim.
    #
    # Both default empty, so every existing adapter and every existing
    # construction of a TurnEvent keeps working unchanged.
    event: str = ""
    data: dict | None = None


@dataclass(frozen=True)
class Harness:
    """One agent backend.

    `runnable` is whether a turn can actually be handed to it TODAY, and it is
    deliberately separate from `installed`. The two are equal for every backend
    Rigma ships — all three are wired — but they answer different questions, and
    conflating them is how a probe becomes a silent fallback: something can be
    present on this machine with no adapter written for it yet, and a menu that
    read `installed` as "usable" would offer a turn that cannot be handed over.

    (This docstring used DSH as its example of "probeable but not wired up". DSH
    is wired now, so the example was teaching the reverse of the field's meaning.)
    """

    name: str
    label: str
    # how a turn is handed over, in one line a user can act on
    drives: str
    runnable: bool
    # the module or executable that has to be present
    needs: str
    # how it is pointed at Rigma's own /v1
    wire: str
    # what it does NOT get from Rigma — the honest cost of choosing it
    unsupported: tuple[str, ...] = ()
    # why it is not runnable yet, when it is not
    pending: str = ""
    # how to tell whether it is on this machine, when `needs` is not a module
    # or an executable name. A backend can be present as a source checkout (DSH
    # is), which neither `which` nor `find_spec` can see.
    probe: "Callable[[], bool] | None" = None
    # The build this adapter was measured against, from the adapter module's own
    # `VERIFIED`. Empty means UNKNOWN — never "fine". A backend that updates
    # underneath Rigma is the failure this seam is most exposed to, and nothing
    # in a turn would say so: an additive schema change is absorbed silently and
    # a RENAMED one degrades quietly, with tool calls simply ceasing to appear
    # while the reply still arrives. This is the field that makes the drift
    # sayable.
    verified: str = ""
    # Rigma's own loop, shipped with Rigma. It has no external build to drift
    # from and nothing to have been "measured against", so the unverified wording
    # is not merely unhelpful here — it is wrong, and it made the built-in look
    # like the riskiest option on the list. Named rather than inferred from
    # `verified == ""`, because empty means UNKNOWN for every other backend and
    # conflating the two would hide a real unknown.
    built_in: bool = False
    # R3-HARN-1: whether this backend actually APPLIES the chat's `permission`
    # setting. The field is part of the adapter contract, but a backend with no
    # such notion ignores it — DSH's confinement is its own bundle's business and
    # its sandbox is pinned. The value was accepted, stored, sent to the adapter
    # and dropped, while the UI rendered the "off — no tools at all" selector for
    # every non-native backend. So a user could arm a safety setting and have
    # nothing happen, with nothing on screen saying so. Declared here rather than
    # inferred, because "ignores it" and "honours it" are both legitimate
    # adapter designs and only the adapter knows which it is.
    honours_permission: bool = True
    # What Rigma ADDS to this backend, when it would otherwise be missing it.
    #
    # `unsupported` says what a backend does not get from Rigma; this says the
    # opposite, and both are needed to answer the question a user actually has,
    # which is "what can this thing do for me". It exists because DSH's minimal
    # profile ships almost no model-facing tools — no goals, subagents, todos,
    # skills, plan mode, filesystem access or compaction — and Rigma mounts them
    # with a Cordis patch. Without a line saying so, that is invisible: the
    # capability appears and nothing anywhere explains where it came from, so a
    # user cannot tell Rigma's additions from DSH's own.
    #
    # Declared by the adapter, because only the adapter knows what it mounts.
    capabilities: tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "drives": self.drives,
                "runnable": self.runnable, "installed": installed(self),
                "needs": self.needs, "wire": self.wire, "verified": self.verified,
                "built_in": self.built_in,
                "honours_permission": self.honours_permission,
                "unsupported": list(self.unsupported), "pending": self.pending,
                "capabilities": list(self.capabilities)}


# AUDIT 13-6 / R3-HARN-6: an external agent is a third-party process with its own
# shell and network tools, so `{**os.environ}` handed it HF_TOKEN,
# GEMINI_API_KEY, TAVILY_API_KEY, DEEPSEEK_API_KEY and every other secret the
# owner had exported — readable by `printenv` and by anything the agent spawns.
# Only what a CLI needs to START and find its own caches is passed. A setup that
# genuinely needs one more name opts it in explicitly with
# RIGMA_HARNESS_ENV_PASSTHROUGH (comma-separated names; a trailing `*` passes a
# whole namespace), which keeps the old behaviour reachable without making it
# the default.
#
# This lives here rather than in one adapter because it was fixed for mcode
# first and DSH kept the leak: the same mistake was made twice in two files, and
# a shared list is what stops the third adapter repeating it.
HARNESS_ENV_ALLOWLIST = (
    # launching a child process
    "PATH", "PATHEXT", "SystemRoot", "SystemDrive", "windir", "COMSPEC",
    "ComSpec",
    # temp + home, so the child's own caches and config resolve
    "TEMP", "TMP", "TMPDIR", "HOME", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "APPDATA", "LOCALAPPDATA", "PROGRAMDATA",
    # locale / terminal, so it does not render mojibake
    "LANG", "LC_ALL", "LC_CTYPE", "TERM",
    # TLS + proxy knobs the owner may need behind a corporate proxy
    "NODE_EXTRA_CA_CERTS", "NODE_OPTIONS", "NODE_PATH",
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)

# Legacy name, still honoured so an existing RIGMA_MCODE_ENV_PASSTHROUGH keeps
# working.
_ENV_PASSTHROUGH_VARS = ("RIGMA_HARNESS_ENV_PASSTHROUGH",
                         "RIGMA_MCODE_ENV_PASSTHROUGH")


def harness_env(base: dict | None = None, *, also: tuple = ()) -> dict:
    """A launch environment for an external agent, minus the owner's secrets.

    `base` defaults to `os.environ`. The allowlist above is the floor; anything
    named by `RIGMA_HARNESS_ENV_PASSTHROUGH` (or the older
    `RIGMA_MCODE_ENV_PASSTHROUGH`) is added on top.

    `also` names variables THIS adapter genuinely needs, so the list stays a
    shared floor instead of growing a special case per caller. It is for a name
    the child cannot work without, not for convenience: DSH resolves its
    provider key through `apiKeyEnv: DEEPSEEK_API_KEY`, so dropping that would
    leave the agent unable to reach the local server at all.
    """
    src = os.environ if base is None else base
    env = {k: src[k] for k in HARNESS_ENV_ALLOWLIST if k in src}
    for k in also:
        if k in src:
            env[k] = src[k]
    extra = ""
    for var in _ENV_PASSTHROUGH_VARS:
        extra = (src.get(var) or "").strip()
        if extra:
            break
    for name in (n.strip() for n in extra.split(",")):
        if not name:
            continue
        if name.endswith("*"):          # a namespace, e.g. MY_AGENT_*
            pre = name[:-1]
            env.update({k: v for k, v in src.items() if k.startswith(pre)})
        elif name in src:
            env[name] = src[name]
    return env


def rigma_import_root() -> str:
    """The directory the RUNNING `rigma` package was imported from.

    A child started as `python -m rigma.<module>` must import this same code, and
    `harness_env` drops PYTHONPATH — which is exactly how a checkout is put ahead
    of a pip-installed copy (Run.bat does it). Without this on the child's path
    the DSH runner resolved `rigma` to the installed 0.10.0 and died with "No
    module named rigma._dsh_runner", and the MCP server silently ran old code.
    """
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# The repair and verification machinery is the reason a strong local-model
# agent works at all here, and no external harness inherits it.
#
# R6-MCP: the first entry used to read "Rigma's tools (image-by-reference, undo,
# sample_files, RAG, methods)" and named RAG and undo as left behind. They are
# not — both reach BOTH backends now, as MCP tools (`search_my_documents`,
# `remember`, `recall`, `undo_last_change`) rather than as native ones. The
# distinction that survives is BETWEEN the four tools that are exposed over MCP
# and the rest of Rigma's roster, which is not exposed at all: image-by-reference,
# sample_files and the methods/verification tools have no MCP equivalent, so a
# harness turn still cannot reach them. Saying "RAG is left behind" would now be
# a false negative, which is the same class of defect as the plan-mode claim.
_LEAVES_BEHIND = (
    "Rigma's tools that have no MCP equivalent (image-by-reference, "
    "sample_files, the methods/verification suite) — the four that DO have one "
    "(search_my_documents, remember, recall, undo_last_change) reach the agent "
    "through Rigma's own MCP server",
    "the weak-model repair layer (tool-call rescue, JSON repair, fuzzy paths)",
    "the run progress log, artifact verification and one-action-per-turn guard",
)


def _dsh_available() -> bool:
    """DSH arrives as a source checkout, not as a module or a binary name.

    `RIGMA_DSH_HOME` names it; the adapter also accepts the checkout this
    project was developed against. Imported lazily so this module stays free of
    the adapter's own imports — and DEFENSIVELY, because a probe that raises
    would take down the whole harness menu rather than report one backend
    missing."""
    try:
        from . import harness_dsh
    except Exception:
        return False
    return harness_dsh.available()


def _mcode_available() -> bool:
    """MiniMax Code is a CLI on PATH, or wherever `RIGMA_MCODE_BIN` points.

    A probe rather than `needs="mcode"` so the menu and the adapter agree: the
    adapter honours the env override, and `shutil.which` alone would call a
    backend missing that the adapter can in fact run."""
    try:
        from . import harness_mcode
    except Exception:
        return False
    return harness_mcode.available()


def _verified_of(module: str) -> str:
    """The build an adapter declares it was measured against, or "".

    Imported DEFENSIVELY for the same reason the probes are: an adapter that
    cannot be imported must cost one menu entry, not the whole menu. Empty is a
    real answer — it means nobody has verified that backend — so it is never
    turned into a default.
    """
    try:
        mod = importlib.import_module(f".{module}", __package__)
    except Exception:
        return ""
    return str(getattr(mod, "VERIFIED", "") or "")


BACKENDS: dict[str, Harness] = {
    NATIVE: Harness(
        name=NATIVE,
        label="Rigma (built in)",
        drives="in-process, in this server",
        runnable=True,
        needs="",
        wire="speaks to llama-server directly",
        built_in=True,
    ),
    DSH: Harness(
        name=DSH,
        label="DeepSeek Harness",
        drives="a subprocess turn: the SDK drives the dsh CLI over stdio JSON-RPC",
        runnable=True,
        needs="the DeepSeek Harness checkout (set RIGMA_DSH_HOME)",
        probe=_dsh_available,
        verified=_verified_of("harness_dsh"),
        # The design note had this wrong. `sdk-minimal` already mounts
        # llm-deepseek; what it does NOT do is set `protocol`, which defaults to
        # `messages` (Anthropic) and would POST /v1/messages at an OpenAI
        # server. So the wire is a one-file patch, not a custom composition.
        wire="sdk-minimal, patched to protocol: chat-completions and to a "
             "context window that fits this machine, pointed at <rigma>/v1",
        unsupported=_LEAVES_BEHIND + (
            "streaming: the SDK reports a turn's text when it ENDS, so the "
            "reply lands in one piece instead of token by token",
            "cancel: there is no wire-level cancel, so Stop ends the "
            "subprocess and the turn with it",
            "the sandbox: sdk-minimal pins danger-full-access with its "
            "workspace at the process cwd, so a confined profile does not "
            "survive the seam",
            # R5-PLANMODE: plan mode was CLAIMED under `capabilities` below and is
            # not available. It was also MOUNTED, which was worse than a wrong
            # sentence — the plugin registers `exit_plan_mode` unconditionally
            # while `/plan`, the only way in, lives inside `inject(['commands'])`,
            # and `dsh-commands` is not mounted. So the model was handed a tool
            # whose first act is to throw. The row is gone from the patch; the
            # claim moves here.
            "plan mode, so it can propose before it changes anything — the plugin "
            "needs `dsh-commands` (the `/plan` command, the only way IN) and "
            "`dsh-user-questions` (`exit_plan_mode`'s review channel, the only way "
            "OUT), and neither is mounted in this build. Mounting it anyway gave "
            "the model an `exit_plan_mode` tool that could only fail, so it is "
            "not mounted at all",
            "asking you a question mid-turn: `ask_user_question` needs the same "
            "missing `dsh-user-questions` channel",
            # R6-WEBFETCH: the OTHER half of the old web decision. Named rather than
            # left implied, because the absence of a search tool is otherwise
            # indistinguishable from an oversight — and because a reader who sees
            # `web_fetch` in the capability list will reasonably ask where
            # `web_search` went.
            # F6: named because the capability EXISTS upstream and is off here, which
            # is exactly the case a capability menu is for. Left unsaid it would read
            # as an oversight, and a reader comparing against DSH's docs would file it
            # as a gap.
            "letting the model choose its subagents' model: DSH gates that behind "
            "`modelSelectionSettings`, which is off, and the choice would be between "
            "one route — Rigma points DSH at a single local model, so you pick the "
            "model in Rigma's own UI instead",
            "web search: every search provider DSH ships reads an API key "
            "(`dsh-web-search-deepseek` wants DEEPSEEK_API_KEY), and Rigma's premise "
            "is a local model — so `web_fetch` is mounted and `web_search` is not",
        ),
        # R3-HARN-1: `harness_dsh.run_turn` accepts `permission` and ignores it
        # by design — DSH's confinement is its own bundle's business. Declared so
        # the UI can say that instead of rendering a selector that does nothing.
        honours_permission=False,
        # What Rigma mounts into the minimal profile. Named one by one rather
        # than summarised, because the list IS the answer to "is anything
        # missing" — and because every entry is something `sdk-minimal` does
        # not ship, so its absence would be silent. Kept in step with
        # data/dsh/agent-capabilities.patch.yml, which
        # tests/test_dsh_capabilities.py checks row by row.
        capabilities=(
            "goals, with a round driver and a tool to set, read and revise one",
            "subagents, spawned in-process or forked from this conversation",
            "a todo list the model maintains and the transcript renders",
            "skills, discoverable and loadable by the model itself "
            "(read from ~/.rigma/skills, the directory the Skills page writes)",
            "filesystem tools: read, write, edit, glob and grep",
            "agent instructions from AGENTS.md, and context compaction",
            # R6-MCP: mounted by a generated patch, so the row is not in the
            # shipped capability file and this sentence is the only place the
            # capability is declared. Named as the model sees them.
            "Rigma's own tools, as an MCP server: `mcp__rigma__search_my_documents`, "
            "`mcp__rigma__remember`, `mcp__rigma__recall` and "
            "`mcp__rigma__undo_last_change`, mounted through `dsh-mcp-client`",
            # R6-WEBFETCH: mounted, and it is keyless — which is why it is here and
            # search is not. `web_search` is declared under `unsupported` below, so the
            # two halves of the old single decision are now visible separately.
            "`web_fetch`, to read a public HTTP(S) page",
        ),
    ),
    MCODE: Harness(
        name=MCODE,
        label="MiniMax Code",
        # R6-ACP: this comment used to read "use exec's versioned NDJSON stream,
        # and treat ACP as a later upgrade — ACP would additionally mean
        # implementing the ACP *client* side." The client now EXISTS
        # (`harness_mcode_acp.py`), so the upgrade is built rather than pending.
        # The turn is still driven over `exec`, deliberately: swapping the live
        # transport is a separate, riskier change than building the client, and the
        # client is what makes the control plane reachable at all.
        drives="a subprocess turn: `mcode exec --output-format stream-json`. The "
               "same chat can instead run over mcode's Agent Client Protocol (ACP) — the "
               "`transport` selector on this chat switches it — which is what makes "
               "its session control plane and an answerable permission prompt "
               "reachable. `exec` stays the DEFAULT, because it is the wire that has "
               "been exercised against a real engine",
        runnable=True,
        needs="mcode",
        probe=_mcode_available,
        verified=_verified_of("harness_mcode"),
        wire="`mcode provider add --base-url <rigma>/v1 --api-format "
             "openai-completions --model <model> --api-key-env <var> "
             "--context-limit N --output-limit N --use`, into a Rigma-owned "
             "MINIMAX_DATA_DIR; the turn then runs as "
             "`--model custom_provider:rigma/<model>`. `--use` is required: it "
             "selects the provider, and an unselected one leaves mcode "
             "demanding a MiniMax login for a turn that never leaves this "
             "machine. It also tests the endpoint first, which costs one "
             "throwaway call to Rigma's own /v1",
        unsupported=_LEAVES_BEHIND + (
            "driveable only as a subprocess: it has no library API",
            "its own tool roster: a real turn sent the model 18 tool schemas, "
            "plus whatever MCP adds — Rigma can subtract from that set, never "
            "replace it",
            "its own system prompt: Rigma's is not passed through",
            "the sandbox: a headless turn needs `--permission full`, so a "
            "confined profile does not survive the seam",
            # R5-MCODE-DEADEND: the cost of `smart`, which is otherwise the
            # reasonable middle. `mcode exec` has no interaction host, so when
            # `smart` decides to ask, nobody can answer — and mcode's guard refuses
            # to start a session that still has a pending question, so the chat is
            # blocked PERMANENTLY rather than for one turn. The adapter now
            # detects that and says so in the chat, but the choice is what invites
            # it, so it is disclosed where the choice is made.
            "an answered question: if you pick `smart` and it decides to ask, "
            "nobody can answer on this transport and that chat cannot continue "
            "— `full` is the mode that does not invite this",
        ),
        # The other direction: what Rigma hands it that it did not arrive with.
        # mcode owns its own goals, todos, subagents and skills, so unlike DSH
        # this list is NOT about mounting capability — it is about the memory and
        # the workspace crossing the seam.
        capabilities=(
            # The roster is `mcp_server._ROSTER`, and it has FOUR entries.
            # Naming three of them understated it and omitted the one an arm is
            # most likely to want: searching the user's own indexed documents.
            "Rigma's own tools, as an MCP server: `search_my_documents`, "
            "`remember`, `recall` and `undo_last_change`, registered in "
            "mcode's own mcp.json",
            "the chat's workspace and its AGENTS.md, written in before a turn",
            "a session that survives the process, so its plan, subagents and "
            "goals continue across turns instead of restarting each one",
            # R6-ACP: what the ACP client adds. These are capabilities mcode has
            # ALWAYS had and Rigma could not reach, because `exec` is a projection
            # of one turn and these are all session-scoped. Disclosed because the
            # menu's job is to say what is reachable, and "unreachable with no line
            # saying so" is the defect this whole round is about.
            "over ACP, its session control plane — goals, a prompt queue, "
            "steering, delegation of a turn to a child session, plan mode, and "
            "switching the model or the permission policy mid-session — and the "
            "control panel can OPERATE it, not only display it",
            "over ACP, an ANSWERABLE permission prompt — mcode asks and waits, so "
            "a permission request can be granted or refused instead of only "
            "displayed. This is the one interaction channel mcode offers that DSH "
            "does not",
        ),
    ),
}


def installed(h: Harness) -> bool:
    """Whether the thing this backend needs is actually on this machine.

    An executable on PATH, an importable module, or a source checkout — the
    three ways a backend can arrive. `find_spec` raises rather than returning
    None for a malformed name, so a bad `needs` reads as "not installed" and
    not as a crash."""
    if h.probe is not None:
        return h.probe()                # a checkout, not a module or a binary
    if not h.needs:
        return True                     # native needs nothing
    if shutil.which(h.needs):
        return True
    try:
        return importlib.util.find_spec(h.needs) is not None
    except (ImportError, ValueError):
        return False


def known() -> list[str]:
    return sorted(BACKENDS)


def list_harnesses() -> list[dict]:
    """Every backend, with what it would cost to choose it.

    For a UI or a CLI: this is the honest menu, not the set of things that work.
    """
    return [BACKENDS[n].as_dict() for n in known()]


def conformance(name: str | None = None) -> list[dict]:
    """Is each backend still the build this adapter was measured against?

    The question a turn cannot answer for itself, and the one this seam is most
    exposed to. Rigma absorbs an ADDITIVE event-schema change on purpose —
    `map_event` ignores what it does not know, which is what the upstream docs
    ask for — so a released backend can change shape while every turn still
    LOOKS fine. A RENAMED item type degrades further: tool calls stop appearing
    and the reply still arrives, so the transcript reads as a model that chose
    not to use tools. Nothing in a turn would say so. This does.

    `drift` is True when the installed build differs from the verified one, and
    None when either is UNKNOWN — because "I could not tell" and "they agree"
    are different answers, and only one of them is reassuring.
    """
    out: list[dict] = []
    for key in known():
        h = BACKENDS[key]
        mod = _adapter_module(key)
        probe = getattr(mod, "backend_version", None) if mod else None
        version = ""
        if callable(probe):
            try:
                version = str(probe() or "")
            except Exception:
                version = ""       # a backend that cannot answer is a fact
        have, want = version.strip(), h.verified.strip()
        out.append({
            "name": h.name,
            "label": h.label,
            "installed": installed(h),
            "verified": want,
            "version": have,
            "built_in": h.built_in,
            "drift": (have != want) if (have and want) else None,
        })
    if name is not None:
        key = str(name).strip().lower()
        return [r for r in out if r["name"] == key]
    return out


def _adapter_module(name: str):
    """The adapter module for a backend, or None. Same defensive import."""
    mod_name = _ADAPTERS.get(str(name or "").strip().lower())
    if not mod_name:
        return None
    try:
        return importlib.import_module(f".{mod_name}", __package__)
    except Exception:
        return None


def endpoint_for(port: int) -> str:
    """The OpenAI-compatible base URL every backend must be pointed at.

    Rigma's own, always — the point of the seam is a different agent, not a
    different model.
    """
    return f"http://127.0.0.1:{int(port)}/v1"


def resolve(name: str | None, *, port: int | None = None) -> Harness:
    """The backend a session asked for, or raise saying exactly why not.

    Never falls back. A blank name is the native loop, which is the default.
    """
    want = str(name or "").strip().lower() or NATIVE
    h = BACKENDS.get(want)
    if h is None:
        raise HarnessError(
            f"unknown harness {want!r} — known: {', '.join(known())}")
    if not h.runnable:
        raise HarnessError(
            f"{h.label} cannot run a turn yet: {h.pending}. "
            f"Use the built-in harness, or see docs/superpowers/specs/"
            f"2026-09-20-harness-rework-design.md for what wiring it needs.")
    if not installed(h):
        raise HarnessError(
            f"{h.label} needs {h.needs!r} on this machine and it is not "
            f"installed")
    return h


# Which module drives which backend. A backend with an adapter and no entry
# here would be `runnable` on the menu and refuse at the first turn, so this
# table is the other half of that promise.
_ADAPTERS = {DSH: "harness_dsh", MCODE: "harness_mcode"}


# POSIX children must be detached so `kill_tree`'s killpg reaches THEIR group,
# not Rigma's own. A module constant rather than a fresh `os.name` read, so a
# test can flip the platform decision without patching `os.name` — which
# `pathlib` reads to choose WindowsPath vs PosixPath, and which would make
# unrelated code explode on a Windows host.
_DETACH_CHILDREN = os.name != "nt"


def _detached_kwargs() -> dict:
    """Popen kwargs that put a harness child in its OWN process group on POSIX.

    `kill_tree` reaches a tree with `killpg` on POSIX, so a child left in
    RIGMA'S own process group would make a stop take the server down with it.
    Every adapter that can end up in `kill_tree`'s path must spawn detached; this
    is the same idiom as `tools._launch_killable`. Windows needs nothing here —
    `tools._kill_tree` uses `taskkill /T` there, and the native job launcher
    already sets CREATE_NEW_PROCESS_GROUP.
    """
    return {"start_new_session": True} if _DETACH_CHILDREN else {}


@dataclass(frozen=True)
class KillResult:
    """What `kill_tree` established, not just what it tried.

    Truthy exactly when the tree is confirmed gone, so every caller that treated
    the old return as a bool keeps its meaning (`if result:` still works);
    `attempted` and `confirmed` are the extra facts the DSH timeout message needs
    to stay honest. The fields are added, not renamed: `ok` is the same boolean
    the function always returned.
    """

    ok: bool = False
    attempted: bool = False
    confirmed: bool = False

    def __bool__(self) -> bool:
        return self.ok


def kill_tree(proc) -> KillResult:
    """Kill a backend process AND everything it started.

    An adapter is handed a `.cmd` shim on Windows, so the process Rigma holds is
    a SHELL whose real work is a grandchild. Killing only the shell leaves the
    agent running and — the part that actually bites — leaves the pipe to stdout
    OPEN, so the read loop never sees EOF and the turn never ends. That is the
    difference between a stop button and a hang, and it was found by a test that
    hung rather than by reading the code.

    Killing the tree is also what stops an agent's SUBAGENTS. The arm spawns
    child agents of its own, and a stop that leaves them running is not a stop.

    The kill is delegated to `tools._kill_tree`, the ONE tree-kill in this
    codebase (AUDIT F35). It uses `killpg` on POSIX, where `proc.kill()` reaches
    only the direct child, and on Windows it polls the process rather than
    trusting `taskkill`'s exit code — which is 0 even when the tree walk misses a
    re-parented grandchild. Calling `taskkill` here AND inside that helper (as an
    earlier revision did) ran the kill twice; this function runs it once.

    `ok`/`confirmed` mean the process Rigma holds is confirmed dead on EITHER
    platform — the strongest evidence available, and the honest limit of it. The
    old body hardcoded `ok = False` off Windows, so a tree that HAD died was
    reported as "could not be confirmed dead" on every Linux/macOS timeout, while
    on Windows a taskkill exit code of 0 was reported as a tree kill it never
    verified. Both are corrected here.
    """
    attempted = False
    confirmed = False
    pid = int(getattr(proc, "pid", 0) or 0)
    try:
        from . import tools as _tools
        attempted = True
        confirmed = bool(_tools._kill_tree(pid, proc))
    except Exception:
        # `_kill_tree` is not supposed to raise; if it does, the direct child is
        # still ours to take, and `poll` is the only confirmation left.
        try:
            proc.kill()
        except OSError:
            pass
        try:
            confirmed = proc.poll() is not None
        except Exception:
            confirmed = False
    return KillResult(ok=confirmed, attempted=attempted, confirmed=confirmed)


def adapter(name: str):
    """The module that drives `name`, or None for the native loop.

    Every adapter exposes the same one call:

        drive_turn(*, base_url, model, prompt, system_prompt="", session_id="",
                   cwd="", max_tokens=4096, context_window=32768,
                   timeout=1800.0, cancel=None, permission="full")
                   -> Iterator[TurnEvent]

    It is a BLOCKING generator: the caller owns the thread. That is deliberate
    — a backend is a subprocess doing blocking IO, and the seam's contract is
    that a stalled backend stalls a worker, never the event loop.

    `session_id` is the CALLER's session id, for a backend that wants to key its
    own logs on it. `state` is the other direction and belongs to the BACKEND: a
    mutable dict the adapter reads to resume and writes to be remembered. MiniMax
    Code puts its own session id there, which is what lets the next turn
    continue the conversation instead of starting it over — and that continuity
    is most of what makes an external agent worth handing a turn to. Rigma
    persists it per backend, so switching away and back resumes the right one.

    `cancel` is a `threading.Event` the OWNER sets to stop the turn, and an
    adapter that ignores it is not finished. A THREADING event and not an
    asyncio one, because the adapter runs on a worker thread: the chat route
    sets it from the event loop and the adapter reads it where the work is. An
    adapter should WAIT on it rather than poll between output lines — a backend
    that has gone quiet, waiting on a model or on a subagent it spawned, is
    exactly the one worth stopping, and it is the one that emits nothing to
    check against. Stopping is not failing: report it as a notice, and make sure
    whatever the backend needs to resume is still written to `state`, so the
    next turn continues instead of starting over.

    `permission` is how much the backend may do without being asked, as the
    owner chose it for this chat. It is a HINT the backend translates into its
    own vocabulary, not a policy Rigma enforces: mcode maps it to
    `--permission`, and an adapter whose backend has no such notion ignores it.
    The default is "full" because headless has nobody to ask, so a mode that
    wants to ask fails the run — a real trade, which is why it is a per-chat
    choice rather than a constant.

    Imported lazily so a broken or absent adapter cannot take down the menu.
    """
    mod = _ADAPTERS.get(str(name or "").strip().lower())
    if mod is None:
        return None
    return importlib.import_module(f".{mod}", __package__)
