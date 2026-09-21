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
import subprocess
from dataclasses import dataclass
from typing import Callable

NATIVE = "native"
DSH = "dsh"
MCODE = "mcode"


class HarnessError(ValueError):
    """A harness cannot be used: unknown, not installed, or not implemented."""


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


@dataclass(frozen=True)
class Harness:
    """One agent backend.

    `runnable` is whether a turn can actually be handed to it TODAY, and it is
    deliberately separate from `installed`. DSH is a real, studied integration
    whose availability can be probed; that does not make it wired up, and
    conflating the two is how a probe becomes a silent fallback.
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

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "drives": self.drives,
                "runnable": self.runnable, "installed": installed(self),
                "needs": self.needs, "wire": self.wire, "verified": self.verified,
                "unsupported": list(self.unsupported), "pending": self.pending}


# The repair and verification machinery is the reason a strong local-model
# agent works at all here, and no external harness inherits it.
_LEAVES_BEHIND = (
    "Rigma's tools (image-by-reference, undo, sample_files, RAG, methods)",
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
        ),
    ),
    MCODE: Harness(
        name=MCODE,
        label="MiniMax Code",
        # The design note said "acp for streaming, exec for batch". Stale: use
        # exec's versioned NDJSON stream, and treat ACP as a later upgrade —
        # ACP would additionally mean implementing the ACP *client* side.
        drives="a subprocess turn: `mcode exec --output-format stream-json`",
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


def kill_tree(proc) -> None:
    """Kill a backend process AND everything it started.

    An adapter is handed a `.cmd` shim on Windows, so the process Rigma holds is
    a SHELL whose real work is a grandchild. Killing only the shell leaves the
    agent running and — the part that actually bites — leaves the pipe to stdout
    OPEN, so the read loop never sees EOF and the turn never ends. That is the
    difference between a stop button and a hang, and it was found by a test that
    hung rather than by reading the code.

    Killing the tree is also what stops an agent's SUBAGENTS. The arm spawns
    child agents of its own, and a stop that leaves them running is not a stop.
    """
    if os.name == "nt":
        try:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                           capture_output=True, timeout=20)
        except (OSError, subprocess.TimeoutExpired):
            pass                    # fall through: proc.kill still gets the shell
    try:
        proc.kill()
    except OSError:
        pass


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
