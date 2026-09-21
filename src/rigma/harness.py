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
import shutil
from dataclasses import dataclass
from typing import Callable

NATIVE = "native"
DSH = "dsh"
MCODE = "mcode"


class HarnessError(ValueError):
    """A harness cannot be used: unknown, not installed, or not implemented."""


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

    def as_dict(self) -> dict:
        return {"name": self.name, "label": self.label, "drives": self.drives,
                "runnable": self.runnable, "installed": installed(self),
                "needs": self.needs, "wire": self.wire,
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
        runnable=False,
        needs="mcode",
        wire="`mcode provider add --base-url <rigma>/v1 --api-format "
             "openai-completions --context-limit N --output-limit N`",
        unsupported=_LEAVES_BEHIND + (
            "driveable only as a subprocess: it has no library API",
            "its own tool roster: 12 built-in ids plus MCP, which Rigma can "
            "subtract from but never replace",
        ),
        pending="the adapter is not written yet — install it with `npm install "
                "-g @minimax-ai/code` (Node 22.19+ or 24+), then select it; its "
                "stream-json schema is versioned, so the adapter is written "
                "against a stable contract",
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
