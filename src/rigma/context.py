"""Deterministic context reclaim for autonomous runs.

LLM compaction summarises an entire span into prose. That preserves the gist
and destroys exactly what an agent navigates by — the filenames, the error
text, the argument that finally worked — which is why summarising measurably
LENGTHENS agent trajectories rather than shortening them.

Observation masking is the alternative used by long-horizon agent harnesses:
shrink only the environment's replies, and leave the model's own reasoning and
tool calls byte-identical. The trajectory stays intact; only the bulk goes.

Pure functions of a message list. No engine, no network, no inference.
"""
from __future__ import annotations

import re

# how a masked observation reads to the model. It must still name the tool:
# "something happened here" leaves the model unable to tell a finished step
# from one it never started, which is the failure masking exists to avoid.
_MASK = "[{n} chars masked — {status}]"
_MASKED_RE = re.compile(r"\[\d+ chars masked — ")
_PREFIX = "TOOL RESULT "


def session_chars(messages: list[dict]) -> int:
    """Rough size of a message list. Characters, not tokens, deliberately: this
    only has to rank and threshold, and a tokeniser here would mean loading one
    on a machine whose VRAM is fully committed to the model."""
    total = 0
    for m in messages:
        c = m.get("content", "")
        if isinstance(c, str):
            total += len(c)
        elif isinstance(c, list):        # vision parts
            total += sum(len(p.get("text", "")) for p in c
                         if isinstance(p, dict))
    return total


# the legacy sniff demands the full shape `TOOL RESULT <name>: `, not just the
# two words: a human steering message like "TOOL RESULT formats are failing,
# fix your syntax" starts with the bare prefix, and masking IT would erase the
# very correction the user typed. Only the server writes the colon form.
_LEGACY_RE = re.compile(r"^TOOL RESULT \S+: ")


def is_observation(msg: dict) -> bool:
    """A tool result, as opposed to a real user turn or the RUN STATE block.

    Prefers the explicit tag, falls back to the text shape: sessions written
    before the tag existed are on disk in live runs and must still mask.
    """
    if msg.get("role") != "user":
        return False
    if msg.get("kind") == "tool_result":
        return True
    if "kind" in msg:                      # explicitly something else
        return False
    return isinstance(msg.get("content"), str) and \
        bool(_LEGACY_RE.match(msg["content"]))


def _mask_one(msg: dict) -> dict:
    content = msg.get("content", "")
    if not isinstance(content, str) or _MASKED_RE.search(content):
        return msg                        # already masked — idempotent
    tools = msg.get("tools") or []
    names = [t.get("name", "") for t in tools if t.get("name")]
    if not names:                         # legacy: recover the name from the text
        names = re.findall(r"^TOOL RESULT ([^:]+):", content, re.M)
    ok = all(t.get("ok", True) for t in tools) if tools else \
        "error" not in content[:200].lower()
    label = _MASK.format(n=len(content), status="succeeded" if ok else "failed")
    head = ", ".join(dict.fromkeys(names)) or "tool"
    out = dict(msg)
    out["content"] = f"{_PREFIX}{head}: {label}"
    return out


def mask_observations(messages: list[dict], keep_recent: int = 6,
                      budget_chars: int = 24_000) -> tuple[list[dict], int]:
    """Mask oldest observations until the list fits `budget_chars`.

    The last `keep_recent` messages are never masked — those are the results the
    model is actively working from, and masking them would force an immediate
    re-fetch of what it just asked for.

    Returns (new_messages, n_masked). Does not mutate the input.
    """
    out = [dict(m) for m in messages]
    # running total, adjusted as messages shrink. Recomputing session_chars()
    # inside the loop was O(n²) — harmless in tests, but this runs ON the
    # asyncio event loop, and a long trajectory would freeze every other
    # request while it ground through millions of length checks.
    total = session_chars(out)
    if total <= budget_chars:
        return out, 0
    tail_start = len(out) - keep_recent if keep_recent else len(out)
    masked = 0
    for i, msg in enumerate(out):
        if total <= budget_chars:
            break
        if i >= tail_start or not is_observation(msg):
            continue
        new = _mask_one(msg)
        if new["content"] != msg["content"]:
            total -= len(msg.get("content", "")) - len(new["content"])
            out[i] = new
            masked += 1
    return out, masked
