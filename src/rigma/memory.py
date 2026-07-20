"""Agent memory — phase 1: event mining, the anchoring guard, and the store.

The design principle, earned from every failure on 2026-07-19: never make a
weak model responsible for its own bookkeeping. The server observes what
actually happened and writes it down. Rigma already HAS remember/recall tools
and the model has never once called them.

Three things live here, and the division of labour matters:

  mine_events()  detects events in the action trace. Deterministic, cannot
                 hallucinate, and deliberately does NOT write the rule — a
                 literal extractor can only produce literal strings, so left
                 alone it would learn "Comfy_UI_428.png failed" rather than
                 "never retype filenames".

  the guard      refuses to store a raw trace. An autoregressive model that
                 reads a transcript of a failing agent will faithfully
                 SIMULATE a failing agent, so showing it its own failure
                 history primes repetition instead of avoidance. Only the
                 distilled imperative is safe.

  MemoryStore    append-only JSONL. Never load-bearing: every read path
                 degrades to "no memories" rather than raising.

Pure functions plus one file. No engine, no network, no inference.
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

# How far after a failure a success still counts as "the thing that fixed it".
# Wide enough for a real recovery (diagnose, then act), narrow enough that two
# unrelated events 14 actions apart are not welded into a false lesson.
RECOVERY_WINDOW = 4

# Kinds whose whole purpose is naming an artifact, where a literal filename is
# the content rather than noise. Behavioural rules get no such licence.
_GUARD_EXEMPT_KINDS = {"project"}

_WIN_PATH = re.compile(r"[A-Za-z]:[\\/]")
_UNC_PATH = re.compile(r"\\\\[^\\]+\\")
_FILENAME = re.compile(
    r"\S+\.(?:png|jpe?g|webp|gif|bmp|md|txt|json|jsonl|py|csv|gguf|safetensors)\b",
    re.I)
_CALL_SYNTAX = re.compile(r"\w+\([^)]*['\"][^)]*\)")


def looks_like_raw_trace(text: str) -> bool:
    """True when `text` carries verbatim evidence rather than a lesson.

    Deliberately blunt. A false positive costs one memory; a false negative
    puts a failure transcript in front of a model that will imitate it.
    """
    t = str(text or "")
    return bool(_WIN_PATH.search(t) or _UNC_PATH.search(t)
                or _FILENAME.search(t) or _CALL_SYNTAX.search(t))


# --- mining ------------------------------------------------------------------

def mine_events(actions: list[dict]) -> list[dict]:
    """Detect interesting events in an actions.jsonl trace.

    Returns event dicts, NOT memories. Events keep their raw args so the
    distiller has the evidence to generalise from; the guard stops that
    evidence reaching the store.
    """
    events: list[dict] = []
    seen_failures: dict[tuple, int] = {}
    for i, act in enumerate(actions or []):
        if act.get("ok", True):
            continue
        key = (act.get("tool"), act.get("args"))
        seen_failures[key] = seen_failures.get(key, 0) + 1
        # the same call failing twice is a loop forming, not bad luck
        if seen_failures[key] == 2:
            events.append({"kind": "loop", "tool": act.get("tool"),
                           "args": act.get("args"),
                           "count": seen_failures[key]})
        # The recovery is the FIRST success after the failure, and only counts
        # if it came from a different tool. Stopping at the first success
        # matters: without it, any successful action within the window gets
        # welded to an unrelated earlier failure and becomes a false lesson.
        # Same tool succeeding is ordinary retrying and teaches nothing.
        for nxt in (actions[i + 1:i + 1 + RECOVERY_WINDOW]):
            if not nxt.get("ok", True):
                continue                      # still failing — keep looking
            if nxt.get("tool") != act.get("tool"):
                events.append({"kind": "recovery",
                               "failed_tool": act.get("tool"),
                               "failed_args": act.get("args"),
                               "worked_tool": nxt.get("tool"),
                               "worked_args": nxt.get("args")})
            break                             # first success decides, either way
    return events


# --- the store ---------------------------------------------------------------

class MemoryStore:
    """Append-only JSONL. One memory per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)

    def all(self) -> list[dict]:
        """Every memory. A corrupt line is skipped, never raised — a run must
        not fail because memory failed."""
        out: list[dict] = []
        try:
            text = self.path.read_text(encoding="utf-8")
        except (FileNotFoundError, OSError):
            return out
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except (ValueError, TypeError):
                continue
        return out

    def _write_all(self, rows: list[dict]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def add(self, kind: str, text: str, **extra) -> dict:
        """Store a memory. Raises ValueError if it carries a raw trace.

        Phase 1 dedups on exact text only. Semantic dedup needs embeddings AND
        a conflict check — similarity alone merges contradictions, since
        "prefer q8_0" and "prefer f16" score >0.95 — so it waits for phase 2
        rather than being approximated badly here.
        """
        text = str(text or "").strip()
        if not text:
            raise ValueError("empty memory")
        if kind not in _GUARD_EXEMPT_KINDS and looks_like_raw_trace(text):
            raise ValueError(
                "refusing to store a raw trace as a behavioural rule: "
                "a model that reads a failure transcript imitates it. "
                f"Distil it into an imperative first. Got: {text[:80]!r}")
        rows = self.all()
        for r in rows:
            if r.get("kind") == kind and r.get("text") == text:
                r["seen_count"] = r.get("seen_count", 1) + 1
                r["last_seen"] = time.time()
                self._write_all(rows)
                return r
        rec = {"kind": kind, "text": text, "status": "draft",
               "seen_count": 1, "outcome_score": 0,
               "born": time.time(), "last_seen": time.time(), **extra}
        rows.append(rec)
        self._write_all(rows)
        return rec


# --- reading it back ---------------------------------------------------------

def render_pitfall_block(memories: list[dict], limit: int = 5,
                         include_drafts: bool = True) -> str:
    """The run-start block: terse imperatives, never prose.

    2026-07-19 proved that anything discursive injected into the loop gets
    narrated back instead of acted on, so this stays a bulleted list of rules
    and nothing else. An empty store renders nothing at all — an empty header
    would just be context the model feels invited to comment on.

    `include_drafts` defaults True in phase 1 and the spec says drafts must
    never be PINNED. Both are right, and the resolution is the label: nothing
    graduates yet (promotion needs the outcome tracking phase 3 builds), so
    verified-only would render an empty block forever and phase 1 would ship
    inert. An explicitly UNVERIFIED line is a hint, not a foundation, which is
    what the quarantine was protecting against. Flip this default to False when
    graduation exists.
    """
    rows = [m for m in memories or [] if m.get("kind") == "pitfall"]
    if not include_drafts:
        rows = [m for m in rows if m.get("status") == "verified"]
    if not rows:
        return ""
    rows.sort(key=lambda m: (m.get("outcome_score", 0), m.get("seen_count", 0)),
              reverse=True)
    lines = ["WHAT YOU LEARNED BEFORE — these are rules, not suggestions:"]
    for m in rows[:limit]:
        tag = "UNVERIFIED: " if m.get("status") != "verified" else ""
        lines.append(f"  • {tag}{m.get('text', '')}")
    return "\n".join(lines)
