"""Reuse a conversation's computed state by its PREFIX, across sessions and
restarts.

Why this shape and not KV shifting. Qwen3.5/3.8 interleave ~48 state-space
(SSM) layers with ~16 attention layers — the gguf says so directly:
`qwen35.ssm.state_size = 128`, `full_attention_interval = 4`. An SSM layer
holds one fixed-size recurrent state summarising everything so far. There is no
per-token vector to re-position and no way to remove a token from the middle:
the recurrence discards information on purpose, so it cannot be inverted. That
is why `--cache-reuse` is refused by the engine on these models and why
llama.cpp #18497 sits open with nobody assigned. It is not a missing feature.

The same property makes the opposite operation exact. An SSM state IS a
complete summary of its prefix, so truncate-and-extend costs nothing: restore
the state as of message N, send a prompt that begins with those N messages, and
continue. No recomputation, no approximation.

So this caches by prefix and only ever extends:

    snapshot the slot at message boundaries, keyed by hash(config + messages)
    on a new request  -> find the LONGEST cached prefix that matches
                      -> restore it, prefill only the tail

What it cannot do is make an edit near the START of a long document cheap.
Nothing can, on this architecture. Appending, branching, regenerating and
reopening an old chat are the cases it helps, and they are most of them.

Two independent safety layers, in order:

  1. the config fingerprint is mixed into every key, so a snapshot taken under
     a different quant / ctx / cache type / layer split can never be selected —
     its key simply is not the one being looked up. Restoring across a config
     change does not error, it produces fluent text from a history that never
     happened, so this is the one that matters.
  2. llama-server compares the restored slot's tokens against the incoming
     prompt and keeps only the genuinely common prefix. So a stale-but-
     same-config snapshot degrades to ordinary prefix matching rather than
     corrupting anything.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path

# Rough characters per token, only ever used to decide whether a prefix has
# grown enough to be worth another multi-gigabyte snapshot. Never used for
# anything the model sees.
CHARS_PER_TOKEN = 3

# Do not take another snapshot until the prefix has grown by at least this
# much. Snapshots cost seconds to write and gigabytes to keep, so one per turn
# would be far more expensive than the prefill it saves.
MIN_GROWTH_TOKENS = 4096

# Total bytes of snapshots to keep. Well under the free space on this machine,
# and enough for a handful of long conversations.
DEFAULT_BUDGET_BYTES = 40 * 2**30


@dataclass(frozen=True)
class PrefixPoint:
    """A message boundary that could be snapshotted or restored."""
    n_messages: int
    key: str
    approx_tokens: int


def _message_blob(msg: dict) -> str:
    """The parts of a message that change what the model computed.

    Role and text only. Everything else rigma carries on a message — ids,
    timestamps, tool metadata, UI state — does not reach the model, and folding
    it in would make every key unique and the cache useless.
    """
    role = str(msg.get("role", ""))
    content = msg.get("content", "")
    if isinstance(content, list):        # vision parts
        content = "".join(str(p.get("text", "")) for p in content
                          if isinstance(p, dict))
    return f"{role}\x00{content}"


def prefix_keys(messages: list[dict], config_fp: str) -> list[PrefixPoint]:
    """One key per message boundary, each covering everything up to it.

    Chained, so key(i) depends on key(i-1): two conversations that diverge at
    message 3 share keys 1 and 2 and nothing after. That is exactly the
    "longest common prefix" relation, computed once, without comparing texts.
    """
    points: list[PrefixPoint] = []
    running = hashlib.sha256(f"cfg={config_fp}".encode("utf-8")).digest()
    chars = 0
    for i, msg in enumerate(messages, start=1):
        blob = _message_blob(msg)
        chars += len(blob)
        running = hashlib.sha256(running + blob.encode("utf-8")).digest()
        points.append(PrefixPoint(n_messages=i,
                                  key=running.hex()[:16],
                                  approx_tokens=chars // CHARS_PER_TOKEN))
    return points


def best_match(points: list[PrefixPoint],
               available: set[str]) -> PrefixPoint | None:
    """The DEEPEST boundary we hold a snapshot for.

    Deepest, not first: a snapshot at message 40 saves everything a snapshot at
    message 4 would and more. Returns None when nothing matches, which is the
    ordinary case for a new conversation and is not a failure.
    """
    for p in reversed(points or []):
        if p.key in available:
            return p
    return None


def should_snapshot(point: PrefixPoint, last: PrefixPoint | None,
                    min_growth: int = MIN_GROWTH_TOKENS) -> bool:
    """Is this boundary far enough past the last snapshot to be worth one?

    Snapshots are gigabytes and take seconds to write. One per turn would cost
    more than the prefill it saves, so they are spaced by how much the prefix
    has actually grown.
    """
    if point.n_messages <= 0:
        return False
    if last is None:
        return point.approx_tokens >= min_growth
    if point.n_messages <= last.n_messages:
        return False                       # not an extension of that snapshot
    return point.approx_tokens - last.approx_tokens >= min_growth


def snapshot_name(key: str) -> str:
    """Filename llama-server writes, relative to --slot-save-path."""
    return f"pfx-{key}.bin"


def available(save_dir: Path) -> set[str]:
    """Keys we hold a snapshot for."""
    try:
        return {p.stem[4:] for p in save_dir.glob("pfx-*.bin") if p.is_file()}
    except OSError:
        return set()


def touch(save_dir: Path, key: str, meta: dict | None = None) -> None:
    """Record that a snapshot was just useful, so eviction keeps it.

    Eviction is by last USE, not last write: the snapshot of the chapter being
    worked on is the valuable one even if it was taken days ago.
    """
    try:
        p = save_dir / snapshot_name(key)
        if p.is_file():
            now = time.time()
            import os
            os.utime(p, (now, now))
        if meta is not None:
            (save_dir / f"pfx-{key}.json").write_text(
                json.dumps({**meta, "used_at": time.time()}, indent=2),
                encoding="utf-8")
    except OSError:
        pass


def snapshot(port: int, save_dir: Path, key: str, *, slot: int = 0,
             meta: dict | None = None) -> str | None:
    """Write the slot to a prefix snapshot. Returns an error, or None."""
    from .kvcache import slot_action
    err = slot_action(port, slot, "save", snapshot_name(key))
    if err is None:
        touch(save_dir, key, meta)
    return err


def warm(port: int, save_dir: Path, key: str, *, slot: int = 0) -> str | None:
    """Restore a prefix snapshot into the slot. Returns an error, or None.

    Only ever called with a key from `best_match`, so the config fingerprint
    baked into it already matches the running engine.
    """
    if not (save_dir / snapshot_name(key)).is_file():
        return "no snapshot"
    from .kvcache import slot_action
    err = slot_action(port, slot, "restore", snapshot_name(key))
    if err is None:
        touch(save_dir, key)
    return err


def evict(save_dir: Path, budget_bytes: int = DEFAULT_BUDGET_BYTES) -> list[str]:
    """Drop least-recently-used snapshots until the directory fits the budget.

    Nothing else ever deletes these and each is a large fraction of a context,
    so without this the directory grows without bound.
    """
    try:
        blobs = [p for p in save_dir.glob("pfx-*.bin") if p.is_file()]
    except OSError:
        return []
    blobs.sort(key=lambda p: p.stat().st_mtime, reverse=True)   # newest use first
    removed, total = [], 0
    for p in blobs:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        if total + size <= budget_bytes:
            total += size
            continue
        try:
            p.unlink()
            removed.append(p.name)
            meta = p.with_suffix(".json")
            if meta.is_file():
                meta.unlink()
        except OSError:
            continue
    return removed
