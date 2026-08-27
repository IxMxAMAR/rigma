"""Keep the prompt cache across an engine restart.

llama-server already caches prompts well *while it is running*: prefix reuse per
slot, `--cache-reuse` for edits that are not a clean prefix, and idle slots
spilled to host RAM. All of it dies with the process. Unload the engine to free
the card, load it again, and a long conversation re-prefills from zero —
measured on this machine, a 120K window at ~560 t/s is **four minutes** before
the first token.

llama-server can write a slot's KV cache to disk (`--slot-save-path`, which
rigma already passes) and read it back. Nothing called it. This does.

THE DANGEROUS PART, and why the naming works the way it does: a saved KV cache
is only meaningful under the exact configuration that produced it. Restore one
taken at a different context length, cache type, quant, layer split or engine
build and the model does not error — it generates fluent, subtly wrong text
from a cache that does not describe its own history. There is no checksum in
the format that would catch it.

So the file is NAMED for its configuration. `fingerprint()` hashes everything a
cache depends on, and restore only ever asks for the file whose name matches the
configuration being launched right now. A mismatch cannot be restored because
the name it would need does not exist. That is a stronger guarantee than
comparing fields after loading, and it is why the fingerprint is a pure function
with its own tests.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import httpx

# Slot 0 is the user's conversation; slot 1 is Rigma's own aux calls
# (auto-title, compaction, delegate). Only the conversation is worth persisting.
MAIN_SLOT = 0

# Every field a saved cache depends on. Named explicitly rather than "whatever
# is in the dict" so that adding a flag to ComboFlags cannot silently widen what
# counts as the same cache.
FINGERPRINT_FIELDS = (
    "model", "quant", "gguf", "backend", "engine",
    "ctx", "cache_type_k", "cache_type_v",
    "ngl", "n_cpu_moe", "spec_type", "spec_n_max",
)

# How many saved caches to keep. Each is roughly ctx x the KV bytes per token —
# ~2.9GB for a 120K window at q5_1 — so this is disk measured in tens of GB.
KEEP_CACHES = 3


def fingerprint(cfg: dict) -> str:
    """A stable short hash of everything a KV cache is only valid under.

    Pure. Key order does not matter, absent fields are distinct from empty ones,
    and any change to any field in FINGERPRINT_FIELDS produces a different
    hash — which is the whole safety property, since the hash is what decides
    whether a cache on disk may be restored.
    """
    parts = []
    for k in FINGERPRINT_FIELDS:
        v = cfg.get(k)
        parts.append(f"{k}={'\x00' if v is None else v}")
    blob = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def cache_name(fp: str) -> str:
    """Filename llama-server will use, relative to --slot-save-path."""
    return f"kv-{fp}.bin"


def _meta_path(save_dir: Path, fp: str) -> Path:
    return save_dir / f"kv-{fp}.json"


def slot_action(port: int, slot: int, action: str, filename: str,
                timeout: float = 120.0) -> str | None:
    """One call to llama-server's slot save/restore. Returns None on success.

    Shared with prefixcache so there is a single place that knows the shape of
    this endpoint — it is enabled by --slot-save-path and takes the filename
    relative to that directory.
    """
    try:
        r = httpx.post(f"http://127.0.0.1:{port}/slots/{slot}",
                       params={"action": action},
                       json={"filename": filename}, timeout=timeout)
        if r.status_code != 200:
            return f"engine refused the {action} ({r.status_code})"
    except Exception as e:
        return str(e)[:200]
    return None


def config_of(plan, engine: str = "") -> dict:
    """The fingerprint input for a RunPlan. One place, so the launch path and
    the restore path cannot drift into disagreeing about what "the same
    configuration" means."""
    f = plan.flags
    return {
        "model": plan.model_slug, "quant": plan.gguf.quant,
        "gguf": plan.gguf.file, "backend": plan.backend, "engine": engine,
        "ctx": f.ctx, "cache_type_k": f.cache_type_k,
        "cache_type_v": f.cache_type_v, "ngl": f.ngl,
        "n_cpu_moe": f.n_cpu_moe, "spec_type": f.spec_type,
        "spec_n_max": f.spec_n_max,
    }


def save(port: int, save_dir: Path, fp: str, *, meta: dict | None = None,
         slot: int = MAIN_SLOT,
         timeout: float = 120.0) -> tuple[str | None, str | None]:
    """Write the slot's KV cache to disk under `fp`. Returns (fp, error).

    The fingerprint is passed in rather than derived: at unload time the plan
    that produced the running engine is no longer to hand, and re-deriving it
    could silently disagree with what was actually launched — which is exactly
    the mismatch this module exists to prevent."""
    err = slot_action(port, slot, "save", cache_name(fp), timeout)
    if err:
        return None, err
    try:
        _meta_path(save_dir, fp).write_text(
            json.dumps({**{k: (meta or {}).get(k) for k in FINGERPRINT_FIELDS},
                        "saved_at": time.time()}, indent=2), encoding="utf-8")
    except OSError:
        pass          # the cache itself is what matters; metadata is for humans
    return fp, None


def restore(port: int, save_dir: Path, fp: str, *, slot: int = MAIN_SLOT,
            timeout: float = 120.0) -> tuple[bool, str | None]:
    """Load the KV cache saved under EXACTLY this fingerprint.

    Returns (restored, note). Not finding one is the normal case after any
    config change and is not an error.
    """
    blob = save_dir / cache_name(fp)
    if not blob.is_file():
        return False, None
    err = slot_action(port, slot, "restore", cache_name(fp), timeout)
    return (False, err) if err else (True, None)


def prune(save_dir: Path, keep: int = KEEP_CACHES) -> list[str]:
    """Drop all but the `keep` newest caches. Returns what was removed.

    These files are gigabytes each. Without this the directory grows by one
    full context every time the engine is restarted under a new configuration,
    and nothing else ever deletes them.
    """
    try:
        blobs = sorted((p for p in save_dir.glob("kv-*.bin") if p.is_file()),
                       key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []
    removed = []
    for p in blobs[max(0, keep):]:
        try:
            p.unlink()
            removed.append(p.name)
            meta = p.with_suffix(".json")
            if meta.is_file():
                meta.unlink()
        except OSError:
            continue
    return removed
