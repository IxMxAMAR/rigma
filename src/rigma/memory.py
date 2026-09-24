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
import logging
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

log = logging.getLogger(__name__)

# How far after a failure a success still counts as "the thing that fixed it".
# Wide enough for a real recovery (diagnose, then act), narrow enough that two
# unrelated events 14 actions apart are not welded into a false lesson.
RECOVERY_WINDOW = 4

# Kinds whose whole purpose is naming an artifact, where a literal filename is
# the content rather than noise. Behavioural rules get no such licence.
_GUARD_EXEMPT_KINDS = {"project"}

# Hard cap on stored pitfalls. Exact-text dedup cannot catch an LLM distiller
# paraphrasing the same lesson differently every run ("Never type filenames" /
# "Do not manually type file names" / ...), so without a bound the store
# accumulates near-duplicates with flat counts and the pinned top-5 becomes a
# pseudo-random slice of redundant rules. Until phase 2 brings semantic dedup,
# the cap IS the quarantine: overflow evicts the least-proven rule.
MAX_PITFALLS = 24


def _eviction_key(m: dict):
    """Least-proven first: drafts/retired outrank verified, then lowest outcome
    score, then fewest sightings, then oldest."""
    return (m.get("status") == "verified", m.get("outcome_score", 0),
            m.get("seen_count", 0), m.get("last_seen", 0))


def _cap_rows(rows: list[dict], cap: int = MAX_PITFALLS) -> bool:
    """Bound EVERY kind, not just pitfalls. True when anything was evicted.

    AUDIT 10-10: the cap was written for pitfalls only, and the technique path
    (one `"When stuck: " + tech` per advisor-assisted step, serve.py) was added
    later without one. Techniques therefore accumulated forever, and every
    `retrieve` re-scored the whole unbounded list on each step change.
    """
    evicted = False
    by_kind: dict[str, list[dict]] = {}
    for r in rows:
        by_kind.setdefault(str(r.get("kind")), []).append(r)
    for kind, group in by_kind.items():
        while len(group) > cap:
            victim = min(group, key=_eviction_key)
            group.remove(victim)
            rows.remove(victim)
            evicted = True
            log.info("memory: %s cap reached, evicted %r", kind,
                     victim.get("text", "")[:60])
    return evicted

_WIN_PATH = re.compile(r"[A-Za-z]:[\\/]")
_UNC_PATH = re.compile(r"\\\\[^\\]+\\")
# enumerated whitelist rather than "any dotted token": the generic form
# rejects version numbers ("use Python 3.12" would be dropped as a filename).
# The list covers what THIS box's tools actually produce — review found the
# first cut missed .ps1/.log/.bat etc., so "check server.log for the trace"
# passed the guard and would have been pinned into every future run.
_FILENAME = re.compile(
    r"\S+\.(?:png|jpe?g|webp|gif|bmp|md|txt|json|jsonl|py|csv|gguf"
    r"|safetensors|ps1|bat|cmd|log|ya?ml|ini|cfg|toml|pt|pth|ckpt|onnx"
    r"|bin|zip|7z|exe|html?|pdf|sh|js|ts|css)\b",
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
        # identity by full-args hash when the trace carries one; the stored
        # args string is display-truncated to 300 chars and collides for big
        # write_file/run_python payloads that share a prefix
        key = (act.get("tool"), act.get("args_sha") or act.get("args"))
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

# Cross-PROCESS lock. The in-process RLock stops two threads losing each
# other's writes; it does nothing for two Rigma processes (the CLI and the
# server, or two servers) opening the same JSONL. This is an OS advisory lock
# on a sidecar file held across the whole read-modify-write.
#
# A sidecar, not the store itself: _write_all replaces the store by rename, so
# a lock on the store's inode would be dropped by the very write it guards.
# Locks conflict across independent handles (msvcrt.locking / fcntl.flock), so
# two handles in ONE process — what the tests can create without spawning a
# subprocess — prove the primitive.
_FILE_LOCK_TIMEOUT = 10.0


class _FileLock:
    """Advisory exclusive lock over one byte of a lock file.

    acquire() is non-blocking per attempt and polls until `timeout`, returning
    False rather than raising: memory is never load-bearing, so a contended
    store degrades (the caller proceeds and logs) instead of hanging a run.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None

    def _take(self, fd: int) -> None:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, 0, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def _drop(self, fd: int) -> None:
        try:
            if os.name == "nt":
                import msvcrt
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:
            pass

    def acquire(self, timeout: float = _FILE_LOCK_TIMEOUT,
                poll: float = 0.05) -> bool:
        if self._fd is not None:
            return True
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT
                     | getattr(os, "O_BINARY", 0), 0o600)
        deadline = time.monotonic() + max(0.0, timeout)
        while True:
            try:
                self._take(fd)
            except OSError:
                if time.monotonic() >= deadline:
                    os.close(fd)
                    return False
                time.sleep(poll)
                continue
            self._fd = fd
            return True

    def release(self) -> None:
        fd, self._fd = self._fd, None
        if fd is None:
            return
        self._drop(fd)
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> "_FileLock":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


def clean_rows(rows) -> list[dict]:
    """Validate memory rows from a backup document. Pure: raises, never writes.

    A restore must check the WHOLE document before applying any of it, or a
    corrupt tail leaves the store half-replaced — and the store is the only
    copy of months of learned rules. IMP-12.
    """
    clean: list[dict] = []
    for r in rows or []:
        if not isinstance(r, dict):
            raise ValueError("memory rows must be JSON objects")
        mid = str(r.get("id") or "").strip()
        kind = str(r.get("kind") or "").strip()
        text = str(r.get("text") or "").strip()
        if not (mid and kind and text):
            raise ValueError("a memory row needs a non-empty id, kind and text")
        clean.append({**r, "id": mid, "kind": kind, "text": text})
    return clean


class MemoryStore:
    """Append-only JSONL. One memory per line."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        # One lock per store, held across the WHOLE read-modify-write by every
        # writer (add, score_memories, add_consolidated). Reentrant because
        # add_consolidated holds it while calling add(). Real cross-thread
        # writers exist: serve.py dispatches scoring through to_thread into
        # tools.py, and asyncio.ensure_future runs add_consolidated alongside.
        self._lock = threading.RLock()
        # AUDIT 10-1r: the same read-modify-write is also reachable from a
        # second PROCESS, which the RLock cannot see. Held for the whole
        # transaction, and re-entrant within one thread (add_consolidated holds
        # it while calling add) via the depth counter.
        self.lock_path = self.path.with_name(f"{self.path.name}.lock")
        self._flock = _FileLock(self.lock_path)
        self._flock_depth = 0

    def _xlock(self):
        """Both locks, for the duration of one read-modify-write.

        RLock first (cheap, in-process), then the file lock only at the
        outermost depth — a nested acquire from the same thread must not open a
        second descriptor, which the OS would (correctly) see as a conflict and
        block against ourselves.
        """
        @contextmanager
        def _held():
            with self._lock:
                if self._flock_depth == 0:
                    if not self._flock.acquire():
                        log.warning(
                            "memory: could not take the cross-process lock on "
                            "%s — proceeding unsynchronised", self.lock_path)
                self._flock_depth += 1
                try:
                    yield
                finally:
                    self._flock_depth -= 1
                    if self._flock_depth == 0:
                        self._flock.release()
        return _held()

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
        # write-then-rename, never truncate-in-place: write_text() empties the
        # file before refilling it, so a crash mid-write (or the box losing
        # power 19 hours into a run) would leave ZERO memories where months of
        # accumulated rules used to be. os.replace is atomic on Windows and
        # POSIX — the store is always either the old rows or the new rows.
        # The temp name is unique per write: the old fixed memories.tmp let two
        # concurrent writers replace/delete each other's file mid-write
        # (PermissionError WinError 32) and lose one of the writes.
        with self._xlock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_name(
                f"{self.path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
            try:
                tmp.write_text("".join(json.dumps(r) + "\n" for r in rows),
                               encoding="utf-8")
                os.replace(tmp, self.path)
            except BaseException:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                raise

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
        # Held across the read AND the write: two threads that both read the
        # file and then both write it back lose one add entirely. A plain
        # interleaving, single event loop or not. _xlock also takes the
        # cross-process file lock (AUDIT 10-1r).
        with self._xlock():
            rows = self.all()
            for r in rows:
                if r.get("kind") == kind and r.get("text") == text:
                    r["seen_count"] = r.get("seen_count", 1) + 1
                    r["last_seen"] = time.time()
                    self._write_all(rows)
                    return r
            import hashlib
            rec = {"id": hashlib.sha1(f"{kind}:{text}".encode("utf-8", "replace"))
                   .hexdigest()[:12],
                   "kind": kind, "text": text, "status": "draft",
                   "seen_count": 1, "outcome_score": 0,
                   "vec": embed_one(text, purpose="doc"),
                   "born": time.time(), "last_seen": time.time(), **extra}
            rows.append(rec)
            # AUDIT 10-10: bounded for EVERY kind, with the same least-proven
            # eviction key. Nothing but this new line changed if nothing was
            # evicted, so append it (O(1)) instead of rewriting the whole store
            # on every add — the cost used to grow with the store.
            if _cap_rows(rows):
                self._write_all(rows)
            else:
                self._append(rec)
            return rec

    def _append(self, rec: dict) -> None:
        """Append ONE record. Only valid inside _xlock, and only when no
        existing row changed — dedup hits and evictions still go through
        _write_all. A crash mid-append can leave a torn trailing line, which
        `all()` skips; it can never lose a memory the way truncate-in-place
        would."""
        with self._xlock():
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec) + "\n")

    # --- inspect / correct / forget (the trust surface) ----------------------
    # Rigma silently learns rules and injects them into later prompts, so the
    # user must be able to see what it believes, fix a wrong rule and delete an
    # obsolete one. Every mutation is a read-modify-write under ONE lock: a fix
    # that raced a scoring pass would otherwise either lose the fix or write the
    # old text back.

    # Fields a person may correct. `id` is the row's identity and must stay put;
    # `vec` must track `text`, so it is recomputed rather than edited; `born`,
    # `last_seen` and the counters are evidence, not opinion.
    EDITABLE = ("text", "kind", "status", "outcome_score")
    STATUSES = ("draft", "verified", "retired")
    _KIND_RE = re.compile(r"^[a-z][a-z0-9_-]{0,31}$")

    def update(self, mid: str, **fields) -> dict | None:
        """Correct one memory in place. Returns the updated row, or None when
        no such id. Raises ValueError for a value the store would refuse."""
        changes = {k: v for k, v in fields.items() if k in self.EDITABLE}
        if not changes:
            return None
        with self._xlock():
            rows = self.all()
            hit = next((r for r in rows if r.get("id") == mid), None)
            if hit is None:
                return None
            if "text" in changes:
                text = str(changes["text"] or "").strip()
                if not text:
                    raise ValueError("empty memory")
                changes["text"] = text
            kind = str(changes.get("kind", hit.get("kind", "")) or "").strip()
            if "kind" in changes:
                if not self._KIND_RE.match(kind):
                    raise ValueError(
                        "kind: 1-32 chars of a-z, 0-9, _ or -")
                changes["kind"] = kind
            if "status" in changes:
                status = str(changes["status"] or "")
                if status not in self.STATUSES:
                    raise ValueError("status: must be one of "
                                     + "/".join(self.STATUSES))
                changes["status"] = status
            if "outcome_score" in changes:
                try:
                    changes["outcome_score"] = int(changes["outcome_score"])
                except (TypeError, ValueError):
                    raise ValueError("outcome_score: must be an integer") from None
            # The same anchoring guard `add` applies: an edit is another way to
            # put a raw failure transcript into a behavioural rule.
            if ("text" in changes or "kind" in changes) \
                    and kind not in _GUARD_EXEMPT_KINDS \
                    and looks_like_raw_trace(
                        str(changes.get("text", hit.get("text", "")))):
                raise ValueError(
                    "refusing to store a raw trace as a behavioural rule: "
                    "distil it into an imperative first")
            hit.update(changes)
            if "text" in changes:
                hit["vec"] = embed_one(hit["text"], purpose="doc")
            hit["edited"] = time.time()
            self._write_all(rows)
            return hit

    def delete(self, mid: str) -> bool:
        """Forget one memory. Returns whether anything was removed."""
        with self._xlock():
            rows = self.all()
            keep = [r for r in rows if r.get("id") != mid]
            if len(keep) == len(rows):
                return False
            self._write_all(keep)
            return True

    def delete_many(self, ids) -> int:
        """Forget a selection. An empty selection removes NOTHING — a prune
        that lost its ids must not empty the store."""
        want = {str(i) for i in (ids or []) if str(i)}
        if not want:
            return 0
        with self._xlock():
            rows = self.all()
            keep = [r for r in rows if r.get("id") not in want]
            gone = len(rows) - len(keep)
            if gone:
                self._write_all(keep)
            return gone

    def restore(self, rows) -> int:
        """Replace the store with `rows` (a backup). Returns the row count.

        Validated in full BEFORE anything is written: a backup that is corrupt
        half-way through must leave the existing memories untouched, because
        the store is the only copy of months of learned rules. IMP-12.
        """
        clean = clean_rows(rows)
        with self._xlock():
            self._write_all(clean)
        return len(clean)


# --- embeddings (optional, never load-bearing) -------------------------------

# Providers in preference order. nomic-embed-text-v1.5 is the research pick
# (task prefixes match the short-rule/long-query asymmetry; owner authorised
# its download 2026-07-20); bge-small is the fallback already on disk from
# Raggity. STRICTLY offline at import: HF_HUB_OFFLINE is set before fastembed
# loads, so this module itself can never start a download — a missing model
# means lexical-only, which is a working (if weaker) retrieval mode, not an
# error.
_EMBED_MODELS = ["nomic-ai/nomic-embed-text-v1.5", "BAAI/bge-small-en-v1.5"]
_embedder = None
_embedder_tried = False


def get_embedder():
    global _embedder, _embedder_tried
    if _embedder_tried:
        return _embedder
    _embedder_tried = True
    if os.environ.get("RIGMA_MEMORY_EMBED") == "0":
        return None
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    cache = os.path.join(os.environ.get("TEMP", ""), "fastembed_cache")
    try:
        from fastembed import TextEmbedding
        for name in _EMBED_MODELS:
            try:
                _embedder = TextEmbedding(name, cache_dir=cache)
                log.info("memory: dense retrieval via %s", name)
                break
            except Exception:
                continue
    except Exception:
        _embedder = None
    if _embedder is None:
        log.info("memory: no cached embedding model — lexical retrieval only")
    return _embedder


def embed_one(text: str, purpose: str = "doc") -> list | None:
    """purpose: "doc" for a stored rule, "query" for a step description.
    nomic is trained with asymmetric task prefixes — a short imperative rule
    retrieved by a longer task description is exactly the asymmetry they
    exist for. bge ignores unknown prefixes gracefully, so prefix only when
    the active provider is nomic."""
    emb = get_embedder()
    if emb is None:
        return None
    text = str(text)
    if "nomic" in getattr(emb, "model_name", ""):
        text = ("search_query: " if purpose == "query"
                else "search_document: ") + text
    try:
        vec = next(iter(emb.embed([text])))
        return [round(float(x), 5) for x in vec]
    except Exception:
        return None


def _cos(a, b) -> float:
    try:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        return num / (da * db) if da and db else 0.0
    except Exception:
        return 0.0


# --- retrieval (phase 2): metadata filter FIRST, then hybrid ------------------

# ambient cosine between UNRELATED texts on the active embedder (anisotropy).
# Measured on nomic-embed-text-v1.5, 2026-07-21: unrelated 0.39-0.45,
# related 0.60. Dense contributions are rescaled so this reads as zero.
_DENSE_BASELINE = 0.40

_WORD = re.compile(r"[a-z0-9_]+")


def _toks(text: str) -> list[str]:
    """Tokens plus two normalisations the replay eval proved necessary:
    naive plural-stripping ("deliverable" must hit "Deliverables"), and
    snake_case splitting so "view the sampled images" reaches the rule that
    names view_sample — tool names ARE the rare entities retrieval exists
    to catch, and treating them as one opaque token hid them."""
    out = []
    for t in _WORD.findall(str(text).lower()):
        out.append(t)
        if "_" in t:
            out.extend(p for p in t.split("_") if len(p) > 2)
    return [t[:-1] if len(t) > 3 and t.endswith("s") else t for t in out]


def _bm25(query_toks: list[str], docs: list[list[str]],
          k1: float = 1.5, b: float = 0.75) -> list[float]:
    """Tiny BM25 — the store is ≤ a few dozen one-line rules, so a real index
    would be machinery for machinery's sake. Keyword scoring is load-bearing
    here, not a nicety: the rare entities ARE the memory (tool names, flags,
    error strings) and dense embeddings flatten exactly those."""
    import math
    n = len(docs)
    if not n:
        return []
    avg = sum(len(d) for d in docs) / n or 1.0
    df: dict[str, int] = {}
    for d in docs:
        for t in set(d):
            df[t] = df.get(t, 0) + 1
    scores = []
    for d in docs:
        s = 0.0
        for t in query_toks:
            f = d.count(t)
            if not f:
                continue
            idf = math.log(1 + (n - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5))
            s += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * len(d) / avg))
        scores.append(s)
    return scores


def retrieve(rows: list[dict], query: str, kinds: tuple = ("pitfall",
             "technique", "project"), workspace: str = "", k: int = 3) -> list[dict]:
    """Top-k memories for a step. Hard metadata filter BEFORE any scoring —
    post-filtering can return a top-3 entirely from the wrong scope and then
    discard it, a silent recall failure indistinguishable from an empty store."""
    pool = [r for r in rows
            if r.get("kind") in kinds and r.get("status") != "retired"
            and (r.get("kind") != "project" or not r.get("workspace")
                 or r.get("workspace") == workspace)]
    if not pool or not str(query).strip():
        return []
    q = _toks(query)
    lex = _bm25(q, [_toks(r.get("text", "")) for r in pool])
    top_lex = max(lex) if lex and max(lex) > 0 else 1.0
    qv = embed_one(query, purpose="query")
    scored = []
    for r, ls in zip(pool, lex):
        s = 0.5 * (ls / top_lex)
        if qv and r.get("vec"):
            # Anisotropy correction. Embedding spaces are cones, not spheres:
            # two UNRELATED sentences score ~0.39-0.45 cosine on nomic
            # (measured here 2026-07-21), so raw cos * 0.5 put every random
            # memory over the old 0.15 floor and the filter passed everything
            # the moment dense was available — injecting garbage that then got
            # falsely punished by outcome scoring. Subtract the ambient
            # baseline so zero means "unrelated", not "orthogonal".
            s += 0.5 * max(0.0, (_cos(qv, r["vec"]) - _DENSE_BASELINE)
                           / (1.0 - _DENSE_BASELINE))
        scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for s, r in scored[:k] if s >= 0.12]


# --- distillation ------------------------------------------------------------

_DISTIL_PROMPT = (
    "You turn one observed agent failure into ONE reusable rule.\n"
    "Write a single imperative sentence, under 90 characters, telling a future "
    "agent what to do instead.\n"
    "NEVER mention specific filenames, paths, or arguments — a rule about one "
    "file is useless. Generalise to the CLASS of mistake.\n"
    "Reply with the sentence only. No preamble, no quotes.\n\n"
    "Example observation: view_images failed on a hand-typed path, then "
    "view_sample succeeded.\n"
    "Example rule: Never type filenames; pass files by reference with "
    "view_sample.\n\n"
    "Observation: ")


def describe_event(event: dict) -> str:
    """One line of evidence for the distiller. Stays inside this module — it
    carries raw args and must never reach the store."""
    if event.get("kind") == "loop":
        return (f"{event.get('tool')} was called with identical arguments "
                f"{event.get('count')} times and failed every time "
                f"(args: {event.get('args')})")
    return (f"{event.get('failed_tool')} failed (args: "
            f"{event.get('failed_args')}), then {event.get('worked_tool')} "
            "succeeded")


def clean_rule(text: str) -> str:
    """Normalise whatever the distiller replied with into one rule line.

    Not simply the first line: a chatty quantised model answers
    "Here is the rule:\nNever type filenames." despite the prompt, and taking
    line one would store the PREAMBLE as a behavioural rule — junk that passes
    the anchoring guard (no paths in it) and squats in the capped store. Pick
    the first line that reads like an imperative: several words, not ending in
    a colon, not an announcement about the rule it precedes.
    """
    text = (text or "").strip()
    if not text:
        return ""
    lines = [ln.strip().strip('"') for ln in text.splitlines() if ln.strip()]
    _PREAMBLE = re.compile(
        r"^(here('s| is| are)|sure|certainly|the (rule|answer)|answer\b|ok(ay)?\b)",
        re.I)
    for ln in lines:
        if ln.endswith(":") or _PREAMBLE.match(ln):
            continue
        if len(ln.split()) >= 3:
            return ln[:200]
    return lines[0].rstrip(":")[:200] if lines else ""


async def distil(event: dict, complete) -> str:
    """Ask the model to generalise one event into a rule.

    `complete` is an async callable taking a prompt and returning text —
    injected so this is testable without an engine, and swappable for a
    stronger model later exactly as the mission compiler is.
    """
    try:
        return clean_rule(await complete(_DISTIL_PROMPT + describe_event(event)))
    except Exception:
        return ""


async def harvest_run(actions: list[dict], store: MemoryStore, complete,
                      max_rules: int = 3, run_id: str = "") -> list[dict]:
    """Mine a finished run's trace and store what can be distilled.

    THE production entry point — serve.py calls exactly this, so the path the
    tests exercise is the path that ships. An earlier revision had serve
    hand-rolling its own copy of this loop against a private prompt constant,
    which meant the tested code and the running code had quietly diverged.

    Bounded on purpose: a bad run can produce dozens of events, and writing a
    rule for each would swamp the store with near-duplicates. Never raises —
    memory is not load-bearing, and a run that already ended must not report a
    failure because its post-mortem failed.
    """
    written: list[dict] = []
    try:
        events = mine_events(actions)
    except Exception:
        # non-fatal by design, but NEVER invisible: an unlogged broad except
        # already hid a NameError here for a whole session, during which every
        # test passed while memory silently did nothing.
        log.exception("memory: event mining failed")
        return written
    for event in events[:max_rules]:
        rule = await distil(event, complete)
        if not rule:
            continue
        try:
            rec = await add_consolidated(store, "pitfall", rule, complete,
                                         run_id=run_id)
            if rec:
                written.append(rec)
        except ValueError as e:
            # the distiller leaked a path or a literal argument. Dropping it is
            # correct: an un-generalised rule is worthless at best, and the
            # guard exists because a raw trace actively teaches the failure.
            log.info("memory: guard rejected a rule (%s)", e)
            continue
        except Exception:
            log.exception("memory: store write failed")
            continue
    return written


# --- outcome tracking (phase 3) ----------------------------------------------

def score_memories(store: MemoryStore, ids: list[str], delta: int,
                   run_id: str = "") -> None:
    """The librarian's ledger. +1 when a step a memory was injected into
    succeeds, -2 when it fails: a rule must earn its place repeatedly but can
    be discredited quickly. This is the ONLY thing that retires a wrong rule —
    a bad rule that keeps matching keeps being retrieved, and without outcome
    scoring, being wrong made it more prominent.

    NO TIME DECAY EXISTS. An earlier version of this docstring said "time decay
    retires unused memories"; it does not, and never has. `born` is written
    (see `MemoryStore.add`) and read nowhere, and `last_seen` is read only as
    the fourth tiebreak when evicting at the cap. An unused memory is retired by
    nothing but the cap and the Memory UI's delete button. Said plainly here
    because a comment describing a safeguard that does not exist is worse than
    no comment: it is the reason nobody went looking (audit 2026-08-18).

    Graduation rides on the same signal: a draft that helped a run OTHER than
    the one that wrote it has proven it generalises, which is the exact claim
    "verified" makes."""
    if not ids:
        return
    try:
        # lock across the whole read-modify-write: scoring read the file, then
        # wrote the snapshot back, so a +1 committed while add_consolidated
        # awaited the conflict gate was silently clobbered. _xlock adds the
        # cross-process file lock (AUDIT 10-1r).
        with store._xlock():
            rows = store.all()
            hit = False
            for r in rows:
                if r.get("id") in ids:
                    r["outcome_score"] = r.get("outcome_score", 0) + delta
                    r["last_seen"] = time.time()
                    if (delta > 0 and r.get("status") == "draft"
                            and run_id and r.get("born_run")
                            and r["born_run"] != run_id):
                        r["status"] = "verified"
                        log.info("memory: %r graduated to verified",
                                 r.get("text", "")[:60])
                    hit = True
            if hit:
                store._write_all(rows)
    except Exception:
        log.exception("memory: outcome scoring failed")


# --- conflict-gated consolidation (phase 2) ----------------------------------

_CONFLICT_PROMPT = (
    "Two rules for an autonomous agent are shown. Answer with ONE word.\n"
    "Answer CONFLICT if an agent cannot follow both (they demand opposite "
    "actions in the same situation).\n"
    "Answer DUPLICATE if they tell the agent the same thing in different "
    "words.\n"
    "Answer DISTINCT otherwise.\n\n"
    "Rule A: {a}\nRule B: {b}\n\nAnswer:")

# similarity NOMINATES; it never decides. "prefer q8_0 cache" vs "prefer f16
# cache" score >0.95 — identical syntax, opposite instruction — so merging on
# cosine alone would make a rule MORE authoritative for having just been
# contradicted. The gate is one word from the model; without an answer we
# append rather than merge, and the cap bounds the bloat.
_NOMINATE_COS = 0.75


async def add_consolidated(store: MemoryStore, kind: str, text: str,
                           complete, run_id: str = "") -> dict | None:
    """Add a memory through the semantic pipeline. Falls back to plain add()
    when there is nothing to consolidate against or no engine to ask."""
    text = clean_rule(text) if kind == "pitfall" else str(text or "").strip()
    if not text:
        return None
    with store._xlock():
        rows = store.all()
        for r in rows:                  # exact text: reinforce, no LLM needed
            if r.get("kind") == kind and r.get("text") == text:
                return store.add(kind=kind, text=text, born_run=run_id)
    nv = embed_one(text, purpose="doc")
    best, best_cos = None, 0.0
    if nv:
        for r in rows:
            if r.get("kind") != kind or r.get("status") == "retired" \
                    or not r.get("vec"):
                continue
            c = _cos(nv, r["vec"])
            if c > best_cos:
                best, best_cos = r, c
    if best is None or best_cos < _NOMINATE_COS or complete is None:
        return store.add(kind=kind, text=text, born_run=run_id)
    try:
        verdict = str(await complete(_CONFLICT_PROMPT.format(
            a=best.get("text", ""), b=text)) or "").strip().upper()
    except Exception:
        verdict = ""
    # substring, not startswith: a quantised model pads despite the prompt
    # ("Answer: CONFLICT"), and startswith silently defaulted every padded
    # verdict to DISTINCT — bypassing consolidation entirely. CONFLICT is
    # checked first: a reply naming both words is treating them as options,
    # and the destructive reading must not win by accident of ordering — but
    # between the two, a false append (DISTINCT) is recoverable and a false
    # merge is not, so ambiguity falls through to append.
    is_conflict = "CONFLICT" in verdict and "DUPLICATE" not in verdict
    is_duplicate = "DUPLICATE" in verdict and "CONFLICT" not in verdict
    # Both branches below RE-READ inside the lock. `rows`/`best` were read
    # before the await; writing that snapshot back clobbered any outcome score
    # (a +1 from a completed step) committed while the gate was answering.
    if is_conflict:
        # The NEW observation supersedes — but how far depends on the old
        # rule's standing. A wrong CONFLICT verdict against a VERIFIED rule
        # would permanently destroy proven capability in favour of an untested
        # draft (the worst possible trade), so verified rules are DEMOTED to
        # draft rather than retired: still retrievable, must re-earn their
        # status. Only drafts die outright.
        with store._xlock():
            rows = store.all()
            for r in rows:
                if r.get("id") != best.get("id"):
                    continue
                if r.get("status") == "verified":
                    r["status"] = "draft"
                    log.info("memory: %r demoted by conflict with %r",
                             r.get("text", "")[:50], text[:50])
                else:
                    r["status"] = "retired"
                    log.info("memory: %r superseded %r", text[:50],
                             r.get("text", "")[:50])
                break
            store._write_all(rows)
        return store.add(kind=kind, text=text, born_run=run_id)
    if is_duplicate:
        with store._xlock():
            rows = store.all()
            for r in rows:
                if r.get("id") != best.get("id"):
                    continue
                r["seen_count"] = r.get("seen_count", 1) + 1
                r["last_seen"] = time.time()
                best = r
                break
            store._write_all(rows)
        return best
    return store.add(kind=kind, text=text, born_run=run_id)


# --- reading it back ---------------------------------------------------------

def render_pitfall_block(memories: list[dict], limit: int = 5,
                         include_drafts: bool = True) -> str:
    """The run-start block: terse imperatives, never prose.

    2026-07-19 proved that anything discursive injected into the loop gets
    narrated back instead of acted on, so this stays a bulleted list of rules
    and nothing else. An empty store renders nothing at all — an empty header
    would just be context the model feels invited to comment on.

    Drafts are shown (nothing can graduate until phase 3 builds outcome
    tracking; verified-only would render an empty block forever) but they are
    NOT labelled. An earlier revision prefixed drafts with "UNVERIFIED:", and
    review killed it: under a header saying "these are rules, not suggestions"
    the hedge is an epistemic contradiction a sub-40B model resolves badly —
    it either ignores the label (quarantine meaningless) or fixates on it and
    narrates its uncertainty, or worse, decides to TEST the unverified rule.
    What gets pinned gets committed to; the real quarantine is the store cap
    and, in phase 3, graduation. `status: draft` stays in the data model.
    """
    rows = [m for m in memories or [] if m.get("kind") == "pitfall"]
    if not include_drafts:
        rows = [m for m in rows if m.get("status") == "verified"]
    if not rows:
        return ""
    rows.sort(key=lambda m: (m.get("status") == "verified",
                             m.get("outcome_score", 0),
                             m.get("seen_count", 0)), reverse=True)
    lines = ["WHAT YOU LEARNED BEFORE — these are rules, not suggestions:"]
    for m in rows[:limit]:
        lines.append(f"  • {m.get('text', '')}")
    return "\n".join(lines)
