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
import time
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
        # write-then-rename, never truncate-in-place: write_text() empties the
        # file before refilling it, so a crash mid-write (or the box losing
        # power 19 hours into a run) would leave ZERO memories where months of
        # accumulated rules used to be. os.replace is atomic on Windows and
        # POSIX — the store is always either the old rows or the new rows.
        # (No cross-process lock: one Rigma process, one active run at a time,
        # and the store is only written from the run loop's post-mortem.)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r) + "\n" for r in rows),
                       encoding="utf-8")
        os.replace(tmp, self.path)

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
        import hashlib
        rec = {"id": hashlib.sha1(f"{kind}:{text}".encode("utf-8", "replace"))
               .hexdigest()[:12],
               "kind": kind, "text": text, "status": "draft",
               "seen_count": 1, "outcome_score": 0,
               "vec": embed_one(text, purpose="doc"),
               "born": time.time(), "last_seen": time.time(), **extra}
        rows.append(rec)
        # bounded: evict the least-proven pitfall when over cap. Verified rules
        # outrank drafts; among equals, lowest outcome then lowest seen goes.
        pits = [r for r in rows if r.get("kind") == "pitfall"]
        if len(pits) > MAX_PITFALLS:
            evict = min(pits, key=lambda m: (m.get("status") == "verified",
                                             m.get("outcome_score", 0),
                                             m.get("seen_count", 0),
                                             m.get("last_seen", 0)))
            rows.remove(evict)
            log.info("memory: cap reached, evicted %r", evict.get("text", "")[:60])
        self._write_all(rows)
        return rec


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
            s += 0.5 * _cos(qv, r["vec"])
        scored.append((s, r))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [r for s, r in scored[:k] if s >= 0.15]


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
    """Normalise whatever the distiller replied with into one rule line."""
    text = (text or "").strip()
    if not text:
        return ""
    return text.strip('"').splitlines()[0].strip()[:200]


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
    be discredited quickly. Time decay retires unused memories; only THIS
    retires wrong ones — a bad rule that keeps matching keeps being retrieved,
    and without outcome scoring, being wrong made it more prominent.

    Graduation rides on the same signal: a draft that helped a run OTHER than
    the one that wrote it has proven it generalises, which is the exact claim
    "verified" makes."""
    if not ids:
        return
    try:
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
    rows = store.all()
    for r in rows:                      # exact text: reinforce, no LLM needed
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
    if verdict.startswith("CONFLICT"):
        # the NEW observation supersedes: it is more recent evidence. The old
        # rule is retired, never merged — a contradiction collapsed into one
        # row is how a superseded rule gains authority from being wrong.
        best["status"] = "retired"
        store._write_all([r if r.get("id") != best.get("id") else best
                          for r in rows])
        log.info("memory: %r superseded %r", text[:50],
                 best.get("text", "")[:50])
        return store.add(kind=kind, text=text, born_run=run_id)
    if verdict.startswith("DUPLICATE"):
        best["seen_count"] = best.get("seen_count", 1) + 1
        best["last_seen"] = time.time()
        store._write_all([r if r.get("id") != best.get("id") else best
                          for r in rows])
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
