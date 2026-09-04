"""SQLite backing store for chat sessions (~/.rigma/rigma.db, WAL mode).

Why: sessions were one JSON file per chat — fine as storage, but everything
that reads ACROSS chats re-parsed the entire corpus per call: the sidebar
re-loaded every file on every render, and search was a full linear scan per
keystroke. A heavy user (long fiction sessions, images as base64 parts) hits
multi-second sidebar loads within months. One indexed table + an FTS5 index
turns both into single queries.

Scope deliberately narrow (audit 2026-07-21): sessions ONLY. state.json,
calibration.json and run artifacts stay files — cross-process visibility and
hand-debuggability are features there, and none has a cross-entity read
pattern. Memory stays JSONL: it's small, append-only, and defensively coded.

Legacy chat files are imported once per database and LEFT IN PLACE as a
backup; they are never written again.

stdlib sqlite3, zero new dependencies. Every call opens its own connection —
serve.py touches sessions from worker threads, and per-call connections with
WAL are the boring, correct answer.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from .runtime import rigma_home

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions(
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL DEFAULT '',
  updated_at REAL NOT NULL DEFAULT 0,
  use_rag INTEGER NOT NULL DEFAULT 0,
  message_count INTEGER NOT NULL DEFAULT 0,
  body TEXT NOT NULL,
  rev INTEGER NOT NULL DEFAULT 0,
  fts_hash TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC);
"""
# AUDIT F1: docs/audit-2026-09-04-full.md
# Columns added after the table shipped. ALTER TABLE ... ADD COLUMN is the
# entire migration: SQLite records it in the schema without rewriting the
# table, existing rows read back with the DEFAULT (rev 0, no index hash), and
# a database written by any earlier build keeps every row it had.
_MIGRATIONS = (
    ("rev", "ALTER TABLE sessions ADD COLUMN rev INTEGER NOT NULL DEFAULT 0"),
    ("fts_hash",
     "ALTER TABLE sessions ADD COLUMN fts_hash TEXT NOT NULL DEFAULT ''"),
)

# The revision a session was loaded at travels on the session dict under this
# key. Leading underscore, and no member of sessions._SESSION_DEFAULTS, so it
# cannot collide with a session field; upsert_session strips it, so it never
# reaches the stored body.
REV_KEY = "_rev"
_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS session_fts
USING fts5(id UNINDEXED, title, text);
"""

# init tracked per db path, NOT a module global: tests point RIGMA_HOME at a
# fresh tmp dir per test and a global flag would skip their schema creation
_initialised: set[str] = set()
_fts_available: dict[str, bool] = {}


def db_path() -> Path:
    return rigma_home() / "rigma.db"


def connect() -> sqlite3.Connection:
    p = db_path()
    # RIGMA_HOME may point at a directory nothing has created yet — the old
    # file store's chats_dir() used to mkdir it as a side effect
    p.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(p, timeout=10)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=NORMAL")
    key = str(p)
    if key not in _initialised:
        c.executescript(_SCHEMA)
        have = {r[1] for r in c.execute("PRAGMA table_info(sessions)")}
        for col, ddl in _MIGRATIONS:
            if col not in have:
                c.execute(ddl)
        try:
            c.executescript(_FTS_SCHEMA)
            _fts_available[key] = True
        except sqlite3.OperationalError:
            # an sqlite built without FTS5: search degrades to LIKE, nothing
            # else changes
            _fts_available[key] = False
        c.commit()
        _initialised.add(key)
    return c


def has_fts() -> bool:
    return _fts_available.get(str(db_path()), True)


def _fts_text(session: dict, cap: int = 1_000_000) -> str:
    """The searchable text of a session: title + message bodies (vision parts
    contribute their text only).

    AUDIT F16: docs/audit-2026-09-04-full.md — `total` replaces a
    `sum(len(x) for x in parts)` re-walked once per message. That rescan is
    O(n^2) in the message count and measured 206 ms on a ~4000-message
    session, on the same event loop that carries the token stream. The
    accumulator is the number the sum produced, so the text is unchanged.
    """
    parts = [str(session.get("title") or "")]
    total = len(parts[0])
    for m in session.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        text = str(content or "")
        parts.append(text)
        total += len(text)
        if total > cap:
            break
    return "\n".join(parts)[:cap]


def _fts_digest(text: str) -> str:
    """Fingerprint of the text an FTS row was built from. Hashing 1 MB costs
    ~1 ms against the ~105 ms the FTS5 rewrite costs, so it is worth paying to
    answer "did anything searchable actually change?" — a params, notes or
    preset save touches no indexed byte, and a mid-session message edit still
    changes the digest and gets reindexed."""
    return hashlib.blake2b(text.encode("utf-8", "surrogatepass"),
                           digest_size=16).hexdigest()


def upsert_session(session: dict, base_rev: int | None = None) -> int | None:
    """Write the session; return its new revision, or None if `base_rev` lost.

    AUDIT F1: docs/audit-2026-09-04-full.md — `base_rev` is the lost-update
    guard. Given one, the row is written ONLY while its stored rev is still
    exactly that value, and the test lives in the WHERE of the single UPDATE
    that does the write: a SELECT to check followed by a separate UPDATE is
    the same race in a smaller window. None (every caller written before the
    guard existed) keeps the unconditional whole-row upsert this store has
    always done. A missing row counts as lost as well — the delete is itself
    a write that moved the session past `base_rev`, and re-inserting would
    resurrect a chat the user removed.
    """
    sid = session["id"]
    # the revision is bookkeeping ON the row; the stored document stays byte
    # for byte what it was before the column existed
    body = json.dumps({k: v for k, v in session.items() if k != REV_KEY},
                      indent=2)
    title = str(session.get("title") or "")
    row = (title, float(session.get("updated_at") or 0),
           1 if session.get("use_rag") else 0,
           len(session.get("messages", [])), body)
    text = _fts_text(session) if has_fts() else ""
    digest = _fts_digest(text) if has_fts() else ""
    with connect() as c:
        # IMMEDIATE: the rev test, the row write and the index write are one
        # write transaction, so nothing can land between them
        c.execute("BEGIN IMMEDIATE")
        prev = c.execute("SELECT rev, fts_hash FROM sessions WHERE id=?",
                         (sid,)).fetchone()
        if base_rev is None:
            c.execute(
                "INSERT INTO sessions(id, title, updated_at, use_rag, "
                "message_count, body, rev, fts_hash) "
                "VALUES(?,?,?,?,?,?,0,?) "
                "ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
                "updated_at=excluded.updated_at, use_rag=excluded.use_rag, "
                "message_count=excluded.message_count, body=excluded.body, "
                "fts_hash=excluded.fts_hash, rev=sessions.rev+1",
                (sid, *row, digest))
            new_rev = 0 if prev is None else int(prev[0]) + 1
        else:
            cur = c.execute(
                "UPDATE sessions SET title=?, updated_at=?, use_rag=?, "
                "message_count=?, body=?, fts_hash=?, rev=rev+1 "
                "WHERE id=? AND rev=?",
                (*row, digest, sid, int(base_rev)))
            if cur.rowcount != 1:
                c.execute("ROLLBACK")
                return None
            new_rev = int(base_rev) + 1
        if has_fts() and (prev is None or prev[1] != digest):
            c.execute("DELETE FROM session_fts WHERE id=?", (sid,))
            c.execute("INSERT INTO session_fts(id, title, text) "
                      "VALUES(?,?,?)", (sid, title, text))
    return new_rev


def get_session_row(session_id: str) -> tuple[str, int] | None:
    """(body, rev) for a session, or None. The rev is what a later
    `upsert_session(..., base_rev=rev)` tests against."""
    with connect() as c:
        row = c.execute("SELECT body, rev FROM sessions WHERE id=?",
                        (session_id,)).fetchone()
    return (row[0], int(row[1])) if row else None


def get_session_body(session_id: str) -> str | None:
    row = get_session_row(session_id)
    return row[0] if row else None


def delete_session(session_id: str) -> bool:
    with connect() as c:
        cur = c.execute("DELETE FROM sessions WHERE id=?", (session_id,))
        if has_fts():
            c.execute("DELETE FROM session_fts WHERE id=?", (session_id,))
    return cur.rowcount > 0


def list_summaries() -> list[dict]:
    with connect() as c:
        rows = c.execute(
            "SELECT id, title, updated_at, use_rag, message_count "
            "FROM sessions ORDER BY updated_at DESC").fetchall()
    return [{"id": r[0], "title": r[1], "updated_at": r[2],
             "use_rag": bool(r[3]), "message_count": r[4]} for r in rows]


def known_ids() -> set[str]:
    with connect() as c:
        return {r[0] for r in c.execute("SELECT id FROM sessions")}


def search_sessions(query: str) -> list[tuple[str, str]]:
    """(session_id, snippet) hits, newest first. FTS5 prefix match when
    available; substring LIKE over the stored body as the fallback (also used
    when FTS finds nothing — FTS can't match inside a word, and the old
    file-scan search could)."""
    q = query.strip()
    if not q:
        return []
    hits: list[tuple[str, str]] = []
    if has_fts():
        # quoted prefix query: user text is a literal, never FTS syntax
        fts_q = '"' + q.replace('"', '""') + '"*'
        try:
            with connect() as c:
                rows = c.execute(
                    "SELECT f.id, snippet(session_fts, 2, '', '', '…', 12) "
                    "FROM session_fts f JOIN sessions s ON s.id = f.id "
                    "WHERE session_fts MATCH ? "
                    "ORDER BY s.updated_at DESC LIMIT 50",
                    (fts_q,)).fetchall()
            hits = [(r[0], r[1]) for r in rows]
        except sqlite3.OperationalError:
            hits = []
    if not hits:
        like = f"%{q}%"
        with connect() as c:
            rows = c.execute(
                "SELECT id, body FROM sessions "
                "WHERE body LIKE ? OR title LIKE ? "
                "ORDER BY updated_at DESC LIMIT 50",
                (like, like)).fetchall()
        ql = q.lower()
        for sid, body in rows:
            pos = body.lower().find(ql)
            snippet = body[max(0, pos - 40):pos + len(q) + 60].strip() \
                if pos != -1 else q
            hits.append((sid, snippet))
    return hits
