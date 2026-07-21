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
  body TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS sessions_updated ON sessions(updated_at DESC);
"""
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
    contribute their text only)."""
    parts = [str(session.get("title") or "")]
    for m in session.get("messages", []):
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(p.get("text", "") for p in content
                               if isinstance(p, dict))
        parts.append(str(content or ""))
        if sum(len(x) for x in parts) > cap:
            break
    return "\n".join(parts)[:cap]


def upsert_session(session: dict) -> None:
    body = json.dumps(session, indent=2)
    with connect() as c:
        c.execute(
            "INSERT INTO sessions(id, title, updated_at, use_rag, "
            "message_count, body) VALUES(?,?,?,?,?,?) "
            "ON CONFLICT(id) DO UPDATE SET title=excluded.title, "
            "updated_at=excluded.updated_at, use_rag=excluded.use_rag, "
            "message_count=excluded.message_count, body=excluded.body",
            (session["id"], str(session.get("title") or ""),
             float(session.get("updated_at") or 0),
             1 if session.get("use_rag") else 0,
             len(session.get("messages", [])), body))
        if has_fts():
            c.execute("DELETE FROM session_fts WHERE id=?", (session["id"],))
            c.execute("INSERT INTO session_fts(id, title, text) "
                      "VALUES(?,?,?)",
                      (session["id"], str(session.get("title") or ""),
                       _fts_text(session)))


def get_session_body(session_id: str) -> str | None:
    with connect() as c:
        row = c.execute("SELECT body FROM sessions WHERE id=?",
                        (session_id,)).fetchone()
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
