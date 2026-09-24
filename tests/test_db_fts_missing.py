"""R3: a database without the FTS table must still be able to store a chat.

`has_fts()` defaults to True for any path it has not decided about, and the
only thing that ever decides is `connect()` — once per process, guarded by
`_initialised`. So a process that opens a database whose `session_fts` table is
absent believes FTS is available. That state is reachable without any exotic
setup: a build whose sqlite lacks FTS5 creates the `sessions` table and records
`_fts_available[key] = False` in that process only; the file itself is left
without `session_fts`. The next process to open it (a Python with FTS5, a
restart, a different interpreter) defaults to True and then runs

    DELETE FROM session_fts WHERE id=?

for every save. `upsert_session` had no guard, so the OperationalError escaped
as a 500 — the chat was never stored at all. `search_sessions` already degrades
to LIKE, which is what the missing table is supposed to mean.
"""
import sqlite3

import pytest

from rigma import db, sessions


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _drop_the_fts_table():
    c = sqlite3.connect(db.db_path())
    c.execute("DROP TABLE IF EXISTS session_fts")
    c.commit()
    c.close()
    # a fresh process's view: nothing decided about this path yet
    db._fts_available.clear()


def test_a_session_still_saves_when_the_fts_table_is_missing(home):
    sessions.create("first")            # initialises the db, creates the table
    _drop_the_fts_table()
    s = sessions.create("after the drop")
    s["messages"].append({"role": "user", "content": "hello"})
    sessions.save(s)                    # used to raise OperationalError
    assert sessions.load(s["id"])["messages"] == [
        {"role": "user", "content": "hello"}]


def test_a_session_still_deletes_when_the_fts_table_is_missing(home):
    s = sessions.create("doomed")
    _drop_the_fts_table()
    assert sessions.delete(s["id"]) is True
    assert sessions.load(s["id"]) is None


def test_search_degrades_to_like_when_the_fts_table_is_missing(home):
    s = sessions.create("findable")
    s["messages"].append({"role": "user", "content": "a rare phrase"})
    sessions.save(s)
    _drop_the_fts_table()
    hits = sessions.search("rare phrase")
    assert [h["id"] for h in hits] == [s["id"]]


def test_the_flag_is_cleared_so_later_writes_do_not_retry_the_missing_table(home):
    sessions.create("first")
    _drop_the_fts_table()
    assert db.has_fts() is True        # the wrong belief, before the first write
    s = sessions.create("second")
    sessions.save(s)
    assert db.has_fts() is False, (
        "the missing table must be remembered, not rediscovered on every save")
    sessions.save(s)                   # and the next save must be clean too
