"""Audit 2026-09-04, foundation findings: the lost-update guard on session
writes (F1/F2/F4/F15), the index cost per save (F16), the export that dropped
the compacted half of a manuscript (F6), the DRY allowance that never reached
the engine (F3), and the session id that could name a file outside ~/.rigma
(F40)."""
import json
import logging
import re
import sqlite3
import threading
from pathlib import Path

import pytest

from rigma import db, hangar, sessions


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


# --------------------------------------------------------------- F1: writes

def test_stale_writer_is_refused_and_the_first_write_survives():
    s = sessions.create("novel")
    first = sessions.load(s["id"])          # two writers, one snapshot each
    second = sessions.load(s["id"])
    first["messages"].append({"role": "user", "content": "chapter one"})
    sessions.save(first, base_rev=first[sessions.REV_KEY])

    second["messages"].append({"role": "user", "content": "clobber"})
    with pytest.raises(sessions.StaleWriteError):
        sessions.save(second, base_rev=second[sessions.REV_KEY])

    stored = sessions.load(s["id"])
    assert [m["content"] for m in stored["messages"]] == ["chapter one"]
    # the refused write left the row exactly where the winner put it
    assert stored[sessions.REV_KEY] == first[sessions.REV_KEY]


def test_unguarded_save_still_overwrites_exactly_as_before():
    """Every caller written before the guard passes no base_rev and must keep
    its old behaviour — last writer wins, stale snapshot or not."""
    s = sessions.create("novel")
    stale = sessions.load(s["id"])
    fresh = sessions.load(s["id"])
    fresh["messages"].append({"role": "user", "content": "landed first"})
    sessions.save(fresh)
    stale["messages"].append({"role": "user", "content": "old snapshot wins"})
    sessions.save(stale)

    got = sessions.load(s["id"])
    assert [m["content"] for m in got["messages"]] == ["old snapshot wins"]
    assert got[sessions.REV_KEY] == 2       # create, then two saves


def test_the_revision_never_reaches_the_stored_body():
    s = sessions.create("novel")
    loaded = sessions.load(s["id"])
    assert loaded[sessions.REV_KEY] == 0
    rev = sessions.save(loaded, base_rev=loaded[sessions.REV_KEY])
    assert rev == 1 and loaded[sessions.REV_KEY] == 1   # snapshot kept honest

    body = json.loads(db.get_session_body(s["id"]))
    assert sessions.REV_KEY not in body
    # and the key cannot collide with a session field
    assert sessions.REV_KEY not in sessions._SESSION_DEFAULTS
    assert sessions.REV_KEY not in sessions.MUTABLE_FIELDS


def test_a_guarded_save_never_resurrects_a_deleted_session():
    s = sessions.create("novel")
    snap = sessions.load(s["id"])
    sessions.delete(s["id"])
    with pytest.raises(sessions.StaleWriteError):
        sessions.save(snap, base_rev=snap[sessions.REV_KEY])
    assert sessions.load(s["id"]) is None


def test_two_threads_on_one_revision_leave_exactly_one_winner():
    s = sessions.create("novel")
    snaps = [sessions.load(s["id"]), sessions.load(s["id"])]
    refused = []

    def writer(snap, text):
        snap["messages"].append({"role": "user", "content": text})
        try:
            sessions.save(snap, base_rev=snap[sessions.REV_KEY])
        except sessions.StaleWriteError:
            refused.append(text)

    threads = [threading.Thread(target=writer, args=(snaps[i], f"writer {i}"))
               for i in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(refused) == 1
    stored = sessions.load(s["id"])
    assert len(stored["messages"]) == 1
    assert stored["messages"][0]["content"] not in refused


def test_reload_and_extend_keeps_both_writers_work():
    s = sessions.create("saga")
    s["messages"] = [{"role": "user", "content": "one"}]
    sessions.save(s)
    turn = list(sessions.load(s["id"])["messages"])      # this turn's snapshot

    other = sessions.load(s["id"])                       # a long generation:
    other["title"] = "renamed mid-generation"            # someone else writes
    other["messages"].append({"role": "user", "content": "queued while busy"})
    sessions.save(other, base_rev=other[sessions.REV_KEY])

    turn.append({"role": "assistant", "content": "the reply"})
    merged = sessions.reload_and_extend(s["id"], turn, since=1)
    sessions.save(merged, base_rev=merged[sessions.REV_KEY])

    got = sessions.load(s["id"])
    assert [m["content"] for m in got["messages"]] == [
        "one", "queued while busy", "the reply"]
    assert got["title"] == "renamed mid-generation"


def test_reload_and_extend_amends_a_continued_reply_in_place():
    s = sessions.create("saga")
    s["messages"] = [{"role": "user", "content": "one"},
                     {"role": "assistant", "content": "half a reply"}]
    sessions.save(s)
    turn = json.loads(json.dumps(sessions.load(s["id"])["messages"]))

    other = sessions.load(s["id"])
    other["messages"].append({"role": "user", "content": "landed meanwhile"})
    sessions.save(other)

    turn[1]["content"] = "half a reply and the rest"     # `continue` extends it
    merged = sessions.reload_and_extend(s["id"], turn, since=2, amend_from=1)
    assert [m["content"] for m in merged["messages"]] == [
        "one", "half a reply and the rest", "landed meanwhile"]


def test_reload_and_extend_after_a_compaction_keeps_the_new_text():
    s = sessions.create("saga")
    s["messages"] = [{"role": "user", "content": "one"},
                     {"role": "assistant", "content": "half a reply"}]
    sessions.save(s)
    turn = json.loads(json.dumps(sessions.load(s["id"])["messages"]))

    compacted = sessions.load(s["id"])                   # compaction ran while
    compacted["archive"] = compacted["messages"]         # this turn generated
    compacted["digest"] = "summary of it all"
    compacted["messages"] = [{"role": "user", "content": "kept tail"}]
    sessions.save(compacted)

    turn[1]["content"] = "half a reply and the rest"
    turn.append({"role": "user", "content": "the new one"})
    merged = sessions.reload_and_extend(s["id"], turn, since=2, amend_from=1)
    # index alignment is gone, so the amendment is appended rather than
    # dropped — losing generated text is worse than a duplicate in `archive`
    contents = [m["content"] for m in merged["messages"]]
    assert contents == ["kept tail", "half a reply and the rest", "the new one"]
    assert merged["digest"] == "summary of it all"


def test_reload_and_extend_returns_none_for_a_deleted_session():
    s = sessions.create("saga")
    sessions.delete(s["id"])
    assert sessions.reload_and_extend(s["id"], [{"role": "user",
                                                 "content": "x"}], 0) is None


def test_an_old_database_migrates_in_place(home):
    """A database written before the rev column must keep every row, read
    back at rev 0, and take the guard from there."""
    con = sqlite3.connect(home / "rigma.db")
    con.executescript("""
      CREATE TABLE sessions(
        id TEXT PRIMARY KEY, title TEXT NOT NULL DEFAULT '',
        updated_at REAL NOT NULL DEFAULT 0, use_rag INTEGER NOT NULL DEFAULT 0,
        message_count INTEGER NOT NULL DEFAULT 0, body TEXT NOT NULL);
      CREATE INDEX sessions_updated ON sessions(updated_at DESC);
      CREATE VIRTUAL TABLE session_fts USING fts5(id UNINDEXED, title, text);
    """)
    body = json.dumps({"id": "old1", "title": "old story", "updated_at": 1.0,
                       "messages": [{"role": "user",
                                     "content": "the wyvern circled"}]})
    con.execute("INSERT INTO sessions(id, title, updated_at, use_rag, "
                "message_count, body) VALUES(?,?,?,?,?,?)",
                ("old1", "old story", 1.0, 0, 1, body))
    con.execute("INSERT INTO session_fts(id, title, text) VALUES(?,?,?)",
                ("old1", "old story", "old story\nthe wyvern circled"))
    con.commit()
    con.close()

    got = sessions.load("old1")
    assert got is not None and got["title"] == "old story"
    assert got[sessions.REV_KEY] == 0
    assert [x["id"] for x in sessions.list_sessions()] == ["old1"]
    assert [h["id"] for h in sessions.search("wyvern")] == ["old1"]

    got["messages"].append({"role": "assistant", "content": "it landed"})
    assert sessions.save(got, base_rev=0) == 1
    assert len(sessions.load("old1")["messages"]) == 2


# ------------------------------------------------------------ F16: indexing

def test_search_follows_an_edit_and_a_rename():
    s = sessions.create("first title")
    s["messages"].append({"role": "user",
                          "content": "the wyvern circled the tower"})
    sessions.save(s)
    assert [h["id"] for h in sessions.search("wyvern")] == [s["id"]]

    # an edit changes neither the title nor the message count — the index
    # still has to follow it
    s["messages"][0]["content"] = "the basilisk circled the tower"
    sessions.save(s)
    assert sessions.search("wyvern") == []
    assert [h["id"] for h in sessions.search("basilisk")] == [s["id"]]

    s["title"] = "renamed chat"
    sessions.save(s)
    assert [h["id"] for h in sessions.search("renamed")] == [s["id"]]
    assert sessions.search("first title") == []


def test_a_save_that_changes_no_text_skips_the_index_rewrite():
    s = sessions.create("story")
    s["messages"].append({"role": "user", "content": "the wyvern circled"})
    sessions.save(s)
    # tamper with the index so any rewrite is observable
    with db.connect() as c:
        c.execute("UPDATE session_fts SET text=? WHERE id=?",
                  ("sentinelword", s["id"]))

    s["params"] = {"temperature": 0.4}          # nothing searchable changed
    sessions.save(s)
    with db.connect() as c:
        kept = c.execute("SELECT text FROM session_fts WHERE id=?",
                         (s["id"],)).fetchone()[0]
    assert kept == "sentinelword"               # ~105 ms per MB not paid

    s["messages"].append({"role": "user", "content": "and landed"})
    sessions.save(s)
    with db.connect() as c:
        rebuilt = c.execute("SELECT text FROM session_fts WHERE id=?",
                            (s["id"],)).fetchone()[0]
    assert "wyvern" in rebuilt and "and landed" in rebuilt


def test_fts_text_is_byte_identical_to_the_summing_version():
    def summing(session, cap):
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

    session = {"title": "t" * 7,
               "messages": [{"role": "user", "content": "m" * 13}
                            for _ in range(40)]
               + [{"role": "user", "content": None},
                  {"role": "user", "content": [{"type": "text", "text": "part"},
                                               {"type": "image_url"}, "junk"]}]}
    for cap in (0, 1, 12, 37, 200, 10_000, 1_000_000):
        assert db._fts_text(session, cap) == summing(session, cap)


# ---------------------------------------------------------------- F6: export

def test_export_carries_the_archive_and_the_digest():
    s = sessions.create("saga")
    s["archive"] = [{"role": "user", "content": "chapter one happened"},
                    {"role": "assistant", "content": "and so did chapter two"}]
    s["digest"] = "Everything before chapter one, summarised."
    s["messages"] = [{"role": "user", "content": "chapter three"}]
    sessions.save(s)

    md = sessions.export_markdown(sessions.load(s["id"]))
    assert md.index("chapter one happened") < md.index("and so did chapter two")
    assert md.index("and so did chapter two") < md.index("Everything before")
    assert md.index("Everything before") < md.index("chapter three")
    assert "> Everything before chapter one, summarised." in md
    assert "---" in md


def test_export_without_compaction_is_unchanged():
    s = {"title": "tale", "system_prompt": "be a bard",
         "messages": [{"role": "user", "content": "sing"}]}
    assert sessions.export_markdown(s) == \
        "# tale\n\n> be a bard\n\n**You:**\n\nsing\n"


# ------------------------------------------------------------------- F3: DRY

def _shipped_allowance() -> int:
    """The allowance hangar deliberately ships — never hand-copied here."""
    return int(hangar._DRY_BASELINE["dry_allowed_length"])


def test_the_dry_allowance_survives_validation_end_to_end():
    probed = hangar.params_from_probe({"sampling": {"temperature": 0.7}})
    effective = sessions.effective_params({"params": {}}, None, probed)
    assert effective["dry_allowed_length"] == _shipped_allowance()
    # the samplers that always got through are still there
    assert effective["dry_multiplier"] == probed["dry_multiplier"]
    assert effective["dry_penalty_last_n"] == probed["dry_penalty_last_n"]


def test_curated_cards_ship_dry_with_an_allowance():
    cards = Path(sessions.__file__).parent / "data" / "registry" / "models"
    for name in ("qwen3-vl-8b.json", "qwen3.6-35b-a3b.json"):
        params = json.loads((cards / name).read_text(
            encoding="utf-8"))["default_params"]
        assert params.get("dry_multiplier")          # DRY is switched on...
        assert sessions.validate_params(params)["dry_allowed_length"] == \
            _shipped_allowance()                     # ...with an allowance


def test_the_ui_slider_agrees_with_the_server_range():
    js = (Path(sessions.__file__).parent / "data" / "ui" / "panels.js"
          ).read_text(encoding="utf-8")
    m = re.search(r'\["dry_allowed_length",\s*([\d.]+),\s*([\d.]+),', js)
    assert m, "the advanced-sampling slider is gone"
    lo, hi = float(m.group(1)), float(m.group(2))
    assert lo <= _shipped_allowance() <= hi
    assert (lo, hi) == tuple(
        float(x) for x in sessions.PARAM_RANGES["dry_allowed_length"])


def test_dropping_an_authored_model_default_is_logged(caplog):
    with caplog.at_level(logging.WARNING, logger="rigma.sessions"):
        out = sessions.effective_params({}, None, {"dry_allowed_length": 999})
    assert "dry_allowed_length" not in out
    assert [r.levelno for r in caplog.records] == [logging.WARNING]
    assert "dry_allowed_length" in caplog.records[0].getMessage()

    caplog.clear()          # stored junk stays silent, as it always was
    with caplog.at_level(logging.WARNING, logger="rigma.sessions"):
        sessions.effective_params({"params": {"dry_allowed_length": 999}})
    assert caplog.records == []


# -------------------------------------------------------------- F40: id path

def test_a_session_id_cannot_escape_the_chats_directory(home):
    outside = home / "budget.json"
    outside.write_text("the owner's file", encoding="utf-8")
    # what "..%5C..%5Cbudget" looks like by the time it reaches _path: uvicorn
    # decodes before routing and Starlette's {param} converter allows "\"
    for evil in ("..\\..\\budget", "../../budget", "\\\\server\\share\\x",
                 "C:\\Windows\\System32\\x", "", "con"):
        with pytest.raises(sessions.SessionIdError):
            sessions._path(evil)
        assert sessions.delete(evil) is False
    assert outside.read_text(encoding="utf-8") == "the owner's file"


def test_a_real_session_id_still_names_its_legacy_backup(home):
    s = sessions.create("saga")
    backup = sessions.chats_dir() / f"{s['id']}.json"
    backup.write_text(json.dumps({"id": s["id"], "title": "saga",
                                  "messages": []}), encoding="utf-8")
    assert sessions._path(s["id"]) == backup.resolve()
    assert sessions.delete(s["id"]) is True
    assert not backup.exists()      # or the next import resurrects the chat
