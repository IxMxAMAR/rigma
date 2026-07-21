"""Phase 5: SQLite session store — imports legacy files, keeps the public
API byte-identical, and turns search into an indexed query."""
import json

import pytest

from rigma import db, sessions


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_roundtrip_and_listing():
    a = sessions.create("first chat")
    b = sessions.create("second chat")
    b["messages"].append({"role": "user", "content": "hello there"})
    sessions.save(b)
    got = sessions.load(b["id"])
    assert got["title"] == "second chat"
    assert got["messages"][0]["content"] == "hello there"
    listing = sessions.list_sessions()
    assert listing[0]["id"] == b["id"]          # newest first
    assert listing[0]["message_count"] == 1
    assert {s["id"] for s in listing} == {a["id"], b["id"]}


def test_legacy_files_imported_once_and_kept(tmp_path):
    d = tmp_path / "sessions" / "chats"
    d.mkdir(parents=True)
    legacy = {"id": "legacy0001", "title": "old story",
              "updated_at": 123.0, "messages": [
                  {"role": "user", "content": "the dragon speaks"}]}
    (d / "legacy0001.json").write_text(json.dumps(legacy), encoding="utf-8")
    got = sessions.load("legacy0001")
    assert got is not None and got["title"] == "old story"
    # defaults backfilled on load, exactly like the file era
    assert got["max_tool_rounds"] == 1000
    # the file is a backup now — still there, never rewritten
    assert (d / "legacy0001.json").exists()
    assert "legacy0001" in {s["id"] for s in sessions.list_sessions()}


def test_delete_kills_legacy_backup_too(tmp_path):
    d = tmp_path / "sessions" / "chats"
    d.mkdir(parents=True)
    (d / "gone123456.json").write_text(json.dumps(
        {"id": "gone123456", "title": "x", "messages": []}), encoding="utf-8")
    assert sessions.load("gone123456") is not None
    assert sessions.delete("gone123456") is True
    assert sessions.load("gone123456") is None
    # without this the next import would resurrect the deleted chat
    assert not (d / "gone123456.json").exists()


def test_search_by_message_body_and_title():
    s = sessions.create("dragon epic")
    s["messages"].append({"role": "assistant",
                          "content": "the wyvern circled the tower"})
    sessions.save(s)
    sessions.create("unrelated")
    by_title = sessions.search("dragon")
    assert [h["id"] for h in by_title] == [s["id"]]
    by_body = sessions.search("wyvern")
    assert [h["id"] for h in by_body] == [s["id"]]
    assert by_body[0]["snippet"]


def test_search_matches_inside_words():
    # FTS can't match mid-word; the LIKE fallback preserves the old
    # substring behaviour
    s = sessions.create("chat")
    s["messages"].append({"role": "user", "content": "supercalifragilistic"})
    sessions.save(s)
    assert [h["id"] for h in sessions.search("califragi")] == [s["id"]]


def test_search_vision_parts_text():
    s = sessions.create("pics")
    s["messages"].append({"role": "user", "content": [
        {"type": "text", "text": "look at this zeppelin"},
        {"type": "image_url", "image_url": {"url": "data:image/png;..."}}]})
    sessions.save(s)
    assert [h["id"] for h in sessions.search("zeppelin")] == [s["id"]]


def test_empty_query_and_no_hits():
    sessions.create("something")
    assert sessions.search("") == []
    assert sessions.search("qqqzzzyyy") == []


def test_duplicate_still_works():
    s = sessions.create("orig")
    s["messages"].append({"role": "user", "content": "keep me"})
    sessions.save(s)
    dup = sessions.duplicate(s["id"])
    assert dup["id"] != s["id"]
    assert dup["title"] == "orig (copy)"
    assert sessions.load(dup["id"])["messages"][0]["content"] == "keep me"


def test_db_survives_weird_titles_and_quotes():
    s = sessions.create('he said "hello" — 100% legit\'s')
    sessions.save(s)
    hits = sessions.search('"hello"')
    assert any(h["id"] == s["id"] for h in hits)


def test_fts_availability_flag_is_bool():
    sessions.create("x")           # forces db init
    assert isinstance(db.has_fts(), bool)
