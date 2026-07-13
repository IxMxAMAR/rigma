from rigma import sessions


def test_create_save_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create()
    assert len(s["id"]) == 12 and s["title"] == "New chat"
    assert s["messages"] == [] and s["use_rag"] is False
    s["messages"].append({"role": "user", "content": "hi"})
    sessions.save(s)
    got = sessions.load(s["id"])
    assert got["messages"] == [{"role": "user", "content": "hi"}]
    assert got["updated_at"] >= got["created_at"]


def test_load_missing_returns_none(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    assert sessions.load("nope") is None


def test_delete(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create()
    assert sessions.delete(s["id"]) is True
    assert sessions.load(s["id"]) is None
    assert sessions.delete(s["id"]) is False


def test_list_sessions_newest_first_skips_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    a = sessions.create(title="first")
    b = sessions.create(title="second")
    b["messages"].append({"role": "user", "content": "x"})
    sessions.save(b)  # save() touches updated_at -> b is newest
    (sessions.chats_dir() / "garbage.json").write_text("{not json", encoding="utf-8")
    out = sessions.list_sessions()
    assert [s["title"] for s in out] == ["second", "first"]
    assert out[0]["message_count"] == 1 and "messages" not in out[0]
    assert a["id"] in [s["id"] for s in out]
