import os
import time

from rigma import sessions, state
from rigma.models import UseCase
from rigma.registry import Registry


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
    assert list(sessions.chats_dir().glob("*.tmp")) == []


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
    time.sleep(0.02)
    b = sessions.create(title="second")
    b["messages"].append({"role": "user", "content": "x"})
    sessions.save(b)  # save() touches updated_at -> b is newest
    (sessions.chats_dir() / "garbage.json").write_text("{not json", encoding="utf-8")
    out = sessions.list_sessions()
    assert [s["title"] for s in out] == ["second", "first"]
    assert out[0]["message_count"] == 1 and "messages" not in out[0]
    assert a["id"] in [s["id"] for s in out]


def test_build_messages_session_prompt_wins():
    s = {"system_prompt": "be a pirate", "messages": [{"role": "user", "content": "hi"}]}
    out = sessions.build_messages(s, default_prompt="be boring")
    assert out[0] == {"role": "system", "content": "be a pirate"}
    assert out[1]["content"] == "hi" and len(out) == 2


def test_build_messages_falls_back_to_default():
    s = {"system_prompt": "", "messages": []}
    out = sessions.build_messages(s, default_prompt="be helpful")
    assert out == [{"role": "system", "content": "be helpful"}]


def test_build_messages_no_prompt_at_all():
    s = {"system_prompt": "", "messages": [{"role": "user", "content": "hi"}]}
    assert sessions.build_messages(s) == [{"role": "user", "content": "hi"}]


def _fake_reg(**prompts):
    return Registry([], {}, {}, {k: UseCase(name=k, system_prompt=v)
                                 for k, v in prompts.items()})


def test_default_prompt_from_state_use_case(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), use_case="creative")
    reg = _fake_reg(general="G", creative="C")
    assert sessions.default_prompt(reg) == "C"


def test_default_prompt_no_state_falls_to_general(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    assert sessions.default_prompt(_fake_reg(general="G")) == "G"


def test_default_prompt_unknown_use_case_is_empty(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid(), use_case="exotic")
    assert sessions.default_prompt(_fake_reg(general="G")) == ""


def test_create_has_cockpit_fields(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create()
    assert s["preset_id"] == "" and s["params"] == {} and s["notes"] == ""


def test_validate_params_whitelists_and_ranges():
    ok = sessions.validate_params({"temperature": 1.2, "max_tokens": 512,
                                   "bogus": 1})
    assert ok == {"temperature": 1.2, "max_tokens": 512}
    import pytest as _pytest
    with _pytest.raises(ValueError, match="temperature"):
        sessions.validate_params({"temperature": 9.0})
    with _pytest.raises(ValueError, match="max_tokens"):
        sessions.validate_params({"max_tokens": 0})
    with _pytest.raises(ValueError, match="max_tokens"):
        sessions.validate_params({"max_tokens": True})


def test_effective_params_session_over_preset():
    sess = {"params": {"temperature": 0.3}}
    preset = {"params": {"temperature": 1.5, "top_p": 0.9}}
    assert sessions.effective_params(sess, preset) == {"temperature": 0.3,
                                                       "top_p": 0.9}
    assert sessions.effective_params({}, None) == {}
    junk = {"params": {"temperature": 99.0, "top_p": 0.5}}
    assert sessions.effective_params(junk, None) == {"top_p": 0.5}


def test_build_messages_preset_layer():
    s = {"system_prompt": "", "messages": []}
    p = {"system_prompt": "PRESET"}
    assert sessions.build_messages(s, "DEFAULT", p)[0]["content"] == "PRESET"
    s2 = {"system_prompt": "MINE", "messages": []}
    assert sessions.build_messages(s2, "DEFAULT", p)[0]["content"] == "MINE"
    assert sessions.build_messages(s, "DEFAULT")[0]["content"] == "DEFAULT"


def test_build_messages_notes_second_system_message():
    s = {"system_prompt": "SYS", "notes": "The dragon is named Ember.",
         "messages": [{"role": "user", "content": "hi"}]}
    out = sessions.build_messages(s)
    assert out[0] == {"role": "system", "content": "SYS"}
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith("Story notes (authoritative):")
    assert "Ember" in out[1]["content"] and out[2]["content"] == "hi"


def test_build_messages_sanitizes_to_role_content():
    s = {"system_prompt": "", "messages": [
        {"role": "assistant", "content": "pick me",
         "variants": ["other take"], "secret": True}]}
    out = sessions.build_messages(s)
    assert out == [{"role": "assistant", "content": "pick me"}]


def test_build_messages_notes_alone_is_sole_system_message():
    s = {"system_prompt": "", "notes": "N", "messages": []}
    out = sessions.build_messages(s)
    assert len(out) == 1 and out[0]["role"] == "system"
    assert out[0]["content"] == "Story notes (authoritative):\nN"


def test_build_messages_missing_keys_default_sanely():
    s = {"messages": [{}, {"content": "only content"}]}
    out = sessions.build_messages(s)
    assert out == [{"role": "user", "content": ""},
                   {"role": "user", "content": "only content"}]


def test_search_titles_and_bodies(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    a = sessions.create(title="dragon tale")
    b = sessions.create(title="other")
    b["messages"] = [{"role": "user", "content": "the DRAGON returns"}]
    sessions.save(b)
    sessions.create(title="unrelated")
    hits = sessions.search("dragon")
    ids = {h["id"] for h in hits}
    assert ids == {a["id"], b["id"]}
    hit_b = next(h for h in hits if h["id"] == b["id"])
    assert "DRAGON" in hit_b["snippet"] and "messages" not in hit_b
    assert sessions.search("") == []


def test_duplicate_deep_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create(title="orig")
    s["messages"] = [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "yo", "variants": ["v"]}]
    s["notes"] = "N"
    sessions.save(s)
    d = sessions.duplicate(s["id"])
    assert d["id"] != s["id"] and d["title"] == "orig (copy)"
    assert d["messages"] == s["messages"] and d["notes"] == "N"
    d["messages"][0]["content"] = "mutated"
    assert sessions.load(s["id"])["messages"][0]["content"] == "hi"  # deep
    assert sessions.duplicate("nope") is None


def test_export_markdown(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create(title="tale", system_prompt="be a bard")
    s["messages"] = [{"role": "user", "content": "sing"},
                     {"role": "assistant", "content": "la la"}]
    sessions.save(s)
    md = sessions.export_markdown(sessions.load(s["id"]))
    assert md.startswith("# tale")
    assert "> be a bard" in md
    assert "**You:**\n\nsing" in md and "**Model:**\n\nla la" in md
