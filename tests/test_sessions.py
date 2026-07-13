import os

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
