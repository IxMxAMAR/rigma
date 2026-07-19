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


def test_build_messages_coalesces_system_blocks():
    # mission + prompt + notes + digest must produce EXACTLY ONE system message
    # at the head — strict templates (Qwen3) raise on a second system message.
    s = {"system_prompt": "be an agent", "mission": "ship the thing",
         "notes": "canon here", "digest": "earlier chat",
         "messages": [{"role": "user", "content": "go"}]}
    out = sessions.build_messages(s)
    systems = [m for m in out if m["role"] == "system"]
    assert len(systems) == 1                       # not four
    assert out[0]["role"] == "system"
    c = out[0]["content"]
    assert c.index("CORE DIRECTIVE") < c.index("be an agent") < c.index("earlier chat")
    assert "ship the thing" in c and "canon here" in c
    assert out[-1] == {"role": "user", "content": "go"}


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


def test_build_messages_notes_folded_into_single_system():
    s = {"system_prompt": "SYS", "notes": "The dragon is named Ember.",
         "messages": [{"role": "user", "content": "hi"}]}
    out = sessions.build_messages(s)
    systems = [m for m in out if m["role"] == "system"]
    assert len(systems) == 1                       # one block, not two
    assert out[0]["content"].startswith("SYS")
    assert "Story notes (authoritative):" in out[0]["content"]
    assert "Ember" in out[0]["content"] and out[1]["content"] == "hi"


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


def test_build_messages_injects_digest_after_notes():
    s = {"system_prompt": "SYS", "notes": "N", "digest": "Earlier: dragons.",
         "messages": [{"role": "user", "content": "hi"}]}
    out = sessions.build_messages(s)
    systems = [m for m in out if m["role"] == "system"]
    assert len(systems) == 1                        # coalesced, order preserved
    c = out[0]["content"]
    assert c.index("SYS") < c.index("Story notes") < c.index("EARLIER CONVERSATION")
    assert "Earlier: dragons." in c
    # the digest is framed as REFERENCE, not as a fresh instruction — otherwise
    # the model restarts finished work after a compaction
    assert "REFERENCE ONLY" in c and "not a new instruction" in c
    assert "tools remain fully active" in c      # or it narrates instead
    assert "END OF SUMMARY" in c
    assert out[1]["content"] == "hi"


def test_mutable_fields_has_digest_not_archive():
    assert "digest" in sessions.MUTABLE_FIELDS
    assert "archive" not in sessions.MUTABLE_FIELDS


def test_build_messages_preserves_content_parts():
    parts = [{"type": "text", "text": "what is this?"},
             {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,AAA"}}]
    s = {"system_prompt": "", "messages": [
        {"role": "user", "content": parts, "extra_key": True}]}
    out = sessions.build_messages(s)
    assert out == [{"role": "user", "content": parts}]  # parts verbatim, extras stripped


def test_export_markdown_image_placeholder(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create(title="v")
    s["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "look"},
        {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    sessions.save(s)
    md = sessions.export_markdown(sessions.load(s["id"]))
    assert "look" in md and "[image]" in md and "data:" not in md


def test_extended_sampler_whitelist():
    ok = sessions.validate_params({
        "dry_multiplier": 0.8, "dry_base": 1.75, "dry_allowed_length": 2,
        "xtc_probability": 0.5, "xtc_threshold": 0.1, "top_n_sigma": 1.0})
    assert len(ok) == 6 and ok["dry_allowed_length"] == 2
    import pytest as _p
    with _p.raises(ValueError, match="xtc_probability"):
        sessions.validate_params({"xtc_probability": 2.0})


def test_search_survives_parts_content(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    s = sessions.create(title="viz chat")
    s["messages"] = [{"role": "user", "content": [
        {"type": "text", "text": "find the dragon here"},
        {"type": "image_url", "image_url": {"url": "data:..."}}]}]
    sessions.save(s)
    hits = sessions.search("dragon")
    assert len(hits) == 1 and "dragon" in hits[0]["snippet"]


def test_build_messages_drops_empty_assistant_turns():
    # a tool-only action persists an assistant message with no text; this Qwen3
    # template 400s on it ("Unable to generate parser for this template"), and
    # it carries no information for the model either way
    s = {"system_prompt": "SYS", "messages": [
        {"role": "user", "content": "go"},
        {"role": "assistant", "content": ""},
        {"role": "user", "content": "TOOL RESULT list_directory: ok"},
        {"role": "assistant", "content": "   "},
        {"role": "assistant", "content": "real answer"},
    ]}
    out = sessions.build_messages(s)
    assert all(not (m["role"] == "assistant" and not m["content"].strip())
               for m in out)
    assert [m["role"] for m in out] == ["system", "user", "user", "assistant"]
