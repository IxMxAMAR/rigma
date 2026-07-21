"""Standing rules join the ONE leading system message (spec §7.7)."""
import pytest

from rigma import methods, sessions


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _method_with_rules():
    saved, errs = methods.save_user({
        "id": "ruled", "name": "Ruled", "tagline": "t",
        "apply": {"system_prompt": "base prompt", "params": {},
                  "effort": "auto", "use_tools": False, "allow_code": False,
                  "notes_template": ""},
        "rules": [{"id": "r1", "kind": "standing", "text": "Be terse."},
                  {"id": "r2", "kind": "standing", "text": "Cite files."},
                  {"id": "r3", "kind": "trigger",
                   "on": {"event": "turn_ended"},
                   "do": {"mode": "nudge", "text": "not a standing rule"}}]})
    assert errs == [], errs
    return saved


def test_standing_rules_appear_in_the_system_message():
    _method_with_rules()
    s = sessions.create("t")
    methods.apply_to_session(s, "ruled")
    msgs = sessions.build_messages(s)
    assert msgs[0]["role"] == "system"
    sys = msgs[0]["content"]
    assert "Be terse." in sys and "Cite files." in sys
    assert "not a standing rule" not in sys


def test_there_is_still_exactly_one_system_message():
    _method_with_rules()
    s = sessions.create("t")
    methods.apply_to_session(s, "ruled")
    s["notes"] = "some notes"
    s["digest"] = "some digest"
    s["messages"] = [{"role": "user", "content": "hi"}]
    msgs = sessions.build_messages(s)
    assert [m["role"] for m in msgs].count("system") == 1
    assert msgs[0]["role"] == "system"


def test_a_method_without_rules_changes_nothing():
    # every BUILT-IN ships a standing rule now, so use a rule-less user method
    methods.save_user({
        "id": "bare", "name": "Bare", "tagline": "t",
        "apply": {"system_prompt": "just the prompt", "params": {},
                  "effort": "auto", "use_tools": False, "allow_code": False,
                  "notes_template": ""}})
    s = sessions.create("t")
    methods.apply_to_session(s, "bare")
    assert "METHOD RULES" not in sessions.build_messages(s)[0]["content"]


def test_an_unknown_method_id_is_survivable():
    s = sessions.create("t")
    s["method"] = "deleted_yesterday"
    msgs = sessions.build_messages(s)
    assert isinstance(msgs, list)


def test_pending_nudges_ride_as_one_trailing_user_message():
    s = sessions.create("t")
    s["messages"] = [{"role": "user", "content": "hi"},
                     {"role": "assistant", "content": "hello"}]
    s["pending_nudges"] = ["update the bible", "check continuity"]
    msgs = sessions.build_messages(s)
    assert msgs[-1]["role"] == "user"
    assert "update the bible" in msgs[-1]["content"]
    assert "check continuity" in msgs[-1]["content"]
    # never a second system message — strict templates 400 on that
    assert [m["role"] for m in msgs].count("system") <= 1


def test_no_nudges_means_no_extra_message():
    s = sessions.create("t")
    s["messages"] = [{"role": "user", "content": "hi"}]
    before = len(sessions.build_messages(s))
    s["pending_nudges"] = []
    assert len(sessions.build_messages(s)) == before
