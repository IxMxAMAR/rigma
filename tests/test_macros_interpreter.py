"""The step interpreter: one primitive behind macros and workflows (spec §4.1).

No pytest-asyncio in this repo -- async coverage runs through asyncio.run()
inside ordinary sync tests, the same way tests/test_mission.py does it.
"""
import asyncio

import pytest

from rigma import macros, sessions


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


class Runner:
    """Fakes for everything model-facing, so the interpreter is testable
    with no engine and no network."""
    def __init__(self, reply="MODEL SAID", aux="AUX SAID"):
        self.events, self.turns, self.aux_prompts = [], [], []
        self.reply, self.aux = reply, aux

    async def emit(self, event, data):
        self.events.append((event, data))

    async def drive_turn(self, session):
        self.turns.append(list(session["messages"]))
        session["messages"].append({"role": "assistant",
                                    "content": self.reply})
        return self.reply

    async def aux_complete(self, prompt, max_tokens=120):
        self.aux_prompts.append(prompt)
        return self.aux


def _method():
    return {"id": "book", "vars": {"bible": {"label": "B", "default": "b.md",
                                             "kind": "path"}}}


def _run(macro, runner, session=None, **kw):
    """Returns (result, session)."""
    s = session if session is not None else sessions.create("Chapter 3")
    out = asyncio.run(macros.run_macro(
        s, _method(), macro, emit=runner.emit, drive_turn=runner.drive_turn,
        aux_complete=runner.aux_complete, tool_ctx={}, **kw))
    return out, s


def test_tool_step_runs_through_cached_run_and_records_the_result(monkeypatch):
    seen = {}

    def fake(name, args, ctx):
        seen["call"] = (name, args)
        return "FILE BODY"
    monkeypatch.setattr("rigma.tools.cached_run", fake)
    r = Runner()
    out, _ = _run({"id": "m", "label": "M", "steps": [
        {"kind": "tool", "name": "read_file",
         "args": {"path": "{{bible}}"}}]}, r)
    assert seen["call"] == ("read_file", {"path": "b.md"})
    assert out["results"] == ["FILE BODY"]
    assert ("tool", {"id": "m0", "name": "read_file",
                     "args": {"path": "b.md"}}) in r.events
    assert any(e == "tool_result" for e, _ in r.events)


def test_chat_prompt_appends_a_user_message_and_drives_a_turn():
    r = Runner()
    out, s = _run({"id": "m", "label": "M", "steps": [
        {"kind": "prompt", "text": "write more"}]}, r)
    assert s["messages"][0] == {"role": "user", "content": "write more"}
    assert out["results"] == ["MODEL SAID"]
    assert len(r.turns) == 1


def test_aux_prompt_never_touches_the_transcript():
    r = Runner()
    out, s = _run({"id": "m", "label": "M", "steps": [
        {"kind": "prompt", "to": "aux", "text": "summarise"}]}, r)
    assert s["messages"] == []
    assert r.turns == []
    assert out["results"] == ["AUX SAID"]
    assert r.aux_prompts == ["summarise"]


def test_later_steps_see_earlier_results():
    r = Runner()
    _run({"id": "m", "label": "M", "steps": [
        {"kind": "prompt", "to": "aux", "text": "a"},
        {"kind": "prompt", "to": "aux", "text": "got {{step:0}}"}]}, r)
    assert r.aux_prompts[1] == "got AUX SAID"


def test_note_append_and_replace():
    r = Runner()
    s = sessions.create("t")
    s["notes"] = "old"
    _run({"id": "m", "label": "M", "steps": [
        {"kind": "note", "op": "append", "text": "added"}]}, r, session=s)
    assert s["notes"] == "old\nadded"
    _run({"id": "m", "label": "M", "steps": [
        {"kind": "note", "op": "replace", "text": "fresh"}]}, r, session=s)
    assert s["notes"] == "fresh"


def test_settings_step_applies_validated_params_only():
    r = Runner()
    _, s = _run({"id": "m", "label": "M", "steps": [
        {"kind": "settings",
         "set": {"effort": "on", "params": {"temperature": 0.2,
                                            "banana": 99}}}]}, r)
    assert s["effort"] == "on"
    assert s["params"]["temperature"] == 0.2
    assert "banana" not in s["params"]


def test_a_quoted_false_does_not_switch_code_or_tools_on():
    """`bool("false")` is True, so a method-builder model that quotes its
    booleans turned code execution ON and left it on for the chat (09-6)."""
    r = Runner()
    _, s = _run({"id": "m", "label": "M", "steps": [
        {"kind": "settings",
         "set": {"allow_code": "false", "use_tools": "0"}}]}, r)
    assert s["allow_code"] is False
    assert s["use_tools"] is False


def test_boolean_words_are_coerced_explicitly():
    r = Runner()
    _, s = _run({"id": "m", "label": "M", "steps": [
        {"kind": "settings",
         "set": {"allow_code": "yes", "use_tools": "TRUE"}}]}, r)
    assert s["allow_code"] is True and s["use_tools"] is True
    _, s2 = _run({"id": "m", "label": "M", "steps": [
        {"kind": "settings",
         "set": {"allow_code": "", "use_tools": "off"}}]}, r)
    assert s2["allow_code"] is False and s2["use_tools"] is False


def test_a_quoted_false_is_not_read_as_an_elevation():
    """The gate and the interpreter must read the value the same way: a preview
    that says "switched on" for a declared false is the bug's other half."""
    steps = [{"kind": "settings", "set": {"allow_code": "false"}}]
    assert macros.is_effectful(steps, allow_code=False) is False
    line = macros.preview_line({"id": "m", "label": "M", "steps": steps}, {})
    assert "switched on" not in line


def test_new_chat_carries_only_the_listed_fields():
    r = Runner()
    s = sessions.create("Chapter 3")
    s.update(notes="the bible", workspace="/w", method="book",
             system_prompt="secret")
    sessions.save(s)
    out, _ = _run({"id": "m", "label": "M", "steps": [
        {"kind": "new_chat", "carry": ["notes", "method"],
         "title": "{{title_next}}"}]}, r, session=s)
    nid = out["new_session_id"]
    assert nid
    fresh = sessions.load(nid)
    assert fresh["notes"] == "the bible" and fresh["method"] == "book"
    assert fresh["workspace"] == ""          # not carried
    assert fresh["title"] == "Chapter 4"
    assert fresh["title_source"] == "auto"


def test_the_session_is_saved_after_a_mutating_run():
    r = Runner()
    s = sessions.create("t")
    _run({"id": "m", "label": "M", "steps": [
        {"kind": "note", "op": "replace", "text": "written"}]}, r, session=s)
    assert sessions.load(s["id"])["notes"] == "written"


def test_a_failing_tool_step_stops_the_macro_and_reports(monkeypatch):
    monkeypatch.setattr("rigma.tools.cached_run",
                        lambda n, a, c: "error: no such file")
    r = Runner()
    out, _ = _run({"id": "m", "label": "M", "steps": [
        {"kind": "tool", "name": "read_file", "args": {"path": "x"}},
        {"kind": "prompt", "text": "never reached"}]}, r)
    assert len(out["results"]) == 1
    assert r.turns == []
    assert any(e == "error" for e, _ in r.events)


def test_every_step_emits_a_macro_step_event():
    r = Runner()
    _run({"id": "m", "label": "M", "steps": [
        {"kind": "note", "op": "append", "text": "a"},
        {"kind": "note", "op": "append", "text": "b"}]}, r)
    steps = [d for e, d in r.events if e == "macro_step"]
    assert [s["index"] for s in steps] == [0, 1]
    assert steps[0]["kind"] == "note"
