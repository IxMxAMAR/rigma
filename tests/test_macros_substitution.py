"""Placeholders, safety classification and the trust store (spec §4.2, §5)."""
import pytest

from rigma import macros


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _session(**over):
    s = {"id": "s1", "title": "Chapter 3", "notes": "",
         "messages": [{"role": "user", "content": "go"},
                      {"role": "assistant", "content": "the reply"}]}
    s.update(over)
    return s


def _method(**over):
    m = {"id": "book", "vars": {"bible": {"label": "Bible",
                                          "default": "b.md", "kind": "path"}}}
    m.update(over)
    return m


def test_vars_resolve_from_defaults():
    ctx = macros.build_context(_session(), _method())
    assert macros.substitute({"path": "{{bible}}"}, ctx) == {"path": "b.md"}


def test_last_reply_is_the_last_assistant_message():
    ctx = macros.build_context(_session(), _method())
    assert macros.substitute("said: {{last_reply}}", ctx) == "said: the reply"


def test_step_results_resolve_by_index():
    ctx = macros.build_context(_session(), _method(),
                               results=["first", "second"])
    assert macros.substitute("{{step:1}}/{{step:0}}", ctx) == "second/first"


def test_pending_step_reference_resolves_empty_not_literal():
    ctx = macros.build_context(_session(), _method(), results=[])
    assert macros.substitute("[{{step:0}}]", ctx) == "[]"


def test_ask_resolves_from_answers():
    ctx = macros.build_context(_session(), _method(),
                               answers={"Which file": "notes.md"})
    assert macros.substitute("{{ask:Which file}}", ctx) == "notes.md"


def test_title_next_increments_the_first_integer():
    ctx = macros.build_context(_session(title="Chapter 3"), _method())
    assert macros.substitute("{{title_next}}", ctx) == "Chapter 4"


def test_title_next_falls_back_to_the_title_when_there_is_no_number():
    ctx = macros.build_context(_session(title="Draft"), _method())
    assert macros.substitute("{{title_next}}", ctx) == "Draft"


def test_transcript_is_assistant_prose_only():
    s = _session(messages=[{"role": "user", "content": "SECRET PROMPT"},
                           {"role": "assistant", "content": "prose one"},
                           {"role": "assistant", "content": "prose two"}])
    ctx = macros.build_context(s, _method())
    out = macros.substitute("{{transcript}}", ctx)
    assert "prose one" in out and "prose two" in out
    assert "SECRET" not in out


def test_substitution_is_deep_and_non_mutating():
    ctx = macros.build_context(_session(), _method())
    src = {"args": {"paths": ["{{bible}}", "fixed"]}}
    out = macros.substitute(src, ctx)
    assert out == {"args": {"paths": ["b.md", "fixed"]}}
    assert src["args"]["paths"][0] == "{{bible}}"


def test_unknown_placeholder_is_left_alone():
    ctx = macros.build_context(_session(), _method())
    assert macros.substitute("{{mystery}}", ctx) == "{{mystery}}"


def test_asks_lists_labels_in_order_without_duplicates():
    steps = [{"kind": "prompt", "text": "{{ask:B}} {{ask:A}}"},
             {"kind": "prompt", "text": "{{ask:B}}"}]
    assert macros.asks(steps) == ["B", "A"]


def test_read_only_steps_are_not_effectful():
    steps = [{"kind": "tool", "name": "read_file", "args": {"path": "x"}},
             {"kind": "prompt", "to": "aux", "text": "hi"},
             {"kind": "note", "op": "append", "text": "n"},
             {"kind": "settings", "set": {"effort": "on"}}]
    assert macros.is_effectful(steps) is False


def test_a_chat_prompt_is_effectful_because_the_model_can_reach_tools():
    """A macro made only of prompts is NOT read-only: the turn it drives can
    call write_file on the model's own initiative."""
    steps = [{"kind": "prompt", "text": "tidy up my folder"}]
    assert macros.is_effectful(steps) is True


def test_a_chat_prompt_is_safe_when_the_session_cannot_run_code():
    """Every write/exec tool is registered needs="code", so with allow_code
    off the turn is never offered one (this is the roleplay posture)."""
    steps = [{"kind": "prompt", "text": "advance the scene"}]
    assert macros.is_effectful(steps, allow_code=False) is False


def test_an_aux_prompt_is_always_safe():
    steps = [{"kind": "prompt", "to": "aux", "text": "recap"}]
    assert macros.is_effectful(steps, allow_code=True) is False


def test_a_write_tool_makes_a_macro_effectful():
    steps = [{"kind": "tool", "name": "write_file", "args": {"path": "x"}}]
    assert macros.is_effectful(steps) is True


def test_mcp_tools_are_always_effectful():
    steps = [{"kind": "tool", "name": "mcp__fs__read", "args": {}}]
    assert macros.is_effectful(steps) is True


def test_new_chat_is_effectful():
    assert macros.is_effectful([{"kind": "new_chat", "carry": []}]) is True


def test_preview_names_the_real_substituted_file():
    ctx = macros.build_context(_session(), _method())
    macro = {"label": "Finish chapter", "steps": [
        {"kind": "tool", "name": "write_file", "args": {"path": "{{bible}}"}}]}
    line = macros.preview_line(macro, ctx)
    assert line.startswith("Finish chapter:")
    assert "b.md" in line and "{{" not in line


def test_trust_persists_per_macro():
    assert macros.is_trusted("book", "finish_chapter") is False
    macros.trust("book", "finish_chapter")
    assert macros.is_trusted("book", "finish_chapter") is True
    assert macros.is_trusted("book", "other") is False
    assert macros.trust_path().is_file()
