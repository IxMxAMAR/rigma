"""The method document: normalization and validation (spec §4)."""
from rigma import method_schema as ms

TOOLS = {"read_file", "write_file", "grep", "run_shell"}


def _doc(**over):
    base = {"id": "demo", "name": "Demo", "tagline": "a demo",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": "n"}}
    base.update(over)
    return base


def test_normalize_fills_every_optional_key():
    out = ms.normalize(_doc())
    assert out["vars"] == {} and out["rules"] == []
    assert out["macros"] == [] and out["workflows"] == []
    assert out["guide"] == [] and out["builtin"] is False
    assert out["version"] == 1 and out["extends"] is None
    # the input is not mutated
    assert "vars" not in _doc()


def test_normalize_assigns_missing_component_ids():
    out = ms.normalize(_doc(macros=[{"label": "Run tests", "steps": []},
                                    {"label": "Run tests", "steps": []}]))
    ids = [m["id"] for m in out["macros"]]
    assert ids == ["run_tests", "run_tests_2"]


def test_normalize_keeps_explicit_ids():
    out = ms.normalize(_doc(macros=[{"id": "keep_me", "label": "L",
                                     "steps": []}]))
    assert out["macros"][0]["id"] == "keep_me"


def test_valid_document_has_no_errors():
    doc = ms.normalize(_doc(
        vars={"bible": {"label": "Bible", "default": "b.md", "kind": "path"}},
        rules=[{"kind": "standing", "text": "be brief"}],
        macros=[{"label": "Read it", "steps": [
            {"kind": "tool", "name": "read_file",
             "args": {"path": "{{bible}}"}},
            {"kind": "prompt", "text": "summarise {{step:0}}"}]}]))
    assert ms.validate(doc, TOOLS) == []


def test_unknown_tool_is_rejected_by_name():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "tool", "name": "teleport", "args": {}}]}]))
    errs = ms.validate(doc, TOOLS)
    assert any("teleport" in e for e in errs)


def test_unknown_step_kind_is_rejected():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "sing", "text": "la"}]}]))
    assert any("sing" in e for e in ms.validate(doc, TOOLS))


def test_undefined_var_placeholder_is_rejected():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "tool", "name": "read_file",
         "args": {"path": "{{nowhere}}"}}]}]))
    assert any("nowhere" in e for e in ms.validate(doc, TOOLS))


def test_forward_step_reference_is_rejected():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "prompt", "text": "{{step:3}}"}]}]))
    assert any("step:3" in e for e in ms.validate(doc, TOOLS))


def test_ask_and_builtin_placeholders_need_no_declaration():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "prompt",
         "text": "{{ask:Which file}} {{last_reply}} {{selection}} "
                 "{{transcript}} {{title_next}}"}]}]))
    assert ms.validate(doc, TOOLS) == []


def test_bad_id_is_rejected():
    assert any("id" in e for e in ms.validate(ms.normalize(_doc(id="No Way!")),
                                              TOOLS))


def test_duplicate_component_ids_are_rejected():
    # normalize() dedupes, so build the collision directly to prove validate()
    # is itself defensive -- the builder tools can hand it anything
    doc = ms.normalize(_doc())
    doc["macros"] = [{"id": "a", "label": "A", "steps": []},
                     {"id": "a", "label": "B", "steps": []}]
    assert any("duplicate" in e.lower() for e in ms.validate(doc, TOOLS))


def test_trigger_rule_needs_a_known_event():
    doc = ms.normalize(_doc(rules=[{"kind": "trigger",
                                    "on": {"event": "full_moon"},
                                    "do": {"mode": "nudge", "text": "hi"}}]))
    assert any("full_moon" in e for e in ms.validate(doc, TOOLS))


def test_trigger_run_mode_must_name_an_existing_macro():
    doc = ms.normalize(_doc(rules=[{"kind": "trigger",
                                    "on": {"event": "turn_ended"},
                                    "do": {"mode": "run", "macro": "ghost"}}]))
    assert any("ghost" in e for e in ms.validate(doc, TOOLS))


def test_new_chat_carry_is_restricted():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "new_chat", "carry": ["notes", "everything"]}]}]))
    assert any("everything" in e for e in ms.validate(doc, TOOLS))


def test_settings_step_rejects_unknown_field():
    doc = ms.normalize(_doc(macros=[{"label": "X", "steps": [
        {"kind": "settings", "set": {"effort": "on", "banana": 1}}]}]))
    assert any("banana" in e for e in ms.validate(doc, TOOLS))


def test_find_placeholders_walks_nested_structures():
    got = ms.find_placeholders({"a": ["{{x}}", {"b": "{{ask:Pick a file}}"}]})
    assert ("x", "") in got and ("ask", "Pick a file") in got


def test_slugify_dedupes_against_taken():
    assert ms.slugify("Finish chapter", set()) == "finish_chapter"
    assert ms.slugify("Finish chapter", {"finish_chapter"}) == "finish_chapter_2"
    assert ms.slugify("!!!", set()) == "item"
