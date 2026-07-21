"""Builder tools, and a creation chat that is offered nothing else (spec §6)."""
import pytest

from rigma import method_drafts, methods, tools


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def ctx():
    d = method_drafts.new_draft()
    return {"method_draft_id": d["id"], "allow_code": True,
            "workspace": ""}


def run(name, args, ctx):
    return tools.run_tool(name, args, ctx)


# --- the gate is the safety property --------------------------------------

def test_builder_only_offers_nothing_but_builder_tools():
    specs = tools.tool_specs(allow_code=True, workspace="/w",
                             builder_only=True)
    names = {s["function"]["name"] for s in specs}
    assert names, "the creation chat must be offered SOME tools"
    assert "write_file" not in names and "run_shell" not in names
    assert "create_method" in names and "save_method" in names


def test_normal_sessions_are_never_offered_builder_tools():
    names = {s["function"]["name"] for s in
             tools.tool_specs(allow_code=True, workspace="/w")}
    assert "create_method" not in names
    assert "save_method" not in names
    assert "write_file" in names          # ...and still get the real ones


def test_every_builder_tool_is_gated_on_the_capability():
    builders = [t for t in tools._REGISTRY.values()
                if t.needs == "method_builder"]
    assert len(builders) >= 8
    for t in builders:
        assert t.safe is False, t.name    # they all write a draft


# --- building a method ----------------------------------------------------

def test_a_whole_method_can_be_built_and_saved(ctx):
    assert not run("create_method", {"name": "Poem writing",
                                     "tagline": "short verse"},
                   ctx).startswith("error")
    assert not run("set_method_prompt",
                   {"text": "You write short poems."}, ctx).startswith("error")
    assert not run("define_macro", {"label": "Another verse", "steps": [
        {"kind": "prompt", "text": "Write one more verse."}]},
        ctx).startswith("error")
    out = run("save_method", {}, ctx)
    assert not out.startswith("error"), out
    saved = next(m for m in methods.catalog() if m["name"] == "Poem writing")
    assert saved["macros"][0]["label"] == "Another verse"
    assert "You write short poems." in saved["apply"]["system_prompt"]


def test_define_macro_rejects_an_unknown_tool_and_persists_nothing(ctx):
    out = run("define_macro", {"label": "Bad", "steps": [
        {"kind": "tool", "name": "teleport", "args": {}}]}, ctx)
    assert out.startswith("error") and "teleport" in out
    assert method_drafts.load(ctx["method_draft_id"])["macros"] == []


def test_define_macro_rejects_an_undeclared_variable(ctx):
    out = run("define_macro", {"label": "Bad", "steps": [
        {"kind": "tool", "name": "read_file",
         "args": {"path": "{{nowhere}}"}}]}, ctx)
    assert out.startswith("error") and "nowhere" in out


def test_set_var_then_the_same_macro_is_accepted(ctx):
    run("set_var", {"key": "nowhere", "label": "A file",
                    "default": "x.md", "kind": "path"}, ctx)
    out = run("define_macro", {"label": "Fine", "steps": [
        {"kind": "tool", "name": "read_file",
         "args": {"path": "{{nowhere}}"}}]}, ctx)
    assert not out.startswith("error"), out


def test_define_rule_standing_and_trigger(ctx):
    assert not run("define_rule", {"kind": "standing",
                                   "text": "Be brief."},
                   ctx).startswith("error")
    assert not run("define_rule", {
        "kind": "trigger", "on": {"event": "turn_ended"},
        "do": {"mode": "nudge", "text": "Check the notes."}},
        ctx).startswith("error")
    d = method_drafts.load(ctx["method_draft_id"])
    assert len(d["rules"]) == 2


def test_define_rule_rejects_an_unknown_event(ctx):
    out = run("define_rule", {"kind": "trigger",
                              "on": {"event": "full_moon"},
                              "do": {"mode": "nudge", "text": "x"}}, ctx)
    assert out.startswith("error") and "full_moon" in out


def test_define_workflow_works_like_define_macro(ctx):
    out = run("define_workflow", {"label": "Long job", "steps": [
        {"kind": "prompt", "text": "step one"}]}, ctx)
    assert not out.startswith("error"), out
    assert method_drafts.load(ctx["method_draft_id"])["workflows"]


def test_remove_component_drops_by_id_and_errors_on_unknown(ctx):
    run("define_macro", {"label": "Doomed", "steps": [
        {"kind": "prompt", "text": "hi"}]}, ctx)
    d = method_drafts.load(ctx["method_draft_id"])
    mid = d["macros"][0]["id"]
    assert not run("remove_component", {"id": mid}, ctx).startswith("error")
    assert method_drafts.load(ctx["method_draft_id"])["macros"] == []
    assert run("remove_component", {"id": "ghost"}, ctx).startswith("error")


def test_preview_method_changes_nothing(ctx):
    run("create_method", {"name": "Peek", "tagline": "t"}, ctx)
    before = method_drafts.load(ctx["method_draft_id"])
    out = run("preview_method", {}, ctx)
    assert "Peek" in out and not out.startswith("error")
    assert method_drafts.load(ctx["method_draft_id"]) == before


def test_save_method_reports_what_is_missing_instead_of_failing_silently(ctx):
    out = run("save_method", {}, ctx)      # no name, no prompt
    assert out.startswith("error")
    assert "name" in out or "system_prompt" in out


def test_a_bare_method_with_no_components_is_legal(ctx):
    run("create_method", {"name": "Minimal", "tagline": "t"}, ctx)
    run("set_method_prompt", {"text": "Just a prompt."}, ctx)
    assert not run("save_method", {}, ctx).startswith("error")


def test_builder_tools_without_a_draft_say_so(ctx):
    out = run("create_method", {"name": "X", "tagline": "y"},
              {"allow_code": True})
    assert out.startswith("error") and "draft" in out.lower()


def test_every_builder_tool_returns_a_string(ctx):
    for name in ("create_method", "set_method_prompt", "set_var",
                 "define_rule", "define_macro", "define_workflow",
                 "remove_component", "preview_method", "save_method"):
        assert isinstance(run(name, {}, ctx), str), name
