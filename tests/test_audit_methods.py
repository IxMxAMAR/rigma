"""Methods, macros and runs: the failures found in the 2026-09-04 audit.

F42 macro confirm gate judged against a flag the macro itself changes
F43 one type-confused method file taking down the whole Methods panel
F40 a URL path parameter interpolated straight into a filename
"""
import json
from types import SimpleNamespace

import pytest

from rigma import macros, method_drafts, methods, presets, runs


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


# --- F42: the confirm gate --------------------------------------------------

def test_a_settings_step_that_turns_code_on_is_effectful():
    """The roleplay posture ships allow_code off. A macro that flips it back
    on grants the turn write_file/run_shell and LEAVES it on afterwards, so
    the step is effectful in its own right -- not merely a prelude."""
    steps = [{"kind": "settings", "set": {"allow_code": True}}]
    assert macros.is_effectful(steps, allow_code=False) is True


def test_a_prompt_is_judged_against_the_value_the_macro_will_have_set():
    steps = [{"kind": "settings", "set": {"allow_code": True}},
             {"kind": "prompt", "text": "tidy up my folder"}]
    assert macros.is_effectful(steps, allow_code=False) is True


def test_a_truthy_non_bool_counts_the_way_the_interpreter_reads_it():
    """run_macro stores bool(st[f]); the gate must agree with it."""
    steps = [{"kind": "settings", "set": {"allow_code": 1}}]
    assert macros.is_effectful(steps, allow_code=False) is True


def test_turning_tools_on_is_effectful():
    steps = [{"kind": "settings", "set": {"use_tools": True}}]
    assert macros.is_effectful(steps, allow_code=False) is True


def test_turning_code_off_narrows_the_rest_of_the_step_list():
    """The re-evaluation runs both ways: once the macro has switched code off,
    a later chat prompt genuinely cannot be offered a write tool."""
    steps = [{"kind": "settings", "set": {"allow_code": False}},
             {"kind": "prompt", "text": "advance the scene"}]
    assert macros.is_effectful(steps, allow_code=True) is False


def test_a_settings_step_that_touches_neither_flag_stays_read_only():
    steps = [{"kind": "settings", "set": {"effort": "on"}},
             {"kind": "prompt", "to": "aux", "text": "recap"}]
    assert macros.is_effectful(steps, allow_code=False) is False


def test_the_preview_names_the_elevation_instead_of_saying_read_only():
    ctx = macros.build_context({"messages": [], "title": "Chapter 1"},
                               {"vars": {}})
    macro = {"id": "helper", "label": "Helper", "steps": [
        {"kind": "settings", "set": {"allow_code": True}},
        {"kind": "prompt", "text": "tidy up my folder"}]}
    line = macros.preview_line(macro, ctx, allow_code=False)
    assert "read-only" not in line
    assert "code" in line


# --- F43: one bad file must not take down the panel -------------------------

# every shape ms.normalize() reaches for with a bare int()/list()/dict()/.items()
_TYPE_CONFUSED = [
    {"version": "two"},
    {"vars": {"tone": "text"}},
    {"vars": []},
    {"guide": 5},
    {"apply": []},
    {"macros": "nope"},
    {"rules": [7]},
]


def _write_method_file(home, stem, doc):
    d = home / "methods"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{stem}.json").write_text(json.dumps(doc), encoding="utf-8")


@pytest.mark.parametrize("field", _TYPE_CONFUSED)
def test_one_type_confused_file_costs_one_method_not_the_catalog(home, field):
    _write_method_file(home, "broken", {"id": "broken", "name": "B", **field})
    ids = [m["id"] for m in methods.catalog()]
    assert "broken" not in ids            # the one bad file is skipped
    assert "book" in ids                  # every other method still lists
    assert methods.get("book") is not None


def test_a_valid_neighbour_survives_a_broken_file(home):
    _write_method_file(home, "broken", {"id": "broken", "guide": 5})
    saved, errs = methods.save_user(
        {"id": "mine", "name": "Mine",
         "apply": {"system_prompt": "write", "effort": "auto"}})
    assert errs == [] and saved is not None
    assert "mine" in [m["id"] for m in methods.catalog()]


@pytest.mark.parametrize("field", _TYPE_CONFUSED)
def test_a_type_confused_import_returns_errors_not_a_500(field):
    doc = {"id": "hostile", "name": "Hostile",
           "apply": {"system_prompt": "x", "effort": "auto"}}
    doc.update(field)
    saved, errs = methods.save_user(doc)
    assert saved is None and errs, field


def test_a_type_confused_draft_file_loads_as_none(home):
    d = home / "method_drafts"
    d.mkdir(parents=True, exist_ok=True)
    (d / "draft_dead.json").write_text(
        json.dumps({"id": "draft_dead", "guide": 5}), encoding="utf-8")
    assert method_drafts.load("draft_dead") is None


# --- F40: a path parameter is not a filename --------------------------------

_HOSTILE = ("../outside", "..\\outside", "..%5Coutside", "/etc/outside")


def test_a_traversing_preset_id_cannot_read_outside_the_presets_dir(home):
    (home / "outside.json").write_text(
        json.dumps({"id": "outside", "name": "secret"}), encoding="utf-8")
    presets.presets_dir()
    for hostile in _HOSTILE:
        assert presets.load(hostile) is None, hostile


def test_a_traversing_preset_id_cannot_delete_outside_the_presets_dir(home):
    outside = home / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    presets.presets_dir()
    for hostile in _HOSTILE:
        assert presets.delete(hostile) is False, hostile
    assert outside.exists()


def test_saving_a_preset_with_a_traversing_id_refuses(home):
    with pytest.raises(ValueError):
        presets.save({"id": "../escape", "name": "x", "params": {}})
    assert not (home / "escape.json").exists()


def test_presets_still_round_trip(home):
    p = presets.create("Mine", "you are helpful")
    assert presets.load(p["id"])["name"] == "Mine"
    listed = presets.list_presets(SimpleNamespace(use_cases={}))
    assert [x["id"] for x in listed] == [p["id"]]
    assert presets.delete(p["id"]) is True
    assert presets.load(p["id"]) is None


def test_a_traversing_method_id_cannot_delete_outside_the_methods_dir(home):
    outside = home / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    methods.methods_dir()
    for hostile in _HOSTILE:
        assert methods.delete_user(hostile) is False, hostile
    assert outside.exists()


def test_a_traversing_draft_id_cannot_read_or_delete_outside(home):
    outside = home / "outside.json"
    outside.write_text(json.dumps({"id": "outside", "name": "n"}),
                       encoding="utf-8")
    method_drafts.drafts_dir()
    for hostile in _HOSTILE:
        assert method_drafts.load(hostile) is None, hostile
        assert method_drafts.delete(hostile) is False, hostile
    assert outside.exists()


def test_drafts_still_round_trip():
    d = method_drafts.new_draft("Wip")
    assert method_drafts.load(d["id"])["name"] == "Wip"
    assert method_drafts.delete(d["id"]) is True
    assert method_drafts.load(d["id"]) is None


def test_a_traversing_run_id_makes_no_directory(home):
    for hostile in ("../evil", "..\\evil", "evil/../../evil"):
        with pytest.raises(ValueError):
            runs.run_dir(hostile, create=True)
        assert runs.load(hostile) is None, hostile
    assert not (home / "evil").exists()


def test_reading_a_run_that_does_not_exist_creates_nothing(home):
    rid = "20260101-000000-abcdef"
    assert runs.load(rid) is None
    assert runs.read_plan(rid) == []
    assert runs.read_actions(rid) == []
    assert runs.get_log_tail(rid) == ""
    assert runs.load_live(rid) == {}
    assert not (home / "runs" / rid).exists()


def test_a_real_run_still_writes_every_file(home):
    r = runs.create("mission", "sess1")
    d = runs.run_dir(r["id"])
    assert (d / "run.json").exists() and (d / "plan.json").exists()
    assert (d / "progress.md").exists() and (d / "outputs").is_dir()
    runs.plan_add(r["id"], "step one")
    runs.append_progress(r["id"], "did a thing", "do the next")
    runs.append_action(r["id"], "read_file", {"path": "x"}, True)
    runs.save_live(r["id"], {"beat": 1})
    assert runs.read_plan(r["id"])[0]["text"] == "step one"
    assert runs.read_actions(r["id"])[0]["tool"] == "read_file"
    assert runs.load_live(r["id"])["beat"] == 1
    assert "did a thing" in runs.get_log_tail(r["id"])
    assert runs.load(r["id"])["mission"] == "mission"
