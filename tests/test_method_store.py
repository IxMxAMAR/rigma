"""User methods on disk, merged with the built-ins (spec §4)."""
import json

import pytest

from rigma import methods


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _doc(mid="mine", name="Mine"):
    return {"id": mid, "name": name, "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""}}


def test_builtins_are_normalized_and_marked():
    for m in methods.builtins():
        assert m["builtin"] is True
        assert isinstance(m["macros"], list)
        assert "vars" in m and "rules" in m and "workflows" in m


def test_normalize_does_not_mutate_the_module_literals():
    methods.builtins()
    methods.builtins()
    assert "vars" not in methods.METHODS[0]


def test_save_then_get_roundtrips():
    saved, errs = methods.save_user(_doc())
    assert errs == [] and saved is not None
    assert saved["builtin"] is False
    got = methods.get("mine")
    assert got["name"] == "Mine"
    assert (methods.methods_dir() / "mine.json").is_file()


def test_save_rejects_an_invalid_document():
    saved, errs = methods.save_user({"id": "bad id", "name": ""})
    assert saved is None and errs
    assert not list(methods.methods_dir().glob("*.json"))


def test_user_method_overrides_a_builtin_in_place():
    before = [m["id"] for m in methods.catalog()]
    methods.save_user(_doc(mid="book", name="My book method"))
    after = methods.catalog()
    assert [m["id"] for m in after] == before          # same order, same count
    book = next(m for m in after if m["id"] == "book")
    assert book["name"] == "My book method"
    assert book["builtin"] is False


def test_user_methods_append_after_builtins():
    methods.save_user(_doc())
    cat = methods.catalog()
    assert cat[-1]["id"] == "mine"


def test_corrupt_file_is_skipped_not_fatal():
    methods.methods_dir().joinpath("broken.json").write_text(
        "{not json", encoding="utf-8")
    assert isinstance(methods.catalog(), list)
    assert methods.get("broken") is None


def test_delete_removes_only_user_methods():
    methods.save_user(_doc())
    assert methods.delete_user("mine") is True
    assert methods.get("mine") is None
    assert methods.delete_user("book") is False       # built-in survives
    assert methods.get("book") is not None


def test_apply_still_works_on_the_new_shape():
    from rigma import sessions
    s = sessions.create("t")
    out = methods.apply_to_session(s, "book")
    assert out["method"] == "book"
    assert out["params"]["temperature"] == 0.85


def test_apply_works_for_a_user_method():
    from rigma import sessions
    methods.save_user(_doc())
    s = sessions.create("t")
    out = methods.apply_to_session(s, "mine")
    assert out["system_prompt"] == "p" and out["method"] == "mine"


def test_saved_file_is_the_normalized_document():
    methods.save_user(_doc())
    raw = json.loads((methods.methods_dir() / "mine.json")
                     .read_text(encoding="utf-8"))
    assert raw["version"] == 1 and raw["rules"] == []
