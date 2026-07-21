"""A draft is the same document as a method, at an earlier stage (spec §6)."""
import pytest

from rigma import method_drafts, methods


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _valid(d):
    d["name"] = "My method"
    d["tagline"] = "does a thing"
    d["apply"] = {"system_prompt": "be useful", "params": {},
                  "effort": "auto", "use_tools": True, "allow_code": True,
                  "notes_template": ""}
    return method_drafts.save(d)


def test_new_draft_roundtrips():
    d = method_drafts.new_draft(name="Draft one")
    assert d["id"]
    got = method_drafts.load(d["id"])
    assert got is not None and got["name"] == "Draft one"


def test_a_draft_is_not_in_the_method_catalog():
    d = method_drafts.new_draft(name="Hidden")
    assert all(m["id"] != d["id"] for m in methods.catalog())


def test_promote_refuses_an_invalid_draft_and_writes_nothing():
    d = method_drafts.new_draft()          # no name, no apply -> invalid
    out, errs = method_drafts.promote(d["id"])
    assert out is None and errs
    assert methods.get(d["id"]) is None
    assert method_drafts.load(d["id"]) is not None    # draft survives


def test_promote_creates_the_method_and_clears_the_draft():
    d = _valid(method_drafts.new_draft(name="Keeper"))
    out, errs = method_drafts.promote(d["id"])
    assert errs == [] and out is not None
    assert methods.get(out["id"])["name"] == "My method"
    assert method_drafts.load(d["id"]) is None


def test_promoted_method_is_a_user_method_not_a_builtin():
    d = _valid(method_drafts.new_draft())
    out, _ = method_drafts.promote(d["id"])
    assert out["builtin"] is False


def test_save_persists_edits():
    d = method_drafts.new_draft(name="A")
    d["tagline"] = "changed"
    method_drafts.save(d)
    assert method_drafts.load(d["id"])["tagline"] == "changed"


def test_delete_removes_a_draft():
    d = method_drafts.new_draft()
    assert method_drafts.delete(d["id"]) is True
    assert method_drafts.load(d["id"]) is None
    assert method_drafts.delete("nope") is False


def test_a_corrupt_draft_file_is_skipped_not_fatal():
    method_drafts.drafts_dir().joinpath("broken.json").write_text(
        "{not json", encoding="utf-8")
    assert method_drafts.load("broken") is None


def test_draft_ids_do_not_collide():
    ids = {method_drafts.new_draft()["id"] for _ in range(5)}
    assert len(ids) == 5
