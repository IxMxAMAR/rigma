"""OD-15 option 1: `POST /api/restore` must TRULY replace the store.

The Backup/Restore card promised "replaces the whole store… anything not in the
file is gone". Measured behaviour before this change: memory WAS replaced
(`store.restore(rows)`), but settings were MERGED (`app_settings.save` does
`cur = load(); cur.update(clean)`) and user methods were MERGED by id
(`methods.save_user` per file method — it deletes nothing). A probe restoring a
document with one method, `settings: {}` and `memory: []` left a second method
(`keepme`) and `idle_unload_minutes=12` in place.

The accepted resolution makes the route destructive in a new way — anything
created after the backup is deleted — so the A11 snapshot/rollback must cover
the new delete/reset steps, and a failed restore must leave the store EXACTLY
as it was. That is the owner's-data guarantee these tests pin.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import app_settings, methods
from rigma.memory import MemoryStore
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


@pytest.fixture
def client(home):
    # raise_server_exceptions=False: a failed restore answers a NAMED 500, not
    # a traceback (A11).
    return TestClient(build_app(upstream_port=1),
                      raise_server_exceptions=False)


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]}]}


def _seed(home, method_ids=("mine",), minutes=12):
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    app_settings.save({"idle_unload_minutes": minutes})
    for mid in method_ids:
        saved, errs = methods.save_user(_doc(mid))
        assert saved is not None and not errs, errs
    return store


def _snapshot(home, store):
    """EVERY byte the restore can touch: the settings file, every user method
    file (name -> exact bytes) and the memory store. The rollback proof compares
    this whole mapping, not one field."""
    snaps = {
        "settings": (app_settings.settings_path().read_bytes()
                     if app_settings.settings_path().exists() else None),
        "memory": store.path.read_bytes() if store.path.exists() else None,
        "methods": {p.name: p.read_bytes()
                    for p in methods.methods_dir().glob("*.json")},
    }
    return snaps


def test_restore_deletes_methods_and_resets_settings_the_document_omits(
        client, home):
    """One method + `settings: {}` + `memory: []` over TWO methods and a
    non-default setting: the extra method is gone and the setting is reset."""
    store = _seed(home, method_ids=("mine", "keepme"), minutes=12)
    doc = {"rigma_backup": 1, "settings": {},
           "methods": [_doc("mine")], "memory": []}

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200, r.text

    # methods: the document's set is the WHOLE set
    assert methods.get("mine") is not None
    assert methods.get("keepme") is None, "a post-backup method survived"
    assert [m["id"] for m in methods.user_methods()] == ["mine"]
    # settings: the omitted key goes back to its default, not its old value
    assert app_settings.load()["idle_unload_minutes"] == 0.0
    # memory: unchanged semantics, replaced outright
    assert store.all() == []


def test_a_failure_in_the_delete_step_leaves_the_store_exactly_as_it_was(
        client, home, monkeypatch):
    """The owner's-data guarantee. The delete step is the new destructive one;
    when it fails on its SECOND call the first delete must be undone and every
    store put back byte for byte."""
    store = _seed(home, method_ids=("mine", "keepme", "other"), minutes=12)
    doc = {"rigma_backup": 1,
           "settings": {"idle_unload_minutes": 99},
           "methods": [_doc("aaa")],
           "memory": [{"id": "mem-1", "kind": "technique",
                       "text": "Prefer q8_0."}]}

    before = _snapshot(home, store)

    real = methods.delete_user
    calls = {"n": 0}

    def flaky(mid):
        calls["n"] += 1
        if calls["n"] == 2:              # after the first file is really gone
            raise OSError("no space left on device")
        return real(mid)

    monkeypatch.setattr(methods, "delete_user", flaky)

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 500, r.text
    assert "methods" in r.json()["error"], r.text
    assert calls["n"] == 2, "the delete step was never reached"

    # A FULL snapshot, not one field: settings bytes, every method file's bytes
    # and the memory bytes must all be exactly what they were.
    assert _snapshot(home, store) == before
    # and the document's own method write was rolled back too
    assert not methods._method_file("aaa").exists()
    assert sorted(m["id"] for m in methods.user_methods()) == [
        "keepme", "mine", "other"]


def test_the_happy_path_still_restores_memory(client, home):
    """The replace must not break the section that already worked.

    On the base, the memory half of this passes (memory was always replaced);
    the failure is the two OD-15 clauses — the omitted settings key surviving
    and the un-named method surviving — so this is the regression guard for the
    section the change must NOT disturb.
    """
    store = _seed(home, method_ids=("mine",), minutes=12)
    store.add(kind="technique", text="Prefer q8_0.")
    assert len(store.all()) == 2
    doc = {"rigma_backup": 1, "settings": {},
           "methods": [], "memory": [{"id": "mem-1", "kind": "pitfall",
                                      "text": "Only the file's row."}]}

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200, r.text
    assert r.json()["memory"] == {"before": 2, "after": 1}
    assert [m["text"] for m in store.all()] == ["Only the file's row."]
    assert app_settings.load()["idle_unload_minutes"] == 0.0
    assert methods.user_methods() == []
