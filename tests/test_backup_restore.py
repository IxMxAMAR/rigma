"""IMP-12: settings + methods + memory in one versioned file.

Moving to a new machine, or recovering from a bad experiment, used to mean
copying several files by hand and knowing where they live. The version check
and the validate-everything-first rule are the parts that make a restore safe:
an unknown document must be refused, and a corrupt one must change NOTHING.
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
    return TestClient(build_app(upstream_port=1))


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]}]}


def _seed(home):
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    app_settings.save({"idle_unload_minutes": 12})
    saved, errs = methods.save_user(_doc())
    assert saved is not None and not errs, errs
    return store


def test_backup_round_trip(client, home):
    store = _seed(home)
    doc = client.get("/api/backup").json()
    assert doc["rigma_backup"] == 1
    assert doc["app_version"]
    assert doc["settings"]["idle_unload_minutes"] == 12.0
    assert [m["id"] for m in doc["methods"]] == ["mine"]
    assert [m["text"] for m in doc["memory"]] == ["Never type filenames."]
    # vec is derived from the text and 768 floats per row: not in the file
    assert all("vec" not in m for m in doc["memory"])

    # wipe everything, then restore from the document alone
    store.delete_many([r["id"] for r in store.all()])
    assert methods.delete_user("mine") is True
    app_settings.save({"idle_unload_minutes": 0})

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200
    body = r.json()
    assert body["restored"] is True and body["methods"] == 1
    assert body["memory"] == {"before": 0, "after": 1}
    assert app_settings.load()["idle_unload_minutes"] == 12.0
    assert methods.get("mine")["name"] == "Mine"
    assert [m["text"] for m in store.all()] == ["Never type filenames."]


def test_restore_refuses_an_unknown_version(client, home):
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["rigma_backup"] = 99
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 400 and "version" in r.json()["error"]
    # refused means NOTHING was applied
    assert app_settings.load()["idle_unload_minutes"] == 12.0
    assert len(store.all()) == 1
    assert methods.get("mine") is not None

    # a document with no version at all is the same refusal
    assert client.post("/api/restore",
                       json={"settings": {}}).status_code == 400


def test_restore_validates_the_whole_document_first(client, home):
    """A corrupt memory section must not leave a half-applied restore — the
    settings write is the thing that would have landed."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99          # valid on its own
    doc["memory"] = [{"id": "x"}]                        # no kind/text
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 400 and "memory" in r.json()["error"]
    assert app_settings.load()["idle_unload_minutes"] == 12.0
    assert len(store.all()) == 1

    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99
    doc["methods"] = [{"id": "bad"}]                     # no name/apply
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 400 and "method 0" in r.json()["error"]
    assert app_settings.load()["idle_unload_minutes"] == 12.0

    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = -1          # invalid value
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 400 and "idle_unload_minutes" in r.json()["error"]


def test_restore_replaces_the_memory_store(client, home):
    store = _seed(home)
    doc = client.get("/api/backup").json()
    store.add(kind="technique", text="Prefer q8_0.")
    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200
    assert r.json()["memory"] == {"before": 2, "after": 1}
    assert len(store.all()) == 1
