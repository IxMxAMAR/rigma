"""A11 / R3-3: `/api/restore` was not atomic — a mid-loop write failure left a
half-restored store.

Settings were saved, then each method was written with a plain `write_text`,
then memory was replaced. An `OSError` on the Nth method left settings and the
first N-1 methods applied and memory untouched, and the client saw only a 500.
The validate pass was already all-or-nothing; the APPLY pass now is too, via an
undo log around the stores' own writers, and the 500 names the section that
failed.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import app_settings, atomicio, methods
from rigma.memory import MemoryStore
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


@pytest.fixture
def client(home):
    # raise_server_exceptions=False: A11 is that the endpoint answers a NAMED
    # 500 instead of letting the OSError escape as a bare traceback.
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


def _seed(home):
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="Never type filenames.")
    app_settings.save({"idle_unload_minutes": 12})
    saved, errs = methods.save_user(_doc())
    assert saved is not None and not errs, errs
    return store


def _bytes(p):
    return p.read_bytes() if p.exists() else None


def _error(r):
    try:
        return r.json().get("error", "")
    except Exception:
        return r.text


def test_a_failed_second_section_leaves_the_first_byte_identical(
        client, home, monkeypatch):
    """The settings write is the FIRST section; the method writes are the
    second. Failing the second method write must leave settings byte-identical
    to before (not 99 from the backup) and name the stage."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99
    doc["methods"] = [_doc("aaa"), _doc("bbb")]
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]

    settings_p = app_settings.settings_path()
    memory_p = store.path
    mine_p = methods._method_file("mine")
    before = {"settings": _bytes(settings_p), "memory": _bytes(memory_p),
              "mine": _bytes(mine_p)}

    real = methods.save_user
    calls = {"n": 0}

    def flaky(full):
        calls["n"] += 1
        if calls["n"] == 2:                 # the SECOND method file
            raise OSError("no space left on device")
        return real(full)

    monkeypatch.setattr(methods, "save_user", flaky)

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 500, r.text
    assert "methods" in _error(r), _error(r)

    # the FIRST section is byte-identical to before
    assert _bytes(settings_p) == before["settings"]
    # and so is everything else the restore had not reached or had to undo
    assert _bytes(memory_p) == before["memory"]
    assert _bytes(mine_p) == before["mine"]
    assert not methods._method_file("aaa").exists(), "the first method stayed"
    assert not methods._method_file("bbb").exists()


def test_a_failed_memory_section_rolls_back_settings_and_methods(
        client, home, monkeypatch):
    """The third section is the last write. Its failure must put the first two
    back, or the client is left with a settings change it cannot see."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99
    doc["methods"] = [_doc("aaa")]
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]

    settings_p = app_settings.settings_path()
    memory_p = store.path
    before = {"settings": _bytes(settings_p), "memory": _bytes(memory_p)}

    def boom(self, rows):
        raise OSError("no space left on device")

    monkeypatch.setattr(MemoryStore, "restore", boom)

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 500, r.text
    assert "memory" in _error(r), _error(r)

    assert _bytes(settings_p) == before["settings"]
    assert _bytes(memory_p) == before["memory"]
    assert not methods._method_file("aaa").exists()


def test_a_clean_restore_still_applies_every_section(client, home):
    """The transaction must not turn a good restore into a no-op."""
    store = _seed(home)
    doc = client.get("/api/backup").json()
    doc["settings"]["idle_unload_minutes"] = 99
    doc["methods"] = [_doc("aaa")]
    doc["memory"] = [{"id": "mem-1", "kind": "technique",
                      "text": "Prefer q8_0."}]

    r = client.post("/api/restore", json=doc)
    assert r.status_code == 200, r.text
    assert app_settings.load()["idle_unload_minutes"] == 99.0
    assert methods.get("aaa") is not None
    # OD-15: the document's method set is now the whole set. "mine" was in the
    # store but not in this document, so a true replace deletes it (the old
    # merge kept it and the test said "additive, as before").
    assert methods.get("mine") is None
    assert [m["text"] for m in store.all()] == ["Prefer q8_0."]


def test_atomic_write_bytes_is_byte_exact(home):
    """The undo log rests on this: a CRLF method file must come back CRLF, not
    silently LF, or the rollback is not the file that was there."""
    p = home / "m.json"
    original = b'{\r\n  "id": "mine"\r\n}'
    p.write_bytes(original)
    assert atomicio.atomic_write_bytes(p, b'{"x": 1}') is True
    assert p.read_bytes() == b'{"x": 1}'
    atomicio.atomic_write_bytes(p, original)
    assert p.read_bytes() == original
