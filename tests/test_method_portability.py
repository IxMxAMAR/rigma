"""Methods export and import as plain JSON (spec §11)."""
import os

import pytest
from fastapi.testclient import TestClient

from rigma import macros
from rigma import state as st
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    return tmp_path


@pytest.fixture
def client():
    return TestClient(build_app(upstream_port=1))


def _doc(mid="shared", name="Shared"):
    return {"id": mid, "name": name, "tagline": "t",
            "apply": {"system_prompt": "p", "params": {}, "effort": "auto",
                      "use_tools": True, "allow_code": True,
                      "notes_template": ""},
            "macros": [{"id": "scribble", "label": "Scribble", "steps": [
                {"kind": "tool", "name": "write_file",
                 "args": {"path": "a.txt", "content": "x"}}]}]}


def test_export_returns_the_document(client):
    client.post("/api/methods", json=_doc())
    r = client.get("/api/methods/shared/export")
    assert r.status_code == 200
    assert r.json()["name"] == "Shared"
    assert "attachment" in r.headers.get("content-disposition", "")


def test_a_builtin_exports_too(client):
    assert client.get("/api/methods/book/export").json()["id"] == "book"


def test_export_404s_for_an_unknown_method(client):
    assert client.get("/api/methods/ghost/export").status_code == 404


def test_import_roundtrips(client):
    client.post("/api/methods", json=_doc())
    doc = client.get("/api/methods/shared/export").json()
    client.delete("/api/methods/shared")
    r = client.post("/api/methods/import", json=doc)
    assert r.status_code == 200
    assert client.get("/api/methods/shared").json()["name"] == "Shared"


def test_importing_a_clashing_id_does_not_overwrite(client):
    client.post("/api/methods", json=_doc(name="Mine"))
    r = client.post("/api/methods/import", json=_doc(name="Theirs"))
    assert r.status_code == 200
    assert r.json()["id"] != "shared"
    # the original survives untouched
    assert client.get("/api/methods/shared").json()["name"] == "Mine"


def test_import_validates(client):
    r = client.post("/api/methods/import", json={
        "id": "bad", "name": "Bad", "tagline": "t",
        "apply": {"system_prompt": "p", "params": {}, "effort": "auto",
                  "use_tools": True, "allow_code": True,
                  "notes_template": ""},
        "macros": [{"label": "X", "steps": [
            {"kind": "tool", "name": "teleport", "args": {}}]}]})
    assert r.status_code == 400
    assert any("teleport" in e for e in r.json()["errors"])


def test_import_never_carries_trust(client):
    """A method file is a shareable artifact. Importing someone else's must
    not import their decision to let a macro write to disk unattended."""
    doc = _doc()
    doc["always_allow"] = ["scribble"]          # a hostile / stale key
    doc["macros"][0]["always_allow"] = True
    r = client.post("/api/methods/import", json=doc)
    assert r.status_code == 200
    mid = r.json()["id"]
    assert macros.is_trusted(mid, "scribble") is False
    # and the confirm bar still stands in front of it
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": mid})
    d = client.post(f"/api/sessions/{sid}/macro/preview",
                    json={"macro_id": "scribble"}).json()
    assert d["needs_confirm"] is True


def test_an_imported_method_is_a_user_method(client):
    r = client.post("/api/methods/import", json=_doc(mid="book"))
    assert r.status_code == 200
    assert r.json()["builtin"] is False
    assert r.json()["id"] != "book"      # the built-in is not clobbered
