"""The methods HTTP surface: CRUD, macro preview and macro run (spec §5)."""
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


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]},
                {"id": "scribble", "label": "Scribble", "steps": [
                    {"kind": "tool", "name": "write_file",
                     "args": {"path": "a.txt", "content": "x"}}]}]}


def test_catalog_lists_builtins_with_components(client):
    cat = client.get("/api/methods").json()["methods"]
    ids = {m["id"] for m in cat}
    assert {"coding", "book", "roleplay", "research", "tutor",
            "organize"} <= ids
    for m in cat:
        assert "macros" in m and "rules" in m and "workflows" in m


def test_create_read_delete_a_user_method(client):
    assert client.post("/api/methods", json=_doc()).status_code == 200
    assert client.get("/api/methods/mine").json()["name"] == "Mine"
    assert client.delete("/api/methods/mine").status_code == 200
    assert client.get("/api/methods/mine").status_code == 404


def test_invalid_method_400s_with_instructional_errors(client):
    r = client.post("/api/methods", json={"id": "x", "name": "X",
                                          "apply": {"system_prompt": "p"},
                                          "macros": [{"label": "L", "steps": [
                                              {"kind": "tool",
                                               "name": "teleport"}]}]})
    assert r.status_code == 400
    assert any("teleport" in e for e in r.json()["errors"])


def test_builtins_cannot_be_deleted(client):
    assert client.delete("/api/methods/book").status_code == 400


def test_preview_reports_safe_macro_as_needing_no_confirm(client):
    client.post("/api/methods", json=_doc())
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    d = client.post(f"/api/sessions/{sid}/macro/preview",
                    json={"macro_id": "peek"}).json()
    assert d["effectful"] is False and d["needs_confirm"] is False
    assert d["asks"] == []


def test_preview_reports_effectful_macro_and_names_the_file(client):
    client.post("/api/methods", json=_doc())
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    d = client.post(f"/api/sessions/{sid}/macro/preview",
                    json={"macro_id": "scribble"}).json()
    assert d["effectful"] is True and d["needs_confirm"] is True
    assert "a.txt" in d["preview"]


def test_trusted_macro_no_longer_needs_confirm(client):
    client.post("/api/methods", json=_doc())
    macros.trust("mine", "scribble")
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    d = client.post(f"/api/sessions/{sid}/macro/preview",
                    json={"macro_id": "scribble"}).json()
    assert d["effectful"] is True and d["needs_confirm"] is False


def test_running_an_effectful_macro_without_confirm_is_refused(client):
    client.post("/api/methods", json=_doc())
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    r = client.post(f"/api/sessions/{sid}/macro", json={"macro_id": "scribble"})
    assert r.status_code == 403
    assert "confirm" in r.json()["error"]


def test_run_streams_sse_and_executes(client, monkeypatch):
    monkeypatch.setattr("rigma.tools.cached_run", lambda n, a, c: "FILE BODY")
    client.post("/api/methods", json=_doc())
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    with client.stream("POST", f"/api/sessions/{sid}/macro",
                       json={"macro_id": "peek"}) as r:
        assert r.status_code == 200
        body = "".join(r.iter_text())
    assert "macro_step" in body and "tool_result" in body
    assert "FILE BODY" in body
    assert body.rstrip().endswith("[DONE]")


def test_always_allow_persists_trust(client, monkeypatch):
    monkeypatch.setattr("rigma.tools.cached_run", lambda n, a, c: "ok")
    client.post("/api/methods", json=_doc())
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "mine"})
    with client.stream("POST", f"/api/sessions/{sid}/macro",
                       json={"macro_id": "scribble",
                             "confirm": "always"}) as r:
        "".join(r.iter_text())
    assert macros.is_trusted("mine", "scribble") is True


def test_unknown_macro_404s(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/method", json={"id": "book"})
    assert client.post(f"/api/sessions/{sid}/macro/preview",
                       json={"macro_id": "ghost"}).status_code == 404


def test_method_apply_endpoint_still_works(client):
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/method", json={"id": "coding"})
    assert r.status_code == 200
    got = client.get(f"/api/sessions/{sid}").json()
    assert got["method"] == "coding" and "engineer" in got["system_prompt"]
    assert client.post(f"/api/sessions/{sid}/method",
                       json={"id": "nope"}).status_code == 404
