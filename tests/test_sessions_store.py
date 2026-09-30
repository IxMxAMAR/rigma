"""OD-2: the copy/move destination allowlist and its session-store seed.

`default_write_allowlist` is METADATA-ONLY. It reads the `workspace` of the
sessions already in the store — through SQLite's JSON extractor, so no message
body leaves the database — plus the folders registered as RAG sources. An
empty store seeds nothing, which is exactly the behaviour before the field
existed (an absolute destination then needs the blanket grant).
"""
import json
import pathlib

import pytest
from fastapi.testclient import TestClient

from rigma import serve, sessions, tools


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    return tmp_path


def _mk(parent, name):
    d = parent / name
    d.mkdir()
    return d


def test_default_write_allowlist_is_empty_for_an_empty_home(tmp_path,
                                                            monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "fresh"))
    assert sessions.default_write_allowlist() == []
    assert sessions.default_write_allowlist(home=tmp_path / "fresh") == []


def test_default_write_allowlist_seeds_from_stored_workspaces(home):
    alpha = _mk(home, "alpha")
    beta = _mk(home, "beta")
    s1 = sessions.create("a")
    s1["workspace"] = str(alpha)
    sessions.save(s1)
    s2 = sessions.create("b")
    s2["workspace"] = str(beta)
    sessions.save(s2)
    s3 = sessions.create("c")
    s3["workspace"] = str(alpha)          # a duplicate must be deduped
    sessions.save(s3)

    got = sessions.default_write_allowlist()
    assert got == [str(alpha.resolve()), str(beta.resolve())]
    # the explicit-home form reads that store directly, without RIGMA_HOME
    assert sessions.default_write_allowlist(home=home / "rigma-home") == got
    # a session with no workspace contributes nothing
    assert "" not in got


def test_a_new_session_is_seeded_with_the_folders_already_in_use(home):
    used = _mk(home, "used")
    s1 = sessions.create("first")
    s1["workspace"] = str(used)
    sessions.save(s1)

    s2 = sessions.create("second")        # no explicit allowlist
    assert str(used.resolve()) in s2["write_allowlist"]

    # an explicit list — including an empty one — is honoured as given
    s3 = sessions.create("third", write_allowlist=[])
    assert s3["write_allowlist"] == []


def test_the_seed_includes_rag_source_folders(home):
    rag = home / "rigma-home" / "rag"
    rag.mkdir(parents=True, exist_ok=True)
    src = _mk(home, "documents")
    (rag / "sources.json").write_text(json.dumps([str(src)]),
                                      encoding="utf-8")
    assert sessions.default_write_allowlist() == [str(src.resolve())]


def test_a_file_or_glob_rag_source_is_not_a_write_root(home):
    rag = home / "rigma-home" / "rag"
    rag.mkdir(parents=True, exist_ok=True)
    src = _mk(home, "documents")
    a_file = src / "note.md"
    a_file.write_text("x", encoding="utf-8")
    (rag / "sources.json").write_text(
        json.dumps([str(a_file), str(src / "**" / "*.md")]), encoding="utf-8")
    assert sessions.default_write_allowlist() == []


def test_a_broad_stored_workspace_is_not_seeded(home):
    """ODR-2: one old chat whose workspace was the home dir (or a drive root,
    or an ancestor of %APPDATA%, or a persistence folder) must not hand every
    NEW chat that root as an absolute write destination. The narrow folder is
    the control: the floor must not touch the normal case."""
    import os as _os

    narrow = _mk(home, "narrow")
    broad = {"home": pathlib.Path.home(),
             "drive root": pathlib.Path(pathlib.Path.home().anchor)}
    appdata = _os.environ.get("APPDATA")
    if appdata:
        broad["appdata ancestor"] = pathlib.Path(appdata).parent
        broad["startup"] = (pathlib.Path(appdata) / "Microsoft" / "Windows"
                            / "Start Menu" / "Programs" / "Startup")
    for name, ws in broad.items():
        s = sessions.create(f"broad-{name}")
        s["workspace"] = str(ws)
        sessions.save(s)
    s = sessions.create("narrow")
    s["workspace"] = str(narrow)
    sessions.save(s)

    got = sessions.default_write_allowlist()
    assert str(narrow.resolve()) in got, got
    for name, ws in broad.items():
        assert str(ws.resolve()) not in got, (name, got)


def test_a_relative_stored_workspace_is_ignored_not_resolved(home):
    """ODR-2: `resolve()` used to run BEFORE `is_absolute()`, so the check was
    dead — a stored workspace of `.` became the SERVER's current directory and
    was seeded into every new chat."""
    s = sessions.create("rel")
    s["workspace"] = "."
    sessions.save(s)
    got = sessions.default_write_allowlist()
    assert str(pathlib.Path.cwd().resolve()) not in got, got
    assert got == [], got


def test_write_allowlist_is_a_list_not_a_bool():
    assert "write_allowlist" in sessions.MUTABLE_FIELDS
    assert sessions._FIELD_TYPES.get("write_allowlist") is list
    assert sessions._SESSION_DEFAULTS["write_allowlist"] == []
    # a string must be refused rather than iterated character by character
    with pytest.raises(Exception):
        sessions.validate_field_types({"write_allowlist": "false"})


def test_write_allowlist_round_trips_through_the_session_api(home):
    client = TestClient(serve.build_app(upstream_port=1, default_prompt=""))
    sid = client.post("/api/sessions", json={"title": "w"}).json()["id"]
    root = _mk(home, "drop")
    r = client.post(f"/api/sessions/{sid}",
                    json={"write_allowlist": [str(root)]})
    assert r.status_code == 200, r.text
    assert sessions.load(sid)["write_allowlist"] == [str(root)]
    # the GET surface exposes it too, so a UI can display/clear it
    assert client.get(f"/api/sessions/{sid}").json()["write_allowlist"] == \
        [str(root)]


def test_every_tool_ctx_the_product_builds_names_the_allowlist():
    """A ctx that omits the field reads as "no roots" to the tool, so the
    allowlisted destination would silently fail in the product."""
    serve_src = (pathlib.Path(tools.__file__).parent
                 / "serve.py").read_text(encoding="utf-8")
    assert serve_src.count('"write_allowlist"') >= 3, serve_src.count(
        '"write_allowlist"')
    # the narrowed delegate ctx keeps none of the session's roots
    assert '"write_allowlist": []' in serve_src
