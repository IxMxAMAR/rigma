"""R3 adversarial-review regressions: the tool-surface and store fixes.

Every test here fails against the code as it was before the fix it names. They
are grouped in one file because they share a shape — each one is a case the
existing suite passed *through*, not a case it missed by accident:

* the credential denylist was enforced on the tools that take a NAME and absent
  on the two that take a PATTERN, and both suites only ever tested names;
* `move_files` reached any absolute destination and `test_audit_sec13` asserted
  that as correct;
* `edit_file` had no size cap while `read_file` did, so nothing compared them;
* the spill file was written outside the workspace and the test that covered it
  read the file with `pathlib`, never with `read_file`.
"""
from __future__ import annotations

import json

import pytest

from rigma import atomicio, bench, tools


@pytest.fixture()
def ws(tmp_path):
    """A workspace holding one credential file, one key, and one ordinary file."""
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-super-secret\n",
                                   encoding="utf-8")
    (tmp_path / "secret.pem").write_text("-----BEGIN PRIVATE KEY-----\n",
                                         encoding="utf-8")
    (tmp_path / "app.py").write_text("print('hello')\n", encoding="utf-8")
    ssh = tmp_path / ".ssh"
    ssh.mkdir()
    (ssh / "id_rsa").write_text("SECRETKEYMATERIAL\n", encoding="utf-8")
    return tmp_path


@pytest.fixture()
def ctx(ws):
    return {"workspace": str(ws), "allow_code": True, "profile": "all"}


# --- R3-TOOL-1: the credential denylist belongs in the shared walker ----------

def test_grep_does_not_read_credential_files(ctx):
    """`grep` took a pattern and applied no denylist at all.

    With the default workspace (the home directory) `grep {"pattern": "API_KEY",
    "glob": "**/.env"}` returned the contents of `.env`, while
    `read_file(".env")` refused the same file. The content could then leave
    through `fetch_url`, which is an auto-run GET that the outbound-POST grant
    does not gate.
    """
    for args in ({"pattern": "SECRETKEYMATERIAL"},
                 {"pattern": "BEGIN", "glob": "**/id_rsa"},
                 {"pattern": "API_KEY", "glob": "**/.env"}):
        out = tools.run_tool("grep", args, ctx)
        assert out == "no matches", (args, out)
        assert "SECRET" not in out and "sk-super" not in out


def test_grep_still_finds_ordinary_files(ctx):
    """The denylist must not turn grep into a no-op — a filter that hides
    everything passes the test above while breaking the tool."""
    out = tools.run_tool("grep", {"pattern": "hello"}, ctx)
    assert "app.py" in out and "print" in out


def test_find_files_hides_credential_names(ctx):
    """`find_files` shares the walker, so it inherits the filter."""
    out = tools.run_tool("find_files", {"pattern": "**/*"}, ctx)
    assert "app.py" in out
    assert "id_rsa" not in out and ".env" not in out and "secret.pem" not in out


# --- R3-TOOL-3: the denylist is checked on the RESOLVED hit -------------------

@pytest.mark.parametrize("pattern", [".en*", "sec*", "*.pem", ".ss*"])
def test_read_file_glob_cannot_resolve_onto_a_credential(ctx, pattern):
    """The denylist ran on the literal name the model typed, never on the file
    the glob resolved to. `read_file(".env")` refused; `read_file(".en*")`
    returned the same bytes, because the literal does not exist so the check
    passed it and `_glob_under` tested containment and nothing else."""
    out = tools.run_tool("read_file", {"path": pattern}, ctx)
    assert "sk-super" not in out and "SECRET" not in out and "PRIVATE KEY" not in out
    assert "error" in out, out


def test_read_file_glob_still_reads_an_ordinary_file(ctx):
    out = tools.run_tool("read_file", {"path": "app*"}, ctx)
    assert "print" in out and "hello" in out


# --- R3-TOOL-4: an out-of-workspace write is its own grant --------------------

def test_absolute_write_needs_its_own_grant(ws, tmp_path):
    """`move_files`/`copy_files` reached any absolute destination on disk needing
    only the default-on `allow_code`, so the WEAKER capability (`write_file`,
    always pinned to the workspace) was the more confined one. A planted .bat in
    the Startup folder is persistence, not data loss."""
    (ws / "a.txt").write_text("x", encoding="utf-8")
    dest = tmp_path / "elsewhere"
    base = {"workspace": str(ws), "allow_code": True}

    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)}, base)
    assert out.startswith("error") and "outside the workspace" in out
    assert not dest.exists()

    # reading outside is a DIFFERENT risk and must not grant writing outside
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {**base, "allow_absolute_reads": True})
    assert out.startswith("error"), out

    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "a.txt")], "dest": str(dest)},
                         {**base, "allow_absolute_writes": True})
    assert out.startswith("moved 1 file(s)"), out
    assert (dest / "a.txt").is_file()


def test_relative_transfer_still_works_without_any_grant(ws):
    """The grant gates the OUT-of-workspace case only."""
    (ws / "a.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("move_files", {"paths": ["a.txt"], "dest": "sorted"},
                         {"workspace": str(ws), "allow_code": True})
    assert out.startswith("moved 1 file(s)"), out
    assert (ws / "sorted" / "a.txt").is_file()


def test_the_write_grant_is_a_declared_session_field():
    """A grant the session schema does not name cannot be stored, cannot be
    displayed, and cannot be set — so the check above would be unreachable in
    the product."""
    from rigma import sessions
    assert "allow_absolute_writes" in sessions.MUTABLE_FIELDS
    assert sessions._FIELD_TYPES.get("allow_absolute_writes") is bool
    d = sessions.create("a fresh chat")
    assert d["allow_absolute_writes"] is False
    # the quoted "false" must not read as a grant, exactly like its siblings
    with pytest.raises(Exception):
        sessions.validate_field_types({"allow_absolute_writes": "false"})


def test_every_ctx_the_product_builds_names_the_write_grant():
    """A grant the ctx omits is not "unset" to the tool, it is false — so a
    missing key makes the UI control a silent no-op."""
    import pathlib
    src = pathlib.Path(tools.__file__).parent
    serve = (src / "serve.py").read_text(encoding="utf-8")
    assert serve.count('"allow_absolute_writes"') >= 3, serve.count(
        '"allow_absolute_writes"')
    mcp = (src / "mcp_server.py").read_text(encoding="utf-8")
    assert '"allow_absolute_writes": allow_absolute_writes()' in mcp


# --- R3-TOOL-5: view_image is a read and needs the read grant ----------------

def test_view_image_absolute_needs_the_read_grant(ws, tmp_path):
    """This was the one read path that ignored `allow_absolute_reads`, while
    `view_images(folder=...)`, `list_directory` and `read_file` enforced it. A
    prompt-injected model could base64 any image off the disk, and from there it
    could leave through `fetch_url`."""
    outside = tmp_path.parent / (tmp_path.name + "-outside")
    outside.mkdir()
    shot = outside / "shot.png"
    shot.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)

    base = {"workspace": str(ws), "allow_code": True, "has_vision": True}
    out = tools.run_tool("view_image", {"path": str(shot)}, base)
    assert "outside the workspace" in out, out
    assert tools.IMAGE_SENTINEL not in out

    # the grant restores it (the sentinel is what carries the bytes)
    out = tools.run_tool("view_image", {"path": str(shot)},
                         {**base, "allow_absolute_reads": True})
    assert tools.IMAGE_SENTINEL in out, out


# --- R3-TOOL-6: edit_file has the same ceiling as read_file ------------------

def test_edit_file_refuses_a_file_too_large_to_read(ws):
    """`edit_file` read the whole file and then read it AGAIN for the undo
    snapshot, with no size guard — while `read_file` refused the same file at
    8 MB. A 10 GB log was pulled into the process holding the model's RAM."""
    big = ws / "big.txt"
    big.write_text("NEEDLE\n" + "x" * (tools._EDIT_MAX_BYTES + 1000),
                   encoding="utf-8")
    base = {"workspace": str(ws), "allow_code": True}

    read = tools.run_tool("read_file", {"path": "big.txt"}, base)
    assert "too large" in read

    out = tools.run_tool("edit_file",
                         {"path": "big.txt", "old": "NEEDLE", "new": "PATCHED"},
                         base)
    assert out.startswith("error") and "too large" in out, out
    # and it must not have been touched
    assert big.read_text(encoding="utf-8").startswith("NEEDLE")


def test_edit_file_still_edits_an_ordinary_file(ws):
    out = tools.run_tool("edit_file",
                         {"path": "app.py", "old": "hello", "new": "goodbye"},
                         {"workspace": str(ws), "allow_code": True})
    assert out.startswith("edited"), out
    assert "goodbye" in (ws / "app.py").read_text(encoding="utf-8")


# --- R3-TOOL-7: the spill recovery path must be readable --------------------

def test_spilled_result_is_readable_by_the_tool_it_names(ws):
    """The model was told `read_file path="<rigma_home>/results/..."`, but the
    read gate refuses any absolute path outside the workspace without a grant.
    Every project workspace and every autonomous run therefore got a truncated
    result plus an instruction that could not succeed. The test that covered
    this read the file with `pathlib`, so "a way back" was asserted without
    being exercised."""
    from rigma import serve
    big = "x" * (serve.SPILL_CHARS + 500)
    out = serve._spill_big_result("grep", big, {"workspace": str(ws)})

    line = next(ln for ln in out.splitlines() if "read_file path=" in ln)
    rel = line.split('path="')[1].split('"')[0]
    assert not __import__("pathlib").Path(rel).is_absolute(), rel

    back = tools.run_tool("read_file", {"path": rel, "offset": 1, "limit": 5},
                          {"workspace": str(ws), "allow_code": True})
    assert not back.startswith("error"), back
    assert back.startswith("x")


def test_spilled_result_is_hidden_from_the_models_own_searches(ws):
    from rigma import serve
    serve._spill_big_result("grep", "y" * (serve.SPILL_CHARS + 10),
                            {"workspace": str(ws)})
    found = tools.run_tool("find_files", {"pattern": "**/*"},
                           {"workspace": str(ws), "allow_code": True})
    assert ".rigma-results" not in found, found


# --- R3-STORE-1/2: atomic stores and a record that predates a field ----------

def test_atomic_write_leaves_no_temp_behind(tmp_path):
    p = tmp_path / "x.json"
    atomicio.atomic_write_json(p, {"a": 1})
    assert json.loads(p.read_text(encoding="utf-8")) == {"a": 1}
    assert [f.name for f in tmp_path.iterdir()] == ["x.json"]


def test_atomic_write_survives_a_concurrent_writer(tmp_path):
    """A fixed `<name>.tmp` meant two concurrent writers replaced each other's
    file mid-write, or the rename failed with `PermissionError [WinError 32]` —
    which `memory.py` documents fixing for itself and nothing else did."""
    import threading
    p = tmp_path / "shared.json"
    errs = []

    def writer(n):
        try:
            for _ in range(40):
                atomicio.atomic_write_json(p, {"n": n})
        except Exception as e:          # noqa: BLE001 - the point of the test
            errs.append(e)

    ts = [threading.Thread(target=writer, args=(i,)) for i in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert errs == [], errs
    assert json.loads(p.read_text(encoding="utf-8")) in [{"n": i} for i in range(4)]


def test_prune_keeps_two_models_on_one_gpu():
    """`prune_calibration` grouped by HARDWARE IDENTITY ALONE, so calibrating a
    second model evicted the first — and because `save_calibration` prunes on
    every save, two models on one card oscillated and neither ever stayed
    calibrated."""
    ident = {"id": "digest123", "name": "RX 9070 XT"}
    cal = {f"model{n}:Q4:vulkan:digest123":
           {"hardware": ident, "date": f"2026-09-0{n}"} for n in (1, 2, 3)}
    kept = bench.prune_calibration(cal)
    assert len(kept) == 3, sorted(kept)


def test_prune_still_bounds_growth_across_gpus():
    """The cap must still exist: a third measurement of the SAME model on the
    SAME card keeps only the newest."""
    ident = {"id": "digest123"}
    cal = {f"modelA:Q4:vulkan:digest123:{d}": {"hardware": ident, "date": d}
           for d in ("2026-09-01", "2026-09-02", "2026-09-03")}
    kept = bench.prune_calibration(cal)
    assert list(kept) == ["modelA:Q4:vulkan:digest123:2026-09-03"], sorted(kept)


def test_update_state_tolerates_a_record_that_predates_engine(tmp_path,
                                                              monkeypatch):
    """`_FIELD_DEFAULTS` omitted `engine` while `_write_record` did
    `rec["engine"]`, so `perform_unload()` on a state.json written before the
    field existed raised KeyError — AFTER the engine had already been killed,
    leaving a record claiming a live engine."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import state as st
    st.state_path().write_text(json.dumps(
        {"model": "m", "quant": "Q4", "backend": "vulkan"}), encoding="utf-8")
    st.update_state(unloaded=True)          # must not raise
    assert st.read_state()["unloaded"] is True
