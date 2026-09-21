import sys
from pathlib import Path

import pytest

from rigma import rag

FAKE = Path(__file__).parent / "fake_raggity_server.py"


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("RIGMA_RAGGITY_CMD", f"{sys.executable} {FAKE}")
    return tmp_path


def test_add_source_persists_and_renders_toml(home, monkeypatch):
    (home / "docs").mkdir()
    (home / "kb").mkdir()
    monkeypatch.chdir(home)
    rag.add_source("docs")
    srcs = rag.add_source(str(home / "kb"))
    assert len(srcs) == 2 and all(Path(s).is_absolute() for s in srcs)
    assert rag.add_source("docs") == srcs  # dedupe
    toml_text = (rag.rag_dir() / "raggity.toml").read_text(encoding="utf-8")
    assert 'profile = "low-ram"' in toml_text
    assert 'backend = "external"' in toml_text
    assert "http://127.0.0.1:11500/v1" in toml_text
    # raggity's [sources] schema: include = [glob patterns], forward slashes
    assert "include = [" in toml_text and "/**/*" in toml_text
    assert "paths = [" not in toml_text


def test_raggity_cmd_none_when_absent(home, monkeypatch):
    monkeypatch.delenv("RIGMA_RAGGITY_CMD")
    monkeypatch.setattr(rag.shutil, "which", lambda name: None)
    assert rag.raggity_cmd() is None


def test_sidecar_lifecycle_and_endpoints(home):
    rag.add_source("docs")
    health = rag.ensure_sidecar(port=11597, timeout=30)
    try:
        assert health["status"] == "ok" and health["documents"] == 42
        # idempotent: second call reuses the live server
        assert rag.ensure_sidecar(port=11597)["version"] == "0.12.0"
        r = rag.retrieve("q", port=11597)
        assert r["chunks"][0]["text"] == "alpha"
        a = rag.ask("what is alpha?", port=11597)
        assert a["answer"].startswith("grounded:")
    finally:
        assert rag.stop_sidecar() is True
    assert rag.sidecar_health(port=11597) is None


# --- F52: the record must mean "answering", not "we once ran Popen" ----------
#
# `sidecar.json` used to be written straight after Popen, before the 90s health
# check that could still fail. Everything that decided whether to OFFER document
# search read that file, so a sidecar that never worked still got
# `search_my_documents` advertised to the model on every turn — while
# /api/rag/status, which asks the port, correctly said it was down.

DEAD_PORT = 11596          # nothing listens here in these tests


def test_the_sidecar_is_recorded_only_once_it_answers(home):
    rag.add_source("docs")
    try:
        rag.ensure_sidecar(port=11597, timeout=30)
        assert (rag.rag_dir() / "sidecar.json").is_file(), (
            "a healthy sidecar must be recorded, or nothing can reuse it")
        assert rag.live_sidecar_port() == 11597
    finally:
        rag.stop_sidecar()


def test_a_sidecar_that_never_answers_leaves_no_record(home, monkeypatch):
    # A command that exits immediately: it never becomes healthy, so this is the
    # failure path that used to leave a record of a process that never worked.
    monkeypatch.setenv("RIGMA_RAGGITY_CMD",
                       f"{sys.executable} -c pass")
    rag.add_source("docs")
    with pytest.raises(RuntimeError):
        rag.ensure_sidecar(port=DEAD_PORT, timeout=5)
    assert not (rag.rag_dir() / "sidecar.json").exists(), (
        "a failed start left a record, which is the whole of F52")
    assert rag.live_sidecar_port() is None


def test_a_stale_record_is_not_mistaken_for_a_live_sidecar(home):
    """A record whose process is gone must not read as available, and must not
    be left behind to be discovered again."""
    rag.rag_dir().mkdir(parents=True, exist_ok=True)
    (rag.rag_dir() / "sidecar.json").write_text(
        '{"pid": 999999, "port": %d}' % DEAD_PORT, encoding="utf-8")
    assert rag.recorded_sidecar_port() == DEAD_PORT, (
        "the raw record is still readable — that is what it is for")
    assert rag.live_sidecar_port() is None
    assert not (rag.rag_dir() / "sidecar.json").exists(), (
        "the stale record should be cleared once found to be stale")


def test_a_stale_record_cannot_advertise_document_search(home):
    """The interaction that made F52 matter: Rigma's MCP server gates the arm's
    roster on the sidecar, so a stale record offered the arm a tool that could
    only fail. The other two tools must survive — they do not need documents."""
    from rigma import mcp_server

    rag.rag_dir().mkdir(parents=True, exist_ok=True)
    (rag.rag_dir() / "sidecar.json").write_text(
        '{"pid": 999999, "port": %d}' % DEAD_PORT, encoding="utf-8")

    names = {t["name"] for t in mcp_server.offered()}
    assert "search_my_documents" not in names, (
        "a dead sidecar must not put a document tool in front of the arm")
    assert {"remember", "recall"} <= names, (
        "the memory tools do not need documents and must still be offered")

