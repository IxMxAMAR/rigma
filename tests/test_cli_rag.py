import sys
from pathlib import Path

from typer.testing import CliRunner

import rigma.cli as cli
from rigma import rag

runner = CliRunner()
FAKE = Path(__file__).parent / "fake_raggity_server.py"


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("RIGMA_RAGGITY_CMD", f"{sys.executable} {FAKE}")


def test_rag_add_requires_existing_path(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rag", "add", str(tmp_path / "nope")])
    assert res.exit_code == 1 and "does not exist" in res.output


def test_rag_add_ingests(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    monkeypatch.setattr(rag, "ingest", lambda: "Indexed. added=3\n")
    (tmp_path / "docs").mkdir()
    res = runner.invoke(cli.app, ["rag", "add", str(tmp_path / "docs")])
    assert res.exit_code == 0 and "added=3" in res.output


def test_rag_ask_requires_running_model(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rag", "ask", "hello?"])
    assert res.exit_code == 1 and "rigma up" in res.output


def test_rag_ask_happy_path(tmp_path, monkeypatch):
    import os

    from rigma import state as st
    _env(tmp_path, monkeypatch)
    st.write_state("m", "q", 11500, engine_pid=os.getpid(), ui_pid=os.getpid(),
                   backend="vulkan")
    monkeypatch.setattr(rag, "ensure_sidecar",
                        lambda port=rag.RAG_PORT: {"status": "ok"})
    monkeypatch.setattr(rag, "ask", lambda q, port=rag.RAG_PORT: {
        "answer": "42", "abstained": False, "citations": [{"source": "a.md"}]})
    res = runner.invoke(cli.app, ["rag", "ask", "meaning of life?"])
    assert res.exit_code == 0 and "42" in res.output and "1 citation" in res.output


def test_rag_status_not_running(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    res = runner.invoke(cli.app, ["rag", "status"])
    assert res.exit_code == 0 and "not running" in res.output.lower()
