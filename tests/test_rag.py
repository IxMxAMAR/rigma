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
