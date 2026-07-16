"""Repo packer: folder -> prompt block, with skips and caps."""
import pytest

from rigma import workspace
from rigma.workspace import WorkspaceError


def test_pack_includes_text_skips_binary_and_junk_dirs(tmp_path):
    (tmp_path / "main.py").write_text("print('hi')", encoding="utf-8")
    (tmp_path / "README.md").write_text("# docs", encoding="utf-8")
    (tmp_path / "logo.png").write_bytes(b"\x89PNG\x00\x00")
    node = tmp_path / "node_modules" / "dep"
    node.mkdir(parents=True)
    (node / "index.js").write_text("module.exports={}", encoding="utf-8")
    git = tmp_path / ".git"
    git.mkdir()
    (git / "config").write_text("[core]", encoding="utf-8")
    out = workspace.pack_folder(str(tmp_path))
    assert out["file_count"] == 2                 # main.py + README.md only
    assert 'path="main.py"' in out["content"]
    assert "logo.png" not in out["content"]       # binary skipped
    assert "node_modules" not in out["content"]   # junk dir skipped
    assert ".git" not in out["content"]
    assert not out["truncated"]


def test_pack_caps_total_size_and_flags_truncated(tmp_path):
    for i in range(20):
        (tmp_path / f"f{i}.py").write_text("x" * 1000, encoding="utf-8")
    out = workspace.pack_folder(str(tmp_path), max_total=5000)
    assert out["truncated"] is True
    assert out["chars"] <= 5000


def test_pack_skips_oversized_file(tmp_path):
    (tmp_path / "small.py").write_text("ok", encoding="utf-8")
    (tmp_path / "huge.py").write_text("y" * 300_000, encoding="utf-8")
    out = workspace.pack_folder(str(tmp_path))
    assert "small.py" in out["content"] and "huge.py" not in out["content"]


def test_pack_errors_cleanly(tmp_path):
    with pytest.raises(WorkspaceError, match="not a folder"):
        workspace.pack_folder(str(tmp_path / "ghost"))
    empty = tmp_path / "empty"
    empty.mkdir()
    (empty / "pic.png").write_bytes(b"\x89PNG")
    with pytest.raises(WorkspaceError, match="no readable text"):
        workspace.pack_folder(str(empty))


def test_pack_endpoint(tmp_path, monkeypatch):
    import os
    from http.server import BaseHTTPRequestHandler, HTTPServer
    import threading

    from fastapi.testclient import TestClient
    from rigma.serve import build_app
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    (tmp_path / "proj").mkdir()
    (tmp_path / "proj" / "a.py").write_text("pass", encoding="utf-8")

    class _U(BaseHTTPRequestHandler):
        def do_GET(self): self.send_response(200); self.end_headers()
        def log_message(self, *a): pass
    srv = HTTPServer(("127.0.0.1", 0), _U)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    client = TestClient(build_app(upstream_port=srv.server_address[1]))
    r = client.post("/api/workspace/pack", json={"folder": str(tmp_path / "proj")})
    assert r.status_code == 200 and r.json()["file_count"] == 1
    r = client.post("/api/workspace/pack", json={"folder": ""})
    assert r.status_code == 400
    r = client.post("/api/workspace/pack",
                    json={"folder": str(tmp_path / "nope")})
    assert r.status_code == 400
    srv.shutdown()
