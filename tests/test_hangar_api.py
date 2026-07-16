"""Hangar HTTP surface + default_params layering through a real turn."""
import json
import struct
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app

T_U32, T_STR = 4, 8


def _s(b):
    return struct.pack("<Q", len(b)) + b


def _gguf(tmp_path, name=b"Drop Test 4B", fname="DropTest-Q5_K_M.gguf"):
    kvs = [
        _s(b"general.architecture") + struct.pack("<I", T_STR) + _s(b"qwen3"),
        _s(b"general.name") + struct.pack("<I", T_STR) + _s(name),
        _s(b"qwen3.block_count") + struct.pack("<I", T_U32) + struct.pack("<I", 8),
        _s(b"qwen3.context_length") + struct.pack("<I", T_U32) + struct.pack("<I", 16384),
        _s(b"qwen3.embedding_length") + struct.pack("<I", T_U32) + struct.pack("<I", 1024),
        _s(b"qwen3.attention.head_count") + struct.pack("<I", T_U32) + struct.pack("<I", 16),
        _s(b"qwen3.attention.head_count_kv") + struct.pack("<I", T_U32) + struct.pack("<I", 2),
    ]
    p = tmp_path / fname
    p.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                  + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"\x00" * 256)
    return p


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


class _Echo(BaseHTTPRequestHandler):
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _Echo.last_body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
            b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_models_list_and_path_install(home, tmp_path, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    src = _gguf(tmp_path)
    r = client.post("/api/models/install", json={"path": str(src)})
    assert r.status_code == 200 and r.json()["slug"] == "drop-test-4b"
    r = client.get("/api/models")
    assert r.status_code == 200
    slugs = {m["slug"]: m for m in r.json()["models"]}
    assert slugs["drop-test-4b"]["custom"] is True
    assert slugs["drop-test-4b"]["quants"][0]["on_disk"] is True
    assert r.json()["disk"]["free_gb"] > 0


def test_install_error_is_400_with_message(home, tmp_path, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.post("/api/models/install", json={"path": str(tmp_path / "no.gguf")})
    assert r.status_code == 400 and "no such file" in r.json()["error"]
    r = client.post("/api/models/install", json={"path": ""})
    assert r.status_code == 400


def test_upload_streams_body_and_installs(home, tmp_path, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    src = _gguf(tmp_path, name=b"Uploaded Tune", fname="Uploaded-Q4_0.gguf")
    r = client.post("/api/models/upload?filename=Uploaded-Q4_0.gguf",
                    content=src.read_bytes())
    assert r.status_code == 200 and r.json()["slug"] == "uploaded-tune"
    assert (home / "models" / "Uploaded-Q4_0.gguf").exists()
    # nothing left in the inbox either way
    assert not list((home / "custom" / "incoming").glob("*"))


def test_upload_rejects_traversal_and_junk(home, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.post("/api/models/upload?filename=../../evil.txt", content=b"x")
    assert r.status_code == 400
    r = client.post("/api/models/upload?filename=junk.gguf", content=b"NOPE")
    assert r.status_code == 400 and "not a GGUF" in r.json()["error"]
    assert not list((home / "custom" / "incoming").glob("*"))


def test_delete_and_patch_routes(home, tmp_path, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    client.post("/api/models/install", json={"path": str(_gguf(tmp_path))})
    r = client.patch("/api/models/drop-test-4b",
                     json={"capabilities": ["tools"]})
    assert r.status_code == 200 and r.json()["capabilities"] == ["tools"]
    r = client.patch("/api/models/qwen3.6-35b-a3b",
                     json={"capabilities": ["tools"]})
    assert r.status_code == 400
    r = client.delete("/api/models/drop-test-4b/files/DropTest-Q5_K_M.gguf")
    assert r.status_code == 200
    r = client.delete("/api/models/drop-test-4b")
    assert r.status_code == 200
    slugs = [m["slug"] for m in client.get("/api/models").json()["models"]]
    assert "drop-test-4b" not in slugs


def test_pull_unknown_file_is_400(home, upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.post("/api/models/qwen3.6-35b-a3b/pull",
                    json={"file": "nope.gguf"})
    assert r.status_code == 400


def test_model_default_params_reach_upstream_but_session_wins(home, tmp_path,
                                                              upstream):
    """default_params flow: model card < session override."""
    import os

    from rigma import state as st
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    from rigma.registry import Registry
    spec = ModelSpec(slug="tuned", family="f", kind="dense", n_layers=8,
                     full_attn_layers=8, kv_heads=2, head_dim=64,
                     native_ctx=8192,
                     ggufs=[GgufFile(repo="r", file="t.gguf", bytes=1,
                                     quant="Q4")],
                     use_cases=["general"], cache_type_policy=CachePolicy(),
                     default_params={"temperature": 0.7, "dry_multiplier": 0.8})
    reg = Registry([], {"tuned": spec}, {})
    st.write_state("tuned", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=upstream, registry=reg))
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert _Echo.last_body["temperature"] == 0.7
    assert _Echo.last_body["dry_multiplier"] == 0.8
    client.post(f"/api/sessions/{sid}", json={"params": {"temperature": 1.2}})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "again"})
    assert _Echo.last_body["temperature"] == 1.2          # session wins
    assert _Echo.last_body["dry_multiplier"] == 0.8       # default persists
