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


def test_unload_then_load_roundtrip(home, upstream, monkeypatch):
    """Unload frees the engine but keeps state (UI alive); load relaunches."""
    import os

    from rigma import server_ops
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=999999999,
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    killed = []
    monkeypatch.setattr(st, "kill_pid", lambda pid: killed.append(pid))
    client = TestClient(build_app(upstream_port=upstream))
    r = client.post("/api/server/unload")
    assert r.status_code == 200 and r.json()["unloaded"] is True
    assert killed == [999999999]
    assert st.server_running() is not None          # UI pid keeps state alive
    r = client.get("/api/status")
    assert r.status_code == 200 and r.json()["unloaded"] is True
    r = client.post("/api/server/unload")           # double unload -> 409
    assert r.status_code == 409
    # load relaunches via perform_switch; stub it to observe the call
    monkeypatch.setattr(server_ops, "perform_switch",
                        lambda model, reg=None, prof=None: {"model": model,
                                                            "ok": True})
    r = client.post("/api/server/load")
    assert r.status_code == 200 and r.json()["model"] == "m"


def test_unloaded_chat_gets_honest_error_event(home, upstream, monkeypatch):
    import os

    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=-1, ui_pid=os.getpid(),
                   unloaded=True)
    # upstream port that nothing listens on -> ConnectError
    client = TestClient(build_app(upstream_port=1))
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert "event: error" in r.text and "unloaded" in r.text


def test_hf_endpoints_delegate_and_map_errors(home, upstream, monkeypatch):
    from rigma import hf_browse
    from rigma.hangar import HangarError
    monkeypatch.setattr(hf_browse, "search",
                        lambda q, limit=12: [{"repo": "a/b", "downloads": 1,
                                              "likes": 0, "updated": ""}])
    monkeypatch.setattr(hf_browse, "inspect_repo",
                        lambda rid, reg=None: {"repo": rid, "ggufs": []})
    monkeypatch.setattr(hf_browse, "add_model", lambda rid, reg=None: (_ for _ in ()).throw(
        HangarError("that repo is gated — accept its license")))
    client = TestClient(build_app(upstream_port=upstream))
    assert client.get("/api/hf/search?q=x").json()[0]["repo"] == "a/b"
    assert client.get("/api/hf/search?q=").json() == []
    assert client.get("/api/hf/repo?id=a/b").json()["repo"] == "a/b"
    r = client.post("/api/hf/add", json={"repo": "a/b"})
    assert r.status_code == 400 and "gated" in r.json()["error"]
    assert client.post("/api/hf/add", json={}).status_code == 400


def test_ctx_endpoint_relaunches_at_requested_size(home, upstream,
                                                   monkeypatch):
    """Owner finding 2026-07-18: no way to raise ctx from the UI."""
    import os

    from rigma import server_ops
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=32768)
    seen = {}

    def fake_switch(model, reg=None, prof=None, ctx=None):
        seen.update(model=model, ctx=ctx)
        return {"model": model, "ctx": ctx, "unloaded": False}
    monkeypatch.setattr(server_ops, "perform_switch", fake_switch)
    client = TestClient(build_app(upstream_port=upstream))
    r = client.post("/api/server/ctx", json={"ctx": 131072})
    assert r.status_code == 200 and r.json()["ctx"] == 131072
    assert seen == {"model": "m", "ctx": 131072}
    assert client.post("/api/server/ctx", json={"ctx": 12}).status_code == 400
    assert client.post("/api/server/ctx", json={}).status_code == 400
    def boom(model, reg=None, prof=None, ctx=None):
        raise RuntimeError("ctx 999,999 doesn't fit — tops out around 262,144")
    monkeypatch.setattr(server_ops, "perform_switch", boom)
    r = client.post("/api/server/ctx", json={"ctx": 999999})
    assert r.status_code == 502 and "tops out" in r.json()["error"]


def test_prefill_not_doubled_when_engine_echoes(home, tmp_path, monkeypatch):
    """User-reported 2026-07-18: prefill appeared twice. llama-server echoes
    the assistant prefix, so we must not also stream/prepend it."""
    import json as _json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from rigma import state as st

    class _Echo(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            body = _json.loads(self.rfile.read(n))
            # emulate llama.cpp: echo the trailing assistant prefix, then continue
            pre = ""
            if body["messages"] and body["messages"][-1]["role"] == "assistant":
                pre = body["messages"][-1]["content"]
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for tok in [pre, "How", " be", " ye?"]:
                self.wfile.write(b"data: " + _json.dumps(
                    {"choices": [{"delta": {"content": tok}}]}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=srv.server_address[1]))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"prefill": "ARRRGH, matey! ", "use_tools": False})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    # the streamed body must contain the prefill exactly ONCE, not twice
    assert r.text.count("ARRRGH, matey!") == 1, r.text
    saved = client.get(f"/api/sessions/{sid}").json()
    asst = [m for m in saved["messages"] if m["role"] == "assistant"][-1]
    assert asst["content"].count("ARRRGH, matey!") == 1
    assert "How be ye?" in asst["content"]
    assert saved["prefill"] == ""          # consumed once
    srv.shutdown()


def test_prefill_prepended_if_engine_does_not_echo(home, tmp_path, monkeypatch):
    """Robustness: an engine that streams only the continuation still gets the
    prefill (prepended once, never lost)."""
    import json as _json
    import os
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from rigma import state as st

    class _NoEcho(BaseHTTPRequestHandler):
        def do_POST(self):
            self.rfile.read(int(self.headers.get("content-length", 0)))
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()
            for tok in ["How", " be", " ye?"]:   # continuation only, no echo
                self.wfile.write(b"data: " + _json.dumps(
                    {"choices": [{"delta": {"content": tok}}]}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass
    srv = HTTPServer(("127.0.0.1", 0), _NoEcho)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=srv.server_address[1]))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"prefill": "Sure: ", "use_tools": False})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    saved = client.get(f"/api/sessions/{sid}").json()
    asst = [m for m in saved["messages"] if m["role"] == "assistant"][-1]
    assert asst["content"] == "Sure: How be ye?"   # prepended once
    srv.shutdown()
