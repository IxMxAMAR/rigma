import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app


class _Upstream(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: " + body + b"\n\ndata: [DONE]\n\n")

    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"object": "list"}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _Upstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield srv.server_address[1]
    srv.shutdown()


def test_proxy_get_and_streaming_post(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.get("/v1/models")
    assert r.status_code == 200 and r.json()["object"] == "list"
    # A RAW body, deliberately: the passthrough guarantee is BYTE-for-byte, and
    # `json={"x": 1}` only proved what the TEST CLIENT's serializer emits. That
    # changed between starlette/httpx versions (compact vs `{"x": 1}`) and broke
    # this test with the proxy itself untouched — found by the CI `floors` job
    # (AUDIT F61). Spaces and all, the engine must see exactly these bytes.
    raw = b'{"x":1, "pad" : "  keep   the   spacing  "}'
    r = client.post("/v1/chat/completions", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200
    assert raw.decode() in r.text and "[DONE]" in r.text, r.text


def test_proxy_applies_tool_params_only_when_tools_are_present(
        upstream, tmp_path, monkeypatch):
    """B7: the passthrough stays byte-for-byte unless the request carries
    `tools`. When it does, the tool-call parameter layer (model card <
    RUN_PARAMS < caller) is applied, because at the engine's stock ~0.8 an
    IQ-quant model's call SYNTAX drifts and the strict parser misses it. The
    response path is untouched: this is still an SSE stream."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))  # no running model: layer = RUN_PARAMS
    client = TestClient(build_app(upstream_port=upstream))

    # No `tools` -> the exact bytes, spacing and all (creative generation).
    raw = b'{"model":"m","temperature":0.9,"messages":[]}'
    r = client.post("/v1/chat/completions", content=raw,
                    headers={"content-type": "application/json"})
    assert r.status_code == 200 and raw.decode() in r.text

    # `tools` present -> RUN_PARAMS injected, and the caller's 0.9 wins.
    raw2 = (b'{"model":"m","messages":[],"temperature":0.9,'
            b'"tools":[{"type":"function","function":{"name":"f"}}]}')
    r2 = client.post("/v1/chat/completions", content=raw2,
                     headers={"content-type": "application/json"})
    assert r2.status_code == 200
    assert "text/event-stream" in r2.headers["content-type"]
    assert "[DONE]" in r2.text
    seen = json.loads(r2.text.split("data: ", 1)[1].split("\n", 1)[0])
    assert seen["temperature"] == 0.9          # caller's own value wins
    assert seen["dry_multiplier"] == 0.8       # RUN_PARAMS reached the engine
    assert seen["dry_penalty_last_n"] == 4096
    assert seen["repeat_penalty"] == 1.05
    assert seen["tools"] == [{"type": "function", "function": {"name": "f"}}]

    # An empty `tools` list is not a tool request: still byte-identical.
    raw3 = b'{"model":"m","messages":[],"tools":[]}'
    r3 = client.post("/v1/chat/completions", content=raw3,
                     headers={"content-type": "application/json"})
    assert raw3.decode() in r3.text


def test_v1_passthrough_engine_down_is_openai_502(tmp_path, monkeypatch):
    """01-4: a dead engine must not surface as Starlette's plain-text 500 —
    every OpenAI client parses the body as JSON and then reports a decode
    error, telling the user their client is broken instead of the engine."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    client = TestClient(build_app(upstream_port=1))   # nothing listens
    r = client.post("/v1/chat/completions",
                    json={"model": "x", "messages": []})
    assert r.status_code == 502, r.text
    assert r.headers["content-type"].startswith("application/json")
    assert r.json()["error"]["type"] == "upstream_error"
    assert r.json()["error"]["message"]
    r = client.get("/v1/models")
    assert r.status_code == 502, r.text
    assert r.json()["error"]["type"] == "upstream_error"


def test_root_serves_html(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()


def test_rizz_serves_real_legacy_chat_ui(upstream):
    # the complete legacy app moved to /rizz (owner cutover 2026-07-21);
    # the point of this test is unchanged — the REAL app, not the fallback
    client = TestClient(build_app(upstream_port=upstream))
    body = client.get("/rizz").text
    assert "/ui/app.js" in body and "/ui/style.css" in body and "/ui/md.js" in body
    assert "/ui/store.js" in body


def test_ui_assets_allowlist(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.get("/ui/style.css")
    assert r.status_code == 200 and "text/css" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"
    r = client.get("/ui/md.js")
    assert r.status_code == 200 and "javascript" in r.headers["content-type"]
    assert r.headers["cache-control"] == "no-store"
    r = client.get("/ui/evil.js")
    assert r.status_code == 404 and r.json() == {"error": "not found"}
    r = client.get("/ui/..%2Fserve.py")
    assert r.status_code == 404


def test_api_status_not_running(upstream, tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    client = TestClient(build_app(upstream_port=upstream))
    assert client.get("/api/status").status_code == 404


def test_app_js_served_and_targets_session_api(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    body = client.get("/ui/app.js").text
    assert "/api/sessions" in body and "renderMarkdown" in body
    assert "/v1/chat/completions" not in body  # UI talks session API only
    store = client.get("/ui/store.js").text
    assert "/api/sessions/" in store and "sseParse" in store
    assert "/v1/chat/completions" not in store


def test_runs_list_and_memory_endpoints(tmp_path, monkeypatch):
    # v2 phase 4/5 backends: run history + the memory trust surface
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import runs as _runs
    from rigma.memory import MemoryStore
    client = TestClient(build_app(upstream_port=1))
    assert client.get("/api/runs").json() == []
    r = _runs.create("do the thing", "sess-x")
    lst = client.get("/api/runs").json()
    assert lst and lst[0]["id"] == r["id"] and "do the thing" in lst[0]["mission"]

    store = MemoryStore(tmp_path / "memory" / "memories.jsonl")
    m = store.add(kind="pitfall", text="Never type filenames.")
    rows = client.get("/api/memory").json()
    assert rows and rows[0]["text"] == "Never type filenames."
    assert "vec" not in rows[0]
    client.delete(f"/api/memory/{m['id']}")
    assert client.get("/api/memory").json() == []
