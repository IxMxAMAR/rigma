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
    r = client.post("/v1/chat/completions", json={"x": 1})
    assert r.status_code == 200
    assert 'data: {"x":1}' in r.text and "[DONE]" in r.text


def test_root_serves_html(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    r = client.get("/")
    assert r.status_code == 200
    assert "<html" in r.text.lower() or "<!doctype" in r.text.lower()


def test_root_serves_real_chat_ui(upstream):
    client = TestClient(build_app(upstream_port=upstream))
    body = client.get("/").text
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
