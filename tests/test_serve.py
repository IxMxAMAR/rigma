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
    assert "chat/completions" in body  # real page, not fallback
    assert "api/status" in body


def test_api_status_not_running(upstream, tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    client = TestClient(build_app(upstream_port=upstream))
    assert client.get("/api/status").status_code == 404
