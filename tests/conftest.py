"""Shared fixtures."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _isolate_user_custom_models(monkeypatch, tmp_path_factory):
    """The user's real ~/.rigma/custom installs must never leak into tests
    (live repro 2026-07-17: installing SmolLM2 flipped test_floor_never_fails).
    Reads RIGMA_HOME at call time so tests that set their own home keep the
    normal layout."""
    import rigma.registry as registry
    empty = tmp_path_factory.mktemp("no-custom")
    monkeypatch.setattr(
        registry, "_custom_dir",
        lambda: (Path(os.environ["RIGMA_HOME"]) / "custom" / "models")
        if os.environ.get("RIGMA_HOME") else empty)


class _OpenAIUpstream(BaseHTTPRequestHandler):
    """Streams 'Hel'+'lo' as OpenAI chat chunks; records the last request body."""
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _OpenAIUpstream.last_body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for tok in ("Hel", "lo"):
            chunk = {"choices": [{"delta": {"content": tok}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def oai_upstream():
    _OpenAIUpstream.last_body = None
    srv = HTTPServer(("127.0.0.1", 0), _OpenAIUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield SimpleNamespace(port=srv.server_address[1],
                          last=lambda: _OpenAIUpstream.last_body)
    srv.shutdown()
