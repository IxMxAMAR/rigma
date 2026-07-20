"""Shared fixtures."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture(autouse=True)
def _no_hang_on_run_ui(monkeypatch):
    """serve.run_ui blocks forever serving the UI; a test that reaches it
    unmocked would hang the ENTIRE suite (and squat on port 11500). Fail fast
    instead. Tests that intend to exercise `up`'s serving path patch run_ui
    themselves — their patch runs after this one and wins."""
    import rigma.serve as serve

    def _boom(*a, **k):
        raise RuntimeError("serve.run_ui reached in a test without mocking it "
                           "— `rigma up` was invoked in a way that serves")
    monkeypatch.setattr(serve, "run_ui", _boom, raising=False)


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


@pytest.fixture(autouse=True, scope="session")
def _tests_run_against_the_source_tree():
    """A stale site-packages copy of rigma once shadowed the editable install
    (2026-07-21) and every subsequent test run silently validated the WRONG
    code. If imports ever resolve outside this repo, fail everything loudly."""
    import rigma
    src = str(Path(__file__).resolve().parent.parent / "src")
    got = str(Path(rigma.__file__).resolve())
    if not got.startswith(src):
        pytest.exit(f"rigma imports from {got}, not the source tree {src} — "
                    "run `pip install -e .` and remove the shadowing copy",
                    returncode=3)
