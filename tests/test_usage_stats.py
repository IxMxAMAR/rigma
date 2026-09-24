"""IMP-7: the usage meter reads the odometer in stats.json.

The point is that nothing is recomputed from the session store: a completed
turn is counted once, in the turn loop, and the meter is a read of that file.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_stats_endpoint_reads_a_legacy_file(home):
    """A stats.json written before by_model_turns/last_used existed must still
    serve — and must not invent turns it never counted."""
    (home / "stats.json").write_text(json.dumps(
        {"total_tokens": 1200, "total_turns": 7,
         "by_model": {"alpha": 900, "beta": 300}}), encoding="utf-8")
    d = TestClient(build_app(upstream_port=1)).get("/api/server/stats").json()
    assert d["total_tokens"] == 1200 and d["total_turns"] == 7
    assert d["by_model"] == {"alpha": 900, "beta": 300}
    assert [m["model"] for m in d["models"]] == ["alpha", "beta"]
    assert all(m["turns"] == 0 and m["last_used"] is None for m in d["models"])


def test_stats_endpoint_sorts_and_reports_per_model(home):
    (home / "stats.json").write_text(json.dumps(
        {"total_tokens": 1000, "total_turns": 5,
         "by_model": {"alpha": 100, "beta": 900},
         "by_model_turns": {"alpha": 4, "beta": 1},
         "last_used": {"alpha": 111.0, "beta": 222.0}}), encoding="utf-8")
    d = TestClient(build_app(upstream_port=1)).get("/api/server/stats").json()
    assert [m["model"] for m in d["models"]] == ["beta", "alpha"]
    assert d["models"][0] == {"model": "beta", "tokens": 900, "turns": 1,
                              "last_used": 222.0}


def test_stats_endpoint_survives_a_corrupt_file(home):
    (home / "stats.json").write_text("{not json", encoding="utf-8")
    d = TestClient(build_app(upstream_port=1)).get("/api/server/stats").json()
    assert d["total_tokens"] == 0 and d["models"] == []


class _TimedUpstream(BaseHTTPRequestHandler):
    """Streams one delta then a timings tail with predicted_n (what the
    odometer counts)."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        tail = {"choices": [], "timings": {"predicted_n": 60,
                                           "predicted_per_second": 5.0}}
        self.wfile.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_a_turn_bumps_the_odometer_per_model(home):
    from rigma import state
    state.write_state("m", "q", 11500, engine_pid=os.getpid(),
                      ui_pid=os.getpid())
    srv = HTTPServer(("127.0.0.1", 0), _TimedUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        sid = c.post("/api/sessions", json={}).json()["id"]
        c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
        d = c.get("/api/server/stats").json()
        assert d["total_tokens"] == 60 and d["total_turns"] == 1
        assert d["by_model"] == {"m": 60}
        assert d["by_model_turns"] == {"m": 1}
        assert d["models"][0]["last_used"] > 0
    finally:
        srv.shutdown()
