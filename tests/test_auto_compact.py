"""Auto-compact: when a turn leaves the window ~full, older messages are
summarized before the next turn — reusing the manual /compact machinery."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import sessions
from rigma import state as st
from rigma.serve import build_app


class _Upstream(BaseHTTPRequestHandler):
    prompt_tokens = 950          # reported by the streamed chat turn
    compact_status = 200         # summarizer (non-stream) response status

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        if not body.get("stream"):                      # the summarizer call
            self.send_response(_Upstream.compact_status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"choices": [{"message": {"content": "COMPACT DIGEST"}}]}).encode())
            return
        self.send_response(200)                          # the streamed chat turn
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        sse({"choices": [{"delta": {"content": "ok"}}]})
        sse({"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": _Upstream.prompt_tokens},
             "timings": {"predicted_per_second": 40}})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _seed(client, n=24):
    sid = client.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["messages"] = [{"role": "user", "content": f"m{i}"} for i in range(n)]
    sessions.save(s)
    return sid


def test_auto_compact_fires_when_nearly_full(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 200   # 950/1000 > .92
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert "event: compacted" in r.text
    s = sessions.load(sid)
    assert s["digest"] == "COMPACT DIGEST"
    assert len(s["messages"]) <= 17           # trimmed to the recent tail
    assert s["archive"]                        # older messages preserved


def test_no_compact_below_threshold(home, upstream):
    _Upstream.prompt_tokens = 100              # 100/1000 well under .92
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: compacted" not in r.text
    assert sessions.load(sid)["digest"] == ""


def test_auto_compact_respects_toggle(home, upstream):
    _Upstream.prompt_tokens = 950
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    client.post(f"/api/sessions/{sid}", json={"auto_compact": False})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: compacted" not in r.text
    assert sessions.load(sid)["digest"] == ""


def test_summarizer_failure_does_not_break_turn(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 500
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "[DONE]" in r.text     # turn still completes
    assert "event: compacted" not in r.text
    s = sessions.load(sid)
    assert s["digest"] == "" and s["messages"][-1]["content"] == "ok"  # answer saved


def test_runs_compact_against_a_small_budget():
    from rigma import serve
    assert serve.RUN_CTX_BUDGET == 32768
    # a run session compacts against RUN_CTX_BUDGET, not the engine context
    assert serve.compact_budget({"run_id": "r1"}, 131072) == serve.RUN_CTX_BUDGET
    # ...but never above what the engine actually has
    assert serve.compact_budget({"run_id": "r1"}, 16384) == 16384
    # a normal chat still uses the engine's context
    assert serve.compact_budget({}, 131072) == 131072


def test_compaction_keeps_enough_actions():
    # one action now costs TWO messages (assistant + TOOL RESULT), so the keep
    # window must retain a useful number of ACTIONS, not just messages
    from rigma import serve
    assert serve.AUTO_COMPACT_KEEP >= 16


def test_archive_is_bounded(home, upstream):
    # a run compacts often and re-serialises the whole session each save, so an
    # unbounded archive is real write amplification over a long run
    from rigma import serve as _s
    assert _s.ARCHIVE_MAX <= 1000
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client, n=24)
    s = sessions.load(sid)
    s["archive"] = [{"role": "user", "content": f"old{i}"}
                    for i in range(_s.ARCHIVE_MAX + 50)]
    sessions.save(s)
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 200
    client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert len(sessions.load(sid)["archive"]) <= _s.ARCHIVE_MAX
