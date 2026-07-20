"""Agentic tool loop through the real chat endpoint."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import state as st
from rigma import tools
from rigma.serve import build_app


class _ToolUpstream(BaseHTTPRequestHandler):
    """Streaming upstream (like llama-server). Round 1: stream a tool_call to
    `calculator`. Round 2 (after the tool result is in the messages): stream
    the answer text."""
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        has_tool_result = any(m.get("role") == "tool"
                              for m in body.get("messages", []))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if not has_tool_result:
            # tool_call arrives split across deltas, by index (real behaviour)
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "calculator", "arguments": ""}}]}}]})
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "function": {
                    "arguments": json.dumps({"expression": "6*7"})}}]}}]})
        else:
            for tok in ["The ", "answer ", "is ", "42."]:
                sse({"choices": [{"delta": {"content": tok}}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _ToolUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


def test_tool_loop_calls_tool_then_answers(home, upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": True})
    r = client.post(f"/api/sessions/{sid}/chat",
                    json={"message": "what is 6 times 7?"})
    assert r.status_code == 200
    # the stream carried the tool call, its result, and the final answer
    assert "event: tool\n" in r.text
    assert '"name": "calculator"' in r.text
    assert "event: tool_result\n" in r.text
    assert '"result": "42"' in r.text
    assert '"delta": "42."' in r.text          # final answer streamed as tokens
    # persisted with a tool_trace for re-render
    saved = client.get(f"/api/sessions/{sid}").json()
    last = [m for m in saved["messages"] if m["role"] == "assistant"][-1]
    assert last["content"] == "The answer is 42."
    assert last["tool_trace"][0]["name"] == "calculator"
    assert last["tool_trace"][0]["result"] == "42"


def test_tools_off_takes_the_plain_path(home, upstream):
    # tools explicitly off: no tool defs sent, tool_calls ignored, no execution
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": False})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "event: tool\n" not in r.text


def test_tools_on_by_default(home):
    # a fresh session should have tools enabled without any opt-in
    from rigma import sessions
    assert sessions.create()["use_tools"] is True


class _BadArgsUpstream(BaseHTTPRequestHandler):
    """Round 1: stream a tool_call with BROKEN JSON args. Round 2 (after the
    error is fed back as a tool message): answer."""
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if not has_tool:
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "b1", "type": "function",
                 "function": {"name": "calculator",
                              "arguments": '{"expression": '}}]}}]})  # broken
        else:
            for tok in ["ok ", "done"]:
                sse({"choices": [{"delta": {"content": tok}}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def bad_upstream():
    srv = HTTPServer(("127.0.0.1", 0), _BadArgsUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_malformed_tool_args_fed_back_not_executed(home, bad_upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=bad_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": True})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "2+2?"})
    assert r.status_code == 200
    assert "malformed JSON" in r.text          # error surfaced to the model
    assert "fix the JSON" in r.text            # instructional
    assert '"delta": "done"' in r.text         # still reached a final answer


class _AlwaysToolUpstream(BaseHTTPRequestHandler):
    """Model that NEVER stops calling a tool. Records whether each request
    carried tool defs (the last round must still advertise tools so a stray
    call is parsed, not leaked as raw text)."""
    reqs = []

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        _AlwaysToolUpstream.reqs.append("tools" in body)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {
            "tool_calls": [{"index": 0, "id": "c1", "type": "function",
                            "function": {"name": "calculator",
                                         "arguments": '{"expression":"1+1"}'}}]}}]}
            ).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def always_upstream():
    _AlwaysToolUpstream.reqs = []
    srv = HTTPServer(("127.0.0.1", 0), _AlwaysToolUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_runaway_tool_loop_keeps_tools_on_last_round(home, always_upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=always_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}",
                json={"use_tools": True, "max_tool_rounds": 10})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200
    assert "<tool_call>" not in r.text            # never leaks raw call text
    assert "<function=" not in r.text
    reqs = _AlwaysToolUpstream.reqs
    assert len(reqs) == 10                        # bounded at max_tool_rounds
    assert all(reqs)                              # EVERY round advertised tools
    assert "keep going" in r.text                 # never finishes silently


class _MultiToolUpstream(BaseHTTPRequestHandler):
    """Round 1: emit THREE tool_calls at once. Round 2: answer."""
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        has_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        if not has_tool:
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": i, "id": f"c{i}", "type": "function",
                 "function": {"name": "calculator",
                              "arguments": json.dumps({"expression": f"{i}+1"})}}
                for i in range(3)]}}]})
        else:
            sse({"choices": [{"delta": {"content": "done"}}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def multi_upstream():
    srv = HTTPServer(("127.0.0.1", 0), _MultiToolUpstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


def test_tool_calls_run_in_parallel(home, multi_upstream, monkeypatch):
    import time
    from rigma import tools
    active = {"n": 0, "max": 0}
    lock = threading.Lock()

    def slow(name, args, ctx=None):
        with lock:
            active["n"] += 1
            active["max"] = max(active["max"], active["n"])
        time.sleep(0.3)
        with lock:
            active["n"] -= 1
        return "ok"

    monkeypatch.setattr(tools, "run_tool", slow)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=multi_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": True})
    t0 = time.monotonic()
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    dt = time.monotonic() - t0
    assert r.status_code == 200
    assert active["max"] >= 2          # ran concurrently, not one-by-one
    assert dt < 0.75                   # ~1x (0.3s), not 3x (0.9s)
    assert r.text.count("event: tool_result") == 3


def test_max_tool_rounds_configurable(home, always_upstream):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    _AlwaysToolUpstream.reqs = []
    client = TestClient(build_app(upstream_port=always_upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}",
                json={"use_tools": True, "max_tool_rounds": 3})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert len(_AlwaysToolUpstream.reqs) == 3      # honored the low ceiling
    assert "keep going" in r.text


def test_truncated_tool_call_is_not_executed(tmp_path, monkeypatch):
    """finish_reason=length mid-call: the repairer would balance the braces of
    HALF a write_file and run a partial write as if complete (silent data
    loss). Truncated calls must come back as an error result, never run."""
    import json as _json
    import threading
    from http.server import BaseHTTPRequestHandler, HTTPServer

    from fastapi.testclient import TestClient

    from rigma import state as st
    from rigma.serve import build_app

    class _Cut(BaseHTTPRequestHandler):
        def do_POST(self):
            n = int(self.headers.get("content-length", 0))
            self.rfile.read(n)
            self.send_response(200)
            self.send_header("content-type", "text/event-stream")
            self.end_headers()

            def sse(o):
                self.wfile.write(b"data: " + _json.dumps(o).encode() + b"\n\n")

            # a write_file whose args got cut mid-string, then length stop
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "function": {
                    "name": "write_file",
                    "arguments": '{"path": "ch.txt", "content": "the story beg'
                }}]}}]})
            sse({"choices": [{"delta": {}, "finish_reason": "length"}]})
            self.wfile.write(b"data: [DONE]\n\n")

        def log_message(self, *a):
            pass

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), _Cut)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1, ctx=4096)
        c = TestClient(build_app(upstream_port=srv.server_address[1]))
        s = c.post("/api/sessions", json={}).json()
        ws = tmp_path / "ws"
        ws.mkdir()
        c.post(f"/api/sessions/{s['id']}",
               json={"use_tools": True, "allow_code": True,
                     "workspace": str(ws)})
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "write"})
        assert "CUT OFF" in r.text
        assert not (ws / "ch.txt").exists(), \
            "a truncated write must never reach the disk"
    finally:
        srv.shutdown()
