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
    """Round 1: ask to call `calculator`. Round 2 (after the tool result is in
    the messages): answer with the number."""
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        has_tool_result = any(m.get("role") == "tool"
                              for m in body.get("messages", []))
        if not has_tool_result:
            msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"id": "c1", "type": "function", "function": {
                       "name": "calculator",
                       "arguments": json.dumps({"expression": "6*7"})}}]}
        else:
            msg = {"role": "assistant", "content": "The answer is 42."}
        out = json.dumps({"choices": [{"message": msg}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

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
    assert "The answer is 42." in r.text
    # persisted with a tool_trace for re-render
    saved = client.get(f"/api/sessions/{sid}").json()
    last = [m for m in saved["messages"] if m["role"] == "assistant"][-1]
    assert last["content"] == "The answer is 42."
    assert last["tool_trace"][0]["name"] == "calculator"
    assert last["tool_trace"][0]["result"] == "42"


def test_tools_off_takes_the_plain_path(home, upstream):
    # with tools off the model gets no tool defs; this upstream would still
    # return a tool_call, but the plain path ignores tool_calls entirely
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    client = TestClient(build_app(upstream_port=upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "event: tool\n" not in r.text
