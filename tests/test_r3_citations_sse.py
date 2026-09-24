"""UIUX-22 end to end: the `citations` SSE event actually reaches the client.

`test_r3_citations.py` covers the capture. This covers the half that would fail
silently: the turn loop reads the context AFTER the round and emits the event. If
that wiring is wrong the tool still records and the UI still renders nothing —
which is exactly the state F11-12 found the code in, so it is worth a real turn.
"""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import rag, serve, state as st


class _Engine(BaseHTTPRequestHandler):
    """Round 1: calls search_my_documents. Round 2: answers, so the turn ends."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        used_tool = any(m.get("role") == "tool" for m in body.get("messages", []))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(obj):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")

        if not used_tool:
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": "c1", "type": "function",
                 "function": {"name": "search_my_documents",
                              "arguments": json.dumps({"query": "what?"})}}]}}]})
            sse({"choices": [{"delta": {}, "finish_reason": "tool_calls"}]})
        else:
            sse({"choices": [{"delta": {"content": "grounded answer"}}]})
            sse({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def client(tmp_path, monkeypatch, engine):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    # a live sidecar, without one
    monkeypatch.setattr(rag, "live_sidecar_port", lambda timeout=5.0: 9)
    monkeypatch.setattr(rag, "ask", lambda q, port=None: {
        "answer": "from your notes", "abstained": False,
        "citations": [{"source": "notes/a.md", "page": 2, "text": "evidence"}]})
    c = TestClient(serve.build_app(upstream_port=engine))
    yield c


def _turn(client, use_tools=True):
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": use_tools,
                                              "title": "t"})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "what?"})
    assert r.status_code == 200, r.text
    return r.text


def test_a_grounded_turn_emits_the_sources_to_the_client(client):
    text = _turn(client)
    assert "event: citations" in text, text
    # the payload is JSON on the data line, as every other event is
    line = [ln for ln in text.splitlines()
            if ln.startswith("data: ") and '"sources"' in ln][0]
    payload = json.loads(line[len("data: "):])
    assert payload["sources"] == [
        {"source": "notes/a.md", "snippet": "evidence", "page": 2}]


def test_the_sources_are_emitted_once_per_round_not_once_per_call(client):
    """Several searches in one round are ONE set of sources to the reader. A
    per-call event would render the same chip list twice."""
    text = _turn(client)
    assert text.count("event: citations") == 1, text.count("event: citations")


def test_the_structured_sources_never_enter_the_transcript(client):
    """Display state must not persist.

    Note what is NOT asserted: the filename still appears in the tool RESULT text
    ("sources: notes/a.md"), and that is correct — the model has always been told
    the sources and must keep being told, or the answer stops being grounded. What
    must not persist is the STRUCTURED field: a `sources` array on a message would
    be replayed to the model as history, which is the class of bug that bricked a
    chat with an instant-EOS loop (2026-07-21).
    """
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"use_tools": True, "title": "t"})
    client.post(f"/api/sessions/{sid}/chat", json={"message": "what?"})
    saved = client.get(f"/api/sessions/{sid}").json()

    def keys(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                yield k
                yield from keys(v)
        elif isinstance(obj, list):
            for v in obj:
                yield from keys(v)

    assert "sources" not in set(keys(saved)), (
        "a `sources` field was persisted into the transcript")
    assert "_citations" not in set(keys(saved)), (
        "the context's citations list was persisted into the transcript")
    # and the model's own view is unchanged
    assert "sources: notes/a.md" in json.dumps(saved)


def test_a_turn_with_no_search_emits_no_citations_event(client):
    """An empty event on every turn would be noise the UI has to filter."""
    text = _turn(client, use_tools=False)
    assert "event: citations" not in text, text
