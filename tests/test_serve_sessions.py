import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import sessions
from rigma.serve import build_app


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return TestClient(build_app(upstream_port=1, default_prompt="DEFAULT"))


def test_session_crud_cycle(client):
    s = client.post("/api/sessions", json={"title": "t"}).json()
    assert s["title"] == "t" and s["messages"] == []
    assert client.get("/api/sessions").json()[0]["id"] == s["id"]
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert got == s
    upd = client.post(f"/api/sessions/{s['id']}",
                      json={"system_prompt": "be brief", "use_rag": True,
                            "id": "EVIL"}).json()
    assert upd["system_prompt"] == "be brief" and upd["use_rag"] is True
    assert upd["id"] == s["id"]  # immutable fields ignored
    noop = client.post(f"/api/sessions/{s['id']}")
    assert noop.status_code == 200 and noop.json()["title"] == "t"
    assert client.delete(f"/api/sessions/{s['id']}").status_code == 200
    assert client.get(f"/api/sessions/{s['id']}").status_code == 404


def test_update_truncates_messages(client):
    s = client.post("/api/sessions", json={}).json()
    sess = sessions.load(s["id"])
    sess["messages"] = [{"role": "user", "content": "a"},
                        {"role": "assistant", "content": "b"}]
    sessions.save(sess)
    upd = client.post(f"/api/sessions/{s['id']}",
                      json={"messages": [{"role": "user", "content": "a"}]}).json()
    assert len(upd["messages"]) == 1


def test_missing_session_is_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.post("/api/sessions/nope", json={}).status_code == 404
    assert client.post("/api/sessions/nope").status_code == 404
    assert client.delete("/api/sessions/nope").status_code == 404


def test_chat_turn_injects_prompt_streams_and_persists(tmp_path, monkeypatch,
                                                       oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port,
                             default_prompt="DEFAULT"))
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert '"delta": "Hel"' in r.text and "[DONE]" in r.text
    sent = oai_upstream.last()["messages"]
    assert sent[0] == {"role": "system", "content": "DEFAULT"}
    assert sent[1] == {"role": "user", "content": "hi"}
    got = c.get(f"/api/sessions/{s['id']}").json()
    assert got["title"] == "hi"
    assert got["messages"] == [{"role": "user", "content": "hi"},
                               {"role": "assistant", "content": "Hello"}]


def test_chat_turn_null_message_regenerates(tmp_path, monkeypatch, oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    c.post(f"/api/sessions/{s['id']}",
           json={"messages": [{"role": "user", "content": "again"}]})
    c.post(f"/api/sessions/{s['id']}/chat", json={"message": None})
    got = c.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]


def test_chat_turn_upstream_down_yields_error_event(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1, default_prompt=""))  # nothing listens
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 200  # errors travel inside the stream
    assert "event: error" in r.text and "[DONE]" in r.text
    got = c.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user"]  # no assistant saved


def test_chat_turn_empty_session_400(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    assert c.post(f"/api/sessions/{s['id']}/chat",
                  json={"message": None}).status_code == 400


class _MidStreamCrash(BaseHTTPRequestHandler):
    """Streams one valid delta chunk, then aborts without finishing."""
    protocol_version = "HTTP/1.1"

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("transfer-encoding", "chunked")
        self.end_headers()
        payload = b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n\n'
        self.wfile.write(hex(len(payload))[2:].encode() + b"\r\n"
                         + payload + b"\r\n")
        self.wfile.flush()
        self.connection.shutdown(socket.SHUT_RDWR)
        self.connection.close()  # abort: RST/EOF now, not at read-timeout

    def log_message(self, *a):
        pass


def test_chat_turn_midstream_failure_discards_partial(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), _MidStreamCrash)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        s = c.post("/api/sessions", json={}).json()
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
        assert '"delta": "Hel"' in r.text          # partial delta was relayed
        assert "event: error" in r.text and "[DONE]" in r.text
        got = c.get(f"/api/sessions/{s['id']}").json()
        assert [m["role"] for m in got["messages"]] == ["user"]  # nothing saved
    finally:
        srv.shutdown()


class _RejectingUpstream(BaseHTTPRequestHandler):
    """Rejects every request the way llama-server rejects an over-ctx prompt."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        body = json.dumps({"error": {
            "code": 400, "type": "exceed_context_size_error",
            "message": "request (4992 tokens) exceeds the available context "
                       "size (4096 tokens), try increasing it"}}).encode()
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


class _MidStreamErrorObject(BaseHTTPRequestHandler):
    """200 stream whose first data line is an error object, not a delta."""

    def do_POST(self):
        self.rfile.read(int(self.headers.get("content-length", 0)))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        err = {"error": {"message": "slot unavailable"}}
        self.wfile.write(b"data: " + json.dumps(err).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def _turn_against(handler_cls, tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), handler_cls)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        s = c.post("/api/sessions", json={}).json()
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
        saved = c.get(f"/api/sessions/{s['id']}").json()
        return r, saved
    finally:
        srv.shutdown()


def test_chat_turn_upstream_http_error_surfaces_message(tmp_path, monkeypatch):
    r, saved = _turn_against(_RejectingUpstream, tmp_path, monkeypatch)
    assert "event: error" in r.text and "[DONE]" in r.text
    assert "exceeds the available context size" in r.text
    assert [m["role"] for m in saved["messages"]] == ["user"]  # nothing saved


def test_chat_turn_instream_error_object_surfaces(tmp_path, monkeypatch):
    r, saved = _turn_against(_MidStreamErrorObject, tmp_path, monkeypatch)
    assert "event: error" in r.text and "slot unavailable" in r.text
    assert [m["role"] for m in saved["messages"]] == ["user"]


class _UsageUpstream(BaseHTTPRequestHandler):
    """Streams one delta, then a final usage/timings chunk, then [DONE]."""
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _UsageUpstream.last_body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        chunk = {"choices": [{"delta": {"content": "Hi"}}]}
        self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        tail = {"choices": [], "usage": {"prompt_tokens": 2244},
                "timings": {"predicted_per_second": 15.5}}
        self.wfile.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_chat_turn_sends_params_and_emits_meta(tmp_path, monkeypatch):
    import os as _os
    from rigma import presets, state
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    state.write_state("m", "q", 11500, engine_pid=_os.getpid(),
                      ui_pid=_os.getpid(), ctx=4096)
    srv = HTTPServer(("127.0.0.1", 0), _UsageUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        p = presets.create("hot", "PROMPT", params={"top_p": 0.8})
        s = c.post("/api/sessions", json={}).json()
        c.post(f"/api/sessions/{s['id']}",
               json={"preset_id": p["id"], "params": {"temperature": 0.4}})
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "go"})
        sent = _UsageUpstream.last_body
        assert sent["temperature"] == 0.4 and sent["top_p"] == 0.8
        assert sent["stream_options"] == {"include_usage": True}
        assert sent["messages"][0] == {"role": "system", "content": "PROMPT"}
        assert "event: meta" in r.text
        assert '"prompt_tokens": 2244' in r.text and '"ctx": 4096' in r.text
        assert "15.5" in r.text and "[DONE]" in r.text
    finally:
        srv.shutdown()


def test_chat_turn_no_meta_on_failure(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert "event: error" in r.text and "event: meta" not in r.text


def test_update_rejects_bad_params(client):
    s = client.post("/api/sessions", json={}).json()
    r = client.post(f"/api/sessions/{s['id']}",
                    json={"params": {"temperature": 9.0}})
    assert r.status_code == 400 and "temperature" in r.json()["error"]
    ok = client.post(f"/api/sessions/{s['id']}",
                     json={"params": {"temperature": 0.7},
                           "preset_id": "usecase:general", "notes": "N"})
    assert ok.json()["params"] == {"temperature": 0.7}
    assert ok.json()["preset_id"] == "usecase:general"
    assert ok.json()["notes"] == "N"


def test_chat_turn_continue_extends_trailing_assistant(tmp_path, monkeypatch,
                                                       oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    c.post(f"/api/sessions/{s['id']}",
           json={"messages": [{"role": "user", "content": "story"},
                              {"role": "assistant", "content": "Once upon",
                               "variants": ["draft"]}]})
    r = c.post(f"/api/sessions/{s['id']}/chat",
               json={"message": None, "continue": True})
    assert "[DONE]" in r.text
    got = c.get(f"/api/sessions/{s['id']}").json()
    msgs = got["messages"]
    assert len(msgs) == 2  # extended, not appended
    assert msgs[1]["content"] == "Once uponHello"  # upstream streams "Hel"+"lo"
    assert msgs[1]["variants"] == ["draft"]  # metadata preserved
    sent = oai_upstream.last()["messages"]
    assert sent[-1] == {"role": "assistant", "content": "Once upon"}


def test_chat_turn_continue_without_trailing_assistant_appends(tmp_path,
                                                               monkeypatch,
                                                               oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}/chat",
               json={"message": "hi", "continue": True})
    assert "[DONE]" in r.text
    got = c.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]


def test_search_export_duplicate_routes(client):
    s = client.post("/api/sessions", json={"title": "dragon tale"}).json()
    client.post(f"/api/sessions/{s['id']}",
                json={"messages": [{"role": "user", "content": "fire"},
                                   {"role": "assistant", "content": "smoke"}]})
    hits = client.get("/api/sessions/search?q=dragon").json()
    assert len(hits) == 1 and hits[0]["id"] == s["id"] and hits[0]["snippet"]
    assert client.get("/api/sessions/search?q=").json() == []

    md = client.get(f"/api/sessions/{s['id']}/export?fmt=md")
    assert md.status_code == 200 and md.text.startswith("# dragon tale")
    assert "attachment" in md.headers["content-disposition"]
    js = client.get(f"/api/sessions/{s['id']}/export?fmt=json")
    assert js.json()["id"] == s["id"]
    assert client.get(f"/api/sessions/{s['id']}/export?fmt=evil").status_code == 400
    assert client.get("/api/sessions/nope/export?fmt=md").status_code == 404

    d = client.post(f"/api/sessions/{s['id']}/duplicate").json()
    assert d["title"] == "dragon tale (copy)" and len(d["messages"]) == 2
    assert client.post("/api/sessions/nope/duplicate").status_code == 404


def test_export_nonascii_title(client):
    s = client.post("/api/sessions", json={"title": "攻略チャット🐉"}).json()
    r = client.get(f"/api/sessions/{s['id']}/export?fmt=md")
    assert r.status_code == 200
    cd = r.headers["content-disposition"]
    assert 'filename="chat.md"' in cd and "filename*=UTF-8''" in cd


class _SummarizerUpstream(BaseHTTPRequestHandler):
    """Non-streaming completion that returns a fixed summary."""
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _SummarizerUpstream.last_body = json.loads(self.rfile.read(n))
        body = json.dumps({"choices": [{"message": {
            "role": "assistant", "content": "SUMMARY-OF-OLD-TURNS"}}]}).encode()
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


def test_compact_endpoint(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), _SummarizerUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        s = c.post("/api/sessions", json={}).json()
        msgs = [{"role": "user", "content": f"m{i}"} if i % 2 == 0 else
                {"role": "assistant", "content": f"r{i}"} for i in range(10)]
        c.post(f"/api/sessions/{s['id']}", json={"messages": msgs})
        r = c.post(f"/api/sessions/{s['id']}/compact", json={"keep": 4})
        assert r.status_code == 200
        out = r.json()
        assert out["archived"] == 6
        sess = out["session"]
        assert len(sess["messages"]) == 4
        assert sess["messages"][0]["content"] == "m6"
        assert sess["digest"] == "SUMMARY-OF-OLD-TURNS"
        assert len(sess["archive"]) == 6
        sent = _SummarizerUpstream.last_body
        assert sent["stream"] is False
        assert any("m0" in str(m) for m in sent["messages"])  # old turns included

        # repeat compact merges: old digest is part of the input
        c.post(f"/api/sessions/{s['id']}",
               json={"messages": sess["messages"] + [
                   {"role": "user", "content": "new1"},
                   {"role": "assistant", "content": "new2"}]})
        r2 = c.post(f"/api/sessions/{s['id']}/compact", json={"keep": 2})
        assert r2.status_code == 200
        sent2 = _SummarizerUpstream.last_body
        assert any("SUMMARY-OF-OLD-TURNS" in str(m) for m in sent2["messages"])
        assert len(r2.json()["session"]["archive"]) == 10  # 6 + 4 more
    finally:
        srv.shutdown()


def test_compact_nothing_to_do_400(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    c.post(f"/api/sessions/{s['id']}",
           json={"messages": [{"role": "user", "content": "a"}]})
    assert c.post(f"/api/sessions/{s['id']}/compact",
                  json={"keep": 6}).status_code == 400


def test_compact_upstream_failure_leaves_session_untouched(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=1, default_prompt=""))  # nothing there
    s = c.post("/api/sessions", json={}).json()
    msgs = [{"role": "user", "content": f"m{i}"} for i in range(8)]
    c.post(f"/api/sessions/{s['id']}", json={"messages": msgs})
    r = c.post(f"/api/sessions/{s['id']}/compact", json={"keep": 2})
    assert r.status_code == 502
    got = c.get(f"/api/sessions/{s['id']}").json()
    assert len(got["messages"]) == 8 and not got.get("digest")


class _ThinkingUpstream(BaseHTTPRequestHandler):
    """Streams reasoning_content deltas before content deltas (deepseek fmt)."""
    last_body = None

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _ThinkingUpstream.last_body = json.loads(self.rfile.read(n))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in ({"choices": [{"delta": {"reasoning_content": "hmm "}}]},
                      {"choices": [{"delta": {"reasoning_content": "ok"}}]},
                      {"choices": [{"delta": {"content": "Answer."}}]}):
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


def test_chat_turn_streams_think_events_and_persists_thinking(tmp_path,
                                                              monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    srv = HTTPServer(("127.0.0.1", 0), _ThinkingUpstream)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        c = TestClient(build_app(upstream_port=srv.server_address[1],
                                 default_prompt=""))
        s = c.post("/api/sessions", json={}).json()
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "q"})
        assert "event: think" in r.text and '"delta": "hmm "' in r.text
        assert '"delta": "Answer."' in r.text and "[DONE]" in r.text
        got = c.get(f"/api/sessions/{s['id']}").json()
        last = got["messages"][-1]
        assert last["content"] == "Answer." and last["thinking"] == "hmm ok"
        # thinking must never reach the model on later turns
        from rigma import sessions as sess_mod
        out = sess_mod.build_messages(got)
        assert all(set(m) == {"role", "content"} for m in out)
    finally:
        srv.shutdown()


def test_effort_field_and_request_layer(tmp_path, monkeypatch, oai_upstream):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    c = TestClient(build_app(upstream_port=oai_upstream.port, default_prompt=""))
    s = c.post("/api/sessions", json={}).json()
    r = c.post(f"/api/sessions/{s['id']}", json={"effort": "off"})
    assert r.json()["effort"] == "off"
    assert c.post(f"/api/sessions/{s['id']}",
                  json={"effort": "extreme"}).status_code == 400
    c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    sent = oai_upstream.last()
    assert sent["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in sent
    c.post(f"/api/sessions/{s['id']}", json={"effort": "on"})
    c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi2"})
    sent = oai_upstream.last()
    assert sent["reasoning_effort"] == "high"
    assert "chat_template_kwargs" not in sent
