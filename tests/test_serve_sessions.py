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
