"""R3: a sidecar record must name a raggity sidecar, not just an open port.

`sidecar_health` accepts ANY local HTTP service that answers /healthz with a
200 and parseable JSON, and everything that decides whether document search
exists goes through it:

  * `live_sidecar_port()` returns the recorded port on the strength of that
    answer, so the model is offered `search_my_documents` and Rigma POSTs
    /retrieve and /ask at the stranger.
  * `ensure_sidecar()` returns the stranger's health body as if it were the
    sidecar, records ITS pid in sidecar.json, and a later `stop_sidecar()`
    terminates that pid (10-5 verifies the pid is the one we recorded — it
    cannot know the record was written about the wrong process).

The port is not proof of identity: ports are recycled, and sidecar.json is a
plain file in ~/.rigma/rag that a user is told to hand-edit.
"""
import http.server
import json
import threading

import pytest

from rigma import rag


class _Stranger(http.server.BaseHTTPRequestHandler):
    """Some other local service that happens to answer on the recorded port."""

    def do_GET(self):
        body = json.dumps({"hello": "world"}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass


@pytest.fixture
def stranger():
    srv = http.server.HTTPServer(("127.0.0.1", 0), _Stranger)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield srv.server_address[1]
    finally:
        srv.shutdown()
        srv.server_close()
        t.join(timeout=5)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _record(port, pid=999999):
    rag.rag_dir().mkdir(parents=True, exist_ok=True)
    (rag.rag_dir() / "sidecar.json").write_text(
        json.dumps({"pid": pid, "port": port, "created_at": 1000.0}),
        encoding="utf-8")


def test_a_stranger_on_the_recorded_port_is_not_a_live_sidecar(home, stranger):
    _record(stranger)
    assert rag.recorded_sidecar_port() == stranger
    assert rag.live_sidecar_port() is None, (
        "an unrelated local service was accepted as the raggity sidecar")


def test_a_stranger_is_not_advertised_as_document_search(home, stranger):
    """The consequence that matters: Rigma's MCP roster gates
    search_my_documents on this answer."""
    from rigma import mcp_server
    _record(stranger)
    names = {t["name"] for t in mcp_server.offered()}
    assert "search_my_documents" not in names


def test_a_stranger_is_not_adopted_by_ensure_sidecar(home, stranger):
    """ensure_sidecar used to return early on a truthy health body and then
    record THAT process's pid — so the record named the stranger and a later
    `rag stop` would terminate it. It must refuse to treat the stranger as our
    sidecar, and must leave no record behind."""
    with pytest.raises(RuntimeError):
        rag.ensure_sidecar(port=stranger, timeout=5)
    assert not (rag.rag_dir() / "sidecar.json").exists(), (
        "ensure_sidecar recorded a pid for a process that is not raggity")
    # and the stranger is still running: we did not claim or signal it
    assert rag.sidecar_health(stranger) is None
    import httpx
    assert httpx.get(f"http://127.0.0.1:{stranger}/healthz",
                     timeout=3).status_code == 200


def test_the_real_health_shape_is_still_accepted(home, monkeypatch):
    """The control: raggity's own /healthz body must keep working."""
    real = {"status": "ok", "version": "0.12.0", "index_backend": "lancedb",
            "documents": 42}
    assert rag._is_raggity_health(real) is True
    assert rag._is_raggity_health({"hello": "world"}) is False
    assert rag._is_raggity_health({}) is False
    assert rag._is_raggity_health(None) is False
    assert rag._is_raggity_health(["not", "a", "dict"]) is False
    # an older raggity that reports only a version is still ours
    assert rag._is_raggity_health({"version": "0.9.0"}) is True
