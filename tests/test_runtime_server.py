import http.server
import socket
import sys
import threading
import uuid
from pathlib import Path

import httpx

from rigma.models import ComboFlags, GgufFile, RunPlan
from rigma.runtime import launch_server

_TOKEN_HEADER = "X-Rigma-Fake-Token"


def _free_port() -> int:
    """An ephemeral loopback port the OS just told us was free.

    REC-1: this test used the literal 11599 through the REAL `launch_server`, so
    two concurrent runs shared the port and the product's own `is_healthy()`
    would accept a 200 from the other run's fake_server — or from any local
    service answering /health. Binding an ephemeral port removes the shared
    resource.
    """
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _answers_as_our_child(sp, port: int, token: str) -> bool:
    """True only if the /health 200 on `port` came from the fake we launched.

    `sp.is_healthy()` alone only proves that SOMETHING answered; the per-launch
    token is proof that it was our child (REC-1).
    """
    if sp.proc.poll() is not None:
        return False
    try:
        r = httpx.get(f"http://127.0.0.1:{port}/health", timeout=3)
    except Exception:
        return False
    return r.status_code == 200 and r.headers.get(_TOKEN_HEADER) == token


def test_launch_health_stop(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = RunPlan(model_slug="x",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q8_0"),
                   backend="cpu", flags=ComboFlags(ctx=8192), origin="calculator")
    fake = Path(__file__).parent / "fake_server.py"
    port = _free_port()
    token = uuid.uuid4().hex
    # exe = python; extra args make it run our fake, which ignores the plan args
    sp = launch_server(Path(sys.executable), plan, Path("model.gguf"),
                       port=port, timeout=30,
                       extra_args=[str(fake), "--token", token])
    try:
        assert sp.is_healthy()
        assert sp.url == f"http://127.0.0.1:{port}"
        assert _answers_as_our_child(sp, port, token), (
            "the /health 200 came from a foreign process, not the fake we "
            "launched (REC-1)")
    finally:
        sp.stop()
    assert not sp.is_healthy()


# --- REC-1 regression: a foreign /health 200 is not our child ---------------

class _ForeignHealth(http.server.BaseHTTPRequestHandler):
    """Answers /health with a bare 200 and no token — the impostor."""

    hits: list = []

    def do_GET(self):
        type(self).hits.append(self.path)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def _start_foreign_health():
    """An in-process /health server on an ephemeral port, bound EXCLUSIVELY so
    the launched child cannot also bind it on Windows (SO_REUSEADDR)."""
    class _Srv(http.server.HTTPServer):
        allow_reuse_address = False

    _ForeignHealth.hits = []
    srv = _Srv(("127.0.0.1", 0), _ForeignHealth)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, int(srv.server_address[1])


class _LiveProc:
    def poll(self):
        return None


class _LiveSp:
    proc = _LiveProc()


def test_a_foreign_health_server_is_not_our_child():
    """`is_healthy()` would call this port healthy; the token proof must not."""
    srv, port = _start_foreign_health()
    try:
        assert httpx.get(f"http://127.0.0.1:{port}/health",
                         timeout=3).status_code == 200
        assert not _answers_as_our_child(_LiveSp(), port, "our-token")
    finally:
        srv.shutdown()
        srv.server_close()
    assert _ForeignHealth.hits, "the impostor was never probed"
