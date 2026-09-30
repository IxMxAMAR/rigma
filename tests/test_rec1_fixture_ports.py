"""REC-1 regression: the `oai_server` fixture must yield ITS OWN fake engine.

`43e63de` replaced the hard-coded 11598 with a fresh ephemeral port per attempt,
but readiness was still decided by "somebody answered /health with 200". The
child takes ~0.6 s to die after losing the bind race, so `proc.poll()` is still
None on the first probe and a foreign listener on that port was accepted as
ours — the independent verifier forced `_free_port()` to return a foreign
server's port and the fixed fixture yielded the impostor after 0.67 s.

These tests drive the real fixture body (`oai_server.__wrapped__()`, the
undecorated generator) and the readiness decision helper. The fixture-level
tests fail against the pre-fix fixture: it yields the foreign port on the first
200.
"""
import http.server
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path

import httpx

import test_bench
from test_bench import oai_server

_FAKE = Path(test_bench.__file__).parent / "fake_oai_server.py"
_TOKEN_HEADER = getattr(test_bench, "_TOKEN_HEADER", "X-Rigma-Fake-Token")


class _ForeignHealth(http.server.BaseHTTPRequestHandler):
    """An impostor that looks ready to a naive /health probe: 200, no token."""

    hits: list = []

    def do_GET(self):
        type(self).hits.append(self.path)
        self.send_response(200)
        self.end_headers()

    def log_message(self, *a):
        pass


def _start_foreign_health():
    """A foreign /health server on an ephemeral port.

    Bound EXCLUSIVELY (`allow_reuse_address = False`): on Windows a plain
    HTTPServer sets SO_REUSEADDR, which would let the fixture's child bind the
    same port too and make the test ambiguous. Exclusive binding makes the child
    lose the port outright (WinError 10013 / EADDRINUSE).
    """
    class _Srv(http.server.HTTPServer):
        allow_reuse_address = False

    _ForeignHealth.hits = []
    srv = _Srv(("127.0.0.1", 0), _ForeignHealth)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, int(srv.server_address[1])


def _start_bare_listener():
    """A socket that accepts TCP and never answers HTTP — a taken port that is
    not our server."""
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    s.listen(1)
    return s, int(s.getsockname()[1])


class _LiveProc:
    """Stand-in for a Popen that is still alive, for the probe's poll()."""

    def poll(self):
        return None


def _fixture_body():
    """The undecorated `oai_server` generator. pytest's @fixture wraps the
    function with functools.wraps, so __wrapped__ is the original."""
    return getattr(oai_server, "__wrapped__", oai_server)()


def _drive_fixture_first_port(monkeypatch, first_port):
    """Make the fixture's FIRST `_free_port()` return `first_port`; later picks
    are real ephemeral ports. Returns the generator and the pick counter."""
    real_free_port = test_bench._free_port
    picks = []

    def fake_free_port():
        picks.append(1)
        return first_port if len(picks) == 1 else real_free_port()

    monkeypatch.setattr(test_bench, "_free_port", fake_free_port)
    return _fixture_body(), picks


def test_probe_rejects_a_foreign_200():
    """A 200 without our token is not ours even while our child looks alive."""
    srv, port = _start_foreign_health()
    try:
        assert test_bench._probe_own_server(_LiveProc(), port,
                                            "our-token") == "foreign"
    finally:
        srv.shutdown()
        srv.server_close()


def test_probe_rejects_a_bare_listener():
    """A listener that accepts but never answers is never "ready": the probe
    keeps waiting, and the fixture only moves on when the child dies (or the
    probe budget runs out)."""
    s, port = _start_bare_listener()
    try:
        verdict = test_bench._probe_own_server(_LiveProc(), port, "our-token")
        assert verdict != "ready"
        assert verdict == "waiting"
    finally:
        s.close()


def test_probe_waits_while_nothing_is_bound():
    """An unbound port is "waiting", not "ready" and not "foreign"."""
    port = test_bench._free_port()
    assert test_bench._probe_own_server(_LiveProc(), port, "our-token") == "waiting"


def test_probe_accepts_our_own_child():
    """Positive control: our fake, started with our token, is ready."""
    token = uuid.uuid4().hex
    port = test_bench._free_port()
    proc = subprocess.Popen([sys.executable, str(_FAKE), "--port", str(port),
                             "--token", token])
    try:
        deadline = time.time() + 10
        verdict = "waiting"
        while time.time() < deadline and verdict not in ("ready", "dead"):
            verdict = test_bench._probe_own_server(proc, port, token)
            if verdict != "ready":
                time.sleep(0.1)
        assert verdict == "ready"
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
            proc.wait(timeout=5)


def test_fixture_refuses_a_foreign_listener(monkeypatch):
    """REGRESSION: point the first port pick at an impostor answering 200.

    The fixture must retry on a fresh port and yield its OWN child; it must
    never yield the impostor's port. Before the token check this fails — the
    old fixture yielded `foreign_port` on the first 200."""
    srv, foreign_port = _start_foreign_health()
    gen, picks = _drive_fixture_first_port(monkeypatch, foreign_port)
    try:
        port = next(gen)
        assert port != foreign_port, "fixture yielded a foreign server's port"
        assert len(picks) >= 2, "fixture did not retry away from the impostor"
        # the server it did yield must be its own: it carries the per-child token
        resp = httpx.get(f"http://127.0.0.1:{port}/health", timeout=2)
        assert resp.status_code == 200
        assert resp.headers.get(_TOKEN_HEADER), "yielded server sent no token"
    finally:
        gen.close()
        srv.shutdown()
        srv.server_close()
    assert _ForeignHealth.hits, "the impostor was never probed"


def test_fixture_refuses_a_bare_listener(monkeypatch):
    """Same, for a taken port that accepts but never answers."""
    listener, bare_port = _start_bare_listener()
    gen, picks = _drive_fixture_first_port(monkeypatch, bare_port)
    try:
        port = next(gen)
        assert port != bare_port
        assert len(picks) >= 2
    finally:
        gen.close()
        listener.close()
