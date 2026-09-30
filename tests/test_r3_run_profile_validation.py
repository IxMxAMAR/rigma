"""A7: an unknown run profile must be refused, not silently coerced to "all".

`POST /api/runs` read `profile` and handed it to `runs.create`, which coerced
anything outside `runs.PROFILES` to "all" — the profile that grants the full
network and delete surface. A typo, or a client that sent "read-only", therefore
started the MOST permissive run and was told nothing. The default when the field
is ABSENT stays "all" (the owner's OD-1 choice); a value that is PRESENT must be
one of the four names. `/api/runs` is the only route in serve.py that accepts a
run profile (`POST /api/sessions/{sid}` cannot write `run_profile` — it is not
in `sessions.MUTABLE_FIELDS`), so it is the only route this covers.
"""
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs, serve, sessions, state as st


class _Engine(BaseHTTPRequestHandler):
    """Answers every turn with a short narration, so the run makes no progress
    and ends on its own — these tests are about the profile, not the work."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in ('data: {"choices":[{"delta":{"content":"ok"}}]}\n\n',
                      'data: [DONE]\n\n'):
            self.wfile.write(chunk.encode())
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


_CLIENTS = []


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    yield tmp_path
    try:
        a = runs.active()
        if a:
            runs.set_status(a, "stopped", "test teardown")
    except Exception:
        pass
    while _CLIENTS:
        try:
            _CLIENTS.pop().__exit__(None, None, None)
        except Exception:
            pass


def _client(port):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    c = TestClient(serve.build_app(upstream_port=port))
    c.__enter__()
    _CLIENTS.append(c)
    return c


def _wait(client, rid, timeout=25):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    raise AssertionError(
        f"run {rid} did not reach terminal within {timeout}s; last: {r!r}")


ALLOWED = sorted(runs.PROFILES)


@pytest.mark.parametrize("bad", ["read-only", "", "ALL", "all ", "no_network",
                                 "no network", 123, 1.5, True, None, [], {}])
def test_an_unknown_profile_is_400_and_names_the_allowed_set(engine, bad):
    """Before A7 every one of these started an "all" run (or 500'd on the
    unhashable ones); the value must be refused and the allowed set named."""
    c = _client(engine)
    before = {s["id"] for s in c.get("/api/sessions").json()}
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                  "profile": bad})
    assert r.status_code == 400, f"{bad!r} -> {r.status_code} {r.text}"
    body = r.json()["error"]
    assert "profile" in body, body
    for name in ALLOWED:
        assert name in body, f"{name!r} missing from {body!r}"
    # a refused start must not leave an orphaned chat behind it (01-5's lesson)
    assert {s["id"] for s in c.get("/api/sessions").json()} == before


def test_an_absent_profile_still_defaults_to_all(engine):
    """OD-1: the DEFAULT is the owner's and is unchanged. Absent means "all"."""
    c = _client(engine)
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert runs.load(rid)["profile"] == "all"
    sid = runs.load(rid)["session_id"]
    assert sessions.load(sid)["run_profile"] == "all"
    _wait(c, rid)


@pytest.mark.parametrize("good", ALLOWED)
def test_a_valid_profile_passes_through_unchanged(engine, good):
    c = _client(engine)
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                  "profile": good})
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert runs.load(rid)["profile"] == good
    sid = runs.load(rid)["session_id"]
    assert sessions.load(sid)["run_profile"] == good
    _wait(c, rid)
