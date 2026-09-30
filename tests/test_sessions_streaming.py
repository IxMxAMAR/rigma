"""D3a: a reloaded chat cannot tell "still streaming" from "interrupted".

`_streaming` is the server's in-memory truth for "a turn is running in this
session". No route exposed it, so after a mid-turn reload the UI showed the last
checkpoint's "this reply was interrupted" notice while the turn was still going,
and the rail showed no running indicator.

`STREAMING` is module scope (not a `build_app` local) so this can be stated
without an engine: a test can put an id in it directly. The frontend half — read
the flag, suppress the false notice, poll the checkpoint — is a later wave.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import serve
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(home):
    c = TestClient(build_app(upstream_port=1))
    c.__enter__()
    yield c
    c.__exit__(None, None, None)


def test_both_session_payloads_report_whether_a_turn_is_streaming(client):
    sid = client.post("/api/sessions", json={}).json()["id"]

    # nothing is running: the flag is PRESENT and false. Before the fix the key
    # was simply absent on both payloads.
    assert client.get(f"/api/sessions/{sid}").json()["streaming"] is False
    rows = {r["id"]: r for r in client.get("/api/sessions").json()}
    assert rows[sid]["streaming"] is False

    # a live turn is visible on both payloads the client already reads
    serve.STREAMING.add(sid)
    try:
        assert client.get(f"/api/sessions/{sid}").json()["streaming"] is True
        rows = {r["id"]: r for r in client.get("/api/sessions").json()}
        assert rows[sid]["streaming"] is True
        # …and it is per-session: another chat is not marked by it
        other = client.post("/api/sessions", json={}).json()["id"]
        rows = {r["id"]: r for r in client.get("/api/sessions").json()}
        assert rows[other]["streaming"] is False
    finally:
        serve.STREAMING.discard(sid)

    assert client.get(f"/api/sessions/{sid}").json()["streaming"] is False
    rows = {r["id"]: r for r in client.get("/api/sessions").json()}
    assert rows[sid]["streaming"] is False
