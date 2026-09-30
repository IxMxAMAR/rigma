"""D3a: a reloaded chat cannot tell "still streaming" from "interrupted".

`_streaming` is the server's in-memory truth for "a turn is running in this
session". No route exposed it, so after a mid-turn reload the UI showed the last
checkpoint's "this reply was interrupted" notice while the turn was still going,
and the rail showed no running indicator.

`STREAMING` is module scope (not a `build_app` local) so this can be stated
without an engine: a test can put an id in it directly. The frontend half — read
the flag, suppress the false notice, poll the checkpoint — is a later wave.
"""
import pathlib

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


# --- the name itself, which is the half neither suite pinned ------------------
#
# `serve.py` writes `r["streaming"]` into the list row (:1833) and
# `{**s, "streaming": sid in _streaming}` into the single body (:1911). The
# client reads `s.streaming === true` / `r?.streaming === true`
# (chatStore.ts:923,941,1027) and declares `streaming?: boolean` on both
# payloads (lib/api.ts:14,73).
#
# The test above pins the EMIT, and the client's own vitest files pin the READ
# against a fixture that happens to use the same word. A rename on ONE side
# alone therefore leaves every suite green while the "interrupted" bug comes
# back — which is exactly the silent coupling D3a's follow-up is about. These
# two pin the literal across the language boundary.

STREAMING_KEY = "streaming"


def _frontend_src(rel: str) -> str:
    root = pathlib.Path(__file__).resolve().parents[1] / "frontend-v2" / "src"
    return (root / rel).read_text(encoding="utf-8")


def test_the_client_reads_the_exact_key_the_server_emits(client):
    """The key is read from the LIVE payload; the client must read that name."""
    sid = client.post("/api/sessions", json={}).json()["id"]
    body = client.get(f"/api/sessions/{sid}").json()
    row = next(r for r in client.get("/api/sessions").json() if r["id"] == sid)
    assert isinstance(body.get(STREAMING_KEY), bool)
    assert isinstance(row.get(STREAMING_KEY), bool)

    api = _frontend_src("lib/api.ts")
    # Both payload types the store reads: SessionDetail and SessionSummary.
    assert api.count(f"{STREAMING_KEY}?: boolean;") >= 2, (
        "lib/api.ts no longer declares the flag on both session payloads; the "
        "client would type-check while reading nothing")

    store = _frontend_src("chat/chatStore.ts")
    read = f".{STREAMING_KEY} === true"
    assert store.count(read) >= 2, (
        "chatStore.ts no longer reads the server's streaming flag on both the "
        "list row and the single session")
    # A guard that cannot fail is not a guard: the same expression, against the
    # same source with the key renamed, must report the break.
    broken = store.replace(read, ".is_streaming === true")
    assert broken != store, (
        "the read site moved, so this guard no longer checks it")
    assert read not in broken
