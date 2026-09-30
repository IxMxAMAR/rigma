"""D3a: a reloaded chat cannot tell "still streaming" from "interrupted".

`_streaming` is the server's in-memory truth for "a turn is running in this
session". No route exposed it, so after a mid-turn reload the UI showed the last
checkpoint's "this reply was interrupted" notice while the turn was still going,
and the rail showed no running indicator.

`STREAMING` is module scope (not a `build_app` local) so this can be stated
without an engine: a test can put an id in it directly. The frontend half — read
the flag, suppress the false notice, poll the checkpoint — is
`chatStore.streaming.test.tsx`, which feeds the real payload shape through the
real client paths.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import serve
from rigma.serve import build_app

# The key the client reads on BOTH payloads (`chatStore.ts`, and the
# `streaming?: boolean` declarations in `lib/api.ts`). Named once here so the
# assertion below is on the LIVE payload rather than on the spelling of the
# frontend source: a rename on the server must fail for the reason that matters
# — the client would read nothing — not on a `count()` of a string.
STREAMING_KEY = "streaming"


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
    """The cross-language contract, read from the LIVE payload.

    `serve.py` writes the flag into the list row and into the single-session
    body; the client reads it on both. This asserts the SHAPE the client
    depends on — present, and a `bool` — for an idle session and for a
    streaming one, and that the flag is per-session.

    An earlier version of this file claimed to pin "the name itself across the
    language boundary" by counting `streaming?: boolean;` and
    `.streaming === true` in the frontend source. A count is not a coupling:
    two occurrences anywhere satisfy it, and a correct simultaneous rename on
    both sides fails it. It also could not see the case worth seeing — the
    server dropping the key from the LIST row while the body keeps it.
    """
    sid = client.post("/api/sessions", json={}).json()["id"]

    def both(session_id: str):
        """The two payloads the client loads for one session."""
        body = client.get(f"/api/sessions/{session_id}").json()
        row = next(r for r in client.get("/api/sessions").json()
                   if r["id"] == session_id)
        return body, row

    # nothing is running: the flag is PRESENT and false on BOTH payloads. Before
    # the fix the key was simply absent on both.
    for payload in both(sid):
        assert STREAMING_KEY in payload, (
            "the server stopped emitting the streaming flag on a payload the "
            "client reads")
        assert isinstance(payload[STREAMING_KEY], bool)
        assert payload[STREAMING_KEY] is False

    # a live turn is visible on both payloads the client already reads
    serve.STREAMING.add(sid)
    try:
        for payload in both(sid):
            assert payload[STREAMING_KEY] is True
        # …and it is per-session: another chat is not marked by it
        other = client.post("/api/sessions", json={}).json()["id"]
        for payload in both(other):
            assert payload[STREAMING_KEY] is False
    finally:
        serve.STREAMING.discard(sid)

    for payload in both(sid):
        assert payload[STREAMING_KEY] is False
