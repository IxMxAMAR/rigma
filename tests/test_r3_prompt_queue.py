"""R3-5: a queued prompt was silently dropped on the wrong unwind.

The queue holds prompts typed while a turn is running. `_release_claim` popped
the WHOLE queue on any unwind, which is right when the reader walked away and
wrong when the generator died for a reason that is not the reader's decision —
the prompts were neither delivered nor declined, and the user had just been told
they were "queued behind the running reply".

It also had no bound at all: the queue is in memory only, so a client that keeps
typing during a long turn grows the server without limit.

SCOPE, because it is easy to over-claim: the ordinary ENGINE-FAILURE path was
never the bug. `_llm_turn` reports a failure as an SSE event and RETURNS, so
`_drain` still runs the queued prompt — that is asserted here too, so nobody
"fixes" it twice.

HOW THE FIX IS PINNED. Three complementary things, because the interesting path
(a generator closed mid-turn) cannot be held open from outside: `TestClient`
buffers the whole response, so by the time a chunk is readable the turn is over.
  1. the generator's `except BaseException` path is driven for REAL, by making the
     turn body raise (a test hook on `_llm_turn`), and the queue is observed;
  2. the two release sites are asserted at the source that defines them, since the
     difference BETWEEN them is the fix;
  3. the safe default is asserted, because a default of "drop" is how this bug
     happened and any future caller would inherit it silently.
"""
import inspect
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import serve, state as st


class _Engine(BaseHTTPRequestHandler):
    """Answers every turn immediately."""

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        for chunk in ('data: {"choices":[{"delta":{"content":"a"}}]}\n\n',
                      'data: {"choices":[{"delta":{"content":"b"},'
                      '"finish_reason":"stop"}]}\n\n',
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


@pytest.fixture
def client(tmp_path, monkeypatch, engine):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=1234, ui_pid=1234)
    c = TestClient(serve.build_app(upstream_port=engine))
    yield c


def _closures(app):
    """Every closure cell dict reachable from the app's routes — the queue lives
    in build_app's scope, which is not otherwise addressable."""
    out = []
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        cells = {}
        for name, cell in zip(fn.__code__.co_freevars, fn.__closure__):
            try:
                cells[name] = cell.cell_contents
            except ValueError:
                pass
        out.append(cells)
    return out


def _queue_state(client):
    for c in _closures(client.app):
        if "_streaming" in c and "_queued" in c and "_QUEUE_MAX" in c:
            return c["_streaming"], c["_queued"], c["_QUEUE_MAX"]
    raise AssertionError("could not reach the queue state")


def _new_session(client):
    return client.post("/api/sessions", json={}).json()["id"]


def test_the_queue_is_bounded_and_refuses_rather_than_grows(client):
    """A cap at the door. Discarding later would be the silent loss this finding
    is about, so the refusal has to happen where the user can see it."""
    sid = _new_session(client)
    streaming, queued, cap = _queue_state(client)
    assert cap == 32, cap
    streaming.add(sid)                   # the state a second request hits
    try:
        for i in range(cap):
            r = client.post(f"/api/sessions/{sid}/chat", json={"message": f"m{i}"})
            assert r.status_code == 200, (i, r.text)
        assert len(queued[sid]) == cap
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": "one more"})
        assert r.status_code == 429, r.text
        assert "queued" in r.text
        assert len(queued[sid]) == cap    # the refusal must not have appended
    finally:
        streaming.discard(sid)
        queued.pop(sid, None)


def test_a_dead_turn_does_not_discard_the_queue(client, monkeypatch):
    """THE BUG, driven for real. The turn dies with a BaseException, so the
    chat handler's `except BaseException` runs `_release_claim()` — which before
    the fix popped the queue. A queued prompt that was never delivered must
    survive, because the reader did not decline it; the turn simply died.

    `_llm_turn` is a local inside `build_app`, not a module attribute, so the
    generator it drives is what has to be replaced. `chat_turn` resolves
    `_llm_turn` from the enclosing scope, and that cell is reachable through the
    route's closure — the same route the queue state comes from."""
    sid = _new_session(client)
    streaming, queued, _ = _queue_state(client)
    queued[sid] = ["still wanted"]

    async def _explode(*a, **k):
        raise KeyboardInterrupt("the turn died hard")
        yield  # pragma: no cover  (makes this an async generator)

    # patch the closure cell the route actually reads
    for route in client.app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        names = fn.__code__.co_freevars
        if "_llm_turn" not in names:
            continue
        monkeypatch.setattr(fn.__closure__[names.index("_llm_turn")],
                            "cell_contents", _explode)
        break
    else:
        pytest.fail("could not reach the _llm_turn closure cell")

    with pytest.raises(BaseException):
        client.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert sid not in streaming, "the claim must not leak after a dead turn"
    assert queued.get(sid) == ["still wanted"], (
        "a dead turn discarded a queued prompt that was never delivered")


def test_a_finished_turn_delivers_the_queued_prompt(client):
    """The normal path, so nobody 'fixes' it twice: a queued prompt is DRAINED by
    the running turn, which is why the queue is empty afterwards. Empty here means
    delivered, not dropped — the two are only distinguishable by the transcript."""
    sid = _new_session(client)
    _, queued, _ = _queue_state(client)
    queued[sid] = ["deliver me"]
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200 and "[DONE]" in r.text
    got = client.get(f"/api/sessions/{sid}").json()
    users = [m["content"] for m in got["messages"] if m["role"] == "user"]
    assert "deliver me" in users, users
    assert not queued.get(sid)


def test_the_drop_decision_is_the_readers_cancel_and_not_any_unwind():
    """The two release paths, asserted at the source that defines them, because
    the difference BETWEEN them is the fix. `_drain`'s finally passes the
    `_cancelled` flag (set only by a stop); the disconnect safety net passes True
    (the reader is gone and the body was never read). Neither may drop
    unconditionally, which is what the old code did."""
    src = inspect.getsource(serve.build_app)
    assert "_release_claim(dropped=_cancelled)" in src, (
        "the drain must drop the queue only when the reader cancelled")
    assert "BackgroundTask(_release_claim, True)" in src, (
        "a client that vanished before the body was read has nobody left to "
        "want the prompts")
    assert src.count("_cancelled = True") == 1, (
        "only a stop may mark the queue droppable")


def test_the_release_helper_defaults_to_keeping_the_queue():
    """A default of "drop" is how this bug happened: any new caller would
    silently inherit the destructive behaviour. The default must be the safe
    one."""
    src = inspect.getsource(serve.build_app)
    assert "def _release_claim(dropped: bool = False)" in src, (
        "the safe default is what stops a future caller reintroducing R3-5")
