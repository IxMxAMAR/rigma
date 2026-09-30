"""The HTTP/turn-layer findings of the 2026-09-04 audit.

Every test here reproduces a specific way the server lost work, blocked the
event loop, killed a healthy run, or answered a request it should have
refused. The fake engine is scripted per test: `Engine.script` decides what the
next streamed turn emits, and `Engine.aux` what a non-streaming call (the
summarizer, the titler, the delegate helper) answers with.
"""
import json
import logging
import os
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from rigma import serve
from rigma import sessions
from rigma import state as st
from rigma.serve import build_app

# The served page's own authority. TestClient defaults to "testserver", which
# the guard also accepts, but every request the real UI makes carries these.
BASE = "http://127.0.0.1:11500"


# --- AUDIT F01-1: the rebinding guard was a prefix test -----------------------
def test_a_host_that_merely_starts_with_127_is_not_loopback():
    # A publishable name an attacker can point at 127.0.0.1. The guard used
    # `h.startswith("127.")`, so every one of these was treated as the local UI.
    for h in ("127.0.0.1.evil.example:11500", "127.0.0.1.evil.example",
              "127.evil.example", "127.0.0.1.attacker.test"):
        assert serve._is_local_host(h) is False, h
        assert serve.guard_request(h, "") != "", h


def test_real_loopback_hosts_are_still_accepted():
    for h in ("127.0.0.1:11500", "127.0.0.1", "127.1.2.3:11500",
              "localhost:11500", "localhost", "[::1]:11500", "testserver"):
        assert serve._is_local_host(h) is True, h
        assert serve.guard_request(h, "") == "", h


def test_a_non_loopback_address_is_still_refused():
    for h in ("192.168.1.5:11500", "10.0.0.1", "evil.example:11500",
              "127.0.0.1.5.6"):
        assert serve._is_local_host(h) is False, h


# --------------------------------------------------------------------------
# a scriptable llama-server


class Engine(BaseHTTPRequestHandler):
    # one streamed turn per entry; the last entry repeats once exhausted
    script: list = []
    aux: list = []
    aux_delay = 0.0
    stream_delay = 0.0
    aux_status = 200
    seen: list = []
    on_aux = None            # hook: runs INSIDE the summarizer request

    def _sse(self, obj):
        self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
        self.wfile.flush()

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        Engine.seen.append(body)
        if not body.get("stream"):
            if Engine.on_aux is not None:
                Engine.on_aux()
            if Engine.aux_delay:
                time.sleep(Engine.aux_delay)
            self.send_response(Engine.aux_status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            msg = (Engine.aux.pop(0) if len(Engine.aux) > 1
                   else (Engine.aux[0] if Engine.aux else {"content": "AUX"}))
            self.wfile.write(json.dumps(
                {"choices": [{"message": msg}]}).encode())
            return
        turn = (Engine.script.pop(0) if len(Engine.script) > 1
                else (Engine.script[0] if Engine.script else []))
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        try:
            for chunk in turn:
                self._sse(chunk)
                if Engine.stream_delay:
                    time.sleep(Engine.stream_delay)
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass          # the client hung up: exactly what F7 is about

    def log_message(self, *a):
        pass


def _say(text, **extra):
    """One streamed turn that just says `text`."""
    out = [{"choices": [{"delta": {"content": c}}]} for c in text]
    out.append({"choices": [{"delta": {}}], **extra})
    return out


def _call(name, args):
    """One streamed turn that emits a single tool call."""
    return [{"choices": [{"delta": {"tool_calls": [
        {"index": 0, "id": "c1", "type": "function",
         "function": {"name": name, "arguments": json.dumps(args)}}]}}]},
        {"choices": [{"delta": {}}]}]


@pytest.fixture
def engine():
    Engine.script, Engine.aux, Engine.seen = [_say("ok")], [], []
    Engine.aux_delay = Engine.stream_delay = 0.0
    Engine.aux_status, Engine.on_aux = 200, None
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield SimpleNamespace(port=srv.server_address[1], cls=Engine)
    srv.shutdown()
    Engine.on_aux = None


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _client(port, **kw):
    return TestClient(build_app(upstream_port=port), base_url=BASE, **kw)


# --------------------------------------------------------------------------
# TestClient buffers the WHOLE response before it returns, so it cannot express
# "the tab went away mid-reply" or "a second request arrives while the first is
# still streaming" — the two situations three of these findings are about.
# These drive the ASGI app directly instead.


def _scope(method, path, headers=None, query=b""):
    hdrs = [(b"host", b"127.0.0.1:11500"),
            (b"content-type", b"application/json")]
    hdrs += [(k.lower().encode(), v.encode())
             for k, v in (headers or {}).items()]
    return {"type": "http", "asgi": {"version": "3.0", "spec_version": "2.3"},
            "http_version": "1.1", "method": method, "scheme": "http",
            "path": path, "raw_path": path.encode(), "query_string": query,
            "root_path": "", "client": ("127.0.0.1", 5000),
            "server": ("127.0.0.1", 11500), "headers": hdrs}


async def _request(app, method, path, payload=None):
    body = b"" if payload is None else json.dumps(payload).encode()
    got = {"status": None, "body": b""}
    state = {"sent": False}

    async def receive():
        if state["sent"]:
            return {"type": "http.disconnect"}
        state["sent"] = True
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            got["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            got["body"] += msg.get("body", b"")

    await app(_scope(method, path,
                     {"content-length": str(len(body))}), receive, send)
    return got


async def _stream(app, path, payload, *, stop_after=None):
    """POST an SSE route. `stop_after` body chunks -> the client disconnects,
    which is what a refresh or a sleeping laptop looks like to the server."""
    import asyncio
    body = json.dumps(payload).encode()
    gone = asyncio.Event()
    got = {"status": None, "chunks": []}
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        await gone.wait()
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            got["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            b = msg.get("body", b"")
            if b:
                got["chunks"].append(b)
                if stop_after is not None and len(got["chunks"]) >= stop_after:
                    gone.set()

    await app(_scope("POST", path,
                     {"content-length": str(len(body))}), receive, send)
    got["text"] = b"".join(got["chunks"]).decode("utf-8", "replace")
    return got


async def _stream_until_armed(app, path, payload, armed):
    """POST an SSE route and disconnect the moment `armed` (a threading.Event
    the test sets from inside the engine) is set.

    `_stream(stop_after=N)` disconnects on a CHUNK COUNT, which cannot express
    "while this particular tool call is in flight": chunks keep arriving after
    the eager task has already been collected. Starlette polls `receive()` for
    the disconnect, so arming from the engine side lands the cancellation
    exactly where the test needs it."""
    import asyncio
    body = json.dumps(payload).encode()
    got = {"status": None, "chunks": []}
    state = {"sent": False}

    async def receive():
        if not state["sent"]:
            state["sent"] = True
            return {"type": "http.request", "body": body, "more_body": False}
        while not armed.is_set():
            await asyncio.sleep(0.01)
        return {"type": "http.disconnect"}

    async def send(msg):
        if msg["type"] == "http.response.start":
            got["status"] = msg["status"]
        elif msg["type"] == "http.response.body":
            got["chunks"].append(msg.get("body", b""))

    await app(_scope("POST", path,
                     {"content-length": str(len(body))}), receive, send)
    got["text"] = b"".join(got["chunks"]).decode("utf-8", "replace")
    return got


def _running(ctx=131072):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=ctx)


def _seed(client, n, run=False):
    sid = client.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["messages"] = [{"role": "user", "content": f"m{i}"} for i in range(n)]
    if run:
        s["run_id"] = "r1"
    sessions.save(s)
    return sid


# --------------------------------------------------------------------------
# F39 — no Host / Origin check


def test_a_page_the_owner_visits_cannot_drive_a_body_less_post(home, engine):
    """A cross-origin simple POST needs no preflight, and every body-less POST
    on this server does something: unload, recalibrate, os.startfile, stop a
    run. The browser always sends Origin on a cross-origin POST — use it."""
    c = _client(engine.port)
    r = c.post("/api/server/unload",
               headers={"Origin": "https://news.example"})
    assert r.status_code == 403
    assert "cross-origin" in r.json()["error"]


def test_a_rebound_dns_name_is_refused_even_on_a_get(home, engine):
    """DNS rebinding makes the attacker's page same-origin, so Origin matches
    and GETs carry none at all. The Host header is the only thing left."""
    c = TestClient(build_app(upstream_port=engine.port),
                   base_url="http://rigma.evil.example:11500")
    assert c.get("/api/sessions").status_code == 403
    r = c.post("/api/sessions", json={},
               headers={"Origin": "http://rigma.evil.example:11500"})
    assert r.status_code == 403


def test_the_apps_own_page_is_untouched(home, engine):
    """The whole point: same-origin GET (no Origin header) and same-origin POST
    (Origin identical to Host) must both behave exactly as before."""
    _running()
    c = _client(engine.port)
    assert c.get("/").status_code == 200
    assert c.get("/api/sessions").status_code == 200
    sid = c.post("/api/sessions", json={"title": "t"},
                 headers={"Origin": BASE}).json()["id"]
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"},
               headers={"Origin": BASE, "Referer": BASE + "/"})
    assert r.status_code == 200 and "[DONE]" in r.text
    assert sessions.load(sid)["messages"][-1]["content"] == "ok"
    # localhost is the other address the owner types
    c2 = TestClient(build_app(upstream_port=engine.port),
                    base_url="http://localhost:11500")
    assert c2.get("/api/sessions").status_code == 200


def test_the_guard_rules_directly():
    assert serve.guard_request("127.0.0.1:11500", "") == ""
    assert serve.guard_request("localhost:11500",
                               "http://localhost:11500") == ""
    assert serve.guard_request("[::1]:11500", "") == ""
    assert serve.guard_request("127.0.0.1:11500", "null")
    assert serve.guard_request("127.0.0.1:11500", "https://evil.example")
    assert serve.guard_request("192.168.1.9:11500", "")
    assert serve.guard_request("", "")


def test_an_upload_must_declare_its_size_and_stick_to_it(home, engine):
    c = _client(engine.port)

    def _chunks():
        yield b"GGUF" * 8

    r = c.post("/api/models/upload?filename=x.gguf", content=_chunks())
    assert r.status_code == 411, r.text
    assert not list((home / "custom" / "incoming").glob("*"))
    # ...and a body that outruns its declaration is cut off, not written out
    app = build_app(upstream_port=engine.port)
    got = _asgi_upload(app, declared=4, body=b"G" * 4096)
    assert got["status"] == 413
    assert not list((home / "custom" / "incoming").glob("*"))


def _asgi_upload(app, declared: int, body: bytes) -> dict:
    """Drive the raw ASGI app: httpx rewrites Content-Length to match the body,
    so a lying sender can only be built at this level."""
    import asyncio
    out: dict = {}
    sent = {"n": 0}

    async def receive():
        if sent["n"]:
            return {"type": "http.disconnect"}
        sent["n"] = 1
        return {"type": "http.request", "body": body, "more_body": False}

    async def send(msg):
        if msg["type"] == "http.response.start":
            out["status"] = msg["status"]

    scope = {"type": "http", "asgi": {"version": "3.0"}, "http_version": "1.1",
             "method": "POST", "scheme": "http", "path": "/api/models/upload",
             "raw_path": b"/api/models/upload",
             "query_string": b"filename=x.gguf", "root_path": "",
             "client": ("127.0.0.1", 5000), "server": ("127.0.0.1", 11500),
             "headers": [(b"host", b"127.0.0.1:11500"),
                         (b"content-length", str(declared).encode())]}
    asyncio.run(app(scope, receive, send))
    return out


# --------------------------------------------------------------------------
# F4 — compaction destroyed anything written while it ran


def test_compaction_keeps_a_turn_that_landed_while_it_folded(home, engine):
    """The fold takes minutes. Whatever the user sent and got answered during
    it used to be gone: not in messages, not in archive, not in the digest."""
    _running(ctx=131072)
    c = _client(engine.port)
    sid = _seed(c, 12)
    Engine.aux = [{"content": "DIGEST"}]

    def _meanwhile():
        # a turn completing on this session WHILE the summarizer is thinking
        live = sessions.load(sid)
        live["messages"].append({"role": "user", "content": "next paragraph"})
        live["messages"].append({"role": "assistant", "content": "the reply"})
        sessions.save(live, base_rev=live[sessions.REV_KEY])

    Engine.on_aux = _meanwhile
    r = c.post(f"/api/sessions/{sid}/compact", json={"keep": 2})
    assert r.status_code == 200, r.text
    after = sessions.load(sid)
    tail = [m["content"] for m in after["messages"]]
    assert "next paragraph" in tail and "the reply" in tail
    assert after["digest"] == "DIGEST"
    assert len(after["archive"]) == 10


def test_compact_is_refused_while_the_chat_is_streaming(home, engine):
    """The reverse ordering loses just as much: the turn's own save lands last
    and reverts a digest that cost minutes."""
    import asyncio
    _running()
    Engine.script = [_say("a long reply" * 20)]
    Engine.stream_delay = 0.01
    c = _client(engine.port)
    sid = _seed(c, 4)
    app = build_app(upstream_port=engine.port)

    async def scenario():
        turn = asyncio.create_task(
            _stream(app, f"/api/sessions/{sid}/chat", {"message": "go"}))
        await asyncio.sleep(0.2)                 # the reply is streaming now
        got = await _request(app, "POST", f"/api/sessions/{sid}/compact",
                             {"keep": 2})
        await turn
        return got

    got = asyncio.run(scenario())
    assert got["status"] == 409, got["body"]
    assert "generating" in json.loads(got["body"])["error"]
    assert sessions.load(sid)["digest"] == ""


# --------------------------------------------------------------------------
# F5 — the archive cap destroyed the oldest prose


def test_a_chats_archive_is_never_trimmed(home, engine):
    """The docstring says "never destroyed", the comment calls the archive the
    user's manuscript, and two design specs say "nothing is destroyed"."""
    _running(ctx=1000)
    Engine.script = [_say("ok", usage={"prompt_tokens": 950},
                          timings={"predicted_per_second": 40})]
    Engine.aux = [{"content": "DIGEST"}]
    c = _client(engine.port)
    sid = _seed(c, 24)
    s = sessions.load(sid)
    s["archive"] = [{"role": "user", "content": f"chapter {i}"}
                    for i in range(serve.ARCHIVE_MAX + 50)]
    sessions.save(s)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: compacted" in r.text, r.text[-300:]
    after = sessions.load(sid)
    assert len(after["archive"]) > serve.ARCHIVE_MAX + 50
    assert after["archive"][0]["content"] == "chapter 0"


def test_a_runs_trimmed_archive_is_spilled_before_it_is_dropped(home, engine):
    """A run keeps the cap (it compacts every few turns and re-serialises the
    whole row each time) — but what falls off goes to disk first."""
    _running(ctx=1000)
    Engine.script = [_say("ok", usage={"prompt_tokens": 950},
                          timings={"predicted_per_second": 40})]
    Engine.aux = [{"content": "DIGEST"}]
    c = _client(engine.port)
    sid = _seed(c, 24, run=True)
    s = sessions.load(sid)
    s["archive"] = [{"role": "user", "content": f"obs {i}"}
                    for i in range(serve.ARCHIVE_MAX + 60)]
    sessions.save(s)
    c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    after = sessions.load(sid)
    assert len(after["archive"]) <= serve.ARCHIVE_MAX
    spill = home / "sessions" / "archive" / f"{sid}.jsonl"
    assert spill.exists(), "the cap dropped messages with no copy anywhere"
    rows = [json.loads(x) for x in
            spill.read_text(encoding="utf-8").splitlines()]
    assert rows[0]["content"] == "obs 0"


def test_the_spill_refuses_a_hostile_session_id(home):
    assert serve._spill_archive("../../evil", [{"role": "user", "content": "x"}]) \
        is False
    assert not (home / "sessions" / "archive").exists()


# --------------------------------------------------------------------------
# F7 — a refresh mid-generation lost the whole reply


def test_a_disconnect_mid_generation_keeps_what_was_generated(home, engine):
    """Eight minutes in, 3500 words on screen, the laptop sleeps. Cancelling
    the SSE generator throws CancelledError/GeneratorExit — neither is an
    Exception subclass, so nothing in the turn ever saw it."""
    import asyncio
    _running()
    Engine.script = [_say("word " * 200)]
    Engine.stream_delay = 0.005
    c = _client(engine.port)
    sid = _seed(c, 2)
    app = build_app(upstream_port=engine.port)
    asyncio.run(_stream(app, f"/api/sessions/{sid}/chat",
                        {"message": "write"}, stop_after=12))
    kept = [m for m in sessions.load(sid)["messages"] if m.get("partial")]
    assert kept, "the whole reply was lost"
    assert kept[0]["role"] == "assistant"
    assert kept[0]["content"].startswith("word")
    assert kept[0]["partial"] is True
    # ...and it says what it is, on the field both transcripts already render
    assert "interrupted" in kept[0]["notice"]
    # the prompt is still there and the turn did not also finish
    assert sessions.load(sid)["messages"][-2]["content"] == "write"


def test_an_engine_error_mid_reply_keeps_the_words(home, engine, monkeypatch):
    """The turn never reaches the persist block when the stream breaks, and
    `text` is only assigned at the end of a round — so 3000 words followed by a
    dropped connection used to persist nothing at all."""
    monkeypatch.setattr(serve, "CHECKPOINT_SECS", 0.0)   # checkpoint every delta
    _running()
    # a stream that stops mid-flight: the error arrives inside the SSE body
    Engine.script = [[{"choices": [{"delta": {"content": "half a scene "}}]},
                      {"choices": [{"delta": {"content": "and then"}}]},
                      {"error": {"message": "context shift failed"}}]]
    c = _client(engine.port)
    sid = _seed(c, 2)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "write"})
    assert "event: error" in r.text
    kept = [m for m in sessions.load(sid)["messages"] if m.get("partial")]
    assert kept and kept[0]["content"] == "half a scene and then"


def test_a_thinking_only_turn_still_leaves_nothing(home, engine, monkeypatch):
    """An empty assistant bubble is exactly the shape that poisons later turns
    — a checkpoint holding only reasoning must be taken back out."""
    monkeypatch.setattr(serve, "CHECKPOINT_SECS", 0.0)
    _running()
    Engine.script = [[{"choices": [{"delta":
                                    {"reasoning_content": "hmm..."}}]},
                      {"choices": [{"delta": {}}]}]]
    c = _client(engine.port)
    sid = _seed(c, 2)
    c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    msgs = sessions.load(sid)["messages"]
    assert [m["role"] for m in msgs][-1] == "user"
    assert not [m for m in msgs if m.get("partial")]


def test_a_finished_turn_leaves_no_partial_behind(home, engine):
    _running()
    Engine.script = [_say("all done")]
    c = _client(engine.port)
    sid = _seed(c, 2)
    c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    msgs = sessions.load(sid)["messages"]
    assert not [m for m in msgs if m.get("partial")]
    assert msgs[-1]["content"] == "all done"
    assert len([m for m in msgs if m.get("role") == "assistant"]) == 1


def test_a_turn_that_cannot_be_saved_is_not_reported_as_saved(
        home, engine, monkeypatch, caplog):
    """A1: if every persist attempt loses the CAS race, the finished reply is
    not in the store. `alive` started True and was only ever assigned on
    SUCCESS, so exhaustion left it True and the turn was treated as saved —
    silently, with the reply living only in whatever partial happened to have
    been checkpointed. It has to say so, and leave the words behind."""
    _running()
    Engine.script = [_say("the whole reply")]
    real = sessions.reload_and_extend

    def _always_stale(sid, messages, since, **kw):
        m = real(sid, messages, since, **kw)
        if m is not None:
            m[sessions.REV_KEY] = -1      # every save() now loses the CAS
        return m

    monkeypatch.setattr(sessions, "reload_and_extend", _always_stale)
    c = _client(engine.port)
    sid = _seed(c, 2)
    with caplog.at_level(logging.ERROR, logger="rigma.serve"):
        r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    # surfaced to the user, not swallowed
    assert "event: notice" in r.text
    assert "could not be saved" in r.text
    # ...and logged, with the reason
    assert "could not save" in caplog.text
    msgs = sessions.load(sid)["messages"]
    # never recorded as a finished turn...
    assert not [m for m in msgs
                if m.get("role") == "assistant" and not m.get("partial")]
    # ...but left un-final, so the checkpoint path kept the words
    kept = [m for m in msgs if m.get("partial")]
    assert kept and kept[0]["content"] == "the whole reply"


def test_a_partial_never_reaches_the_model(home, engine):
    """`partial` and `ckpt_id` are bookkeeping; build_messages must strip them
    exactly as it strips notices and stats."""
    built = sessions.build_messages(
        {"messages": [{"role": "assistant", "content": "half a scene",
                       "partial": True, "ckpt_id": "ck1"}]})
    assert built == [{"role": "assistant", "content": "half a scene"}]


# --------------------------------------------------------------------------
# F10 — /api/server probed the hardware on the event loop, every poll


def test_server_info_probes_off_the_loop_and_only_once_per_poll_burst(
        home, engine, monkeypatch):
    """1.449s of PowerShell on the event loop, every 5-15s, is what made a
    streaming reply stop dead and burst."""
    import asyncio

    from rigma import server_ops
    calls = {"n": 0, "on_loop": []}

    def _slow(_registry=None):
        calls["n"] += 1
        try:
            asyncio.get_running_loop()
            calls["on_loop"].append(True)
        except RuntimeError:
            calls["on_loop"].append(False)
        return {"vram_total_mb": 1}

    monkeypatch.setattr(server_ops, "vram_snapshot", _slow)
    monkeypatch.setattr(server_ops, "available_backends",
                        lambda _r=None: [{"name": "vulkan", "ready": True}])
    _running()
    c = _client(engine.port)
    for _ in range(4):
        assert c.get("/api/server").status_code == 200
    assert calls["n"] == 1, "the probe is not cached between polls"
    assert calls["on_loop"] == [False], "the probe ran on the event loop"
    assert c.get("/api/server").json()["vram"] == {"vram_total_mb": 1}


# --------------------------------------------------------------------------
# F11 — one malformed request wedged every engine control


def test_a_malformed_backend_does_not_wedge_the_switch_lock(home, engine,
                                                            monkeypatch):
    """`(body.get("backend") or "").strip()` with no str() raised
    AttributeError between the acquire and the try/finally. Nothing released
    the lock, so Load/Unload/Switch/ctx 409'd forever and _ensure_loaded taxed
    every turn and every /v1 request 30 seconds."""
    from rigma import server_ops
    monkeypatch.setattr(server_ops, "available_backends",
                        lambda _r=None: [{"name": "vulkan", "ready": True}])
    monkeypatch.setattr(server_ops, "perform_unload",
                        lambda: {"unloaded": True})
    _running()
    c = TestClient(build_app(upstream_port=engine.port), base_url=BASE,
                   raise_server_exceptions=False)
    r = c.post("/api/server/ctx", json={"ctx": 8192, "backend": 5})
    assert r.status_code == 400, r.text
    r2 = c.post("/api/server/unload")
    assert r2.status_code != 409, "the switch lock was never released"
    assert r2.status_code == 200


# --------------------------------------------------------------------------
# F12 — in-run compaction was policed by a watchdog it could not satisfy


def test_compaction_heartbeats_so_the_watchdog_measures_the_engine(
        home, engine, monkeypatch):
    """All end-of-turn housekeeping happens after the last meta chunk, so
    _drain_turn policed a multi-minute fold at IDLE_SECS (90s). Two of those
    ended the run as "engine unresponsive" — blaming the engine for work the
    watchdog would not let finish."""
    monkeypatch.setattr(serve, "TICK_SECS", 0.02)
    _running(ctx=1000)
    Engine.script = [_say("ok", usage={"prompt_tokens": 950},
                          timings={"predicted_per_second": 40})]
    Engine.aux = [{"content": "DIGEST"}]
    Engine.aux_delay = 0.25
    c = _client(engine.port)
    sid = _seed(c, 24)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: housekeeping" in r.text, r.text[-400:]
    assert "event: compacted" in r.text
    # ...and that event is what buys the generous budget in the headless drain
    assert b"event: housekeeping" in serve.PREFILL_EVENTS


# --------------------------------------------------------------------------
# F13 — the KV prefix restore blocked the event loop for seconds


def test_prefix_warm_and_snapshot_run_off_the_event_loop(home, engine,
                                                         monkeypatch):
    """kvcache.slot_action uses synchronous httpx with timeout=120, so reading
    a multi-gigabyte snapshot stopped every other request outright."""
    import asyncio
    where = {}

    def _spy(key):
        def _f(_msgs, _chain=""):
            try:
                asyncio.get_running_loop()
                where[key] = "event loop"
            except RuntimeError:
                where[key] = "thread"
        return _f

    monkeypatch.setattr(serve, "_prefix_warm", _spy("warm"))
    monkeypatch.setattr(serve, "_prefix_snapshot", _spy("snapshot"))
    _running()
    c = _client(engine.port)
    sid = _seed(c, 2)
    c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert where == {"warm": "thread", "snapshot": "thread"}


def test_a_snapshot_is_not_skipped_after_the_engine_was_replaced(
        monkeypatch, tmp_path):
    """AUDIT 02-2: `warm_key` asserts what the LIVE engine's slot 0 holds.

    It used to be the bare prefix key, so after a switch away and back — a new
    engine process with an empty slot 0 — the matching snapshot on disk was
    skipped and the whole history re-prefilled: the 20-60K-token stall the
    snapshot exists to avoid. The key now carries the engine generation, so the
    assertion cannot outlive the process it describes.
    """
    from rigma import prefixcache

    restored: list = []
    monkeypatch.setattr(serve, "_prefix_ctx",
                        lambda: (1, tmp_path, "FP", "pid-1"))
    monkeypatch.setattr(prefixcache, "prefix_keys",
                        lambda msgs, fp: [prefixcache.PrefixPoint(
                            n_messages=1, key="K", approx_tokens=9000)])
    monkeypatch.setattr(prefixcache, "available", lambda d: {"K"})
    monkeypatch.setattr(
        prefixcache, "warm",
        lambda port, d, key, slot=0: restored.append(key) or None)
    monkeypatch.setitem(serve._PREFIX_STATE, "warm_key", "")
    monkeypatch.setitem(serve._PREFIX_STATE, "last_points", {})

    msgs = [{"role": "user", "content": "hi"}]
    serve._prefix_warm(msgs)
    assert restored == ["K"]
    assert serve._PREFIX_STATE["warm_key"] == "pid-1:K"

    # the same live engine, same prefix: the slot is already at least as warm
    restored.clear()
    serve._prefix_warm(msgs)
    assert restored == [], "restoring what the slot already holds wastes time"

    # a REPLACEMENT engine process: same snapshot on disk, empty slot 0
    monkeypatch.setattr(serve, "_prefix_ctx",
                        lambda: (1, tmp_path, "FP", "pid-2"))
    serve._prefix_warm(msgs)
    assert restored == ["K"], "the replacement engine's slot 0 is empty"

    # a snapshot just taken leaves slot 0 holding exactly that prefix, so the
    # snapshot side must record the SAME generation-prefixed key or the next
    # turn restores what it already has
    monkeypatch.setattr(prefixcache, "prefix_keys",
                        lambda msgs, fp: [prefixcache.PrefixPoint(
                            n_messages=2, key="K2", approx_tokens=20000)])
    monkeypatch.setattr(prefixcache, "snapshot",
                        lambda port, d, key, slot=0, meta=None: None)
    monkeypatch.setitem(serve._PREFIX_STATE, "last_points", {})
    serve._prefix_snapshot(msgs)
    assert serve._PREFIX_STATE["warm_key"] == "pid-2:K2"


def test_a_long_chat_does_not_suppress_a_new_chats_snapshot(monkeypatch,
                                                            tmp_path):
    """AUDIT 02-5: the growth heuristic used ONE process-wide `last_point`.

    A 60-message/60K-token chat A therefore made a new chat B that had grown
    past the snapshot floor but not past A look like it had not grown at all
    (`point.n_messages <= last.n_messages`), so B was never snapshotted and
    re-prefilled from zero after a restart. Each chain now measures its own
    growth.
    """
    from rigma import prefixcache

    def _point(key, tokens, n):
        return prefixcache.PrefixPoint(n_messages=n, key=key,
                                       approx_tokens=tokens)

    taken: list = []
    monkeypatch.setattr(serve, "_prefix_ctx",
                        lambda: (1, tmp_path, "FP", "gen"))
    monkeypatch.setattr(prefixcache, "snapshot",
                        lambda port, d, key, slot=0, meta=None:
                        taken.append(key) or None)
    monkeypatch.setitem(serve._PREFIX_STATE, "last_points", {})

    # chat A: a long conversation, snapshotted at 60K
    monkeypatch.setattr(prefixcache, "prefix_keys",
                        lambda msgs, fp: [_point("A", 60000, 60)])
    serve._prefix_snapshot([{"role": "user", "content": "a"}], "chat-A")
    assert taken == ["A"]

    # chat B: 30 messages/20K tokens in its OWN chain — well past the 4096-token
    # floor, but "behind" A on both measures. Under the old global `last_point`
    # this was suppressed; it must snapshot now.
    monkeypatch.setattr(prefixcache, "prefix_keys",
                        lambda msgs, fp: [_point("B", 20000, 30)])
    serve._prefix_snapshot([{"role": "user", "content": "b"}], "chat-B")
    assert taken == ["A", "B"], "chat B's snapshot was suppressed by chat A"

    # ...and a second turn of B that has not grown enough is still suppressed,
    # so the per-chain bookkeeping did not just disable the heuristic
    monkeypatch.setattr(prefixcache, "prefix_keys",
                        lambda msgs, fp: [_point("B2", 21000, 31)])
    serve._prefix_snapshot([{"role": "user", "content": "b"}], "chat-B")
    assert taken == ["A", "B"], "the growth floor stopped applying"


# --------------------------------------------------------------------------
# F14 — idle auto-unload killed the engine under a running job


def test_idle_unload_never_fires_while_a_turn_is_streaming(home, engine,
                                                           monkeypatch):
    """activity["last"] was stamped by the chat route and the /v1 proxy only.
    A run reaches the engine through _llm_turn on the internal client and
    bypassed both, so the poller saw an idle server and killed it mid-flight."""
    import asyncio

    from rigma import server_ops
    unloads = []
    monkeypatch.setattr(server_ops, "perform_unload",
                        lambda: unloads.append(1) or {"unloaded": True})
    monkeypatch.setattr(serve, "KEEPALIVE_POLL_SECS", 0.02)
    monkeypatch.setenv("RIGMA_KEEP_ALIVE_MIN", "0.0005")   # 30ms of idle
    _running()
    Engine.script = [_say("word " * 150)]
    Engine.stream_delay = 0.005
    c = _client(engine.port)
    sid = _seed(c, 2)
    app = build_app(upstream_port=engine.port)

    async def scenario():
        async with app.router.lifespan_context(app):
            await _stream(app, f"/api/sessions/{sid}/chat", {"message": "go"})
            during = list(unloads)
            # control: with nothing in flight the poller really does fire, so
            # `during` being empty is not just a loop that never ran
            #
            # 1000 iterations at 20ms is a ~20s ceiling, raised from 200 (~4s)
            # because 200 produced a FALSE RED once in five full-suite runs
            # (2026-09-22): under load the poller did not get scheduled inside 4s, and
            # the failure read as "the engine was unloaded under a running turn" —
            # the exact defect this test guards — when nothing had been unloaded at
            # all. A timing-sensitive control that can fail for load reasons makes a
            # real regression indistinguishable from noise, which is worse than a
            # slower test. The loop still exits the moment the poller fires, so the
            # ceiling costs nothing when the machine is idle.
            for _ in range(1000):
                if unloads:
                    break
                await asyncio.sleep(0.02)
            return during, list(unloads)

    during, after = asyncio.run(scenario())
    assert during == [], "the engine was unloaded under a running turn"
    assert after, ("the keepalive poller never ran, so this test proved nothing. "
                   "That is a BROKEN TEST, not a broken keepalive — if this fires, "
                   "check the poller is still started in the lifespan before "
                   "suspecting the unload guard.")


def test_a_failed_idle_unload_is_logged_and_leaves_the_engine_loaded(
        home, engine, monkeypatch, caplog):
    """A9: the poller's unload sat under `except Exception: pass`, so a failed
    unload left the card occupied and nothing said why. The failure must be
    logged with its reason, and the idle state left exactly as it was —
    `perform_unload` records `unloaded=True` only on success, so a raise must
    neither be reported as an unload nor clear the loaded state."""
    import asyncio

    from rigma import server_ops
    from rigma import state as st

    def _boom():
        raise RuntimeError("unload exploded: the card is still held")

    monkeypatch.setattr(server_ops, "perform_unload", _boom)
    monkeypatch.setattr(serve, "KEEPALIVE_POLL_SECS", 0.02)
    monkeypatch.setenv("RIGMA_KEEP_ALIVE_MIN", "0.0005")   # 30ms of idle
    _running()
    c = _client(engine.port)
    sid = _seed(c, 2)
    app = build_app(upstream_port=engine.port)

    async def scenario():
        async with app.router.lifespan_context(app):
            await _stream(app, f"/api/sessions/{sid}/chat", {"message": "go"})
            # the poller fires once the turn is over and the 30ms clock is out
            for _ in range(1000):
                if any("idle auto-unload failed" in r.getMessage()
                       for r in caplog.records):
                    break
                await asyncio.sleep(0.02)

    with caplog.at_level(logging.WARNING):
        asyncio.run(scenario())

    reasons = [r.getMessage() for r in caplog.records
               if "idle auto-unload failed" in r.getMessage()]
    assert reasons, ("a failed idle unload was not logged, so the card stayed "
                     "occupied with nothing to explain why")
    assert "unload exploded" in reasons[0], reasons
    assert not st.read_state().get("unloaded"), (
        "a failed unload was recorded as an unload")


# --------------------------------------------------------------------------
# F15 — steering a run was confirmed to the user and then discarded


def test_steering_survives_the_loop_writing_its_snapshot_back(home):
    """The loop loads the run, awaits a multi-second engine call, then writes
    that pre-await dict back. inject_run runs on the same loop during the
    await: it loads fresh, appends, saves, and answers "queued"."""
    from rigma import runs as _runs
    run = _runs.create("mission", "sid-1", workspace=str(home))
    snapshot = _runs.load(run["id"])          # what the loop is holding
    # ...meanwhile, on the same event loop
    live = _runs.load(run["id"])
    live.setdefault("steer_queue", []).append("stop going in circles")
    live["paused"] = True
    _runs.save(live)
    # ...and the loop finishes its turn and saves what it was holding
    snapshot.setdefault("steer_queue", []).append("ADVISOR: try the other path")
    serve._save_run_merged(snapshot)
    got = _runs.load(run["id"])
    assert got["steer_queue"] == ["stop going in circles",
                                  "ADVISOR: try the other path"]
    assert got["paused"] is True


def test_a_consumed_steer_entry_is_not_resurrected(home):
    """_driving_message pops the queue and saves immediately, so the merge must
    not read the popped entry back out of its own stale copy."""
    from rigma import runs as _runs
    run = _runs.create("mission", "sid-2", workspace=str(home))
    run["steer_queue"] = ["do the thing"]
    _runs.save(run)
    snapshot = _runs.load(run["id"])
    snapshot["steer_queue"].pop(0)            # what _driving_message does
    _runs.save(snapshot)
    serve._save_run_merged(snapshot)
    assert _runs.load(run["id"])["steer_queue"] == []


# --------------------------------------------------------------------------
# F32 — the delegate helper could run any tool, invisibly


def test_the_delegate_helper_cannot_run_a_tool_it_was_never_offered(
        home, engine, monkeypatch):
    """The allowlist gated what the helper is advertised and the rescue path,
    but not the normal tool_calls path — and run_tool resolves aliases against
    the whole registry, so `bash` reached run_shell inside what the comment
    calls a read-only firewall."""
    from rigma import tools as toolkit
    ran = []

    def _cached_run(name, args, ctx):
        ran.append((name, ctx.get("allow_code"), ctx.get("run_id")))
        return "tool output"

    monkeypatch.setattr(toolkit, "cached_run", _cached_run)
    _running()
    Engine.script = [_call("delegate", {"question": "what is here?"}),
                     _say("the answer")]
    Engine.aux = [{"content": "", "tool_calls": [
        {"id": "d1", "type": "function",
         "function": {"name": "bash",
                      "arguments": json.dumps({"cmd": "rm -rf /"})}}]},
        {"content": "nothing to report"}]
    c = _client(engine.port)
    sid = c.post("/api/sessions", json={}).json()["id"]
    c.post(f"/api/sessions/{sid}",
           json={"use_tools": True, "allow_code": True,
                 "workspace": str(home)})
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "look around"})
    assert r.status_code == 200
    assert not [n for n, _a, _r in ran if n in ("bash", "run_shell")], ran
    # ...and the blocked call is on the record instead of vanishing
    msg = [m for m in sessions.load(sid)["messages"]
           if m.get("delegate_trace")]
    assert msg and msg[0]["delegate_trace"][0]["name"] == "run_shell"
    assert msg[0]["delegate_trace"][0]["blocked"] is True


def test_an_allowed_delegate_tool_runs_with_code_execution_off(
        home, engine, monkeypatch):
    from rigma import tools as toolkit
    ran = []

    def _cached_run(name, args, ctx):
        ran.append((name, ctx.get("allow_code"), ctx.get("run_id")))
        return "file contents"

    monkeypatch.setattr(toolkit, "cached_run", _cached_run)
    _running()
    Engine.script = [_call("delegate", {"question": "read it"}),
                     _say("the answer")]
    Engine.aux = [{"content": "", "tool_calls": [
        {"id": "d1", "type": "function",
         "function": {"name": "read_file",
                      "arguments": json.dumps({"path": "a.txt"})}}]},
        {"content": "it says hello"}]
    c = _client(engine.port)
    sid = c.post("/api/sessions", json={}).json()["id"]
    c.post(f"/api/sessions/{sid}",
           json={"use_tools": True, "allow_code": True,
                 "workspace": str(home)})
    sess = sessions.load(sid)
    sess["run_id"] = "r9"
    sessions.save(sess)
    c.post(f"/api/sessions/{sid}/chat", json={"message": "look"})
    assert ("read_file", False, "") in ran, ran


# --------------------------------------------------------------------------
# F47 — auto-compaction could stop working and say nothing


def test_auto_compact_failure_is_logged_and_announced(home, engine, caplog):
    """A wedged aux slot could fail every fold for N turns while every turn
    still ended cleanly; the first visible symptom was the engine refusing a
    request for exceeding the context, which sends the user to the fit advisor
    to be told their context is too small."""
    _running(ctx=1000)
    Engine.script = [_say("ok", usage={"prompt_tokens": 950},
                          timings={"predicted_per_second": 40})]
    Engine.aux_status = 500
    c = _client(engine.port)
    sid = _seed(c, 24)
    with caplog.at_level("ERROR", logger="rigma.serve"):
        r = c.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "[DONE]" in r.text     # never fatal
    assert "event: notice" in r.text, r.text[-400:]
    assert any("auto-compact failed" in rec.message for rec in caplog.records)
    last = sessions.load(sid)["messages"][-1]
    assert last["content"] == "ok"                        # the turn is intact
    assert "Auto-compaction failed" in last["notice"]     # and it says so


# --------------------------------------------------------------------------
# F53 — the end-of-turn write put a mid-turn SETTINGS change back


def test_every_field_the_patch_endpoint_accepts_survives_a_turn():
    """The merge list is DERIVED from the PATCH surface, so a field added there
    cannot be forgotten here.

    It used to be hand-written, and it had drifted: `update_session` accepts
    every name in `sessions.MUTABLE_FIELDS`, while the merge copied twelve
    names of its own choosing. The eight it missed were silently reverted at
    the end of any turn they were changed during — including `allow_code`, a
    safety control, and `max_tool_rounds`.
    """
    covered = set(serve._MERGE_FROM_STORE) | {"messages", "prefill"}
    missing = set(sessions.MUTABLE_FIELDS) - covered
    assert not missing, (
        "these fields can be changed through POST /api/sessions/{sid} but the "
        "end-of-turn merge does not take them from the stored row, so a "
        f"mid-turn change to them is reverted: {sorted(missing)}")


def test_a_setting_changed_mid_turn_is_not_reverted(home, engine, monkeypatch):
    """A settings change made while a reply is STREAMING is the user's, and the
    turn's own end-of-turn write must not put it back.

    `allow_code` is the one that matters: a user switching code access OFF
    mid-reply had it switched back on when the reply finished (audit F53). The
    write lands in the window between the turn's snapshot and its end-of-turn
    merge, which is the only window where it can be lost — the titler runs
    AFTER the merge, so hooking that instead would prove nothing.

    `harness` is in the same body because it joined the PATCH surface after
    this test was written: switching the backend mid-reply and having the turn
    hand it back would be the same bug, on the field that decides which agent
    owns the next turn.
    """
    import asyncio

    from rigma import harness_dsh
    monkeypatch.setattr(harness_dsh, "available", lambda: True)
    _running(ctx=131072)
    Engine.script = [_say("a reply that takes a moment to arrive")]
    Engine.stream_delay = 0.01
    c = _client(engine.port)
    sid = _seed(c, 6)
    app = build_app(upstream_port=engine.port)

    async def scenario():
        turn = asyncio.create_task(
            _stream(app, f"/api/sessions/{sid}/chat", {"message": "go"}))
        await asyncio.sleep(0.2)              # the reply is streaming now
        edited = await _request(app, "POST", f"/api/sessions/{sid}",
                                {"allow_code": False, "max_tool_rounds": 7,
                                 "workspace": "C:/somewhere",
                                 "harness": "dsh"})
        await turn
        return edited

    edited = asyncio.run(scenario())
    assert edited["status"] == 200, edited

    after = sessions.load(sid)
    assert after["allow_code"] is False, \
        "a safety setting changed mid-turn was reverted by the turn's own write"
    assert after["max_tool_rounds"] == 7
    assert after["workspace"] == "C:/somewhere"
    assert after["harness"] == "dsh", \
        "the backend chosen mid-turn was handed back by the turn's own write"
    assert after["messages"][-1]["role"] == "assistant"   # the turn still landed


# --------------------------------------------------------------------------
# the client is told which backend drove a turn


def test_a_turn_says_which_backend_drove_it(home, engine):
    """The UI badges a reply with the backend that produced it.

    Sent on EVERY turn, the built-in included: a marker that is only sometimes
    present is one the reader has to interpret, and "no badge" would otherwise
    be ambiguous between the built-in and an older server.

    Its own event name, NOT `meta`: `meta` is suppressed on a failed turn
    (a turn that produced nothing must not report a context size), while which
    backend ran is true whether or not it succeeded.
    """
    import asyncio
    _running()
    Engine.script = [_say("hi")]
    c = _client(engine.port)
    sid = _seed(c, 1)
    app = build_app(upstream_port=engine.port)

    got = asyncio.run(
        _stream(app, f"/api/sessions/{sid}/chat", {"message": "go"}))
    assert got["status"] == 200
    assert "event: harness" in got["text"]
    assert '"name": "native"' in got["text"]
    assert '"label": "Rigma (built in)"' in got["text"]


# --------------------------------------------------------------------------
# F36 — eager tool tasks outlived the consumer that started them


def test_abandon_tasks_cancels_pending_and_swallows_every_result():
    """`_abandon_tasks` is the mechanism; it must cancel what is pending and
    retrieve what is done, so nothing surfaces later as "Task exception was
    never retrieved" (AUDIT F36)."""
    import asyncio

    async def main():
        began = asyncio.Event()

        async def slow():
            began.set()
            await asyncio.sleep(30)

        async def boom():
            raise ValueError("tool blew up")

        pending = asyncio.create_task(slow())
        done_ok = asyncio.create_task(asyncio.sleep(0))
        failed = asyncio.create_task(boom())
        await began.wait()
        await asyncio.sleep(0)                 # let boom() fail
        assert failed.done()
        serve._abandon_tasks([pending, done_ok, failed])   # must not raise
        await asyncio.sleep(0)
        assert pending.cancelled()
        assert done_ok.done() and failed.done()

    asyncio.run(main())


def test_a_disconnect_cancels_the_rounds_eager_tool_tasks(home, engine,
                                                          monkeypatch):
    """GeneratorExit / CancelledError are BaseException subclasses, so the
    `if failed:` branch that cancelled eager tool tasks never saw a tab refresh
    or a sleeping laptop — leaving an orphaned `delegate` sub-loop posting to
    the engine's aux slot for up to 5 hops at 240s each (AUDIT F36).

    The disconnect is armed from INSIDE the delegate's aux request, so it lands
    while that eagerly started task is still pending rather than at some
    chunk-count guess."""
    import asyncio
    import threading
    seen: list = []
    real = serve._abandon_tasks

    def spy(tasks):
        seen.append(list(tasks))
        return real(tasks)

    monkeypatch.setattr(serve, "_abandon_tasks", spy)
    _running()
    Engine.script = [_call("delegate", {"question": "what is here?"})]
    Engine.aux_delay = 5.0
    armed = threading.Event()
    Engine.on_aux = armed.set          # runs INSIDE the delegate's aux call
    c = _client(engine.port)
    sid = _seed(c, 2)
    app = build_app(upstream_port=engine.port)
    asyncio.run(_stream_until_armed(app, f"/api/sessions/{sid}/chat",
                                    {"message": "go"}, armed))
    assert armed.is_set(), "the delegate never reached the engine"
    assert any(tasks for tasks in seen), (
        "the eager delegate task was abandoned without being cancelled")


# --------------------------------------------------------------------------
# 02-4 — an external harness turn leaked its pump thread and subprocess


def test_a_disconnect_stops_an_external_agent_and_keeps_the_partial_reply(
        home, engine, monkeypatch):
    """AUDIT 02-4: `_external_turn` had no try/finally.

    A client disconnect throws GeneratorExit / CancelledError at its current
    `yield`, which unwound straight past `await task` and the save. Nothing set
    the adapter's `cancel`, so the harness subprocess — and every subagent it
    owns — kept running and billing against Rigma's /v1 while its events piled
    into a queue with no consumer. The native loop already had this teardown
    (AUDIT F7); the external path did not.
    """
    import asyncio

    from rigma import harness_dsh

    monkeypatch.setenv("RIGMA_HOME", str(home))
    monkeypatch.setattr(harness_dsh, "available", lambda: True)
    cancels: list = []
    stopped = threading.Event()

    def _drive(**kw):
        cancels.append(kw.get("cancel"))
        yield harness_dsh.TurnEvent("text", text="half ")
        cancel = kw.get("cancel")
        deadline = time.time() + 5.0        # never hangs the suite, cancel or not
        while (cancel is not None and not cancel.is_set()
               and time.time() < deadline):
            time.sleep(0.01)
        stopped.set()

    monkeypatch.setattr(harness_dsh, "drive_turn", _drive)
    _running()
    c = _client(engine.port)
    sid = c.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["harness"] = "dsh"
    s["messages"] = [{"role": "user", "content": "go"}]
    sessions.save(s)
    app = build_app(upstream_port=engine.port)

    async def scenario():
        # chunk 1 is the harness badge, 2 the external notice, 3 the first
        # delta — so the disconnect lands with the agent mid-turn
        got = await _stream(app, f"/api/sessions/{sid}/chat",
                            {"message": "go"}, stop_after=3)
        for _ in range(200):                # let the pump thread see the cancel
            if stopped.is_set():
                break
            await asyncio.sleep(0.01)
        return got

    got = asyncio.run(scenario())

    assert got["status"] == 200, got
    assert cancels and cancels[0] is not None, "no cancel event was handed over"
    assert cancels[0].is_set(), "the client disconnect never stopped the agent"
    stored = sessions.load(sid)
    assert stored["messages"][-1]["role"] == "assistant", stored["messages"]
    assert "half " in stored["messages"][-1]["content"], (
        "the partial reply was thrown away with the disconnect")


# --------------------------------------------------------------------------
# R3-CHAT-3 — a chat already over the window could never compact itself


def test_a_chat_over_the_window_compacts_before_sending(home, engine):
    """Auto-compaction fires at the END of a turn from the engine's real
    `prompt_tokens`. That cannot rescue a chat that is ALREADY over the window:
    the turn that would have triggered it never completes, so `prompt_tokens` is
    never reported, so it never runs — and the chat 400s on every message from
    then on with no way out that the UI can offer.
    """
    _running(ctx=4096)
    Engine.script = [_say("recovered"), _say("x")]
    c = _client(engine.port)
    sid = c.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    # ~20 messages of 4K chars each: ~40K estimated tokens against a 4K window
    s["messages"] = [{"role": "user" if i % 2 == 0 else "assistant",
                      "content": f"m{i} " + "x" * 4000} for i in range(20)]
    sessions.save(s)

    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200, r.text
    assert "compacted before sending" in r.text, r.text[:400]
    # the summariser call is non-streaming and lands BEFORE the first streamed one
    kinds = ["aux" if not b.get("stream") else "turn" for b in Engine.seen]
    assert kinds and kinds[0] == "aux", kinds
    assert "turn" in kinds, kinds


def test_a_short_chat_is_not_compacted_before_sending(home, engine):
    """The pre-send check is an estimate, so it must not fire on an ordinary
    chat — a false positive costs a summary the user did not ask for."""
    _running(ctx=131072)
    Engine.script = [_say("fine"), _say("x")]
    c = _client(engine.port)
    sid = _seed(c, 4)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200, r.text
    assert "compacted before sending" not in r.text, r.text[:400]
    kinds = ["aux" if not b.get("stream") else "turn" for b in Engine.seen]
    assert kinds[0] == "turn", kinds


# --------------------------------------------------------------------------
# R3-CHAT-2 — nothing watched for a model repeating itself


def test_a_repeated_identical_call_stops_the_turn(home, engine):
    """The round cap is a runaway BACKSTOP (1000), not a detector.

    Nothing watched for a loop, so a model that kept issuing the same call with
    the same arguments ran to the ceiling — a thousand round trips, each one
    re-prefilling the whole transcript, which on a local card is minutes of GPU
    time and a context full of identical tool results. The chat looks busy the
    whole time.
    """
    _running()
    # the SAME call every round: the fake engine repeats its last script entry
    Engine.script = [_call("sample_files", {"path": "."})]
    c = _client(engine.port)
    sid = _seed(c, 2)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200, r.text
    rounds = len([b for b in Engine.seen if b.get("stream")])
    assert rounds <= serve._REPEAT_CALL_LIMIT + 1, (
        f"the turn ran {rounds} rounds on one repeated call")
    assert "same" in r.text.lower() or "repeat" in r.text.lower(), r.text[:400]


def test_two_different_calls_alternating_also_stop_the_turn(home, engine):
    """Two DIFFERENT calls alternating is the same defect wearing a hat — the
    signature is the SET of (name, args) in the round, not one call."""
    _running()
    Engine.script = [_call("sample_files", {"path": "."}),
                     _call("list_files", {"path": "."})]
    c = _client(engine.port)
    sid = _seed(c, 2)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200, r.text
    rounds = len([b for b in Engine.seen if b.get("stream")])
    assert rounds <= 2 * (serve._REPEAT_CALL_LIMIT + 1), (
        f"an alternating pair ran {rounds} rounds")


def test_a_turn_that_makes_progress_is_never_cut_off(home, engine):
    """The breaker must not touch a long turn that keeps doing something new —
    the counter resets on any round whose call set differs.

    Eight rounds, which is past the repeat limit: if the counter did not reset,
    this turn would be stopped with the loop notice instead of finishing. The
    paths differ per round so each call genuinely differs.
    """
    _running()
    # The fake engine repeats its LAST script entry once one remains, so the
    # final `_say` needs a sentinel after it to actually be reached. The paths
    # differ per round, so every round is genuinely a different call and the
    # breaker's counter has to reset each time.
    Engine.script = [_call("sample_files", {"path": "./" + "a" * (i + 1)})
                     for i in range(8)] + [_say("all eight are done"), _say("x")]
    c = _client(engine.port)
    sid = _seed(c, 2)
    r = c.post(f"/api/sessions/{sid}/chat", json={"message": "go"})
    assert r.status_code == 200, r.text
    rounds = len([b for b in Engine.seen if b.get("stream")])
    assert rounds == 9, f"expected 8 tool rounds + the final one, saw {rounds}"
    # The reply itself, not the stored row: this chat has a tool-result summary
    # as its last stored entry, and the point here is that the turn was allowed
    # to finish with its answer rather than stopped by the breaker. `_say` emits
    # ONE SSE EVENT PER CHARACTER, so the reply has to be reassembled from the
    # deltas before it can be searched for a phrase.
    reply = "".join(re.findall(r'data: \{"delta": "(.*?)"\}', r.text))
    assert "all eight are done" in reply, reply[:200]
    assert "same tool call" not in r.text, "the breaker fired on a progressing turn"
