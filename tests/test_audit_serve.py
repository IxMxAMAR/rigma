"""The HTTP/turn-layer findings of the 2026-09-04 audit.

Every test here reproduces a specific way the server lost work, blocked the
event loop, killed a healthy run, or answered a request it should have
refused. The fake engine is scripted per test: `Engine.script` decides what the
next streamed turn emits, and `Engine.aux` what a non-streaming call (the
summarizer, the titler, the delegate helper) answers with.
"""
import json
import os
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
        def _f(_msgs):
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
            for _ in range(200):
                if unloads:
                    break
                await asyncio.sleep(0.02)
            return during, list(unloads)

    during, after = asyncio.run(scenario())
    assert during == [], "the engine was unloaded under a running turn"
    assert after, "the keepalive poller never ran; the test proves nothing"


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
