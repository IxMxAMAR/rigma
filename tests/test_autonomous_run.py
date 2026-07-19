"""Autonomous Mode end-to-end: the background executor drives a run to a
terminal state against a scripted fake engine. Uses the Starlette TestClient,
whose blocking portal keeps the background asyncio task running between polls."""
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import runs
from rigma import serve
from rigma import state as st


class _Engine(BaseHTTPRequestHandler):
    """Scripted engine. Each /v1/chat/completions call emits the next tool_call
    from `script` (a list of (name, args) or None=narrate/hang)."""
    script = []
    idx = 0
    hang = False
    delay = 0.0

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        i = _Engine.idx
        _Engine.idx += 1
        step = _Engine.script[i] if i < len(_Engine.script) else _Engine.script[-1]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        if _Engine.delay:
            time.sleep(_Engine.delay)            # slow-but-working per turn
        if _Engine.hang:
            time.sleep(1.0)                      # exceed a small IDLE_SECS
        if step is None:                          # narrate, no tools -> "lazy"
            sse({"choices": [{"delta": {"content": "thinking about it"}}]})
        else:
            name, args = step
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"c{i}", "type": "function",
                 "function": {"name": name, "arguments": json.dumps(args)}}]}}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = HTTPServer(("127.0.0.1", 0), _Engine)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _Engine.idx = 0
    _Engine.hang = False
    _Engine.delay = 0.0
    _Engine.script = []


def _client(port):
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    return TestClient(serve.build_app(upstream_port=port))


def _wait(client, rid, timeout=15):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    return client.get(f"/api/runs/{rid}").json()


def test_run_reaches_done_with_verify_once(engine):
    # plan, plan, log, task_complete (rejected -> verify), task_complete (done)
    _Engine.script = [
        ("manage_plan", {"action": "add", "task": "read the folder"}),
        ("manage_plan", {"action": "add", "task": "write the prompts"}),
        ("log_progress", {"done": "read folder", "next": "write prompts"}),
        ("task_complete", {"summary": "all prompts written"}),
        ("task_complete", {"summary": "all prompts written"}),
    ]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "make 100 prompts",
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "done"
    assert r["verified_once"] is True            # first task_complete was gated
    assert "prompts" in r["summary"]
    assert any(t["status"] == "pending" or t["status"] == "done"
               for t in r["plan"])               # the plan was built


def test_run_stalls_on_pure_narration(engine):
    _Engine.script = [None]                       # never uses a tool
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "stalled"
    assert "progress" in r["halt_reason"] or "idle" in r["halt_reason"]


def test_run_freezes_when_engine_hangs(engine, monkeypatch):
    monkeypatch.setattr(serve, "IDLE_SECS", 0.2)
    monkeypatch.setattr(serve, "M_FROZEN", 2)
    _Engine.hang = True
    _Engine.script = [("manage_plan", {"action": "add", "task": "x"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "frozen"


def test_second_run_refused_while_active(engine):
    # deterministic: an active run exists (no live loop) -> start is refused
    from rigma import sessions
    c = _client(engine)
    sess = sessions.create()
    runs.create("existing mission", sess["id"])   # writes active.json
    r2 = c.post("/api/runs", json={"mission": "b", "budget_hours": 1})
    assert r2.status_code == 409 and "active" in r2.json()["error"]


def test_start_requires_a_model(engine, tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "empty"))
    c = TestClient(serve.build_app(upstream_port=engine))   # no write_state
    r = c.post("/api/runs", json={"mission": "x"})
    assert r.status_code == 409 and "model" in r.json()["error"]


def test_run_control_endpoints(engine):
    from rigma import sessions
    c = _client(engine)
    sess = sessions.create()
    rid = runs.create("m", sess["id"])["id"]        # active run, no live loop
    assert c.post(f"/api/runs/{rid}/inject",
                  json={"message": "use bs4"}).json()["queued"] is True
    assert runs.load(rid)["steer_queue"] == ["use bs4"]
    c.post(f"/api/runs/{rid}/pause")
    assert runs.load(rid)["paused"] is True
    c.post(f"/api/runs/{rid}/resume")
    assert runs.load(rid)["paused"] is False
    runs.append_progress(rid, "did x", "do y")
    runs.plan_add(rid, "step one")
    g = c.get(f"/api/runs/{rid}").json()
    assert g["plan"][0]["text"] == "step one" and "did x" in g["log_tail"]
    assert "did x" in c.get(f"/api/runs/{rid}/log").json()["log"]
    c.post(f"/api/runs/{rid}/stop")
    assert runs.load(rid)["status"] == "stopped"
    assert c.get("/api/runs/active").json() == {}    # active released


def test_run_session_uses_agent_system_prompt(engine):
    from rigma import sessions
    c = _client(engine)
    _Engine.script = [None]                       # stalls fast; we check setup
    rid = c.post("/api/runs", json={"mission": "do stuff",
                                    "budget_hours": 1}).json()["id"]
    _wait(c, rid)                                  # let it terminate
    s = sessions.load(runs.load(rid)["session_id"])
    assert "AUTONOMOUS AGENT" in s["system_prompt"]   # not the chat default
    assert "call at least one tool" in s["system_prompt"].lower()
    assert s["effort"] == "off"                        # act, don't <think> in circles
    assert runs.load(rid)["mission"] == "do stuff"    # mission kept on the run
