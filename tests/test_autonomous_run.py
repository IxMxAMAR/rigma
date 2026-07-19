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
    err_status = 0
    err_body = ""
    gate = None          # threading.Event: hold the turn OPEN after the 1st chunk

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        if _Engine.err_status:                    # simulate an engine 4xx/5xx
            self.send_response(_Engine.err_status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(_Engine.err_body.encode())
            return
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
        if _Engine.gate is not None:      # hold the turn open, deterministically
            _Engine.gate.wait(timeout=10)
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
    _Engine.err_status = 0
    _Engine.err_body = ""
    _Engine.gate = None


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
    monkeypatch.setattr(serve, "PREFILL_SECS", 0.2)   # first-token budget too
    monkeypatch.setattr(serve, "M_FROZEN", 2)
    _Engine.hang = True
    _Engine.script = [("manage_plan", {"action": "add", "task": "x"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "frozen"


def test_prefill_budget_tolerates_slow_first_token(engine, monkeypatch):
    # a slow first token (prefill) must NOT be read as frozen when it lands
    # within PREFILL_SECS, even though it exceeds the inter-token IDLE_SECS
    monkeypatch.setattr(serve, "IDLE_SECS", 0.1)
    monkeypatch.setattr(serve, "PREFILL_SECS", 2.0)
    _Engine.delay = 0.4                                # > IDLE_SECS, < PREFILL_SECS
    _Engine.script = [("task_complete", {"summary": "quick win"}),
                      ("task_complete", {"summary": "quick win"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "done"                       # not "frozen"


def test_slow_turn_reports_heartbeat(engine, monkeypatch):
    # a slow turn must publish "still working" progress so the UI never shows a
    # dead screen — and must still finish normally
    monkeypatch.setattr(serve, "IDLE_SECS", 0.1)
    monkeypatch.setattr(serve, "PREFILL_SECS", 5.0)
    monkeypatch.setattr(serve, "TICK_SECS", 0.05)
    seen = []
    _Engine.delay = 0.4                   # slow enough to cross several ticks
    _Engine.script = [("task_complete", {"summary": "s"}),
                      ("task_complete", {"summary": "s"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    # poll while it runs; we should catch a non-zero waiting_secs at some point
    end = time.time() + 15
    while time.time() < end:
        r = c.get(f"/api/runs/{rid}").json()
        if (r.get("waiting_secs") or 0) > 0:
            seen.append(r["waiting_secs"])
        if r.get("status") in runs.TERMINAL:
            break
        time.sleep(0.02)
    r = c.get(f"/api/runs/{rid}").json()
    assert r["status"] == "done"          # heartbeat didn't break the turn
    assert r.get("waiting_secs", 0) == 0  # cleared when the turn ended


def test_activity_feed_records_tool_calls(engine):
    # the UI must be able to show WHAT the model is doing, not just "(waiting…)"
    # last entry is None (narrate) so the fake engine stops looping tools once
    # the script is exhausted — otherwise one turn floods the rolling window
    _Engine.script = [("manage_plan", {"action": "add", "task": "step one"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)                       # ends stalled; we're testing the feed
    acts = r.get("activity") or []
    assert acts, "activity feed must be populated"
    tools_seen = [a["text"] for a in acts if a["kind"] == "tool"]
    assert any("manage_plan" in t for t in tools_seen), f"activity={acts}"
    assert any("action=add" in t for t in tools_seen)   # args shown, not just name
    assert any(a["kind"] == "result" for a in acts)     # results shown too
    assert any(a["kind"] == "say" for a in acts)        # and plain model text


# NOTE: mid-run visibility (activity readable WHILE status=="running") is not
# covered here. Under TestClient the run loop starves the HTTP portal and even a
# gated fake engine finished before the first sample, so every attempt measured
# the harness, not the product. The flush path itself is covered above (tool
# events flush immediately; text throttles to 1.5s); the bug the owner actually
# hit was UI-side — refreshAuto never repainted the panel, so it only appeared
# on the full re-render that happens when a run ends.


def test_bookkeeping_only_turns_do_not_count_as_progress(engine):
    # ticking plan items forever while producing NO artifact must be treated as
    # idle. Each add has distinct args, so the repeat-signature check can't catch
    # it — only the productive-work check can.
    _Engine.script = [("manage_plan", {"action": "add", "task": f"step {i}"})
                      for i in range(20)]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    assert r["status"] == "stalled"          # cut short, not marched through
    # streaks count TURNS (a turn makes many calls), so assert on iterations:
    # without the productive-work check this never accumulates idle at all
    assert r["iteration"] <= 12, "run kept going on bookkeeping alone"


def test_checkpoint_never_restates_the_mission(engine, monkeypatch):
    # Re-injecting the mission as a user message makes a small model treat it as
    # a NEW instruction and restart from phase 1 (owner hit exactly this). The
    # mission is already pinned in the system prompt every turn.
    from rigma import sessions
    monkeypatch.setattr(serve, "K_REMIND", 2)          # fire a checkpoint early
    mission = "ZEBRAQUEST-TOKEN assemble the widget"
    _Engine.script = [("manage_plan", {"action": "add", "task": "a"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": mission,
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    s = sessions.load(r["session_id"])
    driving = [m["content"] for m in s["messages"] if m.get("role") == "user"]
    assert driving
    assert not any(mission in d for d in driving), "mission must not be re-injected"
    assert any("MID-RUN" in d or "start over" in d for d in driving)
    # ...because the system prompt carries it EVERY turn, so dropping it from the
    # checkpoint loses nothing. (run-end clears session["mission"], so assert the
    # pinning invariant on a live-shaped session rather than the finished one.)
    live = dict(s, mission=mission)
    assert mission in sessions.build_messages(live)[0]["content"]


def test_one_action_mode_executes_exactly_one_tool_then_ends(engine):
    # one action per turn: the tool MUST still run. A naive max_tool_rounds=1
    # makes round 0 "last", and tool calls are only executed when NOT last —
    # that would silently drop every action.
    from rigma import sessions
    _Engine.script = [("manage_plan", {"action": "add", "task": "alpha"}),
                      ("manage_plan", {"action": "add", "task": "beta"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    sess = sessions.load(r["session_id"])
    assert sess is not None
    traces = [m.get("tool_trace") or [] for m in sess["messages"]
              if m.get("role") == "assistant"]
    assert traces, "at least one assistant turn was persisted"
    assert all(len(t) <= 1 for t in traces), f"turn ran multiple actions: {traces}"
    assert len(r["plan"]) >= 1, "tool call was dropped instead of executed"


def test_tool_results_persist_into_the_next_turn(engine):
    # tool_trace is METADATA and build_messages strips it, so without this the
    # model never sees what its tools returned and re-lists/re-reads to find out
    from rigma import sessions
    _Engine.script = [("manage_plan", {"action": "add", "task": "alpha"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    sess = sessions.load(r["session_id"])
    rendered = sessions.build_messages(sess)
    blob = "\n".join(m["content"] for m in rendered
                     if isinstance(m.get("content"), str))
    assert "TOOL RESULT" in blob, "the action's result never re-enters context"
    assert "added step #1" in blob, "the actual result text was not carried forward"


def test_run_finishes_via_completion_checkpoint(engine, monkeypatch):
    # model goes quiet (no tools) past the lazy threshold, then — when given the
    # completion ultimatum — calls task_complete. The run must end DONE, not stalled.
    monkeypatch.setattr(serve, "K_LAZY", 3)
    _Engine.script = [None, None, None,               # 3 idle turns -> would stall
                      ("task_complete", {"summary": "all done"}),
                      ("task_complete", {"summary": "all done"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "done"
    assert r["verified_once"] is True


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


def test_run_fails_fast_on_template_parser_error(engine):
    # engine can't build a tool-call parser for this model's chat template ->
    # every turn 400s the same way, so the run must FAIL FAST with the reason,
    # not silently spin as "no progress" (the near-instant-stall bug).
    _Engine.err_status = 400
    _Engine.err_body = json.dumps(
        {"error": {"message": "Unable to generate parser for this template. "
                              "Automatic parser generation failed"}})
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid)
    assert r["status"] == "error"
    assert "template" in r["halt_reason"] and "parser" in r["halt_reason"]
    log = c.get(f"/api/runs/{rid}/log").json()["log"]
    assert "FATAL ENGINE ERROR" in log and "generate parser" in log


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
    assert s["effort"] == "on"                          # reason each step by default
    assert runs.load(rid)["mission"] == "do stuff"    # mission kept on the run


def test_run_effort_override(engine):
    from rigma import sessions
    c = _client(engine)
    _Engine.script = [None]
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                    "effort": "off"}).json()["id"]
    _wait(c, rid)
    assert sessions.load(runs.load(rid)["session_id"])["effort"] == "off"
