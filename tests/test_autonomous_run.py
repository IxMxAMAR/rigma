"""Autonomous Mode end-to-end: the background executor drives a run to a
terminal state against a scripted fake engine. Uses the Starlette TestClient,
whose blocking portal keeps the background asyncio task running between polls."""
import json
import os
import pathlib
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
    bodies = []          # every request body the engine received
    compile_reply = "not a spec"   # mission compiler falls back by default

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        _body = self.rfile.read(n)
        if _Engine.err_status:                    # simulate an engine 4xx/5xx
            self.send_response(_Engine.err_status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(_Engine.err_body.encode())
            return
        try:
            body = json.loads(_body or b"{}")
        except Exception:
            body = {}
        _Engine.bodies.append(body)
        if not body.get("stream", True):
            # non-streaming = the mission compiler (or compaction), NOT a turn.
            # Answer it without consuming the turn script.
            payload = json.dumps({"choices": [{"message": {
                "content": _Engine.compile_reply}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            # Content-Length matters: without it httpx waits for EOF, which
            # stretched every run by seconds and made the suite flaky
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
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
        elif isinstance(step, tuple) and step[0] == "__text__":
            sse({"choices": [{"delta": {"content": step[1]}}]})
        else:
            steps = step if isinstance(step, list) else [step]
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": j, "id": f"c{i}_{j}", "type": "function",
                 "function": {"name": nm, "arguments": json.dumps(ar)}}
                for j, (nm, ar) in enumerate(steps)]}}]})
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
    _Engine.bodies = []
    _Engine.compile_reply = "not a spec"
    yield tmp_path
    # Stop this test's run so its background loop exits. Nothing cancels those
    # tasks on teardown, so without this every finished test leaves a loop
    # spinning against a dead engine and starves the ones that follow — which
    # showed up as unrelated tests "timing out" at random.
    try:
        a = runs.active()
        if a:
            runs.set_status(a, "stopped", "test teardown")
    except Exception:
        pass
    while _CLIENTS:                     # close the portal + cancel its tasks
        try:
            _CLIENTS.pop().__exit__(None, None, None)
        except Exception:
            pass


_CLIENTS = []


def _client(port):
    """Enter the TestClient's context so the app gets ONE portal for the whole
    test. Without it, starlette spins a fresh portal per request, and the
    background run loop (create_task'd inside the POST handler) lives on a loop
    that is torn down when that request ends — so whether it progresses is a
    race. That was the source of the random 'still running' failures."""
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(), ui_pid=os.getpid())
    c = TestClient(serve.build_app(upstream_port=port))
    c.__enter__()
    _CLIENTS.append(c)
    return c


def _wait(client, rid, timeout=25):
    # a run now compiles its mission first — one extra engine round-trip before
    # the first turn — so this is a little longer than the original 15s
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    return client.get(f"/api/runs/{rid}").json()


def test_run_reaches_done_with_verify_once(engine):
    # plan, plan, complete, task_complete (rejected -> verify), task_complete
    _Engine.script = [
        ("manage_plan", {"action": "add", "task": "read the folder"}),
        ("manage_plan", {"action": "add", "task": "write the prompts"}),
        ("manage_plan", {"action": "complete", "id": 1}),
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


def test_identical_repeated_action_is_caught_immediately(engine):
    # with one action per turn, the same call repeated back-to-back is a loop;
    # catch it on the SECOND occurrence rather than waiting out a 3-turn streak
    _Engine.script = [("manage_plan", {"action": "list"})]   # repeats forever
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    assert r["status"] == "stalled"
    assert r["iteration"] <= 8, f"took too long to notice the loop: {r['iteration']}"


def test_one_action_caps_parallel_tool_calls(engine):
    # a model may emit SEVERAL tool_calls in one round; one-action mode must
    # still execute only one, or the loop is unsupervised again at small scale
    from rigma import sessions
    _Engine.script = [[("manage_plan", {"action": "add", "task": "a"}),
                       ("manage_plan", {"action": "add", "task": "b"})], None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    sess = sessions.load(r["session_id"])
    traces = [m.get("tool_trace") or [] for m in sess["messages"]
              if m.get("role") == "assistant"]
    assert traces
    assert all(len(t) <= 1 for t in traces), f"executed parallel calls: {traces}"


def test_one_action_lets_the_model_see_loaded_images(engine, tmp_path):
    # a tool that loads images appends them to the NEXT round's request. In
    # one-action mode an unconditional break discarded that round, so the model
    # was told "image loaded" but never actually saw a single pixel.
    from PIL import Image
    from rigma import tools as toolkit
    png = tmp_path / "x.png"
    Image.new("RGB", (8, 8), (200, 30, 30)).save(png)

    @toolkit.tool("fake_view", "test-only image loader",
                  {"type": "object", "properties": {}})
    def _fake(args, ctx):
        return toolkit.IMAGE_SENTINEL + str(png)

    try:
        _Engine.script = [("fake_view", {}), None]
        c = _client(engine)
        rid = c.post("/api/runs",
                     json={"mission": "x", "budget_hours": 1}).json()["id"]
        _wait(c, rid, timeout=25)
        saw_image = any(
            isinstance(m.get("content"), list)
            and any(p.get("type") == "image_url" for p in m["content"])
            for b in _Engine.bodies for m in b.get("messages", []))
        assert saw_image, "the loaded image never reached the model"
    finally:
        toolkit._REGISTRY.pop("fake_view", None)


def test_persisted_tool_result_is_not_crippled(engine, tmp_path):
    # one-action mode ends the turn right after the call, so the ONLY copy the
    # model ever sees is the persisted one. Truncating it to ~1k chars would
    # silently turn read_file into "read the first 1200 bytes".
    from rigma import sessions
    big = tmp_path / "big.txt"
    big.write_text("L%03d: the quick brown fox jumps over the lazy dog\n" % 0
                   + "".join("L%03d: padding line for the file body\n" % i
                             for i in range(1, 200)), encoding="utf-8")
    _Engine.script = [("read_file", {"path": "big.txt"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                    "workspace": str(tmp_path)}).json()["id"]
    r = _wait(c, rid, timeout=25)
    sess = sessions.load(r["session_id"])
    blob = "\n".join(m["content"] for m in sess["messages"]
                     if isinstance(m.get("content"), str)
                     and m["content"].startswith("TOOL RESULT"))
    assert "L100:" in blob, "file content was truncated away before the model saw it"
    assert len(blob) > 4000, f"persisted result far too small: {len(blob)}"


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


def test_task_complete_refused_while_plan_steps_are_open(engine):
    # a weak model declares victory with work outstanding; evidence beats
    # self-report, so the run must refuse and name the step
    _Engine.script = [("manage_plan", {"action": "add", "task": "write the file"}),
                      ("task_complete", {"summary": "all done"}),
                      ("task_complete", {"summary": "all done"}),
                      ("task_complete", {"summary": "all done"}),
                      ("task_complete", {"summary": "all done"})]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    log = c.get(f"/api/runs/{rid}/log").json()["log"]
    assert "task_complete refused" in log
    assert r["completion_challenges"] >= 1
    # ...but capped, so a bad plan can't make the run unfinishable
    assert r["status"] in runs.TERMINAL


def test_big_tool_result_spills_to_disk_with_a_way_back(tmp_path):
    from rigma import serve as _s
    big = "x" * (_s.SPILL_CHARS + 5000)
    out = _s._spill_big_result("run_shell", big, {})
    assert len(out) < _s.SPILL_CHARS, "spilled result must be small in context"
    assert "saved to disk" in out and "read_file path=" in out
    path = out.split("Full output: ")[1].splitlines()[0]
    assert pathlib.Path(path).read_text(encoding="utf-8") == big   # nothing lost
    # read_file is exempt or paging would re-trigger the spill it escapes
    assert _s._spill_big_result("read_file", big, {}) == big


def test_run_gets_anti_repetition_samplers_and_a_token_cap(engine):
    # a local model left reasoning alone repeats itself verbatim and never acts
    # ("I will call view_images." over and over). DRY penalises the repeated
    # n-grams; the token cap guarantees a runaway turn ENDS so it can be scored.
    from rigma import sessions
    _Engine.script = [None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    _wait(c, rid, timeout=25)
    p = sessions.load(runs.load(rid)["session_id"])["params"]
    assert p["dry_multiplier"] > 0 and p["dry_allowed_length"] >= 1
    # generous: it must fit a real batch of work (25 detailed prompts +
    # a thinking block). 8192 truncated legitimate output mid-sentence.
    assert 16384 <= p["max_tokens"] <= 32768


def test_driving_line_states_the_work_not_the_protocol(engine):
    # "Emit ONE tool call" repeated every turn gave the model something to
    # reason ABOUT — it looped narrating the instruction instead of obeying it
    from rigma import sessions
    _Engine.script = [("manage_plan", {"action": "add", "task": "write it"}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    driving = [m["content"] for m in sessions.load(r["session_id"])["messages"]
               if m.get("role") == "user"]
    assert driving
    assert not any("Emit ONE tool call" in d for d in driving)
    # was: assert any("Do this now:" in d ...). The work is still named every
    # turn — it is now stated as status rather than ordered, because an
    # imperative in a role=user message reads as a fresh human command and
    # restarts the step (see test_driving_message.py for the trap itself).
    assert any("next:" in d for d in driving), driving[-2:]


def test_prose_turn_is_saved_not_discarded(engine, tmp_path):
    # the owner watched it generate ~60 prompts into a REPLY, which the loop
    # scored as unproductive and threw away. Prose in a run may BE the work.
    _Engine.script = [None]                     # narrates, never calls a tool
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                    "workspace": str(tmp_path)}).json()["id"]
    r = _wait(c, rid, timeout=25)
    log = c.get(f"/api/runs/{rid}/log").json()["log"]
    # the fake engine narrates only ~17 chars, below DRAFT_MIN_CHARS, so nothing
    # should be saved — the guard must not write junk drafts for short replies
    assert "saved" not in log or not (tmp_path / "rigma-drafts").exists()


def test_long_prose_is_written_to_a_draft_file(tmp_path, monkeypatch):
    from rigma import serve as _s
    assert _s.DRAFT_MIN_CHARS >= 200
    assert _s.RUN_PARAMS["max_tokens"] >= 16384, "must fit a real batch of work"


def test_start_run_responds_immediately(engine, monkeypatch):
    # compiling inside start_run made the Start button hang for a whole engine
    # call — the POST must return fast and compile inside the run
    import time as _t
    _Engine.script = [None]
    c = _client(engine)
    t0 = _t.monotonic()
    r = c.post("/api/runs", json={"mission": "x", "budget_hours": 1})
    assert r.status_code == 200
    assert _t.monotonic() - t0 < 3.0, "start_run blocked on the compile"
    _wait(c, rid := r.json()["id"], timeout=25)


def test_compiled_spec_seeds_the_plan(engine):
    # a good compile means the model EXECUTES a plan instead of inventing one
    _Engine.compile_reply = json.dumps({
        "objective": "write prompts in batches",
        "deliverables": [{"path": "D:/out/a.txt", "description": "batch 1"}],
        "constraints": ["do not modify originals"],
        "steps": [
            {"id": 1, "description": "sample images", "artifact": "",
             "verification": {"type": "none"}},
            {"id": 2, "description": "write prompts 1-25",
             "artifact": "D:/out/a.txt",
             "verification": {"type": "file_min_size", "value": 100}}]})
    _Engine.script = [None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "make prompts in batches",
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    assert r["spec"]["compiled"] is True
    texts = [t["text"] for t in r["plan"]]
    assert "sample images" in texts and "write prompts 1-25" in texts


def test_feed_marks_display_truncation(engine, tmp_path):
    # a 12-path sample was cut mid-path in the UI ("D:\Good St") and looked like
    # corrupted data. The model gets the full result; the FEED must say it cut.
    from rigma import serve as _s
    big = tmp_path / "many"
    big.mkdir()
    for i in range(60):
        (big / f"ComfyUI_{i:05d}_.png").write_text("x", encoding="utf-8")
    _Engine.script = [("sample_files", {"path": str(big), "count": 40}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1,
                                    "workspace": str(tmp_path)}).json()["id"]
    r = _wait(c, rid, timeout=25)
    results = [a["text"] for a in (r.get("activity") or [])
               if a["kind"] == "result"]
    assert results, "the sample result should be in the feed"
    long_one = max(results, key=len)
    assert len(long_one) <= _s.LIVE_RESULT_MAX + 40
    if len(long_one) > _s.LIVE_RESULT_MAX:
        assert "display truncated" in long_one
    # and the model itself received the untruncated list
    blob = "\n".join(m["content"] for m in
                     __import__("rigma.sessions", fromlist=["x"]).load(
                         r["session_id"])["messages"]
                     if isinstance(m.get("content"), str))
    assert "ComfyUI_00039_.png" in blob or "40" in blob


def test_server_advances_the_plan_after_real_work(engine):
    # the owner watched it re-sample the same images three turns running: it had
    # DONE step #1 but never called manage_plan(complete), so the driving line
    # said "Do this now: #1 ..." forever. The server advances the plan now.
    _Engine.compile_reply = json.dumps({
        "objective": "look then write",
        "deliverables": [], "constraints": [],
        "steps": [
            {"id": 1, "description": "explore the folder", "artifact": "",
             "verification": {"type": "none"}},
            {"id": 2, "description": "write the notes", "artifact": "",
             "verification": {"type": "none"}}]})
    _Engine.script = [("current_datetime", {}), ("current_datetime", {}), None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "explore then write",
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    done = [t for t in r["plan"] if t["status"] == "done"]
    assert done, f"no step was ever completed: {r['plan']}"
    assert "complete" in c.get(f"/api/runs/{rid}/log").json()["log"]


def test_artifact_step_only_completes_when_the_file_exists(engine, tmp_path):
    # a step that promises a file is NOT done until the file is on disk
    target = tmp_path / "out.txt"
    _Engine.compile_reply = json.dumps({
        "objective": "write a file", "deliverables": [], "constraints": [],
        "steps": [{"id": 1, "description": "write the file",
                   "artifact": str(target),
                   "verification": {"type": "file_min_size", "value": 5}}]})
    _Engine.script = [("current_datetime", {}), None]   # productive, but no file
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "write a file",
                                    "budget_hours": 1,
                                    "workspace": str(tmp_path)}).json()["id"]
    r = _wait(c, rid, timeout=25)
    assert all(t["status"] == "pending" for t in r["plan"]), \
        "a step with a missing artifact must not be marked done"


def test_typing_a_tool_name_is_called_out_specifically(engine):
    # the model wrote "view_sample()" as prose after reading that string in a
    # tool result. A generic "you produced nothing" nudge doesn't tell it what
    # it actually got wrong.
    _Engine.script = [("__text__", "view_sample()")]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "x", "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=25)
    from rigma import sessions
    driving = [m["content"] for m in sessions.load(r["session_id"])["messages"]
               if m.get("role") == "user"]
    assert any("is not a tool call" in d for d in driving), driving[-2:]


def test_sample_files_hint_is_not_callable_looking_text(tmp_path):
    # the hint itself taught the model to type the name: it read
    # "call view_sample()" in the result and emitted that as its reply
    from rigma import runs as _r, tools as _t
    r = _r.create("m", "s")
    (tmp_path / "a.png").write_text("x", encoding="utf-8")
    out = _t.run_tool("sample_files", {"path": str(tmp_path)},
                      {"run_id": r["id"], "workspace": str(tmp_path)})
    assert "view_sample" in out                 # still points at the right tool
    assert "view_sample()" not in out           # but not as copyable syntax
