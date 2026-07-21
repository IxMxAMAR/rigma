"""Phase 4 of the field-parity audit: the run lifecycle grows up.

A crashed/stopped run can be restarted from disk; content_check steps are
judged on substance, not just existence; an impossible step gets blocked and
routed around instead of holding the plan hostage.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import mission, runs, serve
from rigma import state as st


class _E4(BaseHTTPRequestHandler):
    """Scripted engine for lifecycle tests. Streaming requests consume
    `script`; non-streaming requests are routed by prompt content (mission
    compile vs content judge vs advisor)."""
    script = []
    idx = 0
    compile_reply = "not a spec"
    judge_replies: list = []          # popped per content-judge call

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if not body.get("stream", True):
            content = str((body.get("messages") or [{}])[0].get("content", ""))
            if "Acceptance criterion" in content:
                reply = (_E4.judge_replies.pop(0)
                         if _E4.judge_replies else "PASS looks fine")
            elif "debugging advisor" in content:
                reply = ("DIAGNOSIS: it keeps reading a file that is not "
                         "there\nNEXT: list the folder first")
            else:
                reply = _E4.compile_reply
            payload = json.dumps({"choices": [{"message": {
                "content": reply}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        i = _E4.idx
        _E4.idx += 1
        step = _E4.script[i] if i < len(_E4.script) else _E4.script[-1]
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        if step is None:
            sse({"choices": [{"delta": {"content": "hmm"}}]})
        else:
            nm, ar = step
            sse({"choices": [{"delta": {"tool_calls": [
                {"index": 0, "id": f"c{i}", "type": "function",
                 "function": {"name": nm, "arguments": json.dumps(ar)}}]}}]})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def engine():
    srv = HTTPServer(("127.0.0.1", 0), _E4)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setenv("RIGMA_MEMORY", "0")     # lifecycle, not memory, here
    _E4.idx = 0
    _E4.script = []
    _E4.compile_reply = "not a spec"
    _E4.judge_replies = []
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


_CLIENTS = []


def _client(port):
    import os
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    c = TestClient(serve.build_app(upstream_port=port))
    c.__enter__()
    _CLIENTS.append(c)
    return c


def _wait(client, rid, timeout=30, until=None):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        r = client.get(f"/api/runs/{rid}").json()
        if until and until(r):
            return r
        if not until and r.get("status") in runs.TERMINAL:
            return r
        time.sleep(0.05)
    return client.get(f"/api/runs/{rid}").json()


# --- units --------------------------------------------------------------------
def test_plan_block_routes_around(tmp_path):
    run = runs.create("m", "sid", workspace=str(tmp_path))
    rid = run["id"]
    runs.plan_add(rid, "impossible step")
    runs.plan_add(rid, "easy step")
    assert runs.plan_block(rid, 1, "file server is down")
    pend = runs.pending_tasks(rid)
    assert [t["id"] for t in pend] == [2]       # blocked step is skipped
    blocked = runs.blocked_tasks(rid)
    assert blocked[0]["blocked_reason"] == "file server is down"


def test_parse_spec_keeps_content_check_criterion():
    spec = mission.parse_spec(json.dumps({
        "objective": "write prompts",
        "steps": [{"id": 1, "description": "write them",
                   "artifact": "out.txt",
                   "verification": {"type": "content_check",
                                    "value": "contains 25 prompts"}}]}))
    assert spec["steps"][0]["verification"]["value"] == "contains 25 prompts"


def test_verify_step_content_check_needs_nonempty(tmp_path):
    step = {"artifact": "x.txt",
            "verification": {"type": "content_check", "value": "whatever"}}
    ok, why = mission.verify_step(step, str(tmp_path))
    assert not ok and "does not exist" in why
    (tmp_path / "x.txt").write_text("", encoding="utf-8")
    ok, why = mission.verify_step(step, str(tmp_path))
    assert not ok and "empty" in why
    (tmp_path / "x.txt").write_text("real content", encoding="utf-8")
    ok, _ = mission.verify_step(step, str(tmp_path))
    assert ok      # existence passes; substance is the judge's job


def test_content_judge_prompt_carries_evidence(tmp_path):
    (tmp_path / "out.txt").write_text("banana banana", encoding="utf-8")
    step = {"description": "write fruit", "artifact": "out.txt",
            "verification": {"type": "content_check",
                             "value": "mentions banana"}}
    p = mission.content_judge_prompt(step, str(tmp_path))
    assert "mentions banana" in p and "banana banana" in p
    assert "PASS or FAIL" in p


def test_restartable_states_are_terminal_but_not_done():
    assert "done" not in runs.RESTARTABLE
    assert runs.RESTARTABLE <= runs.TERMINAL | {"interrupted"}
    assert "interrupted" in runs.TERMINAL       # clears active.json at boot


# --- e2e ----------------------------------------------------------------------
def test_orphaned_run_marked_interrupted(engine, tmp_path):
    run = runs.create("orphan mission", "no-such-session",
                      workspace=str(tmp_path))
    _client(engine)                              # startup hook reconciles
    r = runs.load(run["id"])
    assert r["status"] == "interrupted"


def test_restart_reattaches_and_finishes(engine, tmp_path):
    _E4.script = [("manage_plan", {"action": "add", "task": "step one"}),
                  None]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "small job",
                                    "budget_hours": 1}).json()["id"]
    _wait(c, rid, until=lambda r: r.get("iteration", 0) >= 1)
    c.post(f"/api/runs/{rid}/stop")
    _wait(c, rid, until=lambda r: r.get("status") in runs.TERMINAL)
    assert runs.load(rid)["status"] in runs.RESTARTABLE
    # new life: finish the plan and complete (verify-once gates the first)
    _E4.idx = 0
    _E4.script = [("manage_plan", {"action": "complete", "id": 1}),
                  ("task_complete", {"summary": "did the small job"}),
                  ("task_complete", {"summary": "did the small job"})]
    out = c.post(f"/api/runs/{rid}/restart").json()
    assert out.get("restarted") is True
    r = _wait(c, rid)
    assert r["status"] == "done"
    assert "small job" in r["summary"]


def test_content_check_fail_then_pass(engine, tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    _E4.compile_reply = json.dumps({
        "objective": "write fruit prompts",
        "deliverables": [{"path": "out.txt", "description": "the prompts"}],
        "constraints": [],
        "steps": [{"id": 1, "description": "write the prompts",
                   "artifact": "out.txt",
                   "verification": {"type": "content_check",
                                    "value": "mentions banana"}}]})
    _E4.judge_replies = ["FAIL it never mentions banana",
                         "PASS banana is right there"]
    _E4.script = [
        ("write_file", {"path": "out.txt", "content": "apple pear plum"}),
        ("write_file", {"path": "out.txt", "content": "banana banana",
                        "append": False}),
        ("task_complete", {"summary": "prompts written"}),
        ("task_complete", {"summary": "prompts written"}),
    ]
    c = _client(engine)
    rid = c.post("/api/runs", json={"mission": "write fruit prompts",
                                    "workspace": str(ws),
                                    "budget_hours": 1}).json()["id"]
    r = _wait(c, rid, timeout=40)
    assert r["status"] == "done"
    plan = r["plan"]
    assert plan and plan[0]["status"] == "done"
    # the FAIL verdict was consumed (first judge call rejected the step)
    assert not _E4.judge_replies
