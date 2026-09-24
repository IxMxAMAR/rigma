"""IMP-8: why a run stopped, and what budget is left.

An autonomous run can end for many reasons and the run view showed only
"stopped", so "finished" and "gave up" were indistinguishable. The prose
`halt_reason` already existed; this adds a stable code and the remaining
budget, both on the run record the UI reads.
"""
import time

import pytest
from fastapi.testclient import TestClient

from rigma import runs
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("status,reason,code", [
    ("done", "task_complete (verified)", "completed"),
    ("budget_exhausted", "time budget reached", "time_budget"),
    ("budget_exhausted", "iteration cap reached", "step_cap"),
    ("budget_exhausted", "token budget reached", "token_cap"),
    ("frozen", "engine unresponsive", "frozen_streak"),
    ("stalled", "too many tool errors", "tool_errors"),
    ("stalled", "no progress / idle", "no_progress"),
    ("stalled", "paused waiting for an answer for over 30 minutes", "paused_timeout"),
    ("stopped", "stopped by user", "user_stop"),
    ("stopped", "cancelled", "user_stop"),
    ("error", "session was deleted", "session_deleted"),
    ("error", "loop crashed: boom", "engine_error"),
    ("interrupted", "server restarted mid-run", "interrupted"),
    ("stalled", "something nobody classified", "unknown"),
])
def test_stop_codes(status, reason, code):
    assert runs.stop_code(status, reason) == code


def test_set_status_persists_the_code():
    r = runs.create("m", "s", budget_hours=0.1)
    r["iteration"] = 2000
    runs.set_status(r, "budget_exhausted", "iteration cap reached")
    on_disk = runs.load(r["id"])
    assert on_disk["status"] == "budget_exhausted"
    assert on_disk["stop_reason"] == "step_cap"
    assert on_disk["halt_reason"] == "iteration cap reached"
    assert runs.stop_reason(on_disk) == "step_cap"


def test_an_older_run_without_the_field_still_reports_one():
    r = runs.create("m", "s")
    r["status"] = "frozen"
    r["halt_reason"] = "engine unresponsive"
    r.pop("stop_reason", None)
    runs.save(r, revive=True)
    assert runs.stop_reason(runs.load(r["id"])) == "frozen_streak"


def test_a_running_run_has_no_stop_reason():
    r = runs.create("m", "s")
    assert runs.stop_reason(runs.load(r["id"])) == ""


def test_budget_snapshot():
    r = runs.create("m", "s", budget_hours=1.0, token_cap=100)
    r["iteration"] = 3
    r["tokens_used"] = 40
    b = runs.budget_snapshot(r)
    assert b["steps_used"] == 3 and b["steps_total"] == runs.MAX_ITERS
    assert b["steps_remaining"] == runs.MAX_ITERS - 3
    assert 3500 < b["seconds_remaining"] <= 3600
    assert b["tokens_remaining"] == 60
    # no token cap: "unlimited" must not render as "exhausted"
    r["token_cap"] = 0
    assert runs.budget_snapshot(r)["tokens_remaining"] is None


def test_run_endpoints_expose_reason_and_budget():
    r = runs.create("m", "s", budget_hours=1.0, token_cap=100)
    r["iteration"] = 5
    r["tokens_used"] = 25
    runs.save(r, revive=True)
    c = TestClient(build_app(upstream_port=1))
    got = c.get(f"/api/runs/{r['id']}").json()
    assert got["budget"]["steps_remaining"] == runs.MAX_ITERS - 5
    assert got["budget"]["tokens_remaining"] == 75
    assert got["stop_reason"] == ""          # still running
    listed = c.get("/api/runs").json()[0]
    assert "stop_reason" in listed

    # once it stops, both the code and the prose are on the wire
    stale = runs.load(r["id"])
    stale["deadline"] = time.time() - 1          # the clock really is spent
    runs.set_status(stale, "budget_exhausted", "time budget reached")
    got = c.get(f"/api/runs/{r['id']}").json()
    assert got["stop_reason"] == "time_budget"
    assert got["halt_reason"] == "time budget reached"
    assert got["budget"]["seconds_remaining"] == 0
