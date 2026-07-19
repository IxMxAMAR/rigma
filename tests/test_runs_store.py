"""Autonomous-run store — pure state, no engine."""
import json

import pytest

from rigma import runs


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def test_create_sets_active_and_files():
    r = runs.create("do the thing", "sess1", workspace="", budget_hours=8)
    assert r["status"] == "running" and r["mission"] == "do the thing"
    assert runs.active()["id"] == r["id"]
    d = runs.run_dir(r["id"])
    assert (d / "run.json").exists() and (d / "plan.json").exists()
    assert (d / "progress.md").exists() and (d / "outputs").is_dir()


def test_budget_hours_clamped():
    r = runs.create("m", "s", budget_hours=999)          # clamp to 48h
    assert r["deadline"] - r["started_at"] <= 48 * 3600 + 1
    r2 = runs.create("m", "s", budget_hours=0)            # floor 0.1h
    assert r2["deadline"] - r2["started_at"] >= 0.1 * 3600 - 1


def test_terminal_status_clears_active():
    r = runs.create("m", "s")
    assert runs.active() is not None
    runs.set_status(r, "done")
    assert runs.active() is None                          # pointer released
    assert runs.load(r["id"])["status"] == "done"         # record kept


def test_plan_add_complete_pending():
    r = runs.create("m", "s")
    t1 = runs.plan_add(r["id"], "step one")
    t2 = runs.plan_add(r["id"], "step two")
    assert [t["id"] for t in runs.pending_tasks(r["id"])] == [t1, t2]
    assert runs.plan_complete(r["id"], t1) is True
    assert [t["id"] for t in runs.pending_tasks(r["id"])] == [t2]
    assert "#{} step two".format(t2) in runs.plan_summary(r["id"])


def test_progress_log_and_tail():
    r = runs.create("m", "s")
    for i in range(7):
        runs.append_progress(r["id"], f"did {i}", f"next {i}")
    tail = runs.get_log_tail(r["id"], n=3)
    assert "did 6" in tail and "did 4" in tail and "did 3" not in tail
    assert tail.count("->  next:") == 3


def test_progress_mirrors_to_workspace(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    r = runs.create("m", "s", workspace=str(ws))
    runs.append_progress(r["id"], "x", "y", workspace=str(ws))
    assert (ws / "rigma-progress.md").exists()
    assert "done: x" in (ws / "rigma-progress.md").read_text(encoding="utf-8")


def test_actions_audit_jsonl():
    r = runs.create("m", "s")
    runs.append_action(r["id"], "write_file", {"path": "a.txt"}, True)
    runs.append_action(r["id"], "run_shell", {"command": "ls"}, False)
    lines = (runs.run_dir(r["id"]) / "actions.jsonl").read_text(
        encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    rec = json.loads(lines[0])
    assert rec["tool"] == "write_file" and rec["ok"] is True


def test_budget_exceeded_reasons():
    r = runs.create("m", "s", budget_hours=8)
    assert runs.budget_exceeded(r) == ""
    r["deadline"] = 0                                     # in the past
    assert "time budget" in runs.budget_exceeded(r)
    r = runs.create("m", "s")
    r["iteration"] = runs.MAX_ITERS
    assert "iteration cap" in runs.budget_exceeded(r)
    r = runs.create("m", "s", token_cap=100)
    r["tokens_used"] = 100
    assert "token budget" in runs.budget_exceeded(r)
