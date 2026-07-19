"""Autonomous-run tools + safety guardrails."""
import pytest

from rigma import runs, tools


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


# --- safety: blocklist + profiles -------------------------------------------
def test_destructive_command_blocked():
    for bad in ("format C:", "shutdown /s", "reg delete HKLM\\x", "rm -rf /"):
        out = tools.run_tool("run_shell", {"command": bad}, {"allow_code": True})
        assert "blocked" in out, bad


def test_no_delete_profile_blocks_deletion():
    ctx = {"allow_code": True, "profile": "no-delete"}
    assert "blocked" in tools.run_tool("run_shell", {"command": "del a.txt"}, ctx)
    assert "blocked" in tools.run_tool("run_shell", {"command": "rm a.txt"}, ctx)


def test_no_network_profile_disables_web():
    ctx = {"profile": "no-network"}
    assert "disabled" in tools.run_tool("web_search", {"query": "x"}, ctx)
    assert "disabled" in tools.run_tool("fetch_url", {"url": "http://x"}, ctx)
    base = {t["function"]["name"]
            for t in tools.tool_specs(profile="no-network")}
    assert "web_search" not in base and "fetch_url" not in base


def test_confined_profile_disables_code_exec():
    ctx = {"allow_code": True, "profile": "confined"}
    assert "disabled" in tools.run_tool("run_shell", {"command": "echo hi"}, ctx)
    assert "disabled" in tools.run_tool("run_python", {"code": "print(1)"}, ctx)


# --- run-scoped tools --------------------------------------------------------
def test_run_tools_gated_off_without_run():
    base = {t["function"]["name"] for t in tools.tool_specs()}
    assert "manage_plan" not in base and "task_complete" not in base
    withrun = {t["function"]["name"]
               for t in tools.tool_specs(has_run=True)}
    assert {"manage_plan", "task_complete"} <= withrun
    assert "log_progress" not in withrun   # server writes the log now
    # calling without a run_id in ctx is refused
    assert "only available inside" in tools.run_tool("manage_plan",
                                                     {"action": "list"}, {})


def test_manage_plan_add_complete():
    r = runs.create("m", "s")
    ctx = {"run_id": r["id"]}
    out = tools.run_tool("manage_plan",
                         {"action": "add", "task": "read the folder"}, ctx)
    assert "added step #1" in out
    tools.run_tool("manage_plan", {"action": "add", "task": "write prompts"}, ctx)
    assert len(runs.pending_tasks(r["id"])) == 2
    tools.run_tool("manage_plan", {"action": "complete", "id": 1}, ctx)
    assert len(runs.pending_tasks(r["id"])) == 1


def test_manage_plan_update_rewords_a_step():
    # models reach for action='update' naturally; rejecting it burned tool calls
    r = runs.create("m", "s")
    ctx = {"run_id": r["id"]}
    tools.run_tool("manage_plan", {"action": "add", "task": "draft it"}, ctx)
    out = tools.run_tool("manage_plan",
                         {"action": "update", "id": 1,
                          "task": "Define Core Directive"}, ctx)
    assert "updated" in out
    assert runs.read_plan(r["id"])[0]["text"] == "Define Core Directive"
    assert runs.read_plan(r["id"])[0]["status"] == "pending"   # status untouched
    assert "required" in tools.run_tool("manage_plan",
                                        {"action": "update", "id": 1}, ctx)
    assert "no such step" in tools.run_tool(
        "manage_plan", {"action": "update", "id": 99, "task": "x"}, ctx)


def test_read_file_progress_md_returns_the_log_not_an_error():
    # the model hunts for progress.md and loops on "no such file"; hand it the
    # real log instead (it lives in the run dir, not the workspace)
    import tempfile
    r = runs.create("m", "s")
    runs.append_progress(r["id"], "sampled 20 images", "write the directive")
    ctx = {"run_id": r["id"], "workspace": tempfile.mkdtemp()}
    out = tools.run_tool("read_file", {"path": "progress.md"}, ctx)
    assert not out.startswith("error")
    assert "sampled 20 images" in out and "Do NOT restart" in out
    # a genuinely missing file still errors normally
    assert tools.run_tool("read_file", {"path": "nope.txt"},
                          ctx).startswith("error")


def test_task_complete_acknowledges():
    r = runs.create("m", "s")
    out = tools.run_tool("task_complete", {"summary": "did it"},
                         {"run_id": r["id"]})
    assert "verify" in out.lower()
