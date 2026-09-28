"""mcode's structured tool state: what was there all along and got flattened.

`_flatten` keeps only a tool result's `content` — the prose — and throws away
`details`, which is where mcode puts the goal object, the todo array and every
task id. One line before it was read. These tests pin the extraction against the
shapes mcode actually emits.

The field names are NOT invented. They come from a census of the 43 session
directories on this machine (43 `messages.jsonl`, 108 lines) and from mcode's own
shipped tool schemas in `llm-call.json`:

  todowrite    arguments `todos[]`, items `{content, status, priority}` all
               required; `status` in pending|in_progress|completed|cancelled;
               the list REPLACES, "an empty list clears it"
  update_goal  result `details.goal` = {goalId, sessionId, objective, status,
               createdAt, updatedAt, tokensUsed, timeUsedSeconds, tokenBudget};
               `status` in active|paused|complete|blocked|budget_limited|
               usage_limited. `get_goal` with no goal answers {goal: null}.
  task family  `details` are SNAKE_CASE: task_id, sub_session_id, sub_turn_id,
               agent_name, status

A caveat stated plainly: `todowrite`, `get_goal` and all five `task_*` tools were
NEVER INVOKED in any session on this machine, so their result shapes come from
mcode's shipped code rather than from observed data. The schemas and the field
names are real; the success payloads are not observed. `update_goal` IS observed,
including its real error result.
"""
from __future__ import annotations

import json
import sys

import pytest

from rigma import harness_mcode as mc


def _item(name: str, *, args=None, out=None, status="completed", iid="c1"):
    """One mcode `item.completed` carrying a tool call, as the wire sends it."""
    call = {"id": iid, "name": name, "status": status}
    if args is not None:
        call["input"] = args
    if out is not None:
        call["output"] = out
    return {"type": "item.completed", "item": {"id": iid, "type": "tool_call",
                                               "toolCall": call}}


# --------------------------------------------------------------------------
# `ok`, which the adapter used to say could not be derived


def test_ok_is_derived_from_the_wires_own_status():
    """The adapter's comment said `ok` could not be derived because mcode "does
    not flag failure in the projected item". The status IS on the item; the
    runtime normalises its numeric enum to started/completed/failed."""
    assert mc._ok_of({"status": "completed"}) is True
    assert mc._ok_of({"status": "failed"}) is False


def test_an_unrecognised_status_is_unknown_not_success():
    """The old behaviour claimed success for every result, including failures.
    `None` is what the seam renders as unknown."""
    assert mc._ok_of({"status": "started"}) is None
    assert mc._ok_of({}) is None
    assert mc._ok_of({"status": "something_new"}) is None


def test_a_failed_tool_result_no_longer_claims_success():
    seen: dict = {}
    events = mc.map_event(_item("bash", args={"command": "x"}, out={"content": []},
                                status="failed"), seen)
    results = [e for e in events if e.kind == "tool_result"]
    assert results and results[0].ok is False


# --------------------------------------------------------------------------
# goals


def test_a_goal_result_becomes_a_structured_goal_event():
    events = mc._state_events("update_goal", {"details": {
        "proposal": {"status": "blocked"},
        "goal": {
            "goalId": "g1", "sessionId": "s1", "objective": "ship the bridge",
            "status": "active", "tokensUsed": 120, "tokenBudget": 5000,
            "createdAt": 1, "updatedAt": 2,
        },
    }})
    assert len(events) == 1
    assert events[0].kind == "state"
    assert events[0].event == "goal"
    # The GOAL, not the proposal wrapper — the proposal describes the request.
    assert events[0].data["objective"] == "ship the bridge"
    assert events[0].data["status"] == "active"
    assert "proposal" not in events[0].data


def test_get_goal_with_no_goal_emits_nothing():
    """`{goal: null}` is not a goal. Emitting an empty one would blank a panel
    that was showing something real."""
    assert mc._state_events("get_goal", {"details": {"goal": None}}) == []


def test_a_goal_error_emits_nothing():
    """The one REAL update_goal invocation on this machine is this error."""
    real = {"error": "cannot update goal because this thread has no goal",
            "is_error": True}
    assert mc._state_events("update_goal", {"details": real}) == []


def test_a_missing_details_block_emits_nothing():
    assert mc._state_events("update_goal", {"content": [{"type": "text", "text": "x"}]}) == []
    assert mc._state_events("update_goal", None) == []
    assert mc._state_events("update_goal", "not a dict") == []


# --------------------------------------------------------------------------
# todos


def test_a_todowrite_result_becomes_the_whole_list():
    events = mc._state_events("todowrite", {"details": {"todos": [
        {"content": "read the docs", "status": "completed", "priority": "high"},
        {"content": "write the bridge", "status": "in_progress", "priority": "medium"},
    ], "total": 2, "active": 1, "completed": 1, "cancelled": 0}})
    assert len(events) == 1
    assert events[0].event == "todos"
    assert [t["content"] for t in events[0].data["todos"]] == \
        ["read the docs", "write the bridge"]
    # `priority` is carried through: mcode requires it, and dropping a field the
    # backend considers mandatory is how a schema quietly becomes lossy.
    assert events[0].data["todos"][0]["priority"] == "high"


def test_an_empty_todo_list_is_carried_rather_than_treated_as_absent():
    """mcode documents an empty list as "clears it", so [] is a real
    instruction and must not be confused with "no todos field"."""
    events = mc._state_events("todowrite", {"details": {"todos": []}})
    assert len(events) == 1 and events[0].data["todos"] == []


def test_a_todo_result_with_no_array_emits_nothing():
    assert mc._state_events("todowrite", {"details": {"total": 0}}) == []


# --------------------------------------------------------------------------
# subagents


def test_a_task_result_becomes_a_subagent_event_with_its_ids():
    """mcode emits snake_case; the seam uses camelCase, which is also what
    mcode's own normaliser produces."""
    events = mc._state_events("task", {"details": {
        "agent_name": "explore", "status": "started",
        "task_id": "bg_f73bca5e", "sub_session_id": "mvs_abc", "sub_turn_id": "turn_1",
    }})
    assert len(events) == 1
    assert events[0].event == "subagent"
    assert events[0].data == {
        "taskId": "bg_f73bca5e", "subSessionId": "mvs_abc", "subTurnId": "turn_1",
        "name": "explore", "status": "started",
    }


def test_a_task_event_with_no_id_at_all_emits_nothing():
    assert mc._state_events("task", {"details": {"status": "started"}}) == []


def test_a_bash_result_carrying_a_task_id_is_not_a_subagent():
    """A backgrounded bash call also reports `task_id` — the real captured
    example is `bg_f73bca5e-...` on a bash result. It is a job, not a child
    agent, and the filter is on the tool NAME."""
    assert mc._state_events("bash", {"details": {
        "status": "completed", "task_id": "bg_f73bca5e",
    }}) == []


def test_every_task_tool_is_recognised():
    for name in ("task", "task_append", "task_query", "task_output", "task_stop"):
        events = mc._state_events(name, {"details": {
            "task_id": "bg_1", "status": "running",
        }})
        assert len(events) == 1, name
        assert events[0].event == "subagent", name


# --------------------------------------------------------------------------
# the wiring through map_event


def test_map_event_emits_the_result_then_the_state():
    """Order matters for the transcript: the call, then its outcome, then what
    the outcome MEANT."""
    seen: dict = {}
    events = mc.map_event(_item(
        "todowrite",
        args={"todos": [{"content": "a", "status": "pending", "priority": "low"}]},
        out={"content": [{"type": "text", "text": "success"}],
             "details": {"todos": [{"content": "a", "status": "pending",
                                    "priority": "low"}]}},
    ), seen)
    kinds = [e.kind for e in events]
    assert kinds == ["tool", "tool_result", "state"]
    assert events[2].event == "todos"


def test_a_tool_with_no_structured_state_emits_only_the_two_events():
    seen: dict = {}
    events = mc.map_event(_item("bash", args={"command": "ls"},
                                out={"content": [{"type": "text", "text": "ok"}]}), seen)
    assert [e.kind for e in events] == ["tool", "tool_result"]


def test_the_state_event_is_not_repeated_for_a_repeated_item():
    """The per-turn memo stops one item being announced four times; the state
    must obey it too, or a todo list would render once per update."""
    seen: dict = {}
    item = _item("todowrite", args={"todos": []},
                 out={"details": {"todos": [{"content": "a", "status": "pending"}]}})
    first = mc.map_event(item, seen)
    second = mc.map_event(item, seen)
    assert len([e for e in first if e.kind == "state"]) == 1
    assert second == []


# --------------------------------------------------------------------------
# the MCP registration, which mcode launches to reach Rigma's own tools
#
# `ensure_mcp` runs every turn and points mcode at `python -m rigma.mcp_server`.
# It only ever ASSIGNED that entry when there was something to offer, so an entry
# that was present but wrong was never corrected — the file looks configured, the
# arm silently has no `remember`, `recall` or `undo_last_change`, and nothing
# anywhere says so. A real stale entry naming a different interpreter was found
# on this machine; these tests are that case, made permanent.


@pytest.fixture
def mcp_home(monkeypatch, tmp_path):
    """A Rigma home with mcode's data dir, and a foreign MCP server in the file
    so clobbering config Rigma does not own would be caught."""
    home = tmp_path / "home"
    (home / "mcode").mkdir(parents=True)
    monkeypatch.setenv("RIGMA_HOME", str(home))
    path = home / "mcode" / "mcp.json"
    return path


def _write(path, servers):
    path.write_text(json.dumps({"mcpServers": servers}, indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


def test_a_stale_interpreter_is_repaired_to_the_running_one(mcp_home):
    """The measured case: an entry naming a python that cannot import Rigma."""
    _write(mcp_home, {
        "rigma": {
            "command": r"C:\somewhere\else\Python312\python.exe",
            "args": ["-m", "rigma.mcp_server"],
            "env": {"RIGMA_HOME": str(mcp_home.parent), "RIGMA_MCP_ALLOW_CODE": "1"},
        },
        "someone-else": {"command": "other", "args": []},
    })
    mc.ensure_mcp("C:/chat")
    after = json.loads(mcp_home.read_text(encoding="utf-8"))["mcpServers"]
    assert after["rigma"]["command"] == sys.executable
    assert after["rigma"]["env"]["RIGMA_MCP_WORKSPACE"] == "C:/chat"
    # Rigma manages this file, it does not own it.
    assert "someone-else" in after


def test_a_correct_entry_is_left_exactly_as_it_is(mcp_home):
    """Idempotent, and specifically not rewritten when the only difference is
    the workspace — which moves legitimately from turn to turn."""
    mc.ensure_mcp("C:/chat")
    first = mcp_home.read_bytes()
    stamp = mcp_home.stat().st_mtime_ns

    mc.ensure_mcp("C:/chat")
    assert mcp_home.read_bytes() == first

    mc.ensure_mcp("C:/a/different/chat")
    assert mcp_home.read_bytes() == first, "a moved workspace is not drift"
    assert mcp_home.stat().st_mtime_ns == stamp


def test_a_dropped_code_grant_is_repaired(mcp_home):
    """`RIGMA_MCP_ALLOW_CODE` decides whether `undo_last_change` may be offered,
    so an entry missing it is a real difference and not a formatting one."""
    _write(mcp_home, {"rigma": {
        "command": sys.executable, "args": ["-m", "rigma.mcp_server"],
        "env": {"RIGMA_HOME": str(mcp_home.parent)},
    }})
    mc.ensure_mcp("C:/chat")
    after = json.loads(mcp_home.read_text(encoding="utf-8"))["mcpServers"]
    assert after["rigma"]["env"]["RIGMA_MCP_ALLOW_CODE"] == "1"


def test_a_hand_written_entry_that_is_not_an_object_is_repaired(mcp_home):
    """A user can edit this file. `"rigma": "python -m rigma.mcp_server"` is a
    plausible mistake and must be replaced, not raise inside a turn."""
    _write(mcp_home, {"rigma": "python -m rigma.mcp_server"})
    mc.ensure_mcp("C:/chat")
    after = json.loads(mcp_home.read_text(encoding="utf-8"))["mcpServers"]
    assert isinstance(after["rigma"], dict)
    assert after["rigma"]["command"] == sys.executable


def test_same_registration_ignores_the_workspace_and_nothing_else():
    spec = mc._mcp_spec("C:/chat")
    moved = {**spec, "env": {**spec["env"], "RIGMA_MCP_WORKSPACE": "D:/elsewhere"}}
    assert mc._same_registration(moved, spec) is True
    assert mc._same_registration({**spec, "command": "python"}, spec) is False
    assert mc._same_registration({**spec, "args": []}, spec) is False
    assert mc._same_registration({}, spec) is False
    assert mc._same_registration("nope", spec) is False
    assert mc._same_registration(spec, spec) is True
