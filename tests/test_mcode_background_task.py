"""mcode #8: a background `bash` task was invisible.

WHAT THE DEFECT WAS. `bash` is not in `_TASK_TOOLS`, and `_state_events` returns `[]`
for anything it does not recognise — so a background command produced a `tool` chip and
nothing else. The id and status survived only as legible text inside
`<bash_background task_id="...">`, which is not data: nothing could key a row on it, and
nothing could report whether the task was still running.

WHY THAT MATTERS MORE THAN IT LOOKS. mcode's own guidance is to avoid routine polling —
the background result text says "The owning conversation will automatically resume when
it finishes" — so the one-time chip was the ONLY trace of a task that might run for
minutes. A user watching a quiet turn had no way to tell a running background command
from a hung one.

The shape is verified against mcode's own bundle (`chunk-QWAB5G2D.js` @5981148):
`{tool_name, text, content, details: {...config, status, task_id}}`.

No model is loaded.
"""
from __future__ import annotations

from rigma import harness_mcode as mcode


def _result(**details) -> dict:
    """A tool result as mcode wraps one: content for text, details for structure."""
    return {
        "tool_name": "bash",
        "text": '<bash_background task_id="t1">…</bash_background>',
        "content": [{"type": "text", "text": "accepted"}],
        "details": details,
    }


def test_a_started_background_bash_becomes_a_task_row():
    out = mcode._state_events("bash", _result(task_id="t1", status="started"))
    assert len(out) == 1, out
    ev = out[0]
    assert ev.kind == "state"
    # `subagent` rather than a new event name: mcode's `task_query` calls these "local
    # background tasks", the same family as a delegated child, and the UI already draws
    # a row keyed by `taskId`.
    assert ev.event == "subagent"
    assert ev.data["taskId"] == "t1"
    assert ev.data["status"] == "started"


def test_a_foreground_bash_grows_no_row():
    """A foreground command was never a task, so a row would claim work that does not
    exist."""
    assert mcode._state_events("bash", _result(status="direct_foreground")) == []
    assert mcode._state_events("bash", {"content": [], "details": {}}) == []


def test_a_COMPLETED_bash_job_grows_no_row():
    """The case a pre-existing test already pinned, from a REAL capture
    (`test_mcode_state.py`, status "completed", id `bg_f73bca5e`).

    A finished background job is not a live child agent, and a row for one would claim
    work that is over. My first version of this fix keyed on the presence of `task_id`
    and broke that test — the trigger had to be the START, not the id.
    """
    assert mcode._state_events("bash", _result(
        task_id="bg_f73bca5e", status="completed")) == []


def test_an_unknown_status_grows_no_row():
    """Reporting "running" for a status word this build does not know would invent
    state, which is the failure mode the whole finding is about."""
    assert mcode._state_events("bash", _result(
        task_id="t1", status="some_future_word")) == []


def test_an_auto_promoted_command_is_a_task():
    """mcode promotes a command that outran its timeout WITHOUT restarting it, and says
    so: "The command is still running and was yielded to a managed background task".
    That is a task even though the model asked for a foreground run."""
    out = mcode._state_events("bash", _result(task_id="t2", status="auto_promoted"))
    assert out[0].data["taskId"] == "t2"
    assert out[0].data["status"] == "auto_promoted"


def test_the_row_carries_a_name_worth_reading():
    """A row reading "subagent running" is unreadable the moment there are two, which
    is the same reason `subagents.ts` needed a name for DSH's children."""
    out = mcode._state_events(
        "bash", _result(task_id="t1", status="started", description="run the suite"))
    assert out[0].data["name"] == "run the suite"


def test_it_falls_back_to_a_name_rather_than_nothing():
    out = mcode._state_events("bash", _result(task_id="t1", status="started"))
    assert out[0].data["name"] == "background shell"


def test_a_task_result_with_no_details_is_not_invented():
    """`details` is where mcode puts the id. Without it there is nothing to key on, and
    guessing an id would put a row in the UI that no later event could ever resolve."""
    assert mcode._state_events("bash", {"content": [], "text": "ok"}) == []


def test_the_existing_task_tools_still_work():
    """The new branch sits before the `_TASK_TOOLS` one, so it must not shadow it."""
    out = mcode._state_events("task_output", {
        "content": [], "details": {"task_id": "t9", "status": "completed"},
    })
    assert len(out) == 1, out
    assert out[0].data["taskId"] == "t9"
