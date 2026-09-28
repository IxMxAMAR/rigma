"""The capability bridge: what Rigma adds to DSH, and what it does with it.

TWO INDEPENDENT BLOCKERS, and this file covers both because either one alone
leaves the user with nothing to see:

1. `sdk-minimal` is a STANDALONE Cordis tree that mounts no goal, subagent,
   todo, skill, plan-mode, filesystem or compaction plugin at all. Those
   capabilities did not merely go unreported — they did not exist in Rigma's
   DSH runtime. `agent-capabilities.patch.yml` inserts them.

2. Even once mounted, their events ride inside `session.event` and were dropped
   by `_dsh_runner._notice_text`, which matched six keywords against a
   two-field summary and truncated to 200 characters. The runner now projects
   them as structured events.

A malformed patch row is SKIPPED WITH A WARNING rather than failing the boot, so
a typo silently yields a missing tool. That is why the first test below reads the
patch file and asserts the rows, rather than trusting that a successful boot
means a complete one.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import yaml

from rigma import _dsh_runner as runner
from rigma import harness_dsh


# The rows the bridge is expected to carry, grouped by the capability they buy.
# If a row is dropped from the patch this list is what notices.
EXPECTED_ROWS = {
    "filesystem": ["fs-local", "tool-fs", "tool-fs-search", "tool-str-replace-editor"],
    "instructions": ["agent-instructions"],
    "todos": ["tool-todo"],
    "goals": ["goal", "goal-round-driver", "tool-goal"],
    "skills": ["skill", "skill-filesystem", "tool-skill"],
    "subagents": [
        "subagent", "subagent-spawn-in-process", "subagent-fork-in-process",
        "tool-subagent-control", "tool-subagent-list-agents", "tool-subagent",
        "tool-subagent-fork",
    ],
    "plan": ["plan-mode"],
    "workflow": ["ptc-runtime", "workflow-ptc", "tool-workflow"],
    "context": ["token-meter", "compaction-basic"],
}


def _patch_rows() -> list[dict]:
    path = harness_dsh.capability_patch()
    assert path, "the capability patch is not shipped in the package"
    doc = yaml.safe_load(open(path, encoding="utf-8"))
    rows: list[dict] = []
    for entry in doc:
        rows.extend(entry.get("insert") or [])
    return rows


def test_the_capability_patch_is_shipped_with_the_package():
    """It is read from the package, not the source tree, so the wheel has it."""
    path = harness_dsh.capability_patch()
    assert path.endswith("agent-capabilities.patch.yml")
    from pathlib import Path

    assert Path(path).is_file()


def test_every_capability_the_minimal_profile_omits_is_inserted():
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    missing = {
        group: [rid for rid in ids if rid not in by_id]
        for group, ids in EXPECTED_ROWS.items()
    }
    missing = {g: v for g, v in missing.items() if v}
    assert not missing, f"capabilities the patch no longer inserts: {missing}"


def test_the_two_subagent_tools_are_the_same_plugin_under_two_names():
    """dsh-base does it this way, and the toolName is what the model calls."""
    rows = [r for r in _patch_rows() if r.get("name") == "@deepseek-ai/dsh-tool-subagent"]
    assert len(rows) == 2, rows
    names = {r["config"]["toolName"] for r in rows}
    assert names == {"subagent", "subagent_fork"}
    # `provider` is required by the plugin's schema; omitting it is fatal at
    # mount rather than a silent degradation.
    assert all(r["config"].get("provider") for r in rows)


def test_rows_whose_config_has_no_default_carry_it():
    """These four are `z.<type>().required()` upstream — no default exists."""
    by_id = {str(r.get("id")): r for r in _patch_rows()}
    assert by_id["tool-todo"]["config"]["allowParallelInProgress"] is True
    assert by_id["tool-fs-search"]["config"]["sampleOverCapGlobResults"] is False
    assert by_id["plan-mode"]["config"]["section"].strip()
    for row in _patch_rows():
        if row.get("name") == "@deepseek-ai/dsh-tool-subagent":
            assert row["config"]["provider"]


def test_every_inserted_row_names_a_package():
    """A row with no `name` would be a silent no-op."""
    for row in _patch_rows():
        assert row.get("id"), row
        assert str(row.get("name") or "").startswith("@deepseek-ai/"), row


# --------------------------------------------------------------------------
# The projection: what the runner does with a notification.


def _notif(method: str, payload: dict):
    return SimpleNamespace(method=method, payload=payload)


def _session_event(kind: str, data: dict):
    return _notif("session.event", {"sessionId": "s1", "event": {"type": kind, "data": data}})


def test_a_tool_call_carries_the_tool_name_and_arguments():
    """Before this, the parent was told the literal string "session.event
    tool/call" — every DSH turn rendered zero tool chips and the tool name was
    unrecoverable downstream."""
    out = runner._project(_session_event("tool/call", {
        "name": "read_file", "arguments": {"path": "a.py"}, "callId": "c1",
    }))
    assert out == [{"type": "tool", "name": "read_file",
                    "args": {"path": "a.py"}, "id": "c1"}]


def test_two_calls_to_one_tool_keep_distinct_ids():
    a = runner._project(_session_event("tool/call", {"name": "read", "callId": "c1"}))[0]
    b = runner._project(_session_event("tool/call", {"name": "read", "callId": "c2"}))[0]
    assert a["id"] != b["id"], "the id is what stops the chips collapsing"


def test_a_tool_result_reports_failure_from_the_backend():
    ok = runner._project(_session_event("tool/result", {"callId": "c1", "content": "fine"}))[0]
    bad = runner._project(_session_event("tool/result",
                                         {"callId": "c1", "isError": True}))[0]
    assert ok["ok"] is True and bad["ok"] is False


def test_a_goal_change_is_structured_not_a_progress_line():
    out = runner._project(_session_event("goal/change", {
        "operation": "create",
        "goal": {"id": "g1", "phase": "active", "objective": "ship it"},
        "roundsStarted": 2,
    }))
    assert out[0]["type"] == "state"
    assert out[0]["event"] == "goal/change"
    # The WHOLE envelope is kept: the snapshot AND the round count and the
    # operation, because a UI needs all three to say "round 4 of 60, paused".
    assert out[0]["data"]["goal"]["phase"] == "active"
    assert out[0]["data"]["roundsStarted"] == 2
    assert out[0]["data"]["operation"] == "create"


def test_a_todo_write_carries_the_whole_list():
    out = runner._project(_session_event("todo/write", {
        "todos": [{"content": "a", "status": "pending"}],
    }))
    assert out[0]["data"]["todos"] == [{"content": "a", "status": "pending"}]


def test_plan_mode_and_usage_are_both_carried():
    plan = runner._project(_session_event("plan/mode", {"active": True}))[0]
    assert plan["event"] == "plan/mode" and plan["data"]["active"] is True
    usage = runner._project(_session_event("assistant/message", {
        "usage": {"inputTokens": 10},
    }))[0]
    assert usage["event"] == "usage" and usage["data"]["inputTokens"] == 10


def test_subagent_lifecycle_is_top_level_and_no_longer_filtered_out():
    """`subagent.finished` contains "finish", not "fail", so the old keyword
    filter dropped it — a subagent could start AND end unseen."""
    started = runner._project(_notif("subagent.started", {
        "parentSessionId": "p", "childSessionId": "c",
    }))[0]
    assert started["type"] == "state" and started["event"] == "subagent.started"
    assert started["data"]["childSessionId"] == "c"
    finished = runner._project(_notif("subagent.finished", {
        "childSessionId": "c", "status": "ok",
    }))[0]
    assert finished["event"] == "subagent.finished"


def test_internal_chatter_is_still_dropped():
    """The filter existed for a reason: a one-line turn produced fifteen
    notifications, and forwarding them all buries the reply."""
    for kind in ("agent/inbox/spliced", "step/start", "request/header",
                 "session/title", "delivery-accepted"):
        assert runner._project(_session_event(kind, {})) == [], kind


def test_a_malformed_notification_produces_nothing_and_does_not_raise():
    for bad in (SimpleNamespace(method="x", payload=None),
                SimpleNamespace(method="x", payload={"event": "not-a-dict"}),
                SimpleNamespace(method="x", payload={"event": {"type": 5}})):
        assert runner._project(bad) == []


# --------------------------------------------------------------------------
# The adapter's translation of those events onto the seam.


def test_the_adapter_passes_structured_state_through():
    ev = harness_dsh._event_for({
        "type": "state", "event": "goal/change", "data": {"goal": {"phase": "active"}},
    })
    assert ev is not None
    assert ev.kind == "state"
    assert ev.event == "goal/change"
    assert ev.data == {"goal": {"phase": "active"}}


def test_the_adapter_carries_the_call_id_on_tool_events():
    ev = harness_dsh._event_for({"type": "tool", "name": "read", "args": {}, "id": "c1"})
    assert ev is not None and ev.data == {"id": "c1"}
    res = harness_dsh._event_for({"type": "tool_result", "ok": False, "id": "c1"})
    assert res is not None and res.data == {"id": "c1"} and res.ok is False


def test_the_adapter_still_skips_an_unknown_kind():
    assert harness_dsh._event_for({"type": "invented_by_a_newer_dsh"}) is None


def test_a_turn_event_can_be_built_without_the_new_fields():
    """They were added with defaults, so every existing adapter and call site
    keeps working."""
    from rigma.harness import TurnEvent

    ev = TurnEvent(kind="text", text="hi")
    assert ev.event == "" and ev.data is None


@pytest.mark.parametrize("window,tokens", [(32768, 4096), (65536, 8192)])
def test_the_capability_patch_is_handed_to_the_runner(tmp_path, window, tokens):
    """The parent locates it; the child only receives a path, so there is
    exactly one copy of the file to keep correct."""
    job_path = harness_dsh.capability_patch()
    assert job_path, "the parent must be able to find the patch"
    from pathlib import Path

    assert Path(job_path).is_file()
