"""R6-WORKFLOW: programmatic tool calling reached the UI as a bare string.

THE DEFECT THIS PINS, and it was an ACCIDENT rather than a decision. `tool-workflow/*`
was not in the runner's `_STATE_EVENTS`, so it fell through to the notice path — which
matches on `_NOTICE_WORTHY`, a list containing the substring `"tool"`. `"tool-workflow"`
contains it. So all four lifecycle events were forwarded as the literal string
`"session.event tool-workflow/agent-start"`, and the run's name, each agent's label and
phase, each agent's outcome and the run's stop reason were all discarded.

The interesting part is WHY a substring match is the wrong mechanism for this: the
notice path is deliberately lossy, so anything that lands there by accident is lost
silently. These tests check the events are STRUCTURED, not merely forwarded.

No model is loaded.
"""
from __future__ import annotations

import inspect

from rigma import _dsh_runner as runner
from rigma import serve

# The four, from DSH's own dispositions
# (packages/session/session-format-v0-to-v1/src/dispositions.ts:102-105).
WORKFLOW_EVENTS = (
    "tool-workflow/run-start",
    "tool-workflow/agent-start",
    "tool-workflow/agent-end",
    "tool-workflow/run-end",
)


def test_all_four_lifecycle_events_are_structured():
    """The whole defect in one assertion: each must be a STATE event, so its payload
    survives. A notice event carries one string and nothing else."""
    for name in WORKFLOW_EVENTS:
        assert name in runner._STATE_EVENTS, (
            f"{name} would fall to the notice path and lose its payload")


class _Note:
    """The shape `_project` reads: `method` plus a `payload` carrying the session-log
    envelope. A stand-in for the SDK's notification object, which needs no client."""

    def __init__(self, kind: str, data: dict) -> None:
        self.method = "session.event"
        self.payload = {"event": {"type": kind, "data": data}}


def test_the_payload_survives_the_projection():
    """End to end through `_project`: the run's name must come out as data, not as a
    line of text naming an event."""
    out = runner._project(_Note("tool-workflow/run-start",
                                {"runId": "r1", "name": "refactor"}))
    assert len(out) == 1, out
    assert out[0]["type"] == "state", out
    assert out[0]["event"] == "tool-workflow/run-start"
    assert out[0]["data"] == {"runId": "r1", "name": "refactor"}


def test_it_no_longer_arrives_as_an_opaque_notice():
    """The exact old behaviour, asserted so it cannot come back: a notice carrying the
    literal string "session.event tool-workflow/agent-start"."""
    out = runner._project(_Note("tool-workflow/agent-start",
                                {"runId": "r1", "seq": 0, "label": "scout"}))
    assert out[0]["type"] == "state", out
    assert "tool-workflow" not in out[0].get("text", "")


def test_the_agent_payload_keeps_label_and_seq():
    """`seq` is what pairs an agent's start with its OWN end. Without it the frontend
    would join by position, and agents finish out of order — so the wrong verdict would
    be attached to the wrong agent, which is worse than no verdict."""
    out = runner._project(_Note("tool-workflow/agent-start",
                                {"runId": "r1", "seq": 3, "label": "scout",
                                 "childId": "c1"}))
    assert out[0]["data"]["seq"] == 3
    assert out[0]["data"]["label"] == "scout"
    assert out[0]["data"]["childId"] == "c1"


def test_serve_has_an_arm_for_the_workflow_name():
    """The runner is only hop one of three. Without an arm here the state event reaches
    `serve.py` and is dropped, which is the failure mode the R6-DRIFT guard exists for
    — and this test is the same check applied to a name the guard was not told about."""
    src = inspect.getsource(serve.build_app)
    assert '_ev.startswith("tool-workflow/")' in src, (
        "the runner forwards these but serve.py has no arm, so they are dropped")
    assert 'event="workflow"' in src


def test_the_durable_run_is_written_only_when_it_ends():
    """A write per agent start would be a write storm, and a run restored from a reload
    could never finish — the turn that would have ended it is gone."""
    src = inspect.getsource(serve.build_app)
    assert '_ev == "tool-workflow/run-end"' in src
    # And the agents are accumulated, because a reload has no turn to fold four events
    # into: persisting only what `run-end` carries would store a run with no agents.
    assert '"tool-workflow/agent-start"' in src


def test_the_four_are_one_run_not_four():
    """`runId` is the only thing joining them, so the accumulation must key on it. A row
    per event would show one run four times."""
    src = inspect.getsource(serve.build_app)
    assert '_rid = str(_data.get("runId") or "")' in src
    assert 'r.get("runId") or "") == _rid' in src
