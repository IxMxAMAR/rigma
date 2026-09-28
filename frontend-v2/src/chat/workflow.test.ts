// R6-WORKFLOW: programmatic tool calling, which reached the UI as a bare string.
//
// The defect these tests guard is specific and was an ACCIDENT: `tool-workflow/*` was
// not in the runner's `_STATE_EVENTS`, so it fell to the notice path — which matches on
// the substring "tool", and "tool-workflow" contains it. Four events therefore reached
// the UI as the literal string "session.event tool-workflow/agent-start", and the run's
// name, each agent's label and phase, each agent's outcome and the run's stop reason
// were all discarded.
//
// The fold is where the four are joined, so these pin the joining rules rather than the
// plumbing.
import { describe, expect, it } from "vitest";

import { applyEvent, durableFromEvent, emptyTurn, foldWorkflow } from "./chatStore";
import type { WorkflowRun } from "../lib/api";

/** One `tool-workflow/*` SSE payload, in the shape serve.py sends it. */
function ev(kind: string, data: Record<string, unknown>) {
  return { event: "workflow", data: { event: kind, data } };
}

const runs = (): WorkflowRun[] => [];

describe("R6-WORKFLOW: foldWorkflow joins four events into one run", () => {
  it("opens a run on run-start and names it", () => {
    const out = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "refactor" }).data);
    expect(out).toHaveLength(1);
    expect(out[0].runId).toBe("r1");
    expect(out[0].name).toBe("refactor");
    expect(out[0].done).toBe(false);
    expect(out[0].agents).toEqual([]);
  });

  it("attaches an agent to the run named by runId, not to a new one", () => {
    let out = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "refactor" }).data);
    out = foldWorkflow(out, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "scout", childId: "c1", phase: "explore" }).data);
    expect(out).toHaveLength(1);
    expect(out[0].agents).toHaveLength(1);
    expect(out[0].agents[0]).toMatchObject({
      seq: 0, label: "scout", childId: "c1", phase: "explore", outcome: "",
    });
  });

  it("pairs agent-end with its OWN start by seq, not by order", () => {
    // The pairing is by `seq` because agents finish out of order. Joining by position
    // would give the wrong verdict to the wrong agent, which is worse than no verdict.
    let out = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "x" }).data);
    out = foldWorkflow(out, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "first" }).data);
    out = foldWorkflow(out, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 1, label: "second" }).data);
    // The SECOND agent finishes first.
    out = foldWorkflow(out, ev("tool-workflow/agent-end",
      { runId: "r1", seq: 1, outcome: "failed" }).data);
    expect(out[0].agents[0].outcome).toBe("");
    expect(out[0].agents[1].outcome).toBe("failed");
  });

  it("marks done and records the stop reason on run-end, keeping the agents", () => {
    let out = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "x" }).data);
    out = foldWorkflow(out, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "scout" }).data);
    out = foldWorkflow(out, ev("tool-workflow/run-end",
      { runId: "r1", stopReason: "complete" }).data);
    expect(out[0].done).toBe(true);
    expect(out[0].stopReason).toBe("complete");
    // A run that ended still has its agents: deleting them on completion would erase
    // the thing the panel exists to show.
    expect(out[0].agents).toHaveLength(1);
  });

  it("OPENS a run for an agent-start whose run-start was never seen", () => {
    // The runner only began forwarding these in R6-WORKFLOW, and a resumed session
    // carries a run's later events without its first. Dropping them would hide real
    // activity to avoid drawing a header.
    const out = foldWorkflow(runs(), ev("tool-workflow/agent-start",
      { runId: "orphan", seq: 0, label: "scout" }).data);
    expect(out).toHaveLength(1);
    expect(out[0].name).toBe("");
    expect(out[0].agents).toHaveLength(1);
  });

  it("still records an outcome with no matching start", () => {
    const out = foldWorkflow(runs(), ev("tool-workflow/agent-end",
      { runId: "r1", seq: 7, outcome: "failed" }).data);
    expect(out[0].agents).toHaveLength(1);
    expect(out[0].agents[0].outcome).toBe("failed");
  });

  it("refuses an event with no runId rather than inventing a run for it", () => {
    // Nothing can join it to anything, so a run would have to be fabricated.
    expect(foldWorkflow(runs(), ev("tool-workflow/agent-start",
      { seq: 0, label: "x" }).data)).toHaveLength(0);
  });

  it("keeps two runs apart", () => {
    let out = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "one" }).data);
    out = foldWorkflow(out, ev("tool-workflow/run-start",
      { runId: "r2", name: "two" }).data);
    expect(out).toHaveLength(2);
    // Newest first: a run is usually read while it is happening.
    expect(out[0].name).toBe("two");
    expect(out[1].name).toBe("one");
  });

  it("never mutates the array it was handed", () => {
    // The reducer returns a new turn, so mutating the old one would change state that
    // has already been rendered and defeat the store's identity checks.
    const before = foldWorkflow(runs(), ev("tool-workflow/run-start",
      { runId: "r1", name: "one" }).data);
    const snapshot = JSON.stringify(before);
    foldWorkflow(before, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "scout" }).data);
    expect(JSON.stringify(before)).toBe(snapshot);
  });
});

describe("R6-WORKFLOW: the store's workflow arm", () => {
  it("folds through applyEvent onto the live turn", () => {
    let turn = emptyTurn();
    turn = applyEvent(turn, ev("tool-workflow/run-start",
      { runId: "r1", name: "refactor" }));
    turn = applyEvent(turn, ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "scout" }));
    expect(turn.workflow).toHaveLength(1);
    expect(turn.workflow[0].name).toBe("refactor");
    expect(turn.workflow[0].agents[0].label).toBe("scout");
  });

  it("starts with no runs, so an empty turn does not render the block", () => {
    expect(emptyTurn().workflow).toEqual([]);
  });

  it("persists only a FINISHED run", () => {
    // Mid-flight persistence would write once per agent start, and a run restored from
    // a reload could never finish — the turn that would have ended it is gone.
    expect(durableFromEvent(ev("tool-workflow/run-start",
      { runId: "r1", name: "x" }) as never)).toBeNull();
    expect(durableFromEvent(ev("tool-workflow/agent-start",
      { runId: "r1", seq: 0, label: "s" }) as never)).toBeNull();
    const done = durableFromEvent(ev("tool-workflow/run-end",
      { runId: "r1", stopReason: "complete" }) as never);
    expect(done?.workflow).toHaveLength(1);
    expect(done?.workflow?.[0].done).toBe(true);
    expect(done?.workflow?.[0].stopReason).toBe("complete");
  });
});
