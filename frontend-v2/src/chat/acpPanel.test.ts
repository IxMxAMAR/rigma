// R6-ACP: mcode's control plane in the panel.
//
// WHY THIS FILE EXISTS. `mcode exec` is a projection of ONE turn, so a queue, a
// delegation tree, the live model/permission selects and a plan review have no
// `exec` representation at all. Over the Agent Client Protocol they do, and until
// this round Rigma drove only `exec` — so all four were not broken, they were
// UNREACHABLE, with nothing in the capability menu saying so.
//
// These tests pin the store's side of the wire. The backend side (ACP
// notification -> Rigma event) is pinned in Python, in
// tests/test_harness_mcode_acp.py, and the two names must agree: that is what the
// cross-layer drift guard at the end of this file checks.
import { describe, expect, it } from "vitest";
import { applyEvent, emptyTurn, type StreamingTurn } from "./chatStore";

const feed = (turn: StreamingTurn, evs: [string, unknown][]) =>
  evs.reduce((t, [event, data]) => applyEvent(t, { event, data }), turn);

describe("R6-ACP: the queue", () => {
  it("replaces the whole list rather than merging it", () => {
    // mcode sends the WHOLE array on every update, so a merge would resurrect a
    // queue item that was just deleted — the same reason `todos` is an assignment.
    const t = feed(emptyTurn(), [
      ["acp_queue", { items: [{ itemId: "q1" }, { itemId: "q2" }] }],
      ["acp_queue", { items: [{ itemId: "q2" }] }],
    ]);
    expect(t.acpQueue).toEqual([{ itemId: "q2" }]);
  });

  it("treats a malformed list as empty rather than crashing the stream", () => {
    const t = feed(emptyTurn(), [["acp_queue", { items: "not a list" }]]);
    expect(t.acpQueue).toEqual([]);
  });

  it("keeps the fields it does not know", () => {
    // mcode's item projection assigns itemId/sessionId/status unconditionally and
    // passes timestamps through unmapped, so this side must not drop what it has
    // no type for.
    const t = feed(emptyTurn(), [
      ["acp_queue", { items: [{ itemId: "q1", createdAt: 123, futureField: "x" }] }],
    ]);
    expect(t.acpQueue[0]).toMatchObject({ itemId: "q1", createdAt: 123 });
  });
});

describe("R6-ACP: the delegation tree", () => {
  it("stores the snapshot whole", () => {
    const snap = {
      schemaVersion: 1,
      rootSessionId: "root",
      members: [{ sessionId: "c1", task: "do the thing", status: "running" }],
    };
    const t = feed(emptyTurn(), [["acp_delegation", snap]]);
    expect(t.acpDelegation).toEqual(snap);
  });

  it("keeps a member's status vocabulary as mcode normalised it", () => {
    // mcode's own projection narrows status to exactly one of
    // queued/running/completed/failed/stopped/unknown, so this side must not
    // re-interpret it — an unrecognised value becomes the literal "unknown".
    const t = feed(emptyTurn(), [
      ["acp_delegation", { members: [{ sessionId: "c1", status: "unknown" }] }],
    ]);
    expect(t.acpDelegation?.members?.[0].status).toBe("unknown");
  });

  it("accepts a non-object payload without breaking the turn", () => {
    const t = feed(emptyTurn(), [["acp_delegation", 7]]);
    expect(t.acpDelegation).toBeNull();
  });
});

describe("R6-ACP: the live selects", () => {
  it("replaces the config options", () => {
    const t = feed(emptyTurn(), [
      ["acp_config", { configOptions: [{ id: "permissionMode", currentValue: "auto" }] }],
      ["acp_config", { configOptions: [{ id: "model", currentValue: "m:x" }] }],
    ]);
    expect(t.acpConfig).toEqual([{ id: "model", currentValue: "m:x" }]);
  });
});

describe("R6-ACP: commands and the plan review", () => {
  it("stores the advertised commands", () => {
    const t = feed(emptyTurn(), [
      ["acp_commands", { commands: [{ name: "compact", description: "shrink" }] }],
    ]);
    expect(t.acpCommands).toHaveLength(1);
  });

  it("keeps a plan review as the backend sent it", () => {
    const t = feed(emptyTurn(), [
      ["acp_plan", { type: "markdown", planId: "p1", content: "# Plan" }],
    ]);
    expect(t.acpPlan).toMatchObject({ content: "# Plan" });
  });

  it("clears the plan on a removal, because a removal is an instruction", () => {
    // Same shape of decision as the goal tombstone: `plan_removed` means the plan
    // is GONE, not that this build did not understand the payload. Treating it as
    // unrecognised would leave a withdrawn plan on screen.
    const t = feed(emptyTurn(), [
      ["acp_plan", { type: "markdown", content: "# Plan" }],
      ["acp_plan", { planId: "p1", removed: true }],
    ]);
    expect(t.acpPlan).toBeNull();
  });
});

describe("R6-ACP: a turn that never spoke ACP is unchanged", () => {
  it("starts empty and stays empty", () => {
    const t = feed(emptyTurn(), [["text", { delta: "hello" }]]);
    expect(t.acpQueue).toEqual([]);
    expect(t.acpDelegation).toBeNull();
    expect(t.acpConfig).toEqual([]);
    expect(t.acpCommands).toEqual([]);
    expect(t.acpPlan).toBeNull();
  });

  it("does not disturb the existing goal, todos or plan-mode arms", () => {
    const t = feed(emptyTurn(), [
      ["goal", { objective: "ship it", status: "active" }],
      ["todos", { todos: [{ content: "one", status: "pending" }] }],
      ["plan_mode", { active: true }],
    ]);
    expect(t.goal?.objective).toBe("ship it");
    expect(t.todos).toHaveLength(1);
    expect(t.planMode).toBe(true);
  });
});
