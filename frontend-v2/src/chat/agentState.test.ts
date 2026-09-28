import { describe, expect, it } from "vitest";

import { applyEvent, emptyTurn } from "./chatStore";

// The capability bridge. Every event below is one the SERVER can now emit and
// that the store's `default` arm would otherwise swallow in silence — which is
// the failure mode this whole file exists to prevent, because an unhandled SSE
// name produces no error anywhere; the fact simply never appears.
const ev = (event: string, data: unknown) => ({ event, data });

describe("the backend's structured agent state reaches the turn", () => {
  it("keeps a goal whole, because the phase is the point", () => {
    const t = applyEvent(emptyTurn(), ev("goal", {
      operation: "create",
      goal: {
        id: "g1",
        revision: 3,
        objective: "make the suite green",
        phase: "active",
        maxGoalRounds: 60,
      },
      roundsStarted: 4,
    }));
    // The `goal/change` envelope is {operation, goal: <snapshot>, roundsStarted}.
    // The snapshot is where the phase and objective live.
    expect(t.goal).toMatchObject({ operation: "create" });
    expect(t.goal?.goal).toMatchObject({
      objective: "make the suite green",
      phase: "active",
    });
    // The round count rides the ENVELOPE, not the snapshot. Losing it would
    // make "round 4 of 60" impossible to render.
    expect(t.goal?.roundsStarted).toBe(4);
  });

  // The snapshot is nested under `goal` in the change envelope. A reducer that
  // assumed a bare snapshot would set `goal` to the envelope and the panel would
  // render nothing — which is exactly the bug this pins.
  it("keeps the goal envelope and its nested snapshot together", () => {
    const t = applyEvent(emptyTurn(), ev("goal", {
      operation: "pause",
      goal: { objective: "x", phase: "paused", maxGoalRounds: 10 },
      roundsStarted: 1,
    }));
    expect(t.goal).toHaveProperty("operation", "pause");
    expect(t.goal).toHaveProperty("goal");
    expect((t.goal?.goal as Record<string, unknown>).maxGoalRounds).toBe(10);
  });

  it("replaces the todo list rather than merging it", () => {
    let t = applyEvent(emptyTurn(), ev("todos", {
      todos: [{ content: "first", status: "completed" }],
    }));
    t = applyEvent(t, ev("todos", { todos: [{ content: "second", status: "in_progress" }] }));
    // `todo_write` sends the ENTIRE list every call. A merge here would show
    // finished work that the model has already dropped.
    expect(t.todos).toEqual([{ content: "second", status: "in_progress" }]);
  });

  it("treats a malformed todo list as empty rather than throwing", () => {
    expect(applyEvent(emptyTurn(), ev("todos", { todos: "nope" })).todos).toEqual([]);
    expect(applyEvent(emptyTurn(), ev("todos", {})).todos).toEqual([]);
  });

  it("follows plan mode both ways", () => {
    let t = applyEvent(emptyTurn(), ev("plan_mode", { active: true }));
    expect(t.planMode).toBe(true);
    t = applyEvent(t, ev("plan_mode", { active: false }));
    expect(t.planMode).toBe(false);
  });

  it("folds a subagent start and finish onto one row", () => {
    let t = applyEvent(emptyTurn(), ev("subagent", {
      event: "subagent.started",
      data: { parentSessionId: "p", childSessionId: "c" },
    }));
    expect(t.subagents).toHaveLength(1);
    expect(t.subagents[0].state).toBe("running");
    t = applyEvent(t, ev("subagent", {
      event: "subagent.finished",
      data: { childSessionId: "c", status: "ok", provider: "spawn", stopReason: "completed" },
    }));
    expect(t.subagents).toHaveLength(1);
    expect(t.subagents[0].state).toBe("done");
  });

  it("records the step's token accounting", () => {
    const t = applyEvent(emptyTurn(), ev("usage", { inputTokens: 10, outputTokens: 4 }));
    expect(t.usage).toMatchObject({ inputTokens: 10, outputTokens: 4 });
  });
});

describe("events the server emitted and the store used to drop", () => {
  it("shows a compaction heartbeat instead of a frozen screen", () => {
    const t = applyEvent(emptyTurn(), ev("housekeeping", { note: "compacting the context" }));
    expect(t.housekeeping).toBe("compacting the context");
  });

  it("reports observation masking", () => {
    expect(applyEvent(emptyTurn(), ev("masked", { masked: 7 })).masked).toBe(7);
  });

  it("reports what compaction archived", () => {
    expect(applyEvent(emptyTurn(), ev("compacted", { archived: 12 })).compacted).toBe(12);
  });

  it("reports a prompt waiting behind the running reply", () => {
    expect(applyEvent(emptyTurn(), ev("info", { queued: 2 })).queued).toBe(2);
  });

  it("takes the server's retitle, so the rail need not wait for a reload", () => {
    expect(applyEvent(emptyTurn(), ev("meta", { title: "Fix the fit bug" })).title)
      .toBe("Fix the fit bug");
  });

  // `meta` also carries ctx/timings. Those must NOT become a title or text —
  // the ctx meter is computed from persisted stats, and `chatStore.test.ts`
  // already pins that meta never sets the harness badge.
  it("ignores meta that carries no title", () => {
    const t = applyEvent(emptyTurn(), ev("meta", { ctx: 65536, predicted_per_second: 12 }));
    expect(t.title).toBe("");
    expect(t.text).toBe("");
  });
});

describe("the default arm still behaves", () => {
  it("appends an unlabelled delta to the reply", () => {
    expect(applyEvent(emptyTurn(), ev("message", { delta: "hi" })).text).toBe("hi");
  });

  it("returns the turn unchanged for a payload with no delta", () => {
    const t = emptyTurn();
    expect(applyEvent(t, ev("something_new", { whatever: 1 }))).toBe(t);
  });
});
