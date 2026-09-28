import { describe, expect, it } from "vitest";

import { foldSubagent, type Subagent } from "./subagents";

// The two lifecycle events carry DIFFERENT field sets, and the child's session
// id is the only key in both. Every test below pins a way that can go wrong.
const started = (child: string) => ({
  event: "subagent.started",
  data: { parentSessionId: "parent-1", childSessionId: child },
});

const finished = (child: string, extra: Record<string, unknown> = {}) => ({
  event: "subagent.finished",
  data: {
    provider: "spawn",
    agentId: "agent-1",
    parentSessionId: "parent-1",
    childSessionId: child,
    status: "ok",
    stopReason: "completed",
    ...extra,
  },
});

// mcode reports a task's ids FLAT, from the `details` of a `task` result, and
// its status IS the lifecycle step. Both shapes must fold into the same rows.
describe("foldSubagent — mcode's flat task shape", () => {
  it("starts a row from a task that is running", () => {
    const rows = foldSubagent([], {
      taskId: "bg_1", subSessionId: "s1", name: "explore", status: "started",
    });
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "s1", state: "running" });
  });

  it("closes the row when the task reports a terminal status", () => {
    let rows = foldSubagent([], {
      taskId: "bg_1", subSessionId: "s1", name: "explore", status: "started",
    });
    rows = foldSubagent(rows, {
      taskId: "bg_1", subSessionId: "s1", name: "explore", status: "succeeded",
    });
    expect(rows).toHaveLength(1);
    expect(rows[0].state).toBe("done");
  });

  it("treats mcode's queued and running as live, not as ends", () => {
    for (const status of ["queued", "running"]) {
      const rows = foldSubagent([], { taskId: "bg_1", subSessionId: "s", status });
      expect(rows[0].state).toBe("running");
    }
  });

  it("falls back to the task id when there is no sub-session id", () => {
    const rows = foldSubagent([], { taskId: "bg_9", status: "started" });
    expect(rows[0].id).toBe("bg_9");
  });

  // A bash call also reports a task_id, but it is not a subagent. The extractor
  // filters that on the tool NAME, so anything reaching here has a sub-session
  // id or was named as a task.
  it("ignores a payload with neither id", () => {
    expect(foldSubagent([], { status: "started" })).toEqual([]);
  });
});

describe("foldSubagent", () => {
  it("adds a running row when a child starts", () => {
    const rows = foldSubagent([], started("child-a"));
    expect(rows).toEqual([{ id: "child-a", state: "running" }]);
  });

  it("closes the row the finish event names, keeping one row per child", () => {
    let rows: Subagent[] = [];
    rows = foldSubagent(rows, started("child-a"));
    rows = foldSubagent(rows, finished("child-a"));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      id: "child-a",
      state: "done",
      provider: "spawn",
      status: "ok",
      stopReason: "completed",
    });
  });

  // A repeat start is the same child reported twice. Appending would render two
  // rows for one agent, which is the visible bug this guards.
  it("ignores a repeat start for a child already running", () => {
    let rows: Subagent[] = [];
    rows = foldSubagent(rows, started("child-a"));
    rows = foldSubagent(rows, started("child-a"));
    expect(rows).toHaveLength(1);
  });

  // A child spawned before this turn began, or a dropped start, must still be
  // shown: a finished agent that ran is better than a silent omission.
  it("adds a finished child that was never seen to start", () => {
    const rows = foldSubagent([], finished("child-b"));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "child-b", state: "done" });
  });

  it("keeps siblings apart and closes only the one named", () => {
    let rows: Subagent[] = [];
    rows = foldSubagent(rows, started("child-a"));
    rows = foldSubagent(rows, started("child-b"));
    rows = foldSubagent(rows, finished("child-a"));
    expect(rows.map((r) => [r.id, r.state])).toEqual([
      ["child-a", "done"],
      ["child-b", "running"],
    ]);
  });

  it("carries the child's closing message, clipped", () => {
    const rows = foldSubagent([], finished("child-a", {
      lastAssistantMessage: [
        { type: "text", text: "found " },
        { type: "text", text: "three call sites" },
      ],
    }));
    expect(rows[0].last).toBe("found three call sites");
  });

  it("clips a very long closing message rather than pasting it in", () => {
    const rows = foldSubagent([], finished("child-a", {
      lastAssistantMessage: [{ type: "text", text: "x".repeat(5000) }],
    }));
    expect(rows[0].last).toHaveLength(400);
  });

  it("records an error status as the backend's own word, not inferred", () => {
    const rows = foldSubagent([], finished("child-a", {
      status: "error",
      stopReason: "failed",
    }));
    expect(rows[0].status).toBe("error");
  });

  // A newer DSH naming a lifecycle step this build does not know must not
  // produce a blank row.
  it("ignores an unknown lifecycle event", () => {
    const rows = foldSubagent([], { event: "subagent.paused", data: { childSessionId: "c" } });
    expect(rows).toEqual([]);
  });

  it("ignores a payload with no child id rather than inventing a key", () => {
    expect(foldSubagent([], { event: "subagent.started", data: {} })).toEqual([]);
    expect(foldSubagent([], null)).toEqual([]);
    expect(foldSubagent([], "nonsense")).toEqual([]);
  });

  it("never mutates the list it was given", () => {
    const before: Subagent[] = [{ id: "child-a", state: "running" }];
    const copy = [...before];
    foldSubagent(before, finished("child-a"));
    expect(before).toEqual(copy);
  });
});


// R4-MCODE-2: the shape the SERVER actually produces.
//
// The block above folds mcode's payload FLAT, but the server wraps every
// subagent payload as `{event, data}` (serve.py, the `_ev.startswith("subagent")`
// arm) — for mcode too. So the flat branch was reachable only from these tests,
// and in the running product an mcode subagent rendered NOTHING: the wrapped
// branch read `data.childSessionId`, mcode's data has no such field, and the fold
// returned the list unchanged. These pin the shape that really arrives.
describe("foldSubagent — mcode WRAPPED by the server (the real wire shape)", () => {
  const wrapped = (extra: Record<string, unknown>) => ({
    event: "subagent",
    data: { taskId: "bg_1", subSessionId: "s1", name: "explore", status: "started", ...extra },
  });

  it("starts a row from the wrapped payload", () => {
    const rows = foldSubagent([], wrapped({}));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({ id: "s1", state: "running" });
  });

  it("closes the row on a wrapped terminal status", () => {
    let rows = foldSubagent([], wrapped({}));
    rows = foldSubagent(rows, wrapped({ status: "succeeded" }));
    expect(rows).toHaveLength(1);
    expect(rows[0].state).toBe("done");
  });

  it("keeps the agent name mcode sent, which was previously discarded", () => {
    const rows = foldSubagent([], wrapped({}));
    expect(rows[0].name).toBe("explore");
  });
});

// R4-MCODE-2: DSH carries a subagent's NAME, just not on the lifecycle pair.
//
// `subagent.started` is `{parentSessionId, childSessionId}` and nothing else, so
// a row could only ever read "subagent running". `subagent/descriptor` and
// `subagent/catalog` carry `label` keyed by the same child id, and both were
// forwarded by the server and then dropped by this fold.
describe("foldSubagent — the name DSH sends on subagent/catalog", () => {
  const start = { event: "subagent.started", data: { childSessionId: "c1" } };
  // Verified shape, not guessed: SubagentCatalogEvent is
  // {version, childId, childCreatedAt, mode, label?}
  // (packages/subagent/subagent/src/catalog.ts:24-32). It names the child
  // `childId`, where the lifecycle pair says `childSessionId` — reading only one
  // of the two is how this name was lost.
  const catalog = (extra: Record<string, unknown>) => ({
    event: "subagent/catalog",
    data: { version: 0, childId: "c1", childCreatedAt: 1, mode: "one-shot", ...extra },
  });

  it("applies a label arriving AFTER the start", () => {
    let rows = foldSubagent([], start);
    expect(rows[0].name).toBeUndefined();
    rows = foldSubagent(rows, catalog({ label: "scout" }));
    expect(rows[0].name).toBe("scout");
    expect(rows[0].state).toBe("running");
  });

  it("does not resurrect a row for a label naming an unknown child", () => {
    // A catalog entry for a child this turn never saw is not a reason to invent
    // a row — the fold must not append on a non-lifecycle event.
    expect(foldSubagent([], catalog({ childId: "ghost", label: "x" }))).toEqual([]);
  });

  it("still ignores a payload with neither id", () => {
    expect(foldSubagent([], { event: "subagent", data: { status: "started" } })).toEqual([]);
  });

  it("ignores subagent/descriptor, which carries no child id at all", () => {
    // Worth pinning: `subagent/descriptor` is NOT about a child. Its fields are
    // {mode, version, provider} + {label, agentProvider, agentModel, persona,
    // toolFilter} — it describes the subagent PROVIDER (dispositions.ts:92-95).
    // Its `label` is the provider's name, not a child's, so folding it onto a row
    // would be wrong rather than merely useless.
    const rows = foldSubagent([], start);
    expect(foldSubagent(rows, {
      event: "subagent/descriptor",
      data: { mode: "one-shot", version: 0, provider: "spawn", label: "spawn" },
    })).toEqual(rows);
  });
});
