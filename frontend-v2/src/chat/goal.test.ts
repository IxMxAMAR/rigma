import { describe, expect, it } from "vitest";

import { goalTone, isGoalCleared, normaliseGoal, phaseLabel } from "./goal";

// Two backends, two shapes, one panel. Every case below is a shape that was
// actually observed: DSH's `goal/change` envelope and mcode's flat goal object
// from an `update_goal`/`get_goal` result.
describe("normaliseGoal", () => {
  it("reads DSH's nested envelope", () => {
    const g = normaliseGoal({
      operation: "create",
      goal: { objective: "ship it", phase: "active", maxGoalRounds: 60 },
      roundsStarted: 4,
    });
    expect(g).toMatchObject({
      objective: "ship it",
      phase: "active",
      rounds: 4,
      maxRounds: 60,
    });
  });

  // mcode says `status` where DSH says `phase`, and reports tokens where DSH
  // reports rounds. The panel must not care which backend it is looking at.
  it("reads mcode's flat goal object", () => {
    const g = normaliseGoal({
      goalId: "g1",
      sessionId: "s1",
      objective: "ship it",
      status: "active",
      tokensUsed: 120,
      tokenBudget: 5000,
      createdAt: 1,
      updatedAt: 2,
    });
    expect(g).toMatchObject({
      objective: "ship it",
      phase: "active",
      tokensUsed: 120,
      tokenBudget: 5000,
    });
    // mcode has no round concept at all; null rather than 0, because 0 would
    // render as "round 0 of ..." which is a claim it did not make.
    expect(g?.rounds).toBeNull();
    expect(g?.maxRounds).toBeNull();
  });

  it("lowercases the phase so a capitalised backend still matches", () => {
    expect(normaliseGoal({ objective: "x", phase: "Active" })?.phase).toBe("active");
  });

  it("reads a blocked reason given as an object", () => {
    const g = normaliseGoal({
      goal: { objective: "x", phase: "blocked", blockedReason: { message: "no GPU" } },
    });
    expect(g?.blocked).toBe("no GPU");
  });

  it("reads a blocked reason given as a plain string", () => {
    expect(normaliseGoal({ objective: "x", blockedReason: "no GPU" })?.blocked).toBe("no GPU");
  });

  // A goal with no objective is not a goal. Rendering one would blank a panel
  // that was showing something real.
  it("returns null for a payload that is not a goal", () => {
    expect(normaliseGoal(null)).toBeNull();
    expect(normaliseGoal("nope")).toBeNull();
    expect(normaliseGoal({})).toBeNull();
    expect(normaliseGoal({ goal: null })).toBeNull();
    expect(normaliseGoal({ error: "cannot update goal" })).toBeNull();
    expect(normaliseGoal({ objective: "   " })).toBeNull();
  });

  it("prefers the nested snapshot when a payload has both", () => {
    const g = normaliseGoal({
      objective: "outer",
      goal: { objective: "inner", phase: "active" },
    });
    expect(g?.objective).toBe("inner");
  });

  it("ignores a non-numeric round count rather than rendering NaN", () => {
    expect(normaliseGoal({ objective: "x", roundsStarted: "four" })?.rounds).toBeNull();
    expect(normaliseGoal({ objective: "x", goal: { objective: "x", maxGoalRounds: null } })
      ?.maxRounds).toBeNull();
  });
});

describe("goalTone", () => {
  it("marks a finished goal as good and a blocked one as bad", () => {
    expect(goalTone("complete")).toContain("moss");
    expect(goalTone("blocked")).toContain("red");
    expect(goalTone("active")).toContain("amber");
  });

  // mcode's two extra terminal states are both failures of the budget, not
  // successes, and must not read as neutral.
  it("treats the budget-limited states as bad", () => {
    expect(goalTone("budget_limited")).toContain("red");
    expect(goalTone("usage_limited")).toContain("red");
  });

  it("is neutral for a phase it does not know", () => {
    expect(goalTone("something_new")).toContain("secondary");
    expect(goalTone("")).toContain("secondary");
  });
});

describe("phaseLabel", () => {
  it("makes an underscored phase readable", () => {
    expect(phaseLabel("budget_limited")).toBe("budget limited");
  });

  it("shows an unknown phase rather than hiding it", () => {
    expect(phaseLabel("something_new")).toBe("something new");
  });

  it("says nothing for no phase", () => {
    expect(phaseLabel("")).toBe("");
  });
});

// R4-GOAL-1: a CLEAR is an instruction, and used to be indistinguishable from
// "this payload is not a goal".
//
// `goal/change` is a union. An ordinary change is `{operation, goal: {...}}`, but
// a clear is `{operation: "clear", cleared: true, clearedAt}` with NO `goal` key.
// Both folded to `null`, and the store's `null` case deliberately keeps whatever
// the panel was showing - so clearing a goal left the old objective on screen
// forever, with nothing saying it was stale.
describe("isGoalCleared", () => {
  it("recognises DSH's clear tombstone", () => {
    expect(isGoalCleared({ operation: "clear", cleared: true, clearedAt: "t" })).toBe(true);
  });

  it("accepts either signal alone, since a backend may send one", () => {
    expect(isGoalCleared({ operation: "clear" })).toBe(true);
    expect(isGoalCleared({ cleared: true })).toBe(true);
    expect(isGoalCleared({ operation: "CLEAR" })).toBe(true);
  });

  it("does NOT treat an ordinary change as a clear", () => {
    expect(isGoalCleared({
      operation: "update", goal: { objective: "ship it", phase: "active" },
    })).toBe(false);
  });

  it("does NOT treat 'no goal' as a clear", () => {
    // `get_goal` with no goal answers `{goal: null}`, and an error answers
    // `{error}`. Neither means the goal was cleared, so neither may blank the
    // panel - that is the whole distinction this function exists to draw.
    expect(isGoalCleared({ goal: null })).toBe(false);
    expect(isGoalCleared({ error: "nope" })).toBe(false);
    expect(isGoalCleared({})).toBe(false);
    expect(isGoalCleared(null)).toBe(false);
    expect(isGoalCleared("clear")).toBe(false);
  });

  it("still normalises a clear to null, so the two agree", () => {
    // The tombstone has no objective, so the normaliser refuses it; the store
    // checks isGoalCleared FIRST and only then falls back to this.
    expect(normaliseGoal({ operation: "clear", cleared: true })).toBeNull();
  });
});
