// One goal, two backends, two shapes.
//
// WHY A NORMALISER RATHER THAN TWO RENDER PATHS. The two harnesses report a goal
// in genuinely different shapes, and neither is wrong:
//
//   DSH    a `goal/change` envelope
//          {operation, goal: {objective, phase, maxGoalRounds, blockedReason},
//           roundsStarted, createdAt, updatedAt}
//   mcode  the goal object itself, from an `update_goal`/`get_goal` result
//          {goalId, sessionId, objective, status, createdAt, updatedAt,
//           tokensUsed, timeUsedSeconds, tokenBudget}
//
// They disagree on the nesting, on what the phase is called (`phase` vs
// `status`), and on whether a round count exists at all (DSH has one; mcode does
// not, and reports tokens instead). Branching on the shape inside the component
// would put that knowledge in a render path and duplicate it for every new
// field. Normalising once here means the panel has ONE shape to draw, and a
// backend whose payload this build does not recognise still renders its
// objective rather than nothing.
//
// This is deliberately permissive: an unknown payload keeps whatever fields it
// can be understood to have, and the raw object stays available for the fields
// that have no common name.

export interface NormalGoal {
  /** The one field both backends agree on. Empty means this is not a goal. */
  objective: string;
  /** DSH's `phase`, or mcode's `status`. Lowercased for comparison. */
  phase: string;
  /** DSH's `roundsStarted`. Absent for mcode. */
  rounds: number | null;
  /** DSH's `maxGoalRounds`. Absent for mcode. */
  maxRounds: number | null;
  /** The message from a `blockedReason`, when one is present. */
  blocked: string;
  /** Token accounting, when the backend reported it (mcode does). */
  tokensUsed: number | null;
  tokenBudget: number | null;
}

const EMPTY: NormalGoal = {
  objective: "",
  phase: "",
  rounds: null,
  maxRounds: null,
  blocked: "",
  tokensUsed: null,
  tokenBudget: null,
};

const num = (v: unknown): number | null =>
  typeof v === "number" && Number.isFinite(v) ? v : null;

/**
 * Fold either backend's goal payload into one shape.
 *
 * Returns `null` when there is no objective, because a goal with no objective is
 * not a goal and rendering an empty panel would blank whatever was there.
 */
export function normaliseGoal(payload: unknown): NormalGoal | null {
  if (!payload || typeof payload !== "object") return null;
  const p = payload as Record<string, unknown>;

  // DSH nests the snapshot under `goal`; mcode IS the snapshot. Prefer the
  // nested one when it exists, so a payload that happens to have both is read
  // the way its own backend means it.
  const inner = p.goal;
  const snap = (
    inner && typeof inner === "object" ? inner : p
  ) as Record<string, unknown>;

  const objective = String(snap.objective ?? "").trim();
  if (!objective) return null;

  // DSH says `phase`, mcode says `status`. Both are lowercased because mcode's
  // enum is lowercase already and a backend that capitalises should not produce
  // a second, unmatched colour.
  const phase = String(snap.phase ?? snap.status ?? "").toLowerCase();

  // `blockedReason` is an object with a `message` on DSH, and absent on mcode.
  const br = snap.blockedReason;
  const blocked =
    br && typeof br === "object"
      ? String((br as { message?: unknown }).message ?? "")
      : typeof br === "string"
        ? br
        : "";

  return {
    objective,
    phase,
    rounds: num(p.roundsStarted ?? snap.roundsStarted),
    maxRounds: num(snap.maxGoalRounds),
    blocked,
    tokensUsed: num(snap.tokensUsed),
    tokenBudget: num(snap.tokenBudget),
  };
}

/** A goal's phase, as a colour class. Unknown phases read as neutral. */
export function goalTone(phase: string): string {
  if (phase === "complete" || phase === "completed") return "text-moss";
  if (phase === "active" || phase === "running") return "text-amber";
  if (phase === "blocked" || phase === "budget_limited" || phase === "usage_limited") {
    return "text-red";
  }
  return "text-secondary";
}

/** The phase as a sentence fragment, so `budget_limited` does not read as
 *  `budget limited` by accident and an unknown value is still shown. */
export function phaseLabel(phase: string): string {
  if (!phase) return "";
  return phase.replace(/_/g, " ");
}

/** `{...}` for the empty case is never reached; exported for the tests. */
export const EMPTY_GOAL = EMPTY;
