// How the run view reads a run record (IMP-8).
//
// The record already carries everything: `iteration`, `deadline`, `status` and
// `halt_reason` (runs.create / runs.set_status), and `/api/runs/{rid}` returns
// it whole. The surface showed only "iter N" and a bare status word, so a run
// that finished and a run that gave up looked the same and the remaining budget
// was invisible.

/** runs.MAX_ITERS. Mirrored the same way RESTARTABLE already is: the server
 *  sends `iter_ceiling` only for a RESTARTED run (it raises the cap by a grace
 *  amount), so a normal run's record has no ceiling and the server's own
 *  default is the only truthful one. */
export const STEP_CAP = 2000;

export interface RunBudget {
  iteration: number;
  ceiling: number;
  remainingSteps: number;
  /** Seconds until `deadline`, clamped at 0. 0 also means "no deadline". */
  remainingSeconds: number;
}

export function budget(
  run: { iteration?: number; iter_ceiling?: number; deadline?: number },
  now: number,
): RunBudget {
  const iteration = Math.max(0, Math.floor(run.iteration ?? 0));
  const raw = Math.floor(run.iter_ceiling ?? 0);
  const ceiling = raw > 0 ? raw : STEP_CAP;
  const remainingSeconds =
    run.deadline != null && Number.isFinite(run.deadline)
      ? Math.max(0, Math.floor(run.deadline - now))
      : 0;
  return {
    iteration,
    ceiling,
    remainingSteps: Math.max(0, ceiling - iteration),
    remainingSeconds,
  };
}

/** A duration as the run view states it: "42m", "8h 5m", "under a minute". */
export function remainingTime(s: number): string {
  if (!Number.isFinite(s) || s <= 0) return "none";
  if (s < 60) return "under a minute";
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// A terminal status in plain words. The status word alone does not tell the
// owner whether the run FINISHED or GAVE UP, which is the whole point of IMP-8.
const STOP: Record<string, string> = {
  done: "finished — the mission is complete",
  stalled: "stopped — it stopped making progress",
  frozen: "stopped — it repeated itself without progress",
  budget_exhausted: "stopped — it used up its step or time budget",
  stopped: "stopped — you stopped it",
  interrupted: "interrupted — the server went away mid-run",
  error: "stopped — a step failed",
};

/** Why the run is not running. The server's own `halt_reason` wins when it is
 *  there; otherwise a sentence for the status. */
export function stopSentence(status: string, haltReason?: string): string {
  if (haltReason && haltReason.trim()) return haltReason.trim();
  return STOP[status] ?? `stopped (${status || "unknown"})`;
}
