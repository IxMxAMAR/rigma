// What the agent is working toward, what it plans to do, and who is helping.
//
// WHY THIS IS A COMPONENT AND NOT MORE LINES IN Transcript.tsx. The transcript
// renders a turn's prose and its tool chips. These are a different kind of thing:
// they are the agent's OWN state, reported by the backend rather than written by
// the model, and they outlive the turn that produced them. Keeping them in one
// component also means the whole block can be absent when a backend reports none
// of it — which is the case for the built-in loop, whose plan lives in the
// Autonomous surface instead.
//
// The four facts and their shapes come from the backend, not from inference:
//   goal       — the backend's goal snapshot (phase, objective, revision, rounds)
//   todos      — the LATEST whole-list snapshot; `todo_write` replaces, never merges
//   planMode   — the agent is proposing rather than doing
//   subagents  — child agents this turn started, folded from their lifecycle
//   usage      — the step's token accounting, when the backend reported any

import type { Subagent } from "./subagents";

/** Amber = in flight, moss = settled well, muted = not started, red = failed. */
function todoTone(status: string): string {
  if (status === "completed") return "text-moss";
  if (status === "in_progress") return "text-amber";
  if (status === "cancelled") return "text-muted line-through";
  return "text-muted";
}

function todoGlyph(status: string): string {
  if (status === "completed") return "✓";
  if (status === "in_progress") return "◌";
  if (status === "cancelled") return "×";
  return "○";
}

/** A goal's phase drives its colour, and every phase has its own word. */
function goalTone(phase: string): string {
  if (phase === "complete") return "text-moss";
  if (phase === "active") return "text-amber";
  if (phase === "blocked") return "text-red";
  return "text-secondary";
}

function SubagentRow({ row }: { row: Subagent }) {
  const tone =
    row.state === "running"
      ? "text-amber"
      : row.status === "error"
        ? "text-red"
        : "text-moss";
  const glyph = row.state === "running" ? "◌" : row.status === "error" ? "✕" : "✓";
  const title = [
    row.provider ? `provider ${row.provider}` : "",
    row.stopReason ? `stopped: ${row.stopReason}` : "",
  ]
    .filter(Boolean)
    .join(" · ");
  return (
    <li className="flex items-start gap-2 text-[12px]">
      <span className={`font-mono mt-px ${tone}`} aria-hidden="true">{glyph}</span>
      <span className="flex-1 min-w-0">
        <span className="font-mono text-[11.5px] text-secondary">
          {row.state === "running" ? "subagent running" : "subagent finished"}
        </span>
        {title && <span className="font-mono text-[10.5px] text-muted"> · {title}</span>}
        {row.last && (
          <span className="block text-[12px] text-muted truncate" title={row.last}>
            {row.last}
          </span>
        )}
      </span>
    </li>
  );
}

export default function AgentState({
  goal,
  todos,
  planMode,
  subagents,
  usage,
}: {
  goal: Record<string, unknown> | null;
  todos: { content: string; status: string }[];
  planMode: boolean;
  subagents: Subagent[];
  usage: Record<string, unknown> | null;
}) {
  // The `goal/change` envelope is {operation, goal: <snapshot>, roundsStarted}.
  // Read through the envelope rather than assuming a bare snapshot, so a backend
  // that adds a field does not silently blank the panel.
  const snap = (goal?.goal ?? null) as Record<string, unknown> | null;
  const hasGoal =
    snap !== null && typeof snap.objective === "string" && snap.objective !== "";
  const hasUsage = usage !== null && Object.keys(usage).length > 0;
  if (!hasGoal && todos.length === 0 && !planMode && subagents.length === 0 && !hasUsage) {
    return null;
  }

  const phase = String((snap?.phase as string) ?? "");
  const rounds = goal?.roundsStarted;
  const maxRounds = snap?.maxGoalRounds;
  const blocked = snap?.blockedReason as { message?: unknown } | undefined;

  return (
    <div className="rounded-md bg-panel px-3 py-2 flex flex-col gap-2">
      {hasGoal && (
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em]">
              goal
            </span>
            {phase && (
              <span className={`font-mono text-[10.5px] ${goalTone(phase)}`}>{phase}</span>
            )}
            {rounds != null && (
              <span className="font-mono text-[10.5px] text-muted">
                round {String(rounds)}
                {maxRounds != null ? ` of ${String(maxRounds)}` : ""}
              </span>
            )}
          </div>
          <p className="text-[13px] text-primary">{String(snap?.objective ?? "")}</p>
          {blocked?.message != null && (
            <p className="text-[12px] text-red mt-0.5">{String(blocked.message)}</p>
          )}
        </div>
      )}

      {planMode && (
        <p className="font-mono text-[11px] text-amber">
          plan mode — proposing, not changing anything yet
        </p>
      )}

      {todos.length > 0 && (
        <div>
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-1">
            todos
          </div>
          <ul className="flex flex-col gap-0.5">
            {todos.map((t, i) => (
              <li key={i} className="flex items-start gap-2 text-[12.5px]">
                <span className={`font-mono mt-px ${todoTone(t.status)}`} aria-hidden="true">
                  {todoGlyph(t.status)}
                </span>
                <span className={t.status === "completed" ? "text-muted" : "text-primary"}>
                  {t.content}
                </span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {subagents.length > 0 && (
        <div>
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-1">
            subagents
          </div>
          <ul className="flex flex-col gap-0.5">
            {subagents.map((s) => (
              <SubagentRow key={s.id} row={s} />
            ))}
          </ul>
        </div>
      )}

      {hasUsage && (
        <p className="font-mono text-[10.5px] text-muted">
          {Object.entries(usage!)
            .filter(([, v]) => typeof v === "number")
            .map(([k, v]) => `${k} ${v}`)
            .join(" · ")}
        </p>
      )}
    </div>
  );
}
