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
import { goalTone, phaseLabel, type NormalGoal } from "./goal";
import {
  outcomeLabel,
  outcomeTone,
  sandboxLabel,
  sandboxTone,
  type Governance,
} from "./governance";

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

/** What the harness was allowed to do, and what it asked for.
 *
 *  DISPLAY-ONLY, and that is the transport's property rather than a shortcut
 *  taken here: DSH's SDK wire exposes exactly initialize / session/prompt /
 *  shutdown, so there is no way to ANSWER an approval over it — approval is
 *  decided by the policy engine. A clickable "Allow?" would therefore be a lie
 *  about what this connection can do, so there is none.
 *
 *  What it is instead is the one thing a reader currently cannot see: that an
 *  agent asked for permission, what it was told, and how confined it was. DSH
 *  marks all of these `log-only` — durable, replayable, never in the model
 *  transcript — which is exactly why they belong beside the turn, not in it.
 */
function GovernanceBlock({ gov }: { gov: Governance }) {
  const hasTrail = gov.approvals.length > 0;
  if (!hasTrail && !gov.sandbox && !gov.preset) return null;
  return (
    <div className="border-t border-line pt-1.5 flex flex-col gap-1">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em]">
          permitted
        </span>
        {gov.sandbox && (
          <span
            className={`font-mono text-[10.5px] ${sandboxTone(gov.sandbox)}`}
            title="DSH sandbox/mode — the confinement actually in force"
          >
            {sandboxLabel(gov.sandbox)}
          </span>
        )}
        {gov.preset && (
          <span className="font-mono text-[10.5px] text-muted">
            preset {gov.preset}
          </span>
        )}
      </div>
      {hasTrail && (
        <ul className="flex flex-col gap-0.5">
          {gov.approvals.map((a, i) => (
            <li key={`${a.id}-${String(i)}`} className="text-[11.5px] flex items-start gap-1.5">
              <span
                className={`font-mono mt-px ${
                  a.outcome ? outcomeTone(a.outcome) : "text-amber"
                }`}
                aria-hidden="true"
              >
                {a.outcome ? (a.outcome === "allowed-once" ? "✓" : "✕") : "?"}
              </span>
              <span className="flex-1 min-w-0">
                <span className="font-mono text-secondary">{a.toolName || a.kind}</span>
                {a.outcome ? (
                  <span className={`font-mono ${outcomeTone(a.outcome)}`}>
                    {" "}— {outcomeLabel(a.outcome)}
                  </span>
                ) : (
                  // An ASK with no decision yet. DSH appends exactly one
                  // `decided` per `asked`, so this is genuinely pending — it is
                  // not a lost answer.
                  <span className="font-mono text-amber"> — asked, not yet decided</span>
                )}
                {a.reason && (
                  <span className="block text-muted break-words">{a.reason}</span>
                )}
                {a.policy && (
                  <span className="block text-muted">
                    policy in force: <span className="font-mono">{a.policy}</span>
                  </span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

export default function AgentState({
  goal,
  todos,
  planMode,
  subagents,
  usage,
  governance,
}: {
  goal: NormalGoal | null;
  todos: { content: string; status: string }[];
  planMode: boolean;
  subagents: Subagent[];
  usage: Record<string, unknown> | null;
  governance: Governance;
}) {
  // Already normalised by the store, so this draws ONE shape whatever the
  // backend was: DSH's nested `goal/change` envelope and mcode's flat goal
  // object both arrive here as a NormalGoal.
  const hasGoal = goal !== null && goal.objective !== "";
  const hasUsage = usage !== null && Object.keys(usage).length > 0;
  const hasGov =
    governance.approvals.length > 0 || !!governance.sandbox || !!governance.preset;
  if (
    !hasGoal && todos.length === 0 && !planMode && subagents.length === 0 &&
    !hasUsage && !hasGov
  ) {
    return null;
  }

  const phase = goal?.phase ?? "";
  const rounds = goal?.rounds ?? null;
  const maxRounds = goal?.maxRounds ?? null;
  const blocked = goal?.blocked ?? "";

  return (
    <div className="rounded-md bg-panel px-3 py-2 flex flex-col gap-2">
      {hasGoal && (
        <div>
          <div className="flex items-center gap-2 mb-0.5">
            <span className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em]">
              goal
            </span>
            {phase && (
              <span className={`font-mono text-[10.5px] ${goalTone(phase)}`}>
                {phaseLabel(phase)}
              </span>
            )}
            {rounds != null && (
              <span className="font-mono text-[10.5px] text-muted">
                round {String(rounds)}
                {maxRounds != null ? ` of ${String(maxRounds)}` : ""}
              </span>
            )}
            {/* mcode reports tokens where DSH reports rounds. Shown only when
                a budget exists, because a bare count with nothing to compare it
                to is not information. */}
            {goal?.tokenBudget != null && (
              <span className="font-mono text-[10.5px] text-muted">
                {goal.tokensUsed ?? 0} of {goal.tokenBudget} tokens
              </span>
            )}
          </div>
          <p className="text-[13px] text-primary">{goal?.objective ?? ""}</p>
          {blocked && <p className="text-[12px] text-red mt-0.5">{blocked}</p>}
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

      {/* Last, and separated by a rule: this is about the CONNECTION's
          permissions rather than about the work, and putting it above the goal
          would make confinement read as the turn's subject. */}
      <GovernanceBlock gov={governance} />
    </div>
  );
}
