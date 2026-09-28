// What the agent is working toward, what it plans to do, and who is helping.
//
// WHY THIS IS A COMPONENT AND NOT MORE LINES IN Transcript.tsx. The transcript
// renders a turn's prose and its tool chips. These are a different kind of thing:
// they are the agent's OWN state, reported by the backend rather than written by
// the model. Keeping them in one component also means the whole block can be
// absent when a backend reports none of it — which is the case for the built-in
// loop, whose plan lives in the Autonomous surface instead.
//
// R4-PERSIST: this comment used to say they "outlive the turn that produced them",
// and that is FALSE as rendered. `AgentState` is mounted inside `LiveTurn`, and
// `chatStore` drops the stream when the turn ends, so the block disappears on
// reload even though the agent's own state is durable. The claim is removed rather
// than left standing, because a comment that reads as a guarantee is how a real
// gap survives a review — this one survived several.
//
// The four facts and their shapes come from the backend, not from inference:
//   goal       — the backend's goal snapshot (phase, objective, revision, rounds)
//   todos      — the LATEST whole-list snapshot; `todo_write` replaces, never merges
//   planMode   — the agent is proposing rather than doing
//   subagents  — child agents this turn started, folded from their lifecycle
//   usage      — the step's token accounting, when the backend reported any

import type { Subagent } from "./subagents";
import { goalTone, phaseLabel, type NormalGoal } from "./goal";
import type {
  AcpCommand,
  AcpConfigOption,
  AcpDelegation,
  AcpQueueItem,
} from "./chatStore";
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
          {/* R4-MCODE-2: the child's NAME when the backend supplied one. DSH's
              lifecycle pair carries no name, so before this a row could only say
              "subagent running" — unreadable as soon as two ran at once. The name
              arrives on `subagent/descriptor`/`catalog` (`label`) and on mcode's
              task details (`agent_name`), and both were previously discarded. */}
          {row.name && <span className="text-primary">{row.name}</span>}
          {row.name && " — "}
          {row.state === "running" ? "running" : "finished"}
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

/**
 * The token accounting as one readable line, or "" when there is nothing to say.
 *
 * WHY THIS IS NOT A GENERIC `key value` JOIN. It was one, and it filtered to
 * numbers only — so `durationMs` rendered as a bare `durationMs 4200` with no
 * unit, and `model` and `usageIncomplete` were dropped entirely. Both backends
 * send more than token counts here:
 *
 *   DSH    inputTokens + outputTokens required; totalTokens, cacheReadTokens,
 *          cacheWriteTokens, reasoningTokens optional
 *          (packages/session/session-format-v0-to-v1/src/payload-validation.ts:415-421)
 *   mcode  the same token keys, plus durationMs, model and usageIncomplete,
 *          which its projector spreads in as siblings of `usage`
 *
 * So the known keys get labels and units, `model` is unwrapped from whatever
 * shape the backend used, and anything unrecognised still shows up rather than
 * being silently dropped — a future key must not vanish the way `model` did.
 */
export function usageLine(usage: Record<string, unknown>): string {
  const LABELS: Record<string, string> = {
    inputTokens: "in",
    outputTokens: "out",
    totalTokens: "total",
    cacheReadTokens: "cache read",
    cacheWriteTokens: "cache write",
    reasoningTokens: "reasoning",
  };
  const bits: string[] = [];
  for (const [k, label] of Object.entries(LABELS)) {
    const v = usage[k];
    if (typeof v === "number") bits.push(`${label} ${v}`);
  }
  // A duration without a unit is not information. Sub-second turns are common,
  // so seconds is the readable unit; one decimal is enough to tell 0.4s from 0.9s.
  const ms = usage.durationMs;
  if (typeof ms === "number") bits.push(`${(ms / 1000).toFixed(1)}s`);

  const model = modelName(usage.model);
  if (model) bits.push(model);

  // Anything the list above does not know about, so a new backend key is
  // visible instead of being dropped. Booleans are skipped: `usageIncomplete`
  // is rendered as its own sentence by the caller, where it can be explained.
  const known = new Set([...Object.keys(LABELS), "durationMs", "model",
                         "usageIncomplete"]);
  for (const [k, v] of Object.entries(usage)) {
    if (known.has(k) || typeof v !== "number") continue;
    bits.push(`${k} ${v}`);
  }
  return bits.join(" · ");
}

/** The model's name, from whichever shape the backend used. */
function modelName(model: unknown): string {
  if (typeof model === "string") return model;
  // mcode sends `{modelId: "..."}` (the projector's own fixture agrees), and the
  // name is what a reader wants — not the object.
  if (model && typeof model === "object") {
    const m = model as Record<string, unknown>;
    for (const k of ["modelId", "id", "name", "model"]) {
      if (typeof m[k] === "string") return m[k] as string;
    }
  }
  return "";
}

/** R6-ACP: mcode's control plane, which only the Agent Client Protocol transport
 *  can report.
 *
 *  WHY THIS IS WORTH A PANEL AT ALL. `mcode exec` is a projection of one turn, so
 *  a queue, a delegation tree and the live model/permission selects have no `exec`
 *  representation whatsoever. Over ACP they do — and until this round Rigma drove
 *  only `exec`, so all three were invisible: not broken, simply unreachable, with
 *  nothing in the menu saying so.
 *
 *  Each section renders only when the backend actually reported it, so a turn that
 *  never spoke ACP draws exactly as before. Nothing here is guessed: a missing
 *  field is omitted rather than filled with a default that would read as a fact.
 */
function AcpBlock({
  queue,
  delegation,
  config,
  commands,
  plan,
}: {
  queue: AcpQueueItem[];
  delegation: AcpDelegation | null;
  config: AcpConfigOption[];
  commands: AcpCommand[];
  plan: Record<string, unknown> | null;
}) {
  const members = delegation?.members ?? [];
  const hasPlan =
    plan != null && (typeof plan.content === "string" || typeof plan.uri === "string");
  if (
    queue.length === 0 && members.length === 0 && config.length === 0 &&
    commands.length === 0 && !hasPlan
  ) {
    return null;
  }
  return (
    <div className="border-t border-line pt-1.5 flex flex-col gap-1.5">
      <div className="flex items-center gap-2 flex-wrap">
        <span className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em]">
          acp
        </span>
        {/* The live selects. Shown because these are the two things a session can
            be switched between mid-flight over ACP, and because an `exec` turn
            cannot change either without ending the session. */}
        {config.map((c) => (
          <span key={String(c.id)} className="font-mono text-[10.5px] text-muted">
            {String(c.id)}={configLabel(c.currentValue)}
          </span>
        ))}
      </div>

      {hasPlan && (
        <div>
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-0.5">
            plan review
          </div>
          <pre className="text-[11.5px] text-primary whitespace-pre-wrap break-words max-h-48 overflow-auto">
            {typeof plan!.content === "string"
              ? plan!.content
              : String(plan!.uri ?? "")}
          </pre>
        </div>
      )}

      {members.length > 0 && (
        <div>
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-0.5">
            delegations
          </div>
          <ul className="flex flex-col gap-0.5">
            {members.map((m, i) => (
              <li key={`${String(m.sessionId)}-${String(i)}`}
                  className="text-[12px] flex items-start gap-1.5">
                <span className={`font-mono mt-px ${delegationTone(m.status)}`}
                      aria-hidden="true">
                  {delegationGlyph(m.status)}
                </span>
                <span className="text-primary">
                  {/* The field is literally named `task` — it holds the member's
                      title, not an id, which is worth a comment because `task`
                      reads like one. */}
                  {m.task || m.agentName || String(m.sessionId ?? "")}
                </span>
                {m.status && (
                  <span className="font-mono text-[10.5px] text-muted">{m.status}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {queue.length > 0 && (
        <div>
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-0.5">
            queued ({queue.length})
          </div>
          <ul className="flex flex-col gap-0.5">
            {queue.map((q, i) => (
              <li key={`${String(q.itemId)}-${String(i)}`} className="text-[12px] flex gap-1.5">
                <span className="font-mono text-[10.5px] text-muted">
                  {String(q.status ?? "queued")}
                </span>
                <span className="text-primary">{queueText(q.content)}</span>
                {q.failedReason && (
                  <span className="text-red">{String(q.failedReason)}</span>
                )}
              </li>
            ))}
          </ul>
        </div>
      )}

      {commands.length > 0 && (
        <div className="font-mono text-[10.5px] text-muted">
          {commands.length} command{commands.length === 1 ? "" : "s"} available:{" "}
          {commands.map((c) => `/${String(c.name ?? "")}`).join(" ")}
        </div>
      )}
    </div>
  );
}

/** A configOption's current value, which may be a bare scalar or an object.
 *
 *  mcode's `model` option is an opaque id like
 *  `m:custom_provider%3Arigma:local-test:v:thinking`, so it is shown as-is rather
 *  than prettified — decoding it here would be this side inventing a schema. */
function configLabel(v: unknown): string {
  if (v == null) return "";
  if (typeof v === "string" || typeof v === "number") return String(v);
  if (typeof v === "object") {
    const o = v as Record<string, unknown>;
    const inner = o.value ?? o.id ?? o.name;
    if (inner != null) return String(inner);
  }
  return JSON.stringify(v);
}

/** A queued message's text. mcode passes `content` through unmapped, so it may be
 *  a string, a block list, or absent — and a `[object Object]` in the panel would
 *  be worse than saying nothing. */
function queueText(content: unknown): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content
      .map((b) =>
        b && typeof b === "object" && "text" in (b as Record<string, unknown>)
          ? String((b as Record<string, unknown>).text ?? "")
          : "",
      )
      .join("");
  }
  if (typeof content === "object") {
    const o = content as Record<string, unknown>;
    if (typeof o.text === "string") return o.text;
    if (typeof o.content === "string") return o.content;
  }
  return "";
}

function delegationTone(status?: string): string {
  switch ((status ?? "").toLowerCase()) {
    case "completed":
      return "text-moss";
    case "failed":
      return "text-red";
    case "stopped":
      return "text-muted";
    case "running":
      return "text-amber";
    default:
      return "text-muted";
  }
}

function delegationGlyph(status?: string): string {
  switch ((status ?? "").toLowerCase()) {
    case "completed":
      return "✓";
    case "failed":
      return "✕";
    case "stopped":
      return "−";
    case "running":
      return "▸";
    case "queued":
      return "·";
    default:
      // mcode normalises an unrecognised status to the literal string "unknown",
      // so this arm is reached on purpose rather than by accident.
      return "?";
  }
}

export default function AgentState({
  goal,
  todos,
  planMode,
  subagents,
  usage,
  governance,
  acpQueue = [],
  acpDelegation = null,
  acpConfig = [],
  acpCommands = [],
  acpPlan = null,
}: {
  goal: NormalGoal | null;
  todos: { content: string; status: string }[];
  planMode: boolean;
  subagents: Subagent[];
  /** R5-PERSIST: optional because the DURABLE panel has no usage to show — a
   *  token count describes a step, not the conversation, so it is not persisted.
   *  Required-and-passed-null would work too, but making it optional says why. */
  usage?: Record<string, unknown> | null;
  governance: Governance;
  /** R6-ACP: optional and defaulted, so every existing call site — including the
   *  durable panel, which has none of this — keeps compiling and drawing. */
  acpQueue?: AcpQueueItem[];
  acpDelegation?: AcpDelegation | null;
  acpConfig?: AcpConfigOption[];
  acpCommands?: AcpCommand[];
  acpPlan?: Record<string, unknown> | null;
}) {
  // Already normalised by the store, so this draws ONE shape whatever the
  // backend was: DSH's nested `goal/change` envelope and mcode's flat goal
  // object both arrive here as a NormalGoal.
  const hasGoal = goal !== null && goal.objective !== "";
  const hasUsage = usage != null && Object.keys(usage).length > 0;
  const hasGov =
    governance.approvals.length > 0 || !!governance.sandbox || !!governance.preset;
  const hasAcp =
    acpQueue.length > 0 || (acpDelegation?.members?.length ?? 0) > 0 ||
    acpConfig.length > 0 || acpCommands.length > 0 || acpPlan != null;
  if (
    !hasGoal && todos.length === 0 && !planMode && subagents.length === 0 &&
    !hasUsage && !hasGov && !hasAcp
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
        <div className="font-mono text-[10.5px] text-muted">
          <p>{usageLine(usage!)}</p>
          {/* mcode says when its own token count is not the whole story. A reader
              who cannot see this trusts a number that is low, so it is said
              plainly rather than left in the payload. */}
          {usage!.usageIncomplete === true && (
            <p className="text-amber/80">
              token count incomplete — the backend reported a partial figure
            </p>
          )}
        </div>
      )}

      {/* R6-ACP: the control plane, above governance because it is about the
          WORK (what is queued, what was delegated, which plan) rather than about
          the connection's permissions. */}
      <AcpBlock queue={acpQueue} delegation={acpDelegation} config={acpConfig}
                commands={acpCommands} plan={acpPlan} />

      {/* Last, and separated by a rule: this is about the CONNECTION's
          permissions rather than about the work, and putting it above the goal
          would make confinement read as the turn's subject. */}
      <GovernanceBlock gov={governance} />
    </div>
  );
}
