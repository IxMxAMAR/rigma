// Chat state. The heart is applyEvent(): a PURE reducer from SSE events to
// streaming state, unit-tested in isolation. Chips are keyed by the server's
// call id — the structural fix for the bug class caught live 2026-07-21,
// where order-based matching hung results on the wrong rows. FIFO is only
// the fallback for id-less legacy events.
//
// The same lesson applies one level up: an in-flight TURN is keyed by the
// session that owns it. A single `streaming` slot meant the reply you started
// in one chapter finished into whichever chapter you happened to be reading.
import { create } from "zustand";
import {
  api, type ChatMessage, type SavedAgentState, type SessionSummary,
  type WorkflowAgent, type WorkflowRun,
} from "../lib/api";
import { responseError } from "../lib/listFetch";
import { runMacroStream } from "../lib/methods";
import { streamChat, type SseEvent } from "../lib/sse";
import { DRAFT_KEY, parseDrafts, saveDrafts } from "./drafts";
import { foldSubagent, type Subagent } from "./subagents";
import { isGoalCleared, normaliseGoal, type NormalGoal } from "./goal";
import { EMPTY_GOVERNANCE, foldApproval, type Governance } from "./governance";
import { EMPTY_COMPACTIONS, foldCompaction, type Compactions } from "./compaction";
import { EMPTY_RETRY, foldRetry, type Retry } from "./retry";

export interface Chip {
  id: string;
  name: string;
  args?: unknown;
  result?: string;
  state: "running" | "done";
  /** Whether the call succeeded, when the backend actually said.
   *
   *  `undefined` means UNKNOWN, and it is the honest default rather than `true`:
   *  a backend that does not report an outcome (mcode's adapter used to claim
   *  success for every result, including failures) must not be drawn as a tick.
   *  The chip falls back to reading the result text, which is what it always
   *  did, only now it can do better when it knows better. */
  ok?: boolean | null;
}

/** One grounded passage's origin. `snippet` is a display excerpt, clipped
 *  server-side; it is never fed back to the model. */
export interface Source {
  source: string;
  snippet?: string;
  page?: number;
}

export interface StreamingTurn {
  text: string;
  thinking: string;
  chips: Chip[];
  error: string | null;
  /** set while a macro drives this turn, so the UI can say which step */
  macro: { label: string; index: number; total: number } | null;
  /** Which backend owns this turn, named at its START — an external turn can
   *  run for a minute before its first token, so the badge must not wait for
   *  the reply to exist. Empty means the server never said. */
  harness: string;
  harnessLabel: string;
  /** Server-authored status lines for this turn. They used to be dropped here,
   *  including the one that explains an external backend is driving. */
  notices: string[];
  /** UIUX-22: which of the user's own documents a grounded answer came from.
   *  The sidecar has always returned these and the tool always folded them into
   *  the MODEL's text, so the sources reached the model and never the reader —
   *  backwards for the one feature whose value is "which file said this". */
  sources: Source[];
  /** What the agent is working TOWARD, normalised from whichever backend
   *  reported it. A goal is not a message and not a todo: it outlives the turn
   *  that set it. Both harnesses report one and they disagree on the shape, so
   *  `chat/goal.ts` folds them into this.
   *
   *  R4-PERSIST: it outlives the turn in the AGENT's state, but NOT in Rigma's
   *  UI. This comment used to claim the field outlives the turn, full stop, and
   *  that was wrong in the way that matters: `AgentState` renders inside
   *  `LiveTurn` and the stream is dropped when the turn ends, so a goal is gone
   *  from the screen on reload. Nothing on the server persists or returns it
   *  either — the four SSE emitters are the only places a goal is mentioned in
   *  `src/rigma`. So this is a real gap, and it is recorded rather than papered
   *  over with a comment that reads as a guarantee. */
  goal: NormalGoal | null;
  /** The backend's latest whole-list todo snapshot. `todo_write` REPLACES the
   *  list every call, so this is an assignment, never a merge. */
  todos: { content: string; status: string }[];
  /** Plan mode: the agent is proposing rather than doing. */
  planMode: boolean;
  /** Subagents this turn's backend started, folded from its lifecycle. */
  subagents: Subagent[];
  /** The step's token accounting, when the backend reported any. */
  usage: Record<string, unknown> | null;
  /** What the harness was allowed to do, and what it asked for. DSH records
   *  its governance as `log-only` events — durable, never in the model
   *  transcript — so this belongs beside the turn rather than inside it.
   *
   *  DISPLAY-ONLY, and that is the transport's property, not a shortcut: DSH's
   *  SDK wire has no approval-response method, so approval is decided by the
   *  policy engine. See chat/governance.ts. */
  governance: Governance;
  /** DSH's compaction lifecycle. Rigma's NATIVE compaction reports itself through
   *  `housekeeping`/`masked`/`compacted` below; a DSH turn reported none of it,
   *  so a long turn busy summarising its own context looked hung. Folded rather
   *  than appended because `compaction/start`…`compaction/end` is a bracket. */
  compactions: Compactions;
  /** The most recent LLM retry, if DSH had to make one. `dsh-llm-retry` is a
   *  dependency of sdk-minimal itself, so this fires whether or not anyone asked
   *  for it — and a silent retry against a local engine reads as a frozen turn. */
  retry: Retry | null;
  /** These five names were emitted by the server and dropped here by the
   *  default arm, so a compaction, an observation-masking pass, a queued
   *  prompt and the server's own retitling were all silent. */
  housekeeping: string;
  masked: number;
  compacted: number;
  queued: number;
  title: string;
  /** R6-ACP: mcode's control plane, which `exec` cannot report at all.
   *
   *  These arrive only over the Agent Client Protocol transport. They are kept as
   *  the backend sent them rather than reshaped, because the shapes are mcode's
   *  own (`mcode/session/queue_update` and friends) and a second vocabulary here
   *  would be one more place for the two to disagree. Each is `null` until the
   *  backend reports it, so a turn that never spoke ACP renders exactly as before.
   *
   *  A QUEUE is prompts waiting for a later turn; a DELEGATION snapshot is the
   *  parent/child session tree mcode spawned; configOptions are the live model and
   *  permission selects. All three were unreachable before this round. */
  acpQueue: AcpQueueItem[];
  acpDelegation: AcpDelegation | null;
  acpConfig: AcpConfigOption[];
  acpCommands: AcpCommand[];
  /** ACP's own plan review, when the backend offered one. Distinct from
   *  `todos`, which is the model's checklist, and from `planMode`, which is
   *  whether the agent is proposing rather than doing. */
  acpPlan: Record<string, unknown> | null;
  /** R6-WORKFLOW: programmatic-tool-calling runs, newest first. Folded from four DSH
   *  events by `foldWorkflow`, which is where the joining rules live. */
  workflow: WorkflowRun[];
}

/** One prompt mcode is holding for a later turn.
 *
 *  Every field is optional because mcode's own projection assigns `itemId`,
 *  `sessionId` and `status` unconditionally — so any of them may be absent — and
 *  passes timestamps through unmapped. Typing them as required would be a claim
 *  about a schema this side does not own. */
/** One queued message.
 *
 *  MEASURED from mcode's queue-item mapper (`chunks/chunk-E2AN54L4.js`, `Tve(a)`):
 *
 *      {itemId, sessionId, status, ...source, ...reviewRequest, ...content,
 *       ...attachments, ...modelInfo, ...createdAt, ...expiresAt, ...startedAt,
 *       ...finishedAt, ...failedReason}
 *
 *  The timestamps are mcode's own names (`createdAt`, not `createdAtMs` — unlike the
 *  delegation members, which use the `Ms` suffix). `failedReason` is real: mcode's TUI
 *  renders it as "Couldn't send · <reason>" on a failed row. */
export interface AcpQueueItem {
  itemId?: string;
  sessionId?: string;
  status?: string;
  content?: unknown;
  /** Why the send failed. mcode's TUI shows this, so a row that only said "failed"
   *  was withholding something the backend had already sent. */
  failedReason?: string;
  source?: string;
  createdAt?: unknown;
  expiresAt?: unknown;
  startedAt?: unknown;
  finishedAt?: unknown;
  [k: string]: unknown;
}

/** mcode's delegation snapshot: the parent/child session tree under one root.
 *
 *  `members[].task` is the member's title — the field is literally named `task`,
 *  which is worth knowing because `task` reads like an id. `status` is normalised
 *  by mcode to exactly one of queued/running/completed/failed/stopped/unknown.
 *
 *  EVERY FIELD HERE IS MEASURED, not inferred, from mcode's own member mapper
 *  (`chunks/chunk-M5QJG5VT.js`, `Se(e)`):
 *
 *      {sessionId, parentSessionId,
 *       ...agentName ? {agentName} : {}, ...title ? {task: title} : {},
 *       status: D(e.status),
 *       ...purpose.startsWith("local-background-task:")
 *           ? {backgroundTaskId: purpose.slice(len)} : {},
 *       ...createdAtMs, ...updatedAtMs,
 *       ...errorMessage ? {errorMessage} : {}}
 *
 *  The status normaliser `D(e)` maps EVERY raw status onto exactly those six values, and
 *  `errorMessage` and `backgroundTaskId` are real — each is emitted only when present,
 *  which is why a search for them in one bundle file finds nothing. */
export interface AcpDelegation {
  schemaVersion?: number;
  rootSessionId?: string;
  members?: {
    sessionId?: string;
    parentSessionId?: string;
    agentName?: string;
    task?: string;
    status?: string;
    backgroundTaskId?: string;
    createdAtMs?: number;
    updatedAtMs?: number;
    errorMessage?: string;
  }[];
  [k: string]: unknown;
}

/** One of mcode's live selects — the model or the permission policy. */
export interface AcpConfigOption {
  id?: string;
  name?: string;
  currentValue?: unknown;
  options?: { value?: unknown; name?: string }[];
  [k: string]: unknown;
}

/** A slash command the backend says it has. */
export interface AcpCommand {
  name?: string;
  description?: string;
  [k: string]: unknown;
}

/** Marks a reply the user cut short, so the transcript never reads as if the
 *  model chose to end there. */
export const STOPPED_SUFFIX = "\n\n_(stopped — partial reply)_";

/** Did the server already persist this stopped turn? The abort can land after
 *  the turn finished writing, and appending then would duplicate the reply.
 *  Compared on a prefix: the saved copy has the suffix and may have had
 *  <think> blocks stripped, so it is never character-identical. */
export function alreadySaved(msgs: ChatMessage[], partial: string): boolean {
  const last = msgs[msgs.length - 1];
  if (!last || last.role !== "assistant") return false;
  const saved = typeof last.content === "string" ? last.content : "";
  const head = partial.trim().slice(0, 64);
  return head.length > 0 && saved.includes(head);
}

// AUDIT F49: docs/audit-2026-09-04-full.md
/** Did the server's copy of the transcript keep the prompt we just sent?
 *  A turn that fails persists no reply, and the vision guard rejects BEFORE
 *  the prompt is stored — so the reload that ends every turn can erase the
 *  very message the error is about. Compared against the reloaded tail, not
 *  against a count: a queued or compacted transcript moves the count. */
export function promptSurvived(
  msgs: ChatMessage[], content: ChatMessage["content"],
): boolean {
  // AUDIT F49: docs/audit-2026-09-04-full.md — the prompt is NOT always last.
  // The server persists an assistant message whenever the turn produced text
  // OR a tool trace, so a turn that ran a tool and then failed mid-stream is
  // stored as [... user prompt, assistant "-> tool"]. Inspecting only the tail
  // read that as "the prompt was lost", re-appended it, and — because a
  // following regenerate PUTs the visible list and pops nothing when the tail
  // is a user message — wrote the duplicate into the saved chapter, leaving the
  // model looking at two consecutive user turns. Grounded chat forces tools on,
  // so it is the likeliest way in. Scan back over the turn's own tail instead.
  const want = JSON.stringify(content);
  for (let i = msgs.length - 1; i >= 0 && i >= msgs.length - 8; i--) {
    const m = msgs[i];
    if (m.role === "user" && JSON.stringify(m.content) === want) return true;
  }
  return false;
}

/** Never show the user the word "undefined": a rejected fetch can arrive as a
 *  DOMException, a bare string, or nothing at all. */
export function errText(e: unknown): string {
  const m = e instanceof Error ? e.message : String(e ?? "");
  return m.trim() || "the request failed";
}

/** Copy of a per-session record with one session dropped. A finished turn
 *  leaves no entry behind: a stale key is a turn the UI still thinks is
 *  running. */
function without<T>(rec: Record<string, T>, key: string): Record<string, T> {
  if (!(key in rec)) return rec;
  const next = { ...rec };
  delete next[key];
  return next;
}

/** IMP-1: drafts read once when the store is created, so a reload does not lose
 *  a half-written message. Best-effort — a blocked localStorage (private mode)
 *  must not stop the store from existing. */
function loadDrafts(): Record<string, string> {
  try { return parseDrafts(localStorage.getItem(DRAFT_KEY)); }
  catch { return {}; }
}

export const emptyTurn = (): StreamingTurn => ({
  text: "",
  thinking: "",
  chips: [],
  error: null,
  macro: null,
  harness: "",
  harnessLabel: "",
  notices: [],
  sources: [],
  goal: null,
  governance: EMPTY_GOVERNANCE,
  compactions: EMPTY_COMPACTIONS,
  retry: EMPTY_RETRY,
  todos: [],
  planMode: false,
  subagents: [],
  usage: null,
  acpQueue: [],
  acpDelegation: null,
  acpConfig: [],
  acpCommands: [],
  acpPlan: null,
  workflow: [],
  housekeeping: "",
  masked: 0,
  compacted: 0,
  queued: 0,
  title: "",
});

/** Pure: fold one SSE event into the streaming turn. Returns a NEW object —
 *  functional updates only, so React sees every change and nothing aliases. */
/** Fold one `tool-workflow/*` event into the run it describes.
 *
 *  R6-WORKFLOW. DSH sends four events joined only by `runId`, and joins an agent's
 *  start to its own end by `seq`. This is where that structure is rebuilt, so the
 *  panel can draw a run rather than four lines the reader has to assemble.
 *
 *  THREE DECISIONS WORTH NAMING:
 *
 *  1. **An unknown `runId` OPENS a run.** `agent-start` can arrive without a
 *     `run-start` ever being seen — the runner only began forwarding these in
 *     R6-WORKFLOW, and a resumed session can carry a run's later events without its
 *     first. Treating an unseen id as "ignore" would drop real activity to avoid
 *     drawing a header, so it opens a run with a blank name instead. The `name` is
 *     filled in by `run-start` if it ever arrives.
 *
 *  2. **An `agent-end` for an unseen `seq` still records the agent.** Same reasoning:
 *     an outcome with no start is still a fact, and dropping it would make a run look
 *     shorter than it was.
 *
 *  3. **`run-end` MARKS done rather than deleting.** A finished run is the thing worth
 *     reading; deleting it would make the panel flash and vanish. This is also what
 *     the durable path keys on, so `done` is load-bearing rather than cosmetic.
 *
 *  A new run is PREPENDED, so the newest is at the top — a run is usually read while
 *  it is happening, and the newest is the one happening. */
export function foldWorkflow(
  runs: WorkflowRun[], d: Record<string, unknown>,
): WorkflowRun[] {
  const kind = String(d.event ?? "");
  const data = (d.data ?? {}) as Record<string, unknown>;
  const runId = String(data.runId ?? "");
  // An event with no runId cannot be placed. Refusing it is not a drop: nothing can
  // join it to anything, so a run would have to be invented for it.
  if (!runId) return runs;

  const idx = runs.findIndex((r) => r.runId === runId);
  // Copy-on-write: the reducer must never mutate the array it was handed.
  const next = runs.map((r) => ({ ...r, agents: [...r.agents] }));
  let run = idx >= 0 ? next[idx] : null;
  if (run === null) {
    run = { runId, name: "", stopReason: "", done: false, agents: [] };
    next.unshift(run);
  }

  switch (kind) {
    case "tool-workflow/run-start":
      run.name = String(data.name ?? "");
      break;
    case "tool-workflow/run-end":
      run.done = true;
      run.stopReason = String(data.stopReason ?? "");
      break;
    case "tool-workflow/agent-start": {
      const seq = Number(data.seq ?? -1);
      run.agents.push({
        seq,
        label: String(data.label ?? ""),
        childId: String(data.childId ?? ""),
        phase: String(data.phase ?? ""),
        outcome: "",
      });
      break;
    }
    case "tool-workflow/agent-end": {
      const seq = Number(data.seq ?? -1);
      const agent = run.agents.find((a: WorkflowAgent) => a.seq === seq);
      if (agent) {
        agent.outcome = String(data.outcome ?? "");
      } else {
        // Decision 2: the outcome is still a fact even with no start to attach it to.
        run.agents.push({
          seq,
          label: "",
          childId: "",
          phase: "",
          outcome: String(data.outcome ?? ""),
        });
      }
      break;
    }
    default:
      // An unrecognised `tool-workflow/*` subtype: the run is opened or found, which
      // is the honest outcome — something ran, and this build cannot name what.
      break;
  }
  return next;
}

export function applyEvent(turn: StreamingTurn, ev: SseEvent): StreamingTurn {
  const d = (ev.data ?? {}) as Record<string, unknown>;
  switch (ev.event) {
    case "think":
      return { ...turn, thinking: turn.thinking + String(d.delta ?? "") };
    case "tool": {
      const id = String(d.id ?? `fifo-${turn.chips.length}`);
      return {
        ...turn,
        chips: [
          ...turn.chips,
          { id, name: String(d.name ?? "?"), args: d.args, state: "running" },
        ],
      };
    }
    case "tool_result": {
      const id = d.id == null ? null : String(d.id);
      let matched = false;
      const chips = turn.chips.map((c) => {
        if (matched) return c;
        const hit = id != null ? c.id === id && c.state === "running"
                               : c.state === "running"; // legacy: first open
        if (!hit) return c;
        matched = true;
        return {
          ...c,
          result: String(d.result ?? ""),
          state: "done" as const,
          // Absent on the wire means UNKNOWN, which is not `true`. The native
          // loop sends it; the external path does not always.
          ok: typeof d.ok === "boolean" ? d.ok : null,
        };
      });
      return { ...turn, chips };
    }
    // A macro's steps ride the SAME reducer as a chat turn, so its tool calls
    // render as the ordinary chips — one rendering path, not two.
    case "macro_step":
      return {
        ...turn,
        macro: {
          label: String(d.label ?? ""),
          index: Number(d.index ?? 0),
          total: Number(d.total ?? 0),
        },
      };
    case "macro_done":
      return { ...turn, macro: null };
    // AUDIT F11-12: a `citations` branch and StreamingTurn field lived here.
    // No server code emits the event (grep over src/rigma/*.py finds only the
    // RAG tool's plain-text `sources: …` line) and no component read the field
    // (Transcript renders harness, macro, notices, thinking, chips, text,
    // error). The half was removed rather than left looking like a pipeline
    // that already handles citations; re-adding it needs a producer first.
    // UIUX-22: grounded sources for this turn. The server emits only what each
    // round ADDED, so these append — and an entry with no usable `source` is
    // dropped rather than rendered as an empty chip, because "sourced from
    // nothing" is worse than saying nothing.
    case "citations": {
      const rows = Array.isArray(d.sources) ? d.sources : [];
      const add: Source[] = [];
      for (const raw of rows) {
        if (raw == null || typeof raw !== "object") continue;
        const r = raw as Record<string, unknown>;
        const source = String(r.source ?? "").trim();
        if (!source) continue;
        const page = Number(r.page);
        add.push({
          source,
          snippet: String(r.snippet ?? ""),
          ...(Number.isInteger(page) && page > 0 ? { page } : {}),
        });
      }
      return add.length ? { ...turn, sources: [...turn.sources, ...add] } : turn;
    }
    case "error":
      return { ...turn, error: String(d.message ?? "unknown error") };
    // Which backend owns this turn, named at its start. Its own event name
    // rather than `meta`, which already carries ctx/timings and titles.
    case "harness":
      return d.name != null
        ? {
            ...turn,
            harness: String(d.name),
            harnessLabel: String(d.label ?? d.name),
          }
        : turn;
    case "notice": {
      const note = String(d.note ?? "");
      return note ? { ...turn, notices: [...turn.notices, note] } : turn;
    }
    // The backend's structured state. Each is a fact with its own shape, so
    // none of them is folded into `text`.
    case "approval":
      return { ...turn, governance: foldApproval(turn.governance, d) };
    case "llm_retry":
      // `foldRetry` returns the state unchanged for a non-retry payload, so this
      // arm needs no guard of its own.
      return { ...turn, retry: foldRetry(turn.retry, d) };
    case "compaction":
      // `foldCompaction` returns the state unchanged for a non-compaction
      // payload, so this arm needs no guard of its own.
      return { ...turn, compactions: foldCompaction(turn.compactions, d) };
    case "sandbox":
      return {
        ...turn,
        governance: { ...turn.governance, sandbox: String(d.mode ?? "") },
      };
    case "permission_preset":
      return {
        ...turn,
        governance: { ...turn.governance, preset: String(d.preset ?? "") },
      };
    case "session_title":
      // The SERVER's title, which is authoritative. Rigma's own auto-title is a
      // guess made without it, so adopting this is a correction rather than a
      // second opinion.
      return { ...turn, title: String(d.title ?? "") || turn.title };
    case "goal": {
      // R4-GOAL-1: an explicit CLEAR is an instruction, not an unrecognised
      // payload. DSH sends `{operation: "clear", cleared: true}` with no `goal`
      // key, and treating that like "not a goal" left the cleared objective on
      // screen forever. Only the tombstone clears; everything else keeps the
      // panel as it was, because `get_goal` answers `{goal: null}` and an error
      // answers `{error}` — neither of which means the goal is gone.
      if (isGoalCleared(d)) return { ...turn, goal: null };
      const g = normaliseGoal(d);
      return g ? { ...turn, goal: g } : turn;
    }
    case "todos": {
      const raw = Array.isArray(d.todos) ? d.todos : [];
      return {
        ...turn,
        todos: raw.map((t) => {
          const o = (t ?? {}) as Record<string, unknown>;
          return { content: String(o.content ?? ""), status: String(o.status ?? "pending") };
        }),
      };
    }
    case "plan_mode":
      return { ...turn, planMode: d.active === true };
    // R6-WORKFLOW: four DSH events, one run. See `WorkflowRun`.
    case "workflow":
      return { ...turn, workflow: foldWorkflow(turn.workflow, d) };
    case "subagent":
      return { ...turn, subagents: foldSubagent(turn.subagents, d) };
    // R6-ACP: mcode's control plane over the Agent Client Protocol transport.
    // Each replaces rather than merges, because mcode sends the WHOLE list or
    // snapshot on every update — a merge would resurrect a deleted queue item.
    case "acp_queue": {
      const items = Array.isArray(d.items) ? d.items : [];
      return { ...turn, acpQueue: items as AcpQueueItem[] };
    }
    case "acp_delegation": {
      const snap = d && typeof d === "object" ? (d as AcpDelegation) : null;
      return { ...turn, acpDelegation: snap };
    }
    case "acp_config": {
      const opts = Array.isArray(d.configOptions) ? d.configOptions : [];
      return { ...turn, acpConfig: opts as AcpConfigOption[] };
    }
    case "acp_commands": {
      const cmds = Array.isArray(d.commands) ? d.commands : [];
      return { ...turn, acpCommands: cmds as AcpCommand[] };
    }
    case "acp_plan": {
      // A removal is an instruction, like the goal tombstone: the plan is gone.
      if (d.removed === true) return { ...turn, acpPlan: null };
      return { ...turn, acpPlan: d };
    }
    case "usage":
      return { ...turn, usage: d };
    // Emitted by the server and previously dropped by the default arm.
    case "housekeeping":
      return d.note != null ? { ...turn, housekeeping: String(d.note) } : turn;
    case "masked":
      return d.masked != null ? { ...turn, masked: Number(d.masked) } : turn;
    case "compacted":
      return d.archived != null ? { ...turn, compacted: Number(d.archived) } : turn;
    case "info":
      return d.queued != null
        ? { ...turn, queued: Number(d.queued) }
        : d.note != null
          ? { ...turn, housekeeping: String(d.note) }
          : turn;
    // The server retitles a chat from its first exchange. The rail was never
    // told, so a renamed chat kept its old name until a reload.
    case "meta":
      return d.title != null ? { ...turn, title: String(d.title) } : turn;
    case "message":
    default:
      return d.delta != null
        ? { ...turn, text: turn.text + String(d.delta) }
        : turn;
  }
}

/**
 * R5-PERSIST: the part of an event that OUTLIVES the turn, or null.
 *
 * A separate function from `applyEvent` because the two answer different
 * questions and must not be confused. `applyEvent` is a pure reducer for what is
 * on screen RIGHT NOW; this picks out the facts the server also stored, which
 * must survive the turn ending. Only three of the events are durable, and the
 * two omissions are deliberate:
 *
 *   subagent — a row names a child process belonging to the turn that spawned
 *              it. Restoring one after a reload would draw "running" for a child
 *              that is long gone, so it is turn-scoped on purpose. The server
 *              does not persist it either; this is the same decision, twice.
 *   usage    — a per-step number, not a fact about the conversation, and it
 *              arrives every step.
 *
 * The value stored is the RAW payload, so `chat/goal.ts` stays the single place
 * the two backends' field names are reconciled.
 */
export function durableFromEvent(ev: SseEvent): SavedAgentState | null {
  const d = (ev.data ?? {}) as Record<string, unknown>;
  switch (ev.event) {
    case "goal":
      // A clear stores null, matching what the server does — otherwise a reload
      // would resurrect the objective that was just cleared, since the tombstone
      // has no objective to normalise.
      return { goal: isGoalCleared(d) ? null : d };
    case "todos": {
      const raw = Array.isArray(d.todos) ? d.todos : [];
      return {
        todos: raw.map((t) => {
          const o = (t ?? {}) as Record<string, unknown>;
          return { content: String(o.content ?? ""),
                   status: String(o.status ?? "pending") };
        }),
      };
    }
    case "plan_mode":
      return { plan_mode: d.active === true };
    // R6-WORKFLOW: only a FINISHED run is durable. Persisting a run mid-flight would
    // write on every agent start, and a run restored from a reload could never
    // finish — the turn that would have ended it is gone.
    //
    // Keyed on the EVENT rather than on a `done` field, because `serve.py` sends the
    // four lifecycle events with their own name alongside the payload
    // (`{event, data}`), so there is no `done` on the envelope to read — only the
    // event name says which of the four this is.
    case "workflow":
      return d.event === "tool-workflow/run-end"
        ? { workflow: foldWorkflow([], d) }
        : null;
    default:
      return null;
  }
}

/** Fold a durable fragment onto the saved state for one chat. Keyed by session
 *  so a reload of chat B cannot paint chat A's goal into B's panel. */
function mergeSavedAgent(sid: string, add: SavedAgentState) {
  return (st: ChatState) => ({
    savedAgent: {
      ...st.savedAgent,
      [sid]: { ...(st.savedAgent[sid] ?? {}), ...add },
    },
  });
}

export interface ChatState {
  sessions: SessionSummary[];  currentId: string | null;
  messages: ChatMessage[];
  // AUDIT F1: docs/audit-2026-09-04-full.md
  /** In-flight turns, keyed by the session that owns them. Written only under
   *  the id captured when the request started, so a reply that outlives a
   *  chat switch can never render — or save — anywhere but its own chapter. */
  streams: Record<string, StreamingTurn>;
  /** One controller per in-flight turn: Stop must cancel the chat on screen,
   *  not whichever turn happened to start last. */
  aborts: Record<string, AbortController>;
  /** The take a regenerate set aside, per session — it is folded back in when
   *  THAT session's turn returns, however many chats later. */
  pendingVariants: Record<string, { content: unknown; variants: unknown[] }>;
  /** The last failure worth a sentence on screen. Held here because the turn
   *  that produced it is thrown away one round-trip after it appears. */
  lastError: string | null;
  /** What a command just did, or refused to do.
   *
   *  A separate channel from `lastError` on purpose: "compacted" and "unknown
   *  permission" are not failures, and routing them through the error banner
   *  would train the user to read red as noise. It is also separate from a
   *  turn's `notices`, which are the SERVER's status lines for a turn — a
   *  command can be run with no turn in flight at all, and a turn's stream is
   *  replaced when the next one starts, which would erase the message before it
   *  could be read. */
  notice: string | null;
  images: string[];               // data URIs staged in the composer
  /** Unsent composer text, keyed by the chat it was typed in. The draft is the
   *  one piece of user input that must NOT outlive its chat: a paragraph typed
   *  in "chapter one" and Entered after switching to "chapter two" used to be
   *  sent — and persisted — in the wrong chapter with no warning. */
  drafts: Record<string, string>;
  /** Which agent backend owns the chat on screen. Held here because both the
   *  picker and the transcript badge need it, and the store is the only place
   *  that already knows which chat that is. */
  harness: string;
  permission: string;
  /** R6-ACP-TURN: WHICH of mcode's two wires drives this chat.
   *
   *  `exec` is a projection of one turn; `acp` is a session, so it can report a
   *  queue, a delegation tree, a plan review and the live selects — and it can ASK a
   *  question and wait for the answer. That last one is why this exists: on `exec`
   *  there is no interaction host, so `smart` deciding to ask blocks the chat
   *  PERMANENTLY rather than for one turn.
   *
   *  A session field, and only meaningful for mcode. `exec` stays the default. */
  mcodeTransport: string;
  /** R5-PERSIST: the agent's durable state for the chat on screen, keyed by
   *  session id — the goal, the todo list, and plan mode, as the SERVER stored
   *  them.
   *
   *  WHY THIS IS NOT ON THE TURN. These facts outlive the turn that reported
   *  them, and the turn does not: `streams[sid]` is created when a turn starts
   *  and dropped when it ends, so a panel drawn from it vanishes on reload even
   *  though nothing about the agent's state changed. Keyed by session because a
   *  reload of chat B must not paint chat A's goal.
   *
   *  Held as the RAW wire payloads rather than normalised ones so that
   *  `chat/goal.ts` stays the single place the two backends' field names are
   *  reconciled — the same reason the server stores them unreshaped. */
  savedAgent: Record<string, SavedAgentState>;

  loadSessions: () => Promise<void>;
  search: (q: string) => Promise<void>;
  open: (id: string) => Promise<void>;
  newChat: () => Promise<void>;
  /** The chat to write a per-chat setting to, creating one if none is open.
   *
   *  R3-UI-1. `currentId` is null until a chat is opened OR the first message is
   *  sent, because `send` creates one on demand. Every other per-chat write used
   *  to `if (!currentId) return` instead, so on a fresh page the Sidecar's
   *  controls rendered as live and did nothing at all: the harness dropdown
   *  snapped back and the three grant checkboxes could not be ticked, with no
   *  message saying why. Returning the id (or null on failure) lets a caller do
   *  what `send` already does — make the chat the setting belongs to. */
  ensureSession: () => Promise<string | null>;
  deleteChat: (id: string) => Promise<void>;
  duplicateChat: (id: string) => Promise<void>;
  /** false = the turn never started; the composer keeps what you typed. */
  send: (message: string | null,
         opts?: Record<string, unknown>) => Promise<boolean>;
  runMacro: (macroId: string, confirm?: "run" | "always",
             answers?: Record<string, string>) => Promise<void>;
  regenerate: () => Promise<void>;
  continueTurn: () => Promise<void>;
  flipVariant: (dir: 1 | -1) => Promise<void>;
  addImage: (dataUri: string) => void;
  removeImage: (i: number) => void;
  /** Write the composer draft under the chat it belongs to. `sid` is the
   *  session captured when a send STARTED, so a failed send can put the text
   *  back where it came from even if the user has switched chats since. */
  setDraft: (text: string, sid?: string) => void;
  stop: () => void;
  /** Fold this chat's older turns into a digest. Returns the server's own
   *  refusal message when it refuses, so the caller can SHOW it: the two
   *  refusals (mid-reply, nothing to compact) are informative, and a silent
   *  no-op would read as the command being broken. */
  compactChat: () => Promise<string>;
  clearError: () => void;
  /** Show what a command did. One message at a time: a queue would need
   *  dismissals the user did not ask for. */
  pushNotice: (text: string) => void;
  clearNotice: () => void;
  /** Hand the current chat's turns to another backend. Unlike the other
   *  session patches this one is NOT fire-and-forget: the server refuses a
   *  backend it cannot run and says why, and that reason is the whole value of
   *  validating the write. */
  setHarness: (name: string) => Promise<void>;
  /** How much the agent may do without asking. A trade, not a fact: the
   *  mode that asks fails the run headlessly, so the default is `full`. */
  setPermission: (mode: string) => Promise<void>;
  setMcodeTransport: (wire: string) => Promise<void>;
  /** R7-SETTINGS: record a configOption change the SERVER already accepted.
   *
   *  WHY THIS IS NEEDED AT ALL. The select's value comes from the live turn's
   *  `acpConfig`, which is written only by the `acp_config` SSE event — and that event
   *  comes from a `config_option_update` NOTIFICATION on the connection that ran the
   *  turn. A control operation runs on its OWN short-lived mcode process whose
   *  notifications nobody reads, so the change succeeds and no event ever arrives: the
   *  panel said "done" and the select snapped back to the old value.
   *
   *  Called only AFTER the server confirmed the change, so this is a record of an
   *  accepted fact rather than an optimistic guess.
   */
  setAcpConfigOption: (optionId: string, value: string) => void;
}

/** The live turn for the chat on screen — nothing else may render. */
export const selectStreaming = (s: ChatState): StreamingTurn | null =>
  s.currentId ? s.streams[s.currentId] ?? null : null;

/** Is ANY chat generating? For the engine controls, which serve every
 *  session at once and would cut a background reply off mid-sentence. */
export const selectAnyStreaming = (s: ChatState): boolean =>
  Object.keys(s.streams).length > 0;

/** Fold an update into the turn a session owns — and only while that turn is
 *  still the live one. Events can arrive after the turn ended; resurrecting a
 *  key here is how a finished reply reappears over a later one. */
const patchTurn =
  (sid: string, fn: (t: StreamingTurn) => StreamingTurn) =>
  (st: ChatState): Partial<ChatState> => {
    const cur = st.streams[sid];
    return cur ? { streams: { ...st.streams, [sid]: fn(cur) } } : {};
  };

export const useChat = create<ChatState>((set, get) => ({
  sessions: [],
  currentId: null,
  messages: [],
  streams: {},
  aborts: {},
  pendingVariants: {},
  lastError: null,
  notice: null,
  images: [],
  drafts: loadDrafts(),
  harness: "native",
  permission: "full",
  mcodeTransport: "exec",
  savedAgent: {},

  loadSessions: async () => {
    try {
      set({ sessions: await api.listSessions() });
    } catch (e) {
      set({ lastError: errText(e) });
    }
  },

  search: async (q) => {
    if (!q.trim()) return get().loadSessions();
    // R3-UI-2: a failed search used to leave the FULL list on screen, so the
    // filter appeared to match everything — the reader concludes no chat
    // matches, when in fact the search never ran. Say so instead.
    //
    // `r.ok` is checked BEFORE reading the body: a Response body can only be
    // consumed once, so parsing first left `responseError` nothing to read and
    // it fell back to "server replied 500" instead of the server's sentence.
    try {
      const r = await fetch(`/api/sessions/search?q=${encodeURIComponent(q)}`);
      if (!r.ok) {
        set({ lastError: await responseError(r) });
        return;
      }
      const d: unknown = await r.json();
      if (Array.isArray(d)) set({ sessions: d as SessionSummary[] });
    } catch (e) {
      set({ lastError: errText(e) });
    }
  },

  deleteChat: async (id) => {
    // the turn belongs to a chat that is about to stop existing: cancel it
    // rather than leaving it streaming into a deleted session
    get().aborts[id]?.abort();
    await api.deleteSession(id).catch(() => {});
    set((st) => ({
      streams: without(st.streams, id),
      aborts: without(st.aborts, id),
      pendingVariants: without(st.pendingVariants, id),
      // the draft was typed for a chat that is about to stop existing
      drafts: without(st.drafts, id),
      // R5-PERSIST: and so was the agent state. Same reason as the draft — every
      // other per-chat map is dropped here and this one was added without it. It
      // is not only a leak: if an id were ever reused, the deleted chat's goal
      // would appear in the fresh chat's panel with nothing to explain it.
      savedAgent: without(st.savedAgent, id),
      ...(st.currentId === id ? { currentId: null, messages: [] } : {}),
    }));
    saveDrafts(get().drafts);
    await get().loadSessions();
  },

  duplicateChat: async (id) => {
    // R3-UI-2: this was a total silent no-op on EVERY failure path — no new
    // chat, no error, no state change. `r.ok` was never checked, so even a 404
    // resolved cleanly with `d.id` undefined, and the bare `catch {}` swallowed
    // the rest. A duplicate button that does nothing is indistinguishable from
    // one that worked and made a chat you cannot find.
    try {
      const r = await fetch(`/api/sessions/${id}/duplicate`, { method: "POST" });
      if (!r.ok) {
        set({ lastError: await responseError(r) });
        return;
      }
      const d = (await r.json()) as { id?: string };
      await get().loadSessions();
      if (d.id) await get().open(d.id);
    } catch (e) {
      set({ lastError: errText(e) });
    }
  },

  // AUDIT F1: docs/audit-2026-09-04-full.md — opening another chat no longer
  // touches the in-flight turn. It keeps streaming under its own id, and comes
  // back on screen intact if you return to it.
  open: async (id) => {
    try {
      const s = await api.getSession(id);
      set({ currentId: id, messages: s.messages,
            harness: s.harness ?? "native",
        permission: s.permission ?? "full", mcodeTransport: s.mcode_transport ?? "exec", lastError: null,
        notice: null,
        // R5-PERSIST: the agent's durable state comes back with the chat, so the
        // panel survives a reload. Replaced rather than merged — this is the
        // server's authoritative copy for THIS chat, and a stale local fragment
        // (a goal cleared in another tab) must not win over it.
        savedAgent: { ...get().savedAgent, [id]: s.agent_state ?? {} } });
    } catch (e) {
      // AUDIT F50: a session click that failed used to do nothing at all
      set({ lastError: errText(e) });
    }
  },

  newChat: async () => {
    try {
      const s = await api.createSession();
      set({ currentId: s.id, messages: [], harness: s.harness ?? "native",
        permission: s.permission ?? "full",
      mcodeTransport: s.mcode_transport ?? "exec",
            lastError: null, notice: null });
      await get().loadSessions();
    } catch (e) {
      set({ lastError: errText(e) });
    }
  },

  // R3-UI-1. One place that answers "which chat does this setting belong to",
  // creating it when there is none. `send` has always done this inline; every
  // other per-chat write silently gave up instead, which is why the Sidecar
  // looked broken on a fresh page.
  ensureSession: async () => {
    const sid = get().currentId;
    if (sid) return sid;
    try {
      const s = await api.createSession();
      set({ currentId: s.id, messages: [],
            harness: s.harness ?? "native",
            permission: s.permission ?? "full", mcodeTransport: s.mcode_transport ?? "exec", lastError: null,
            notice: null });
      await get().loadSessions();
      return s.id;
    } catch (e) {
      set({ lastError: errText(e) });
      return null;
    }
  },

  setHarness: async (name) => {
    if (name === get().harness) return;
    // Create the chat rather than refusing: the picker is on screen and looks
    // live, so a click that does nothing is the bug this fixes.
    const sid = await get().ensureSession();
    if (!sid) return;
    const prev = get().harness;
    set({ harness: name, lastError: null });
    try {
      await api.updateSession(sid, { harness: name });
    } catch (e) {
      // Put the old backend back. Leaving the picker showing one the server
      // rejected would make the next turn's failure look like a model problem.
      set({ harness: prev, lastError: errText(e) });
    }
  },

  setPermission: async (mode) => {
    if (mode === get().permission) return;
    const sid = await get().ensureSession();
    if (!sid) return;
    const prev = get().permission;
    set({ permission: mode, lastError: null });
    try {
      await api.updateSession(sid, { permission: mode });
    } catch (e) {
      // Same reasoning as the backend picker: a rejected value that stays on
      // screen makes the NEXT turn's failure look like something else.
      set({ permission: prev, lastError: errText(e) });
    }
  },

  setMcodeTransport: async (wire) => {
    if (wire === get().mcodeTransport) return;
    const sid = await get().ensureSession();
    if (!sid) return;
    const prev = get().mcodeTransport;
    set({ mcodeTransport: wire, lastError: null });
    try {
      await api.updateSession(sid, { mcode_transport: wire });
    } catch (e) {
      // Same reasoning as `setPermission`: a rejected value left on screen makes the
      // next turn's behaviour look like something else entirely — and here the
      // difference is which WIRE ran, so the lie would be about whether a permission
      // prompt could have been answered.
      set({ mcodeTransport: prev, lastError: errText(e) });
    }
  },

  runMacro: async (macroId, confirm, answers) => {
    const sid = get().currentId;
    if (!sid || get().streams[sid]) return;  // one turn at a time, macro or not
    const ctl = new AbortController();
    set((st) => ({
      streams: { ...st.streams, [sid]: emptyTurn() },
      aborts: { ...st.aborts, [sid]: ctl },
      lastError: null,
    }));
    let spawned: string | null = null;
    try {
      await runMacroStream(
        sid,
        { macro_id: macroId, confirm, answers },
        (ev) => {
          if (ev.event === "macro_done") {
            const nid = (ev.data as { new_session_id?: string })
              ?.new_session_id;
            if (nid) spawned = nid;
          }
          set(patchTurn(sid, (t) => applyEvent(t, ev)));
          const durable = durableFromEvent(ev);
          if (durable) set(mergeSavedAgent(sid, durable));
        },
        ctl.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError")
        set(patchTurn(sid, (t) => ({ ...t, error: errText(e) })));
    }
    // the macro wrote real messages server-side; reload so they replace the
    // streamed preview rather than double-rendering
    let msgs: ChatMessage[] | null = null;
    try {
      msgs = (await api.getSession(sid)).messages;
    } catch { /* keep what is on screen */ }
    const failure = get().streams[sid]?.error ?? null;
    // AUDIT F1: the transcript belongs to sid, not to whatever is on screen now
    set((st) => ({
      streams: without(st.streams, sid),
      aborts: without(st.aborts, sid),
      ...(msgs && st.currentId === sid ? { messages: msgs } : {}),
      ...(failure ? { lastError: failure } : {}),
    }));
    // a new_chat step spawned the next chapter: follow it
    if (spawned) {
      await get().loadSessions();
      await get().open(spawned);
    }
  },

  send: async (message, opts) => {
    const active = get().currentId;
    // one turn at a time — per chat, not per app
    if (active && get().streams[active]) return false;
    let id = active;
    if (!id) {
      try {
        id = (await api.createSession()).id;
        set({ currentId: id });
      } catch (e) {
        // AUDIT F50: the composer has already cleared the draft by now — say
        // what happened and let it put the paragraph back.
        set({ lastError: errText(e) });
        return false;
      }
    }
    const sid = id;                 // the owner of this turn, captured once
    const staged = get().images;
    let content: ChatMessage["content"] | null = message;
    if (message != null && staged.length > 0) {
      content = [
        { type: "text", text: message },
        ...staged.map((u) => ({ type: "image_url", image_url: { url: u } })),
      ] as ChatMessage["content"];
      set({ images: [] });
    }
    // optimistic bubble, ownership-guarded like every other write: creating
    // the session is an await, and the rail is clickable across it
    if (content != null)
      set((st) => (st.currentId === sid
        ? { messages: [...st.messages,
                       { role: "user",
                         content: content as ChatMessage["content"] }] }
        : {}));
    const ctl = new AbortController();
    set((st) => ({
      streams: { ...st.streams, [sid]: emptyTurn() },
      aborts: { ...st.aborts, [sid]: ctl },
      lastError: null,
    }));
    let stopped = false;
    try {
      await streamChat(
        sid,
        { message: content, ...(opts ?? {}) },
        (ev) => {
          set(patchTurn(sid, (t) => applyEvent(t, ev)));
          // R5-PERSIST: the same durable half, so the panel is already populated
          // the moment the turn ends rather than waiting for a reload.
          const durable = durableFromEvent(ev);
          if (durable) set(mergeSavedAgent(sid, durable));
        },
        ctl.signal,
      );
    } catch (e) {
      if ((e as Error).name === "AbortError") stopped = true;
      else
        set(patchTurn(sid, (t) => ({ ...t, error: errText(e) })));
    }
    // Read the partial BEFORE the reload below clears it. A stopped turn never
    // reaches the server's persist step, so what was on screen is the only
    // copy — throwing it away was the whole complaint about Stop.
    const mine = get().streams[sid] ?? null;
    const partial = stopped ? mine : null;
    // AUDIT F49: the same for the failure. It lives on the turn object, which
    // the reload below destroys tens of milliseconds after it was rendered.
    const failure = mine?.error ?? null;
    // reload the authoritative transcript; the streamed turn was a preview
    let msgs: ChatMessage[] | null = null;
    try {
      msgs = (await api.getSession(sid)).messages;
      // a completed regenerate folds the replaced reply into variants
      const pv = get().pendingVariants[sid];
      const last = msgs[msgs.length - 1];
      if (pv && last && last.role === "assistant") {
        const variants = [...(last.variants ?? []), ...pv.variants, pv.content]
          .filter(Boolean);
        msgs = [...msgs.slice(0, -1), { ...last, variants }];
        await api.updateSession(sid, { messages: msgs }).catch(() => {});
      }
      // Keep a stopped turn's text — but only if the server didn't already
      // save it (the abort can land after the turn finished persisting, and
      // appending then would duplicate the reply).
      const text = partial?.text?.trim();
      if (text && !alreadySaved(msgs, text)) {
        msgs = [...msgs, {
          role: "assistant",
          content: partial!.text + STOPPED_SUFFIX,
          ...(partial!.thinking ? { thinking: partial!.thinking } : {}),
        } as ChatMessage];
        // ONE writer for this: stop() only aborts, so nothing else is racing
        // this PUT with a different idea of the transcript
        await api.updateSession(sid, { messages: msgs }).catch(() => {});
      }
      // AUDIT F49: a failed turn persists nothing, so the reload would take
      // the prompt off screen along with the reason it failed.
      if (failure && content != null && !promptSurvived(msgs, content))
        msgs = [...msgs, { role: "user", content } as ChatMessage];
    } catch { /* keep what is on screen */ }
    // AUDIT F1: every terminal write is keyed to sid, and the transcript only
    // reaches the pane when sid is still the chat being read.
    set((st) => ({
      streams: without(st.streams, sid),
      aborts: without(st.aborts, sid),
      pendingVariants: without(st.pendingVariants, sid),
      ...(msgs && st.currentId === sid ? { messages: msgs } : {}),
      ...(failure ? { lastError: failure } : {}),
    }));
    void get().loadSessions();
    return true;
  },

  regenerate: async () => {
    const { currentId, messages } = get();
    if (!currentId || get().streams[currentId]) return;
    const msgs = [...messages];
    let old: ChatMessage | null = null;
    if (msgs.length && msgs[msgs.length - 1].role === "assistant")
      old = msgs.pop() ?? null;
    if (!msgs.length) return;
    const sid = currentId;

    // R3-UI-2. The reply is set aside BEFORE the destructive write, not after.
    //
    // `msgs` has the last assistant message POPPED OFF, so the PUT below
    // persists a transcript that no longer contains it. The set-aside take is
    // the only copy left, and it used to be recorded only once the write had
    // already returned AND the ownership guard had passed. So a user who
    // switched chats during that round-trip lost the reply twice over: the
    // server no longer had it, and the take that could have restored it was
    // never recorded. Nothing in the UI could bring it back.
    //
    // Recording it first makes the write recoverable: whatever happens to the
    // guard or the request, the previous reply is still in `pendingVariants`
    // for its own chat. This is the only ordering that cannot lose it.
    set((st) => ({
      pendingVariants: old
        ? { ...st.pendingVariants,
            [sid]: { content: old.content, variants: old.variants ?? [] } }
        : without(st.pendingVariants, sid),
    }));

    // AUDIT F1: the rail is live during that PUT. Without this guard, clicking
    // another chat mid-round-trip paints THIS chat's transcript into that one's
    // pane, and the send below starts a generation in a chat that never asked
    // for one. Bail rather than write: the user moved on.
    if (get().currentId !== sid) return;
    await api.updateSession(sid, { messages: msgs }).catch(() => {});
    set({ messages: msgs });
    await get().send(null);
  },

  continueTurn: async () => {
    const { currentId } = get();
    if (!currentId || get().streams[currentId]) return;
    await get().send(null, { continue: true });
  },

  flipVariant: async (dir) => {
    const { currentId, messages } = get();
    const last = messages[messages.length - 1];
    if (!currentId || !last || last.role !== "assistant") return;
    const pool = [...(last.variants ?? []), last.content];
    if (pool.length < 2) return;
    // rotate: current content goes to the back/front, next one becomes live
    const next = dir === 1 ? pool[0] : pool[pool.length - 2];
    const variants = pool.filter((v) => v !== next);
    const msgs = [...messages.slice(0, -1),
                  { ...last, content: next as ChatMessage["content"],
                    variants }];
    set({ messages: msgs });
    // R3-UI-2: an optimistic write with the failure discarded. The other take
    // appeared immediately and looked saved; on reload it was back to the old
    // one, with nothing anywhere saying the write had failed. Roll back so the
    // screen matches the server, and say what happened.
    try {
      await api.updateSession(currentId, { messages: msgs });
    } catch (e) {
      set({ messages, lastError: errText(e) });
    }
  },

  addImage: (dataUri) => set((st) => ({ images: [...st.images, dataUri] })),
  removeImage: (i) =>
    set((st) => ({ images: st.images.filter((_, j) => j !== i) })),

  setDraft: (text, sid) => {
    const key = sid ?? get().currentId ?? "";
    if ((get().drafts[key] ?? "") === text) return;
    set((st) => {
      const drafts = { ...st.drafts };
      // An empty draft leaves no entry behind, so the rail's unsent marker and
      // the map itself cannot accumulate a key per chat the user ever typed in.
      if (text) drafts[key] = text;
      else delete drafts[key];
      return { drafts };
    });
    // IMP-1: mirror to localStorage so a reload does not lose a half-written
    // message. Best-effort and size-capped — see chat/drafts.ts.
    saveDrafts(get().drafts);
  },

  // Stop ABORTS, and only aborts. Persisting the partial is send()'s job,
  // where the transcript reload already lives — two writers racing over the
  // same message list is how a stopped reply ends up duplicated or lost.
  // It cancels the chat ON SCREEN: the button lives in that chat's composer.
  //
  // But aborting the fetch only drops THIS BROWSER'S END of the stream. The
  // server's turn keeps running: for the native loop the engine generates on,
  // and for an external agent the process — and any subagents it spawned —
  // keeps going while the transcript says "stopped". So the server is told
  // first, and the local abort follows once that request has settled. Order
  // matters: a disconnect processed before the stop would tear down the
  // registry entry the stop route reads, and the stop would silently do
  // nothing.
  stop: () => {
    const { currentId } = get();
    if (!currentId) return;
    void api.stopSession(currentId)
      .catch(() => {})
      .then(() => get().aborts[currentId]?.abort());
  },

  compactChat: async () => {
    const { currentId } = get();
    if (!currentId) return "there is no chat to compact";
    // Refused locally as well as by the server, for the same reason the server
    // refuses: the fold and the turn's save would race, and one of them would
    // erase the other. Catching it here means no request is made at all. The
    // per-session stream entry is the same signal `selectStreaming` reads.
    if (get().streams[currentId]) {
      return "this chat is still replying — compact it after";
    }
    try {
      const out = await api.compactSession(currentId);
      // Reload rather than patching in place: compaction REPLACES the visible
      // transcript with a digest, so a client-side splice would have to
      // reproduce the server's own choice of what survived.
      const fresh = await api.getSession(currentId);
      set({ messages: fresh.messages ?? [] });
      return out.archived > 0
        ? `compacted — ${String(out.archived)} earlier messages folded into a digest`
        : "compacted";
    } catch (e) {
      return errText(e);
    }
  },

  clearError: () => set({ lastError: null }),

  pushNotice: (text) => set({ notice: text }),

  clearNotice: () => set({ notice: null }),

  setAcpConfigOption: (optionId, value) =>
    set((st) => {
      const id = st.currentId;
      if (!id) return {};
      const turn = st.streams[id];
      if (!turn) return {};
      // A NEW array and NEW option objects: `selectStreaming(s)?.acpConfig` is handed to
      // React through a zustand selector, so mutating in place would leave the reference
      // identical and the select would not re-render.
      const acpConfig = turn.acpConfig.map((c) =>
        String(c.id) === optionId ? { ...c, currentValue: value } : c);
      return { streams: { ...st.streams, [id]: { ...turn, acpConfig } } };
    }),
}));
