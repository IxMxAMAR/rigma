// Thin typed wrappers over the existing REST surface. The 38 endpoints are
// the contract; v2 adapts to them, never the reverse.
export interface SessionSummary {
  id: string;
  title: string;
  message_count?: number;
  use_rag?: boolean;
}

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string | { type: string; [k: string]: unknown }[];
  /** `ok` is the server's own record of whether the call succeeded. It was
   *  always persisted and never read: the chip inferred success from the result
   *  text starting with "error", which is a guess. `null` means unknown. */
  tool_trace?: {
    name: string;
    args?: unknown;
    result?: string;
    ok?: boolean | null;
  }[];
  /** What a `delegate` research helper did behind the context firewall. The
   *  server has always recorded this; nothing rendered it, so the exploration
   *  was invisible — one chip, then an answer. UI-only: it never re-enters
   *  context, which is the whole point of the firewall. */
  delegate_trace?: {
    name: string;
    args?: unknown;
    ok?: boolean;
    blocked?: boolean;
    ts?: number;
  }[];
  variants?: unknown[];
  kind?: string;          // "tool_result" = model-context carrier, not UI
  notice?: string;        // server-authored status line — shown, never fed
  /** Which EXTERNAL agent backend wrote this reply, and what to call it. Both
   *  are absent on a reply the built-in loop wrote: the badge explains the
   *  unusual case, so the ordinary one carries no marker. */
  harness?: string;
  permission?: string;
  harness_label?: string;
}

export interface Session {
  id: string;
  title: string;
  messages: ChatMessage[];
  use_rag?: boolean;
  use_tools?: boolean;
  harness?: string;
  permission?: string;
  /** R6-ACP-TURN: which of mcode's two wires drives this chat — `exec` (default) or
   *  `acp`. Only meaningful for mcode, and a SESSION field rather than a turn's,
   *  because the mcode session id belongs to one wire. */
  mcode_transport?: string;
  /** AUDIT 13-2/13-3 per-chat tool grants. None has a server default, so a
   *  chat that has never been toggled reports all three absent — which the UI
   *  must read as OFF, never as granted (see chat/grants.ts). */
  confirm_exec?: boolean;
  allow_absolute_reads?: boolean;
  allow_outbound_post?: boolean;
  /** R5-PERSIST: the agent's durable state — its goal, its todo list, whether it
   *  is in plan mode. Written by the SERVER mid-turn from the backend's own state
   *  events, not by the PATCH surface, and the payloads are the raw wire shapes so
   *  that `chat/goal.ts` stays the single place the two backends' field names are
   *  reconciled.
   *
   *  Deliberately has no `subagents`: a subagent row names a child process that
   *  belongs to the turn that spawned it, so restoring one would claim a child
   *  that is long gone. */
  agent_state?: SavedAgentState;
}

/** R6-ACP-APPROVE: the answer to a permission request a turn is BLOCKED on.
 *
 *  Only mcode over ACP produces one. DSH's SDK wire exposes exactly three methods —
 *  `initialize`, `session/prompt`, `shutdown` — so a DSH approval is display-only and
 *  this call never applies to it. The server refuses with 409 when the chat is not
 *  waiting, or when `requestId` names a request that is no longer the one in flight,
 *  so a caller must surface that rather than assume the decision was applied. */
export interface ApprovalAnswer {
  allow: boolean;
  /** Quotes the request being answered, so a stale card cannot decide a different
   *  question than the one on screen. */
  requestId?: string;
}

/** The durable half of the agent's state. See `Session.agent_state`. */
/** R6-WORKFLOW: one programmatic-tool-calling run, folded from four DSH events.
 *
 *  WHY IT IS FOLDED RATHER THAN APPENDED. DSH reports a run as four separate session
 *  events — `run-start`, `agent-start`, `agent-end`, `run-end` — joined only by a
 *  `runId`, and an agent's start is joined to its own end by `seq`. A list of four
 *  lines would make the reader reconstruct the run themselves, which is exactly what
 *  the UI is for.
 *
 *  Until R6-WORKFLOW all four reached the UI as the literal string
 *  "session.event tool-workflow/agent-start", and that was an ACCIDENT: the runner's
 *  notice filter matches on the substring "tool", which "tool-workflow" contains. */
export interface WorkflowAgent {
  /** `agent-start`'s `seq`, which pairs the start with its own end. */
  seq: number;
  /** The agent's name, from `agent-start`. */
  label: string;
  /** The child session that ran it, when DSH named one. */
  childId: string;
  /** `agent-start`'s optional phase. */
  phase: string;
  /** `agent-end`'s verdict — empty while the agent is still running. */
  outcome: string;
}

export interface WorkflowRun {
  runId: string;
  /** `run-start`'s name for the run. */
  name: string;
  /** `run-end`'s reason, empty while the run is still going. */
  stopReason: string;
  /** Finished, from the presence of `run-end`. */
  done: boolean;
  agents: WorkflowAgent[];
}

export interface SavedAgentState {
  goal?: Record<string, unknown> | null;
  todos?: { content: string; status: string }[];
  plan_mode?: boolean;
  /** R6-WORKFLOW: programmatic tool calling, which Rigma mounts and therefore fires.
   *
   *  Persisted for the same reason `todos` is: a run is a record of what this
   *  conversation DID, and a reload that dropped it would show a chat whose agents had
   *  never run. Keyed by `runId` in the live turn and a list here, because the durable
   *  copy is written once at the end of a run rather than folded event by event. */
  workflow?: WorkflowRun[];
}

/** One agent backend, and what choosing it would cost.
 *
 *  `runnable` (can a turn be handed to it?) is deliberately separate from
 *  `installed` (is it on this machine?): a backend can be present and still not
 *  wired up, and collapsing the two is how a probe becomes a silent fallback. */
export interface HarnessInfo {
  name: string;
  label: string;
  drives: string;
  runnable: boolean;
  installed: boolean;
  needs: string;
  wire: string;
  /** The build this adapter was MEASURED against. Empty means nobody has
   *  checked — which is not the same as "it matches". `rigma harness` compares
   *  it with whatever is installed. */
  verified: string;
  /** Rigma's own loop. It ships with Rigma, so it has no separate build to
   *  drift from and the "nothing has been verified" wording would be wrong —
   *  it made the built-in read as the least trustworthy option. */
  built_in?: boolean;
  /** What the backend ACTUALLY is, when the caller asked for a check. Empty
   *  when nobody asked. */
  version?: string;
  /** `true` it moved, `false` it still matches, `null` one side could not say.
   *  `null` is not `false` — "I could not tell" is a different sentence to the
   *  owner than "it agrees". Absent when no check was run. */
  drift?: boolean | null;
  /** R3-HARN-1: whether this backend APPLIES the chat's `permission` setting.
   *  `permission` is part of the adapter contract, but a backend with no such
   *  notion ignores it — DSH's confinement is its own bundle's business. The
   *  UI used to render the "off — no tools at all" selector for every
   *  non-native backend, so a user could arm a safety setting and have nothing
   *  happen. Optional: an older server that does not send it is treated as
   *  `true`, which keeps the selector (the pre-fix behaviour) rather than
   *  silently removing a control. */
  honours_permission?: boolean;
  unsupported: string[];
  /** What RIGMA ADDS to this backend, when it would otherwise be missing it.
   *  The other half of `unsupported`: that says what the backend does not get
   *  from Rigma, this says what Rigma mounts into it. Both are needed to answer
   *  "what can this actually do for me". DSH's minimal profile ships almost no
   *  model-facing tools, so without this line the goals, subagents and todos
   *  Rigma patches in appear with nothing saying where they came from.
   *  Optional: an older server does not send it. */
  capabilities?: string[];
  pending: string;
}

export interface HarnessMenu {
  built_in: string;
  endpoint: string;
  harnesses: HarnessInfo[];
}

async function j<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { error?: string };
    throw new Error(e.error ?? `server replied ${r.status}`);
  }
  return r.json() as Promise<T>;
}

export const api = {
  listSessions: () => j<SessionSummary[]>("GET", "/api/sessions"),
  getSession: (id: string) => j<Session>("GET", `/api/sessions/${id}`),
  createSession: () => j<Session>("POST", "/api/sessions", {}),
  updateSession: (id: string, patch: Record<string, unknown>) =>
    j<Session>("POST", `/api/sessions/${id}`, patch),
  /** Answer the permission request this chat is waiting on.
   *
   *  R6-ACP-APPROVE. This is the other half of the ACP handshake: the client's reader
   *  thread blocks on an Event and this call sets it. Left unanswered, mcode waits
   *  forever and the TRANSPORT wedges — which is why the turn says it is waiting
   *  rather than silently proceeding. */
  answerApproval: (id: string, answer: ApprovalAnswer) =>
    j<{ ok: boolean; requestId?: string; allow?: boolean }>(
      "POST", `/api/sessions/${id}/approval`, answer),
  deleteSession: (id: string) => j<unknown>("DELETE", `/api/sessions/${id}`),
  /** Stop the turn running in this chat. Distinct from aborting the fetch that
   *  is reading it: that only drops this browser's end, and the backend keeps
   *  working — for an external agent, keeps running and keeps spawning
   *  subagents while the UI says stopped. `stopped` says whether anything was
   *  actually running, so the UI can be honest about what it did. */
  stopSession: (id: string) =>
    j<{ ok: boolean; stopped: boolean }>(
      "POST", `/api/sessions/${id}/stop`, {}),
  /** The honest menu: every backend, including the ones that cannot run a turn
   *  yet, each with what it would cost.
   *
   *  `check` additionally asks each backend what version it actually is, which
   *  costs a subprocess per backend — so the caller asks for the plain list
   *  first and this one after, and the warning arrives late rather than the
   *  menu. */
  listHarnesses: (check = false) =>
    j<HarnessMenu>("GET", `/api/harnesses${check ? "?check=1" : ""}`),
  /** Fold this chat's older turns into a digest. The server refuses while the
   *  chat is mid-reply (409) and when there is nothing to fold (400), and both
   *  refusals are worth surfacing verbatim: a silent no-op would read as the
   *  command not working. */
  compactSession: (id: string, keep = 6) =>
    j<{ session: Session; archived: number }>(
      "POST", `/api/sessions/${id}/compact`, { keep }),
};
