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
  tool_trace?: { name: string; args?: unknown; result?: string }[];
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
  unsupported: string[];
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
   *  yet, each with what it would cost. */
  listHarnesses: () => j<HarnessMenu>("GET", "/api/harnesses"),
};
