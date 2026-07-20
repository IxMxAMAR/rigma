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
}

export interface Session {
  id: string;
  title: string;
  messages: ChatMessage[];
  use_rag?: boolean;
  use_tools?: boolean;
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
};
