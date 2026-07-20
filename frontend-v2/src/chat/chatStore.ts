// Chat state. The heart is applyEvent(): a PURE reducer from SSE events to
// streaming state, unit-tested in isolation. Chips are keyed by the server's
// call id — the structural fix for the bug class caught live 2026-07-21,
// where order-based matching hung results on the wrong rows. FIFO is only
// the fallback for id-less legacy events.
import { create } from "zustand";
import { api, type ChatMessage, type SessionSummary } from "../lib/api";
import { streamChat, type SseEvent } from "../lib/sse";

export interface Chip {
  id: string;
  name: string;
  args?: unknown;
  result?: string;
  state: "running" | "done";
}

export interface StreamingTurn {
  text: string;
  thinking: string;
  chips: Chip[];
  citations: unknown[];
  error: string | null;
}

export const emptyTurn = (): StreamingTurn => ({
  text: "",
  thinking: "",
  chips: [],
  citations: [],
  error: null,
});

/** Pure: fold one SSE event into the streaming turn. Returns a NEW object —
 *  functional updates only, so React sees every change and nothing aliases. */
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
        return { ...c, result: String(d.result ?? ""), state: "done" as const };
      });
      return { ...turn, chips };
    }
    case "citations":
      return { ...turn, citations: (d.citations as unknown[]) ?? [] };
    case "error":
      return { ...turn, error: String(d.message ?? "unknown error") };
    case "message":
    default:
      return d.delta != null
        ? { ...turn, text: turn.text + String(d.delta) }
        : turn;
  }
}

interface ChatState {
  sessions: SessionSummary[];
  currentId: string | null;
  messages: ChatMessage[];
  streaming: StreamingTurn | null;
  abort: AbortController | null;

  loadSessions: () => Promise<void>;
  open: (id: string) => Promise<void>;
  newChat: () => Promise<void>;
  send: (message: string | null, opts?: Record<string, unknown>) => Promise<void>;
  stop: () => void;
}

export const useChat = create<ChatState>((set, get) => ({
  sessions: [],
  currentId: null,
  messages: [],
  streaming: null,
  abort: null,

  loadSessions: async () => {
    set({ sessions: await api.listSessions() });
  },

  open: async (id) => {
    const s = await api.getSession(id);
    set({ currentId: id, messages: s.messages, streaming: null });
  },

  newChat: async () => {
    const s = await api.createSession();
    set({ currentId: s.id, messages: [], streaming: null });
    await get().loadSessions();
  },

  send: async (message, opts) => {
    const { currentId, streaming } = get();
    if (streaming) return; // one turn at a time
    let id = currentId;
    if (!id) {
      const s = await api.createSession();
      id = s.id;
      set({ currentId: id });
    }
    if (message != null)
      set((st) => ({
        messages: [...st.messages, { role: "user", content: message }],
      }));
    const ctl = new AbortController();
    set({ streaming: emptyTurn(), abort: ctl });
    try {
      await streamChat(
        id,
        { message, ...(opts ?? {}) },
        (ev) =>
          set((st) => ({
            streaming: st.streaming ? applyEvent(st.streaming, ev) : null,
          })),
        ctl.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError")
        set((st) => ({
          streaming: st.streaming
            ? { ...st.streaming, error: (e as Error).message }
            : null,
        }));
    }
    // reload the authoritative transcript; the streamed turn was a preview
    try {
      const s = await api.getSession(id);
      set({ messages: s.messages, streaming: null, abort: null });
    } catch {
      set({ streaming: null, abort: null });
    }
    void get().loadSessions();
  },

  stop: () => {
    get().abort?.abort();
  },
}));
