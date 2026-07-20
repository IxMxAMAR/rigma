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
  images: string[];               // data URIs staged in the composer
  pendingVariant: { content: unknown; variants: unknown[] } | null;

  loadSessions: () => Promise<void>;
  search: (q: string) => Promise<void>;
  open: (id: string) => Promise<void>;
  newChat: () => Promise<void>;
  deleteChat: (id: string) => Promise<void>;
  duplicateChat: (id: string) => Promise<void>;
  send: (message: string | null, opts?: Record<string, unknown>) => Promise<void>;
  regenerate: () => Promise<void>;
  continueTurn: () => Promise<void>;
  flipVariant: (dir: 1 | -1) => Promise<void>;
  addImage: (dataUri: string) => void;
  removeImage: (i: number) => void;
  stop: () => void;
}

export const useChat = create<ChatState>((set, get) => ({
  sessions: [],
  currentId: null,
  messages: [],
  streaming: null,
  abort: null,
  images: [],
  pendingVariant: null,

  loadSessions: async () => {
    set({ sessions: await api.listSessions() });
  },

  search: async (q) => {
    if (!q.trim()) return get().loadSessions();
    try {
      const r = await fetch(`/api/sessions/search?q=${encodeURIComponent(q)}`);
      const d: unknown = await r.json();
      if (r.ok && Array.isArray(d)) set({ sessions: d as SessionSummary[] });
    } catch { /* keep list */ }
  },

  deleteChat: async (id) => {
    await api.deleteSession(id).catch(() => {});
    if (get().currentId === id)
      set({ currentId: null, messages: [], streaming: null });
    await get().loadSessions();
  },

  duplicateChat: async (id) => {
    try {
      const r = await fetch(`/api/sessions/${id}/duplicate`, { method: "POST" });
      const d = (await r.json()) as { id?: string };
      await get().loadSessions();
      if (d.id) await get().open(d.id);
    } catch { /* nothing */ }
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
    const staged = get().images;
    let content: ChatMessage["content"] | null = message;
    if (message != null && staged.length > 0) {
      content = [
        { type: "text", text: message },
        ...staged.map((u) => ({ type: "image_url", image_url: { url: u } })),
      ] as ChatMessage["content"];
      set({ images: [] });
    }
    if (content != null)
      set((st) => ({
        messages: [...st.messages,
                   { role: "user", content: content as ChatMessage["content"] }],
      }));
    const ctl = new AbortController();
    set({ streaming: emptyTurn(), abort: ctl });
    try {
      await streamChat(
        id,
        { message: content, ...(opts ?? {}) },
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
      let msgs = s.messages;
      // a completed regenerate folds the replaced reply into variants
      const pv = get().pendingVariant;
      const last = msgs[msgs.length - 1];
      if (pv && last && last.role === "assistant") {
        const variants = [...(last.variants ?? []), ...pv.variants, pv.content]
          .filter(Boolean);
        msgs = [...msgs.slice(0, -1), { ...last, variants }];
        await api.updateSession(id, { messages: msgs }).catch(() => {});
      }
      set({ messages: msgs, streaming: null, abort: null,
            pendingVariant: null });
    } catch {
      set({ streaming: null, abort: null, pendingVariant: null });
    }
    void get().loadSessions();
  },

  regenerate: async () => {
    const { currentId, messages, streaming } = get();
    if (!currentId || streaming) return;
    const msgs = [...messages];
    let old: ChatMessage | null = null;
    if (msgs.length && msgs[msgs.length - 1].role === "assistant")
      old = msgs.pop() ?? null;
    if (!msgs.length) return;
    await api.updateSession(currentId, { messages: msgs }).catch(() => {});
    set({
      messages: msgs,
      pendingVariant: old
        ? { content: old.content, variants: old.variants ?? [] }
        : null,
    });
    await get().send(null);
  },

  continueTurn: async () => {
    if (!get().currentId || get().streaming) return;
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
    await api.updateSession(currentId, { messages: msgs }).catch(() => {});
  },

  addImage: (dataUri) => set((st) => ({ images: [...st.images, dataUri] })),
  removeImage: (i) =>
    set((st) => ({ images: st.images.filter((_, j) => j !== i) })),

  stop: () => {
    get().abort?.abort();
  },
}));
