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
import { api, type ChatMessage, type SessionSummary } from "../lib/api";
import { runMacroStream } from "../lib/methods";
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
  /** set while a macro drives this turn, so the UI can say which step */
  macro: { label: string; index: number; total: number } | null;
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

export const emptyTurn = (): StreamingTurn => ({
  text: "",
  thinking: "",
  chips: [],
  citations: [],
  error: null,
  macro: null,
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

export interface ChatState {
  sessions: SessionSummary[];
  currentId: string | null;
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
  images: string[];               // data URIs staged in the composer

  loadSessions: () => Promise<void>;
  search: (q: string) => Promise<void>;
  open: (id: string) => Promise<void>;
  newChat: () => Promise<void>;
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
  stop: () => void;
  clearError: () => void;
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
  images: [],

  loadSessions: async () => {
    try {
      set({ sessions: await api.listSessions() });
    } catch (e) {
      set({ lastError: errText(e) });
    }
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
    // the turn belongs to a chat that is about to stop existing: cancel it
    // rather than leaving it streaming into a deleted session
    get().aborts[id]?.abort();
    await api.deleteSession(id).catch(() => {});
    set((st) => ({
      streams: without(st.streams, id),
      aborts: without(st.aborts, id),
      pendingVariants: without(st.pendingVariants, id),
      ...(st.currentId === id ? { currentId: null, messages: [] } : {}),
    }));
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

  // AUDIT F1: docs/audit-2026-09-04-full.md — opening another chat no longer
  // touches the in-flight turn. It keeps streaming under its own id, and comes
  // back on screen intact if you return to it.
  open: async (id) => {
    try {
      const s = await api.getSession(id);
      set({ currentId: id, messages: s.messages, lastError: null });
    } catch (e) {
      // AUDIT F50: a session click that failed used to do nothing at all
      set({ lastError: errText(e) });
    }
  },

  newChat: async () => {
    try {
      const s = await api.createSession();
      set({ currentId: s.id, messages: [], lastError: null });
      await get().loadSessions();
    } catch (e) {
      set({ lastError: errText(e) });
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
        (ev) =>
          set(patchTurn(sid, (t) => applyEvent(t, ev))),
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
    await api.updateSession(sid, { messages: msgs }).catch(() => {});
    // AUDIT F1: the rail is live during that PUT. Without this guard, clicking
    // another chat mid-round-trip paints THIS chat's transcript into that one's
    // pane, and the send below starts a generation in a chat that never asked
    // for one — the same defect F1 describes, through a one-round-trip window
    // instead of a whole generation. Bail rather than write: the user moved on.
    if (get().currentId !== sid) return;
    set((st) => ({
      messages: msgs,
      // AUDIT F1: the set-aside take belongs to THIS chat. As one global slot
      // it was folded into whichever session's turn returned first.
      pendingVariants: old
        ? { ...st.pendingVariants,
            [sid]: { content: old.content,
                     variants: old.variants ?? [] } }
        : without(st.pendingVariants, sid),
    }));
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
    await api.updateSession(currentId, { messages: msgs }).catch(() => {});
  },

  addImage: (dataUri) => set((st) => ({ images: [...st.images, dataUri] })),
  removeImage: (i) =>
    set((st) => ({ images: st.images.filter((_, j) => j !== i) })),

  // Stop ABORTS, and only aborts. Persisting the partial is send()'s job,
  // where the transcript reload already lives — two writers racing over the
  // same message list is how a stopped reply ends up duplicated or lost.
  // It cancels the chat ON SCREEN: the button lives in that chat's composer.
  stop: () => {
    const { currentId } = get();
    if (currentId) get().aborts[currentId]?.abort();
  },

  clearError: () => set({ lastError: null }),
}));
