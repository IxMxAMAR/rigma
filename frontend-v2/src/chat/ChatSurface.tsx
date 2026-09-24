// The Chat surface: inner rail (sessions) + transcript + composer.
// Right sidecar (params, grounding) arrives in Phase 4.
import { useEffect, useRef, useState } from "react";
import FloatWindow from "../FloatWindow";
import { Attach, Close, Download, Duplicate } from "../Icon";
import { CHAT_FILTER_EVENT } from "../lib/chatFilter";
import MacroStrip from "./MacroStrip";
import ModelPicker from "./ModelPicker";
import { useApp } from "../store";
import Sidecar from "./Sidecar";
import Transcript from "./Transcript";
import { selectStreaming, useChat } from "./chatStore";
import { isSendKey, isStopKey, isTypingTarget } from "./keyboard";

function SessionRail() {
  const sessions = useChat((s) => s.sessions);
  const currentId = useChat((s) => s.currentId);
  const open = useChat((s) => s.open);
  const newChat = useChat((s) => s.newChat);
  const deleteChat = useChat((s) => s.deleteChat);
  const duplicateChat = useChat((s) => s.duplicateChat);
  const search = useChat((s) => s.search);
  const drafts = useChat((s) => s.drafts);
  const [q, setQ] = useState("");
  const timer = useRef<number | null>(null);
  const filterRef = useRef<HTMLInputElement>(null);

  // IMP-9: `/` focuses the filter when the user is not already typing, and the
  // palette's "Filter chats" command reaches it through Ctrl+K. Ctrl+K itself
  // stays the palette's — see lib/chatFilter.ts for why.
  useEffect(() => {
    const focus = () => {
      filterRef.current?.focus();
      filterRef.current?.select();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "/" || e.ctrlKey || e.metaKey || e.altKey) return;
      if (isTypingTarget(e.target as HTMLElement | null)) return;
      e.preventDefault();
      focus();
    };
    window.addEventListener(CHAT_FILTER_EVENT, focus);
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener(CHAT_FILTER_EVENT, focus);
      window.removeEventListener("keydown", onKey);
    };
  }, []);
  return (
    <aside className="w-[230px] shrink-0 bg-panel/60 flex flex-col border-r border-white/5">
      <div className="p-2 flex flex-col gap-1.5">
        <button
          onClick={() => void newChat()}
          className="w-full rounded-md bg-surface hover:bg-float px-3 py-1.5 text-[13px] text-left"
        >
          + new chat
        </button>
        <input
          ref={filterRef}
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            if (timer.current) window.clearTimeout(timer.current);
            timer.current = window.setTimeout(
              () => void search(e.target.value), 250);
          }}
          placeholder="search chats…  (/)"
          aria-label="Search chats"
          className="w-full rounded-md bg-surface px-3 py-1 text-[12.5px] outline-none placeholder:text-muted"
        />
      </div>
      <nav className="flex-1 overflow-y-auto px-2 pb-2" aria-label="Chats">
        {sessions.map((s) => (
          <div key={s.id}
               className={`group flex items-center rounded-md ${
                 s.id === currentId ? "bg-surface" : "hover:bg-white/5"}`}>
            <button
              onClick={() => void open(s.id)}
              aria-current={s.id === currentId ? "true" : undefined}
              className={`flex-1 min-w-0 text-left px-3 py-1.5 text-[13px] truncate ${
                s.id === currentId ? "text-primary" : "text-secondary"}`}
              title={s.title}
            >
              {s.title || "untitled"}
            </button>
            {/* An unsent draft is otherwise invisible: you cannot tell whether
                the paragraph belongs to this chat or the last one. Shown at
                rest, not on hover — it is state, not an action. */}
            {drafts[s.id]?.trim() ? (
              <span role="img" aria-label={`unsent draft in ${s.title}`}
                    title="unsent draft"
                    className="shrink-0 pr-1 text-amber">•</span>
            ) : null}
            {/* Always rendered, not `hidden group-hover:flex`: a display:none
                control is outside the tab order, so export/duplicate/delete
                could not be reached by keyboard at all, and a touch device has
                no hover state so they did not exist there either (AUDIT F11-8).
                Muted at rest; full contrast on row hover or when one of them
                takes focus. */}
            <span className="flex items-center gap-0.5 pr-1.5 shrink-0 opacity-60
                             group-hover:opacity-100 focus-within:opacity-100">
              <a href={`/api/sessions/${s.id}/export?fmt=md`} download
                 title="export as markdown" aria-label={`export ${s.title}`}
                 className="p-0.5 text-muted hover:text-amber"><Download /></a>
              <button onClick={() => void duplicateChat(s.id)}
                      title="duplicate" aria-label={`duplicate ${s.title}`}
                      className="p-0.5 text-muted hover:text-primary"><Duplicate /></button>
              <button onClick={() => {
                        if (window.confirm(`Delete "${s.title || "untitled"}"?`))
                          void deleteChat(s.id);
                      }}
                      title="delete" aria-label={`delete ${s.title}`}
                      className="p-0.5 text-muted hover:text-red"><Close /></button>
            </span>
          </div>
        ))}
      </nav>
    </aside>
  );
}

function ContextMeter() {
  const messages = useChat((s) => s.messages);
  const currentId = useChat((s) => s.currentId);
  const ctx = useApp((s) => s.server?.ctx ?? 0);
  const [busy, setBusy] = useState(false);
  // engine-reported prompt_tokens from the last message that HAS stats —
  // but if newer messages exist past that point (tool-heavy turns could
  // miss stats), extend with a char-based estimate so the meter never
  // freezes on a stale number (owner report 2026-07-21: stuck at 8%)
  let ptoks = 0;
  let statsAt = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    const st = (messages[i] as { stats?: { prompt_tokens?: number } }).stats;
    if (st?.prompt_tokens) { ptoks = st.prompt_tokens; statsAt = i; break; }
  }
  let estimated = false;
  if (statsAt < messages.length - 1) {
    let extra = 0;
    for (let i = statsAt + 1; i < messages.length; i++) {
      const c = messages[i].content;
      extra += typeof c === "string" ? c.length
        : (c as unknown[]).length * 400;
    }
    if (extra > 0) { ptoks += Math.round(extra / 3); estimated = true; }
  }
  if (!ctx || !ptoks) return null;
  const frac = Math.min(1, ptoks / ctx);
  return (
    <div className="flex items-center gap-2 px-1 pb-1">
      <div className="w-28 h-1 rounded-full bg-surface overflow-hidden">
        <div className={`h-full ${frac > 0.85 ? "bg-red" : frac > 0.6 ? "bg-amber" : "bg-moss"}`}
             style={{ width: `${frac * 100}%` }} />
      </div>
      <span
        className="font-mono text-[10.5px] text-muted"
        title={"context window: how full the model's memory of this chat is. "
               + "Set at engine launch (Engine page) — not related to max "
               + "tokens, which caps reply length."}
      >
        ctx {estimated ? "~" : ""}{Math.round(frac * 100)}% of {Math.round(ctx / 1024)}K
      </span>
      {frac > 0.6 && currentId && (
        <button
          disabled={busy}
          onClick={async () => {
            setBusy(true);
            await fetch(`/api/sessions/${currentId}/compact`,
                        { method: "POST" }).catch(() => {});
            setBusy(false);
            void useChat.getState().open(currentId);
          }}
          className="font-mono text-[10.5px] text-amber hover:underline disabled:opacity-40"
        >
          {busy ? "compacting…" : "compact"}
        </button>
      )}
    </div>
  );
}

function Composer() {
  const send = useChat((s) => s.send);
  const stop = useChat((s) => s.stop);
  const streaming = useChat(selectStreaming);
  const currentId = useChat((s) => s.currentId);
  const images = useChat((s) => s.images);
  const addImage = useChat((s) => s.addImage);
  const removeImage = useChat((s) => s.removeImage);
  // The draft is keyed to the chat on screen (chatStore.drafts). As component
  // state it outlived the chat it was written for, so a paragraph typed in one
  // chapter and Entered after a switch was sent — and saved — in the other.
  const draft = useChat((s) => s.drafts[s.currentId ?? ""] ?? "");
  const setDraft = useChat((s) => s.setDraft);
  const ref = useRef<HTMLTextAreaElement>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const stage = (files: FileList | null) => {
    for (const f of files ?? []) {
      if (!f.type.startsWith("image/")) continue;
      const rd = new FileReader();
      rd.onload = () => typeof rd.result === "string" && addImage(rd.result);
      rd.readAsDataURL(f);
    }
  };

  const submit = () => {
    const text = draft.trim();
    if (!text || streaming) return;
    // The key is captured here, not read back after the await: send() creates a
    // session when there is none, and the rail stays clickable across it.
    const key = useChat.getState().currentId ?? "";
    setDraft("", key);
    // AUDIT F50: docs/audit-2026-09-04-full.md — the draft was cleared before
    // send()'s first await. A backend that was restarting took the typed
    // paragraph with it: no bubble, no error, no spinner, just an empty box.
    void send(text).then((started) => { if (!started) setDraft(text, key); });
  };

  // autosize: content height up to ~7 lines
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 168) + "px";
  }, [draft]);

  // IMP-9: switching chats puts the caret where the next message goes.
  useEffect(() => {
    ref.current?.focus();
  }, [currentId]);

  // IMP-9: Esc stops the turn — except when the palette owns Esc (it closes on
  // Esc) or focus is in a single-line input cancelling its own edit. See
  // chat/keyboard.ts for the rule.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (!isStopKey(e, { streaming: streaming !== null,
                          paletteOpen: useApp.getState().paletteOpen })) return;
      e.preventDefault();
      stop();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [streaming, stop]);

  return (
    <div className="shrink-0 px-6 pb-5 pt-2">
      <div className="max-w-[760px] mx-auto">
      <ContextMeter />
      <MacroStrip />
      {images.length > 0 && (
        <div className="flex gap-2 pb-2">
          {images.map((u, i) => (
            <div key={i} className="relative">
              <img src={u} alt={`attachment ${i + 1}`}
                   className="h-14 w-14 object-cover rounded-md" />
              <button onClick={() => removeImage(i)}
                      aria-label="remove image"
                      className="absolute -top-1.5 -right-1.5 w-4 h-4 rounded-full bg-float text-muted hover:text-red text-[10px] leading-none">
                ×
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-end gap-2 rounded-xl bg-surface px-3 py-2 focus-within:bg-float">
        <input ref={fileRef} type="file" accept="image/*" multiple hidden
               onChange={(e) => { stage(e.target.files); e.target.value = ""; }} />
        <button
          onClick={() => fileRef.current?.click()}
          aria-label="Attach images"
          title="attach images (needs a vision model)"
          className="shrink-0 text-muted hover:text-secondary pb-1"
        >
          <Attach />
        </button>
        <textarea
          ref={ref}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onPaste={(e) => {
            if (e.clipboardData?.files?.length) stage(e.clipboardData.files);
          }}
          onKeyDown={(e) => {
            if (isSendKey(e)) {
              e.preventDefault();
              submit();
            }
          }}
          rows={1}
          placeholder="Message the model…  (Enter to send, Shift+Enter for newline)"
          aria-label="Message"
          className="flex-1 bg-transparent resize-none outline-none text-[14px] placeholder:text-muted py-1"
        />
        {streaming ? (
          <button
            onClick={stop}
            className="shrink-0 rounded-md bg-red/15 text-red px-3 py-1.5 text-[13px] font-semibold"
          >
            stop
          </button>
        ) : (
          <button
            onClick={submit}
            disabled={!draft.trim()}
            className="shrink-0 rounded-md bg-amber/15 text-amber px-3 py-1.5 text-[13px] font-semibold disabled:opacity-40"
          >
            send
          </button>
        )}
      </div>
      {/* IMP-9: the key hints stay visible once the placeholder is gone. */}
      <div className="flex items-center gap-3 px-1 pt-1 font-mono text-[10.5px] text-muted">
        <span>enter send</span>
        <span>shift+enter newline</span>
        {streaming && <span className="text-amber">esc stop</span>}
      </div>
      </div>
    </div>
  );
}

export default function ChatSurface() {
  const loadSessions = useChat((s) => s.loadSessions);
  const [sidecar, setSidecar] = useState(false);
  useEffect(() => {
    void loadSessions();
  }, [loadSessions]);
  return (
    <div className="flex-1 flex min-w-0 min-h-0">
      <SessionRail />
      <div className="flex-1 flex flex-col min-w-0 min-h-0 relative">
        <button
          onClick={() => setSidecar(!sidecar)}
          aria-expanded={sidecar}
          aria-label="This chat's settings"
          title="this chat — grounding, sampling, workspace"
          className="absolute top-2 right-3 z-10 rounded-md bg-surface/80 hover:bg-float px-2 py-0.5 font-mono text-[12px] text-secondary"
        >
          ⚙
        </button>
        <Transcript />
        <Composer />
      </div>
      <FloatWindow id="chat-settings" title="this chat" open={sidecar}
                   onClose={() => setSidecar(false)}>
        <Sidecar />
      </FloatWindow>
      <ModelPicker />
    </div>
  );
}
