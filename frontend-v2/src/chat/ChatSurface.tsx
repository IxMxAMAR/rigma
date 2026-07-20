// The Chat surface: inner rail (sessions) + transcript + composer.
// Right sidecar (params, grounding) arrives in Phase 4.
import { useEffect, useRef, useState } from "react";
import { useApp } from "../store";
import Sidecar from "./Sidecar";
import Transcript from "./Transcript";
import { useChat } from "./chatStore";

function SessionRail() {
  const sessions = useChat((s) => s.sessions);
  const currentId = useChat((s) => s.currentId);
  const open = useChat((s) => s.open);
  const newChat = useChat((s) => s.newChat);
  const deleteChat = useChat((s) => s.deleteChat);
  const duplicateChat = useChat((s) => s.duplicateChat);
  const search = useChat((s) => s.search);
  const [q, setQ] = useState("");
  const timer = useRef<number | null>(null);
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
          value={q}
          onChange={(e) => {
            setQ(e.target.value);
            if (timer.current) window.clearTimeout(timer.current);
            timer.current = window.setTimeout(
              () => void search(e.target.value), 250);
          }}
          placeholder="search chats…"
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
            <span className="hidden group-hover:flex items-center pr-1 shrink-0">
              <a href={`/api/sessions/${s.id}/export?fmt=md`} download
                 title="export as markdown" aria-label={`export ${s.title}`}
                 className="px-1 text-muted hover:text-amber text-[12px]">↧</a>
              <button onClick={() => void duplicateChat(s.id)}
                      title="duplicate" aria-label={`duplicate ${s.title}`}
                      className="px-1 text-muted hover:text-primary text-[12px]">⧉</button>
              <button onClick={() => {
                        if (window.confirm(`Delete "${s.title || "untitled"}"?`))
                          void deleteChat(s.id);
                      }}
                      title="delete" aria-label={`delete ${s.title}`}
                      className="px-1 text-muted hover:text-red text-[12px]">×</button>
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
  // the engine's own prompt_tokens from the last completed turn — stored on
  // the assistant message by the server
  let ptoks = 0;
  for (let i = messages.length - 1; i >= 0; i--) {
    const st = (messages[i] as { stats?: { prompt_tokens?: number } }).stats;
    if (st?.prompt_tokens) { ptoks = st.prompt_tokens; break; }
  }
  if (!ctx || !ptoks) return null;
  const frac = Math.min(1, ptoks / ctx);
  return (
    <div className="flex items-center gap-2 px-1 pb-1">
      <div className="w-28 h-1 rounded-full bg-surface overflow-hidden">
        <div className={`h-full ${frac > 0.85 ? "bg-red" : frac > 0.6 ? "bg-amber" : "bg-moss"}`}
             style={{ width: `${frac * 100}%` }} />
      </div>
      <span className="font-mono text-[10.5px] text-muted">
        {Math.round(frac * 100)}% of {Math.round(ctx / 1024)}K
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
  const streaming = useChat((s) => s.streaming);
  const images = useChat((s) => s.images);
  const addImage = useChat((s) => s.addImage);
  const removeImage = useChat((s) => s.removeImage);
  const [draft, setDraft] = useState("");
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
    setDraft("");
    void send(text);
  };

  // autosize: content height up to ~7 lines
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 168) + "px";
  }, [draft]);

  return (
    <div className="shrink-0 px-6 pb-5 pt-2">
      <div className="max-w-[760px] mx-auto">
      <ContextMeter />
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
          className="shrink-0 text-muted hover:text-secondary text-[15px] pb-0.5"
        >
          ⊕
        </button>
        <textarea
          ref={ref}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          onPaste={(e) => {
            if (e.clipboardData?.files?.length) stage(e.clipboardData.files);
          }}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
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
          aria-label="Chat settings"
          className="absolute top-2 right-3 z-10 rounded-md bg-surface/80 hover:bg-float px-2 py-0.5 font-mono text-[12px] text-secondary"
        >
          {sidecar ? "⇥" : "⚙"}
        </button>
        <Transcript />
        <Composer />
      </div>
      <Sidecar open={sidecar} />
    </div>
  );
}
