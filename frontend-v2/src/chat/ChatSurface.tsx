// The Chat surface: inner rail (sessions) + transcript + composer.
// Right sidecar (params, grounding) arrives in Phase 4.
import { useEffect, useRef, useState } from "react";
import Sidecar from "./Sidecar";
import Transcript from "./Transcript";
import { useChat } from "./chatStore";

function SessionRail() {
  const sessions = useChat((s) => s.sessions);
  const currentId = useChat((s) => s.currentId);
  const open = useChat((s) => s.open);
  const newChat = useChat((s) => s.newChat);
  return (
    <aside className="w-[220px] shrink-0 bg-panel/60 flex flex-col border-r border-white/5">
      <div className="p-2">
        <button
          onClick={() => void newChat()}
          className="w-full rounded-md bg-surface hover:bg-float px-3 py-1.5 text-[13px] text-left"
        >
          + new chat
        </button>
      </div>
      <nav className="flex-1 overflow-y-auto px-2 pb-2" aria-label="Chats">
        {sessions.map((s) => (
          <button
            key={s.id}
            onClick={() => void open(s.id)}
            aria-current={s.id === currentId ? "true" : undefined}
            className={`w-full text-left px-3 py-1.5 rounded-md text-[13px] truncate ${
              s.id === currentId
                ? "bg-surface text-primary"
                : "text-secondary hover:bg-white/5"
            }`}
            title={s.title}
          >
            {s.title || "untitled"}
          </button>
        ))}
      </nav>
    </aside>
  );
}

function Composer() {
  const send = useChat((s) => s.send);
  const stop = useChat((s) => s.stop);
  const streaming = useChat((s) => s.streaming);
  const [draft, setDraft] = useState("");
  const ref = useRef<HTMLTextAreaElement>(null);

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
      <div className="max-w-[760px] mx-auto flex items-end gap-2 rounded-xl bg-surface px-3 py-2 focus-within:bg-float">
        <textarea
          ref={ref}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
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
