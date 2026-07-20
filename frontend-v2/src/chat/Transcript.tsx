// Transcript: persisted messages + the live streaming turn. Thinking blocks
// collapse once the reply starts; chips expand to show their result.
import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../lib/api";
import Markdown from "./Markdown";
import { useChat, type Chip, type StreamingTurn } from "./chatStore";

function ChipRow({ chip }: { chip: Chip }) {
  return (
    <details className="rounded-md bg-surface/60 open:bg-surface">
      <summary className="flex items-center gap-2 px-3 py-1.5 cursor-pointer list-none font-mono text-[12px]">
        <span
          className={
            chip.state === "running"
              ? "text-amber animate-pulse"
              : chip.result?.startsWith("error")
                ? "text-red"
                : "text-moss"
          }
          aria-label={chip.state}
        >
          {chip.state === "running" ? "◌" : chip.result?.startsWith("error") ? "✕" : "✓"}
        </span>
        <span className="font-semibold text-primary">{chip.name}</span>
        <span className="text-muted truncate">{previewArgs(chip.args)}</span>
      </summary>
      {chip.result && (
        <pre className="px-3 pb-2 pt-1 text-[12px] font-mono text-secondary whitespace-pre-wrap break-words max-h-56 overflow-y-auto">
          {chip.result}
        </pre>
      )}
    </details>
  );
}

// identity, never payload — same rule the legacy chips learned the hard way
function previewArgs(args: unknown): string {
  if (!args || typeof args !== "object") return "";
  const a = args as Record<string, unknown>;
  for (const k of ["path", "paths", "pattern", "question", "query", "cmd", "task", "action"]) {
    if (a[k] !== undefined) {
      const v = Array.isArray(a[k]) ? (a[k] as unknown[]).join(", ") : String(a[k]);
      const size = typeof a.content === "string" ? ` · ${(a.content as string).length} chars` : "";
      return v.slice(0, 56) + (v.length > 56 ? "…" : "") + size;
    }
  }
  return Object.keys(a).join(", ").slice(0, 56);
}

function Thinking({ text, live }: { text: string; live: boolean }) {
  const [open, setOpen] = useState(live);
  useEffect(() => {
    if (!live) setOpen(false);
  }, [live]);
  if (!text) return null;
  return (
    <div className="rounded-md bg-panel/70">
      <button
        className="w-full text-left px-3 py-1.5 font-mono text-[11.5px] text-muted hover:text-secondary"
        onClick={() => setOpen(!open)}
        aria-expanded={open}
      >
        {open ? "▾" : "▸"} thinking
      </button>
      {open && (
        <div className="px-3 pb-2 text-[12.5px] text-muted whitespace-pre-wrap max-h-64 overflow-y-auto">
          {text}
        </div>
      )}
    </div>
  );
}

function Bubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === "user";
  const text =
    typeof m.content === "string"
      ? m.content
      : m.content
          .map((p) => (p.type === "text" ? String(p.text ?? "") : "[image]"))
          .join(" ");
  // server-side bookkeeping messages (TOOL RESULT …) are machine chatter in
  // an agentic chat; render them compactly, not as fake user turns
  const isMachine = isUser && /^(TOOL RESULT |### RUN STATE)/.test(text);
  if (isMachine)
    return (
      <div className="font-mono text-[11.5px] text-muted px-1 truncate" title={text}>
        {text.split("\n")[0]}
      </div>
    );
  return (
    <div className={isUser ? "flex justify-end" : ""}>
      <div
        className={
          isUser
            ? "max-w-[78%] rounded-lg bg-surface px-4 py-2.5 text-[14px] whitespace-pre-wrap break-words"
            : "max-w-full text-[14px]"
        }
      >
        {isUser ? text : <Markdown text={text} />}
      </div>
    </div>
  );
}

function LiveTurn({ turn }: { turn: StreamingTurn }) {
  return (
    <div className="flex flex-col gap-2">
      <Thinking text={turn.thinking} live={turn.text === ""} />
      {turn.chips.length > 0 && (
        <div className="flex flex-col gap-1">
          {turn.chips.map((c) => (
            <ChipRow key={c.id} chip={c} />
          ))}
        </div>
      )}
      {turn.text && <Markdown text={turn.text} />}
      {turn.error && (
        <div className="rounded-md bg-red/10 text-red px-3 py-2 text-[13px]">
          {turn.error}
        </div>
      )}
      {!turn.text && !turn.error && (
        <div className="font-mono text-[12px] text-muted animate-pulse">…</div>
      )}
    </div>
  );
}

const WINDOW = 150;

export default function Transcript() {
  const messages = useChat((s) => s.messages);
  const streaming = useChat((s) => s.streaming);
  const currentId = useChat((s) => s.currentId);
  const [shown, setShown] = useState(WINDOW);
  useEffect(() => setShown(WINDOW), [currentId]);
  const endRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  useEffect(() => {
    if (stick.current) endRef.current?.scrollIntoView({ block: "end" });
  }, [messages, streaming]);

  return (
    <div
      className="flex-1 overflow-y-auto px-6 py-4"
      onScroll={(e) => {
        const el = e.currentTarget;
        stick.current = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
      }}
    >
      <div className="max-w-[760px] mx-auto flex flex-col gap-4">
        {messages.length === 0 && !streaming && (
          <div className="text-center pt-24">
            <div className="font-mono text-[12px] text-muted uppercase tracking-[0.1em] mb-2">
              new conversation
            </div>
            <p className="text-secondary text-[13.5px]">
              Send a message to the loaded model. Ctrl+K for commands.
            </p>
          </div>
        )}
        {messages.length > shown && (
          <button
            onClick={() => setShown((n) => n + WINDOW)}
            className="self-center rounded-md bg-surface hover:bg-float px-3 py-1 font-mono text-[12px] text-secondary"
          >
            show {Math.min(WINDOW, messages.length - shown)} earlier messages
          </button>
        )}
        {messages.slice(-shown).map((m, i) => (
          <Bubble key={messages.length - shown + i} m={m} />
        ))}
        {streaming && <LiveTurn turn={streaming} />}
        <div ref={endRef} />
      </div>
    </div>
  );
}
