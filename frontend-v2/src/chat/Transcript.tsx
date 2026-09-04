// Transcript: persisted messages + the live streaming turn. Thinking blocks
// collapse once the reply starts; chips expand to show their result.
import { useEffect, useRef, useState } from "react";
import type { ChatMessage } from "../lib/api";
import Markdown from "./Markdown";
import {
  selectStreaming, useChat, type Chip, type StreamingTurn,
} from "./chatStore";

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
  const body = useRef<HTMLDivElement>(null);
  const stick = useRef(true);
  useEffect(() => {
    if (!live) setOpen(false);
  }, [live]);
  useEffect(() => {
    // follow the newest thinking while streaming, unless the user scrolled
    // up to read something (same stickiness rule as the transcript)
    const el = body.current;
    if (el && live && stick.current) el.scrollTop = el.scrollHeight;
  }, [text, live, open]);
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
        <div
          ref={body}
          onScroll={(e) => {
            const el = e.currentTarget;
            stick.current =
              el.scrollHeight - el.scrollTop - el.clientHeight < 40;
          }}
          className="px-3 pb-2 text-[12.5px] text-muted whitespace-pre-wrap max-h-64 overflow-y-auto"
        >
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
          .filter((p) => p.type === "text")
          .map((p) => String(p.text ?? ""))
          .join(" ");
  // attached images render as real thumbnails, not a "[image]" placeholder
  // (owner report 2026-07-21) — the data URIs are right there in the parts
  const images =
    typeof m.content === "string"
      ? []
      : m.content
          .filter((p) => p.type === "image_url")
          .map((p) =>
            String((p as { image_url?: { url?: string } }).image_url?.url ?? ""))
          .filter(Boolean);
  // server-side bookkeeping messages (TOOL RESULT …) are machine chatter in
  // an agentic chat; render them compactly, not as fake user turns
  const isMachine = isUser && /^(TOOL RESULT |### RUN STATE)/.test(text);
  if (isMachine)
    return (
      <div className="font-mono text-[11.5px] text-muted px-1 truncate" title={text}>
        {text.split("\n")[0]}
      </div>
    );
  // finished turns keep their tool chips: the trace is persisted precisely
  // so it can re-render (chips used to vanish the moment a reply completed —
  // owner assumed it was a design choice; it was a v2 parity gap)
  const trace = (!isUser && m.tool_trace) || [];
  return (
    <div className={isUser ? "flex justify-end" : ""}>
      <div
        className={
          isUser
            ? "max-w-[78%] rounded-lg bg-surface px-4 py-2.5 text-[14px] whitespace-pre-wrap break-words"
            : "max-w-full text-[14px]"
        }
      >
        {trace.length > 0 && (
          <div className="flex flex-col gap-1 mb-2">
            {trace.map((t, i) => (
              <ChipRow
                key={i}
                chip={{ id: `trace-${i}`, name: t.name, args: t.args,
                        result: t.result, state: "done" }}
              />
            ))}
          </div>
        )}
        {images.length > 0 && (
          <div className="flex flex-wrap gap-2 mb-2">
            {images.map((src, i) => (
              <img
                key={i}
                src={src}
                alt={`attachment ${i + 1}`}
                className="max-h-48 max-w-full rounded-md object-contain"
              />
            ))}
          </div>
        )}
        {isUser ? text : <Markdown text={text} />}
        {m.notice && (
          <p className="text-[12px] italic text-muted mt-1.5">{m.notice}</p>
        )}
      </div>
    </div>
  );
}

function MessageActions({ m }: { m: ChatMessage }) {
  const regenerate = useChat((s) => s.regenerate);
  const continueTurn = useChat((s) => s.continueTurn);
  const flipVariant = useChat((s) => s.flipVariant);
  const takes = (m.variants?.length ?? 0) + 1;
  return (
    <div className="flex items-center gap-3 pt-1 opacity-0 group-hover/msg:opacity-100 font-mono text-[11.5px] text-muted">
      <button onClick={() => void regenerate()} className="hover:text-amber">
        regenerate
      </button>
      <button onClick={() => void continueTurn()} className="hover:text-amber">
        continue
      </button>
      {takes > 1 && (
        <span className="flex items-center gap-1">
          <button onClick={() => void flipVariant(-1)} aria-label="previous take"
                  className="hover:text-primary">◂</button>
          {takes} takes
          <button onClick={() => void flipVariant(1)} aria-label="next take"
                  className="hover:text-primary">▸</button>
        </span>
      )}
    </div>
  );
}

function Working({ label }: { label: string }) {
  // a hairline "…" was invisible on a 2K screen (owner report 2026-07-21):
  // the working state earns a real indicator — dot, words, elapsed time
  const [secs, setSecs] = useState(0);
  useEffect(() => {
    const t0 = Date.now();
    const t = setInterval(() => setSecs(Math.floor((Date.now() - t0) / 1000)),
                          1000);
    return () => clearInterval(t);
  }, []);
  return (
    <div className="flex items-center gap-2.5 py-1">
      <span className="w-2.5 h-2.5 rounded-full bg-amber animate-pulse" />
      <span className="font-mono text-[12.5px] text-secondary">
        {label}{secs >= 3 ? ` — ${secs}s` : ""}
      </span>
    </div>
  );
}

function LiveTurn({ turn }: { turn: StreamingTurn }) {
  return (
    <div className="flex flex-col gap-2">
      {turn.macro && (
        <p className="font-mono text-[11px] text-muted">
          {turn.macro.label} — step {turn.macro.index + 1} of{" "}
          {turn.macro.total}
        </p>
      )}
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
        <Working
          label={turn.chips.some((c) => c.state === "running")
            ? "running tools"
            : turn.thinking
              ? "thinking"
              : "generating"}
        />
      )}
    </div>
  );
}

// AUDIT F49: docs/audit-2026-09-04-full.md — the reason a turn failed used to
// live on the streaming object, which the end-of-turn reload destroys. On a
// 3-second engine error the screen was left reading as if the model had simply
// chosen not to answer. It now outlives the turn and says so here.
function TurnError({ text, onClose }: { text: string; onClose: () => void }) {
  return (
    <div role="alert"
         className="flex items-start gap-2 rounded-md border border-red/30 bg-red/10 px-3 py-2 text-[12.5px] text-red">
      <span className="shrink-0" aria-hidden="true">▲</span>
      <span className="flex-1 min-w-0 whitespace-pre-wrap">{text}</span>
      <button onClick={onClose} aria-label="dismiss error"
              className="shrink-0 text-red/70 hover:text-red leading-none">×</button>
    </div>
  );
}

const WINDOW = 150;

export default function Transcript() {
  const messages = useChat((s) => s.messages);
  const streaming = useChat(selectStreaming);
  const currentId = useChat((s) => s.currentId);
  const lastError = useChat((s) => s.lastError);
  const clearError = useChat((s) => s.clearError);
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
        {messages.slice(-shown).map((m, i) => {
          const abs = Math.max(0, messages.length - shown) + i;
          // tool-result carriers exist for the MODEL's next-turn context;
          // the user sees the same data as persistent chips on the reply
          if (m.kind === "tool_result") return null;
          const isLastAssistant =
            abs === messages.length - 1 && m.role === "assistant";
          return (
            <div key={abs} className="group/msg">
              <Bubble m={m} />
              {isLastAssistant && !streaming && <MessageActions m={m} />}
            </div>
          );
        })}
        {streaming && <LiveTurn turn={streaming} />}
        {lastError && <TurnError text={lastError} onClose={clearError} />}
        <div ref={endRef} />
      </div>
    </div>
  );
}
