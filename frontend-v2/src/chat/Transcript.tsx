// Transcript: persisted messages + the live streaming turn. Thinking blocks
// collapse once the reply starts; chips expand to show their result.
import { useCallback, useEffect, useRef, useState } from "react";
import { api, apiStatus, type ChatMessage } from "../lib/api";
import Markdown from "./Markdown";
import {
  errText, selectStreaming, useChat, type Chip, type Source, type StreamingTurn,
} from "./chatStore";
import { formatArgs, previewArgs } from "./toolChip";
import AgentState from "./AgentState";
import { EMPTY_GOVERNANCE, questionRefusal } from "./governance";
import { normaliseGoal } from "./goal";
import { compactionLine, running } from "./compaction";
import { retryLine } from "./retry";
import { argHint, delegateSentence, summariseDelegate } from "./delegate";
import { chipOutcome } from "./toolChip";
import { commandPrompt } from "./commands";
import { liveTail } from "./reattach";

// How each outcome is drawn. `unknown` is deliberately NOT moss — see
// `chipOutcome` in toolChip.ts for why an unreported result is not a success.
const OUTCOME_TONE: Record<string, string> = {
  running: "text-amber animate-pulse",
  failed: "text-red",
  ok: "text-moss",
  unknown: "text-muted",
};

const OUTCOME_GLYPH: Record<string, string> = {
  running: "◌", failed: "✕", ok: "✓", unknown: "?",
};

function ChipRow({ chip }: { chip: Chip }) {
  const outcome = chipOutcome(chip);
  return (
    <details className="rounded-md bg-surface/60 open:bg-surface">
      <summary className="flex items-center gap-2 px-3 py-1.5 cursor-pointer list-none font-mono text-[12px]">
        <span
          className={OUTCOME_TONE[outcome]}
          aria-label={outcome}
          title={
            outcome === "unknown"
              ? "this backend did not report whether the call succeeded"
              : undefined
          }
        >
          {OUTCOME_GLYPH[outcome]}
        </span>
        <span className="font-semibold text-primary">{chip.name}</span>
        <span className="text-muted truncate">{previewArgs(chip.args)}</span>
      </summary>
      {formatArgs(chip.args) && (
        <pre className="px-3 pt-1 text-[12px] font-mono text-muted whitespace-pre-wrap break-words max-h-40 overflow-y-auto">
          {formatArgs(chip.args)}
        </pre>
      )}
      {chip.result && (
        <pre className="px-3 pb-2 pt-1 text-[12px] font-mono text-secondary whitespace-pre-wrap break-words max-h-56 overflow-y-auto">
          {chip.result}
        </pre>
      )}
    </details>
  );
}

// identity, never payload — same rule the legacy chips learned the hard way.
// The rules live in `toolChip.ts` so they can be tested without a renderer.


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

function Bubble({ m, tailNotice }: { m: ChatMessage; tailNotice?: string }) {
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
                        result: t.result, state: "done", ok: t.ok }}
              />
            ))}
          </div>
        )}
        {/* What the research helper did. The server recorded this from the
            start and nothing drew it, so a delegated answer arrived with no
            sign of the work behind it. Collapsed by default: it is the receipt,
            not the answer, and the answer is what the reader came for. */}
        {!isUser && (m.delegate_trace?.length ?? 0) > 0 && (
          <details className="rounded-md bg-surface px-2.5 py-1.5 text-[11.5px] mb-2">
            <summary className="cursor-pointer text-secondary">
              {delegateSentence(summariseDelegate(m.delegate_trace ?? []))}
            </summary>
            <ul className="mt-1 flex flex-col gap-0.5 pl-3 text-muted leading-snug">
              {(m.delegate_trace ?? []).map((d, i) => (
                <li key={i} className="font-mono text-[11px]">
                  <span className={d.blocked ? "text-red" : d.ok === false ? "text-red" : "text-moss"}>
                    {d.blocked ? "✕" : d.ok === false ? "✕" : "✓"}
                  </span>{" "}
                  {d.name}
                  {argHint(d.args) && <span className="text-muted"> {argHint(d.args)}</span>}
                  {d.blocked && <span className="text-red"> — not available to a helper</span>}
                </li>
              ))}
            </ul>
          </details>
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
        {!isUser && m.harness && m.harness !== "native" && (
          <div className="pt-1.5">
            <HarnessBadge name={m.harness} label={m.harness_label} />
          </div>
        )}
        {/* D3b: the tail of a reloaded transcript may be a mid-turn CHECKPOINT,
            and the server's own notice on it says "interrupted". While the
            session is still `streaming` that sentence is false, so the tail's
            sentence is passed in. `undefined` (not "") leaves an ordinary
            message's notice exactly as the server wrote it. */}
        {(() => {
          const notice = tailNotice !== undefined ? tailNotice : m.notice;
          return notice ? (
            <p className="text-[12px] italic text-muted mt-1.5">{notice}</p>
          ) : null;
        })()}
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
    <div className="flex items-center gap-3 pt-1 opacity-0 group-hover/msg:opacity-100
                    group-focus-within/msg:opacity-100 font-mono text-[11.5px] text-muted">
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

/** Which agent backend produced a reply.
 *
 *  Absent for a built-in turn, and it renders nothing for one: the badge exists
 *  to explain the unusual case, and stamping "Rigma (built in)" on every bubble
 *  would be noise the reader learns to skip — which is exactly what makes a
 *  real marker easy to miss. The tooltip carries the cost, because "driven by
 *  DeepSeek Harness" is not self-explanatory: that turn had no tool-call
 *  repair, no undo and no artifact check behind it. */
function HarnessBadge({ name, label }: { name: string; label?: string }) {
  if (!name || name === "native") return null;
  return (
    <span
      className="font-mono text-[10.5px] text-amber bg-amber/15 rounded px-1.5 py-0.5"
      title={`${label || name} drove this turn with its own tools and system `
             + "prompt. Rigma's tool-call repair, fuzzy path recovery and "
             + "artifact verification did not apply to it."}
    >
      {label || name}
    </span>
  );
}

function LiveTurn({ turn }: { turn: StreamingTurn }) {
  // R6-ACP-APPROVE: the one place an approval can be ANSWERED. Read here rather than
  // passed down, so the handler always names the chat actually on screen — a stale
  // closure over a previous id would send the answer to the wrong turn.
  //
  // The server refuses a requestId it is not waiting on, so a race (the turn ended,
  // or the request timed out) surfaces as a 409 rather than as a silent no-op. That
  // refusal is deliberately NOT swallowed: a click that did nothing must not look
  // like one that worked.
  const answerApproval = useCallback(async (requestId: string, allow: boolean,
                                             answer?: Record<string, unknown>) => {
    const sid = useChat.getState().currentId;
    if (!sid) return;
    try {
      // B4: exactly one of the two shapes. The route refuses a question answered
      // with `allow` (409) and a permission answered with an object, so the choice
      // is made here from what the card submitted rather than by sending both.
      await api.answerApproval(sid, answer !== undefined
        ? { requestId, answer } : { allow, requestId });
    } catch (e) {
      const status = apiStatus(e);
      // B4b/OD-12: a 409 on a QUESTION is the server saying it is no longer
      // waiting on it — the window closed, or it was already answered. Fold the
      // row terminal (the same state an expiry event produces) so the dead form
      // stops being clickable, and say so rather than repeating the wire's own
      // sentence. A permission's error handling is untouched.
      if (answer !== undefined && status === 409) {
        useChat.getState().expireQuestion(requestId);
        useChat.setState({ lastError: questionRefusal(errText(e), status) });
      } else {
        useChat.setState({ lastError: errText(e) });
      }
    }
  }, []);

  // R6-ACP-CONTROL-ROW: a per-row control action on the LIVE turn.
  //
  // Live only, and that is the whole point of passing it here rather than to every
  // `AgentState`: a durable turn's queue and delegation tree are HISTORY, and the ids in
  // them name a session that may be long gone. The live render is the only place where
  // "steer this queued message" has a turn to steer.
  //
  // Read from the store at click time, like `answerApproval` above, so the handler always
  // names the chat actually on screen.
  const rowOp = useCallback(async (op: string, params: Record<string, unknown>) => {
    const sid = useChat.getState().currentId;
    if (!sid) return;
    try {
      await api.control(sid, op, params);
    } catch (e) {
      // Surfaced, not swallowed: dropping a queued message is destructive, so a silent
      // failure would leave the user believing something happened. (This said "stop a
      // child" too, which was the per-delegate action that no longer exists —
      // `delegation_stop` is session-wide and lives in the control panel.)
      useChat.setState({ lastError: errText(e) });
    }
  }, []);

  // B6c: run a command the BACKEND advertised.
  //
  // Straight to `send()`, NOT through the composer: the invocation is a
  // `session/prompt` whose text is the command, and `parseSlash` knows only
  // Rigma's own roster — an advertised name that collides with one of them
  // (`/compact`) would be intercepted and Rigma's command would run instead.
  //
  // `send()` refuses while a reply is streaming, and this panel is only drawn
  // during one, so that refusal is the COMMON case rather than an edge: it is
  // surfaced through the chat's notice line instead of being swallowed, because
  // a click that does nothing is indistinguishable from a broken button. (The
  // server would queue a new prompt behind the running reply — serve.py:5017 —
  // but the store's one-turn-at-a-time guard stops it before the request; see
  // the B6c report.)
  const runCommand = useCallback((name: string) => {
    void useChat.getState().send(commandPrompt(name)).then((ok) => {
      if (!ok) {
        useChat.getState().pushNotice(
          `/${name} was not sent — this chat is already producing a reply.`,
        );
      }
    });
  }, []);

  // The still-open compaction for THIS turn, if any. Computed here rather than
  // inline in the JSX so it is one lookup per render, not one per condition.
  const runningCompaction = running(turn.compactions);
  // DSH had to retry the model request. Silent in every other surface, and
  // against a local engine a silent retry reads as a frozen turn.
  const retry = turn.retry ? retryLine(turn.retry) : null;
  return (
    <div className="flex flex-col gap-2">
      {/* Named at the START of the turn, so this is on screen while an external
          backend is still working — it can run for a minute before its first
          token. */}
      {turn.harness && turn.harness !== "native" && (
        <HarnessBadge name={turn.harness} label={turn.harnessLabel} />
      )}
      {turn.macro && (
        <p className="font-mono text-[11px] text-muted">
          {turn.macro.label} — step {turn.macro.index + 1} of{" "}
          {turn.macro.total}
        </p>
      )}
      {/* Server-authored status for this turn. These were being dropped by the
          store's reducer, which is why the line explaining an external backend
          never appeared even though the server had always sent it. */}
      {turn.notices.map((n, i) => (
        <p key={i} className="text-[12px] italic text-muted">{n}</p>
      ))}
      {/* The agent's own state, reported by the backend rather than written
          by the model: its goal, its todo list, whether it is in plan mode,
          the subagents it started, and what the step cost. Rendered above the
          reply because it is the context the reply is happening in. */}
      {/* R6-ACP-APPROVE: the one place an approval can be ANSWERED. Declared here,
          where the panel that draws the buttons is rendered, and it reads the CURRENT
          session at click time — a stale closure over a previous id would send the
          answer to the wrong turn.

          The server refuses a requestId it is not waiting on, so a race (the turn
          ended, or the request timed out) surfaces as a 409 rather than as a silent
          no-op. That refusal is deliberately NOT swallowed: a click that did nothing
          must not look like one that worked. */}
      <AgentState
        goal={turn.goal}
        todos={turn.todos}
        planMode={turn.planMode}
        subagents={turn.subagents}
        usage={turn.usage}
        governance={turn.governance}
        acpQueue={turn.acpQueue}
        acpDelegation={turn.acpDelegation}
        acpConfig={turn.acpConfig}
        acpCommands={turn.acpCommands}
        acpPlan={turn.acpPlan}
        workflow={turn.workflow}
        onAnswerApproval={answerApproval}
        onRowOp={rowOp}
        onRunCommand={runCommand}
      />
      {/* Compaction, observation-masking and the prompt queue. All three were
          emitted by the server and dropped by the store's default arm, so a
          long turn looked frozen while it was actually working. */}
      {turn.housekeeping && (
        <p className="font-mono text-[11px] text-muted">{turn.housekeeping}</p>
      )}
      {turn.masked > 0 && (
        <p className="font-mono text-[11px] text-muted">
          masked {turn.masked} earlier observation{turn.masked === 1 ? "" : "s"} to make room
        </p>
      )}
      {turn.compacted > 0 && (
        <p className="font-mono text-[11px] text-muted">
          compacted {turn.compacted} message{turn.compacted === 1 ? "" : "s"} into the summary
        </p>
      )}
      {/* DSH's compaction, beside Rigma's own. The native path reports
          `housekeeping`/`masked`/`compacted` above; a DSH turn reported none of
          it, so a long turn busy summarising its own context looked hung. The
          running line is the point: it is what turns "frozen" into "working". */}
      {retry && (
        <p className="font-mono text-[11px] text-amber">{retry}</p>
      )}
      {runningCompaction && (
        <p className="font-mono text-[11px] text-amber">
          compacting the context — this can take a while
        </p>
      )}
      {turn.compactions.items.filter((c) => !c.open).map((c) => {
        const line = compactionLine(c);
        if (line === null) return null;
        return (
          <p
            key={c.id || "compaction"}
            className={`font-mono text-[11px] ${c.error ? "text-red" : "text-muted"}`}
          >
            {line}
            {c.model && <span> · by {c.model}</span>}
          </p>
        );
      })}
      {turn.queued > 0 && (
        <p className="font-mono text-[11px] text-amber">
          queued behind the running reply — {turn.queued} waiting
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
      {turn.sources.length > 0 && <Sources rows={turn.sources} />}
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

// UIUX-22: which of the user's own documents a grounded answer came from.
//
// This is the whole value of grounding a private corpus — "which file said
// this?" — and it was the one thing the reader could not get: the sidecar
// returned citations on every /ask, the tool folded them into the MODEL's text as
// a "sources: …" line, and the streaming pipeline had no citations branch at all
// (deleted by F11-12 for want of a producer). The model could see the sources;
// the person deciding whether to trust the answer could not.
//
// A disclosure rather than a chip list, because a passage excerpt is long and
// most turns have several — the reader who wants to check gets the snippet, and
// the reader who does not is not made to scroll past it.
function Sources({ rows }: { rows: Source[] }) {
  return (
    <details className="rounded-md bg-surface/60 open:bg-surface">
      <summary className="flex items-center gap-2 px-3 py-1.5 cursor-pointer list-none font-mono text-[12px]">
        <span className="text-moss" aria-hidden="true">◆</span>
        <span className="font-semibold text-primary">
          {rows.length === 1 ? "1 source" : `${rows.length} sources`}
        </span>
        <span className="text-muted">from your documents</span>
      </summary>
      <ul className="px-3 pb-2 pt-1 flex flex-col gap-1.5">
        {rows.map((s, i) => (
          <li key={`${s.source}-${s.page ?? ""}-${i}`} className="min-w-0">
            <span className="font-mono text-[12px] text-secondary break-all">
              {s.source}
              {s.page != null && <span className="text-muted"> p.{s.page}</span>}
            </span>
            {s.snippet && (
              <p className="text-[12px] text-muted leading-snug break-words">
                {s.snippet}
              </p>
            )}
          </li>
        ))}
      </ul>
    </details>
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
  // D3b: is the SERVER still generating this chat? Not `streaming`, which only
  // knows about turns this tab started — that is the whole defect.
  const remote = useChat((s) =>
    s.currentId ? s.remoteStreaming[s.currentId] === true : false);
  const currentId = useChat((s) => s.currentId);
  const lastError = useChat((s) => s.lastError);
  const clearError = useChat((s) => s.clearError);
  const [shown, setShown] = useState(WINDOW);
  useEffect(() => setShown(WINDOW), [currentId]);
  const endRef = useRef<HTMLDivElement>(null);
  const stick = useRef(true);

  // D3b: while the server is generating, re-read the transcript on a bounded
  // timer so the reloaded chat catches each 20 s checkpoint and stops the moment
  // the turn ends. The interval is shorter than `CHECKPOINT_SECS` so a finished
  // turn is noticed within one checkpoint, and it is torn down on a chat switch
  // or unmount — a poll that outlived the chat on screen would write another
  // chat's messages (the ownership guard in `refreshRemote` also refuses).
  //
  // NOT while THIS tab owns the turn: `LiveTurn` is already drawing that reply
  // from the stream, and the server's durable copy carries the same checkpoint as
  // a message, so polling would render the text twice. The dependency is the
  // boolean, not the turn — a turn object changes on every token and would
  // restart the timer constantly, which is a poll that never fires.
  const live = streaming !== null;
  useEffect(() => {
    if (!currentId || !remote || live) return;
    const t = setInterval(() => {
      void useChat.getState().refreshRemote(currentId);
    }, 10000);
    return () => clearInterval(t);
  }, [currentId, remote, live]);

  // The sentence for the tail, and whether the chat is live. See `reattach.ts`.
  const tail = liveTail(messages, remote);

  // R5-PERSIST: the stored durable state for the chat on screen. Read here rather
  // than passed down so the panel and the transcript cannot disagree about which
  // chat they are describing — the same reason `messages` is read here.
  const saved = useChat((s) => (currentId ? s.savedAgent[currentId] : undefined));
  // Normalised here, at the one place it is rendered, so the stored copy stays the
  // raw wire payload and `chat/goal.ts` remains the only place the two backends'
  // field names are reconciled. `undefined` for no goal, which AgentState treats
  // the same as null.
  const durableGoal = saved?.goal ? normaliseGoal(saved.goal) : null;
  // R6-WORKFLOW: a run counts as something to show, so it joins the condition as well
  // as the payload. Leaving it out of the CONDITION would mean a chat whose only
  // durable state is a workflow rendered no durable panel at all.
  const durable = saved && (durableGoal || (saved.todos?.length ?? 0) > 0 ||
                            saved.plan_mode === true ||
                            (saved.workflow?.length ?? 0) > 0)
    ? { goal: durableGoal, todos: saved.todos ?? [], plan: saved.plan_mode === true,
        workflow: saved.workflow ?? [] }
    : null;

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
          // D3b: only the tail checkpoint gets the liveness sentence. `undefined`
          // for every other message keeps its own notice untouched.
          const tailNotice =
            abs === messages.length - 1 && m.partial === true
              ? tail.text : undefined;
          return (
            <div key={abs} className="group/msg">
              <Bubble m={m} tailNotice={tailNotice} />
              {isLastAssistant && !streaming && <MessageActions m={m} />}
            </div>
          );
        })}
        {streaming && <LiveTurn turn={streaming} />}
        {/* R5-PERSIST: the agent's DURABLE state, shown when no turn is live.
            Goals, todos and plan mode outlive the turn that reported them, and the
            panel above is drawn from the turn — so before this, reloading a chat
            lost a goal the agent was still working toward, and the panel silently
            disagreed with the agent.

            Only when NOT streaming, because the live panel already draws these
            from the turn and two copies of the same goal would be a bug, not a
            feature. Nothing is rendered when the chat has no durable state, so a
            chat that never used a backend with goals looks exactly as before. */}
        {!streaming && durable && (
          <AgentState
            goal={durable.goal}
            todos={durable.todos}
            planMode={durable.plan}
            /* R6-WORKFLOW: restored, UNLIKE the ACP block below, and the difference is
               the point. A queue or a delegation snapshot describes a session that is
               running right now, so restoring one would claim live state that is gone.
               A workflow run is a RECORD of work this conversation already did — the
               same class of fact as `todos`, which is also restored — and dropping it
               would show a chat whose agents had never run. */
            workflow={durable.workflow ?? []}
            subagents={[]}
            usage={null}
            governance={EMPTY_GOVERNANCE}
            /* R6-ACP: the ACP control plane is deliberately NOT restored here,
               and that is a decision rather than an omission. A queue is messages
               waiting for the NEXT turn, a delegation snapshot is live session
               state, and configOptions are the selects in force right now — all
               three describe a running session, and none of them survives a
               reload in mcode either. Restoring them from a durable copy would
               show a queue that has already drained. The durable panel is for
               what OUTLIVES the turn, which is the goal and the todos. */
          />
        )}
        {lastError && <TurnError text={lastError} onClose={clearError} />}
        <div ref={endRef} />
      </div>
    </div>
  );
}
