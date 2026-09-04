// The live preview beside a method-creation chat. Every builder tool call
// changes the draft on disk, so this re-reads it whenever a tool finishes.
//
// The spec called for a `method_draft` SSE event. Refetching on tool_result
// is the same liveness without threading new state through _llm_turn — a
// 3.3K-line function whose split was already deferred once
// (docs/backlog-post-audit.md). One fewer thing wedged into that loop.
import { useEffect, useState } from "react";
import { getDraft, promoteDraft, type Method } from "../lib/methods";
import { selectStreaming, useChat } from "./chatStore";

export default function MethodBuilder({ draftId }: { draftId: string }) {
  const streaming = useChat(selectStreaming);
  const loadSessions = useChat((s) => s.loadSessions);
  const [draft, setDraft] = useState<Method | null>(null);
  const [saved, setSaved] = useState<string>("");
  const [error, setError] = useState<string>("");

  // the chip count changes exactly when a builder tool completes
  const doneTools = streaming?.chips.filter((c) => c.state === "done").length
    ?? 0;

  useEffect(() => {
    let alive = true;
    void (async () => {
      const d = await getDraft(draftId);
      if (alive && d) setDraft(d);
    })();
    return () => {
      alive = false;
    };
  }, [draftId, doneTools, streaming === null]);

  if (!draft) return null;

  const save = async () => {
    setError("");
    try {
      const m = await promoteDraft(draftId);
      setSaved(m.name);
      await loadSessions();
    } catch (e) {
      setError((e as Error).message);
    }
  };

  const rules = draft.rules ?? [];
  const macros = draft.macros ?? [];
  const workflows = draft.workflows ?? [];
  const empty = !draft.name && rules.length === 0 && macros.length === 0;

  return (
    <section className="rounded-lg bg-panel p-3 flex flex-col gap-1.5">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em]">
        new method
      </h3>

      {empty ? (
        <p className="text-[11.5px] text-muted">
          Answer in the chat and it will take shape here.
        </p>
      ) : (
        <>
          <p className="text-[12.5px] text-primary">
            {draft.name || "(unnamed)"}
          </p>
          {draft.tagline && (
            <p className="text-[11px] text-secondary">{draft.tagline}</p>
          )}

          {rules.length > 0 && (
            <div className="flex flex-col gap-1 pt-1">
              <p className="font-mono text-[11px] text-muted">rules</p>
              {rules.map((r) => (
                <p key={r.id} className="text-[11.5px] text-secondary">
                  <span className="text-muted">·</span>{" "}
                  {r.kind === "standing" ? r.text : `when ${r.kind}`}
                </p>
              ))}
            </div>
          )}

          {macros.length > 0 && (
            <div className="flex flex-col gap-1 pt-1">
              <p className="font-mono text-[11px] text-muted">buttons</p>
              <div className="flex flex-wrap gap-1.5">
                {macros.map((m) => (
                  <span
                    key={m.id}
                    className="rounded-md bg-surface text-secondary px-2 py-0.5 text-[11.5px]"
                  >
                    {m.label}
                  </span>
                ))}
              </div>
            </div>
          )}

          {workflows.length > 0 && (
            <p className="text-[11px] text-muted pt-1">
              <span className="font-mono">workflows</span>{" "}
              {workflows.map((w) => w.label).join(", ")}
            </p>
          )}
        </>
      )}

      {saved ? (
        <p className="text-[11px] text-moss pt-1">
          saved “{saved}” — pick it from Methods above
        </p>
      ) : (
        <button
          onClick={() => void save()}
          disabled={!draft.name}
          className="rounded-md bg-amber/15 text-amber px-2.5 py-1 text-[12px] font-semibold disabled:opacity-40 mt-1"
        >
          Save method
        </button>
      )}
      {error && <p className="text-[11px] text-red">{error}</p>}
    </section>
  );
}
