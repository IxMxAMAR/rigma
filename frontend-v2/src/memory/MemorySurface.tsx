// The memory trust surface — the first UI the agent-memory system has had.
// "Memory that acts on the model without being inspectable is a debugging
// nightmare" (agent-memory spec). Every learned rule, its evidence, delete.
import { useCallback, useEffect, useState } from "react";

import EmptyState from "../EmptyState";
import InlineError from "../InlineError";
import LoadError from "../LoadError";
import { responseError } from "../lib/listFetch";
import { MEMORY_STATUSES, memoryPatch } from "./edit";

interface MemoryRow {
  id: string;
  kind: string;
  text: string;
  status: string;
  seen_count: number;
  outcome_score: number;
  born?: number;
  born_run?: string;
}

export default function MemorySurface() {
  const [rows, setRows] = useState<MemoryRow[] | null>(null);
  // A refused forget is not a failed load: keeping one `err` for both made a
  // refused delete render as "could not load memory" with a retry that would
  // not help (the list was fine). Two slots, like the other surfaces.
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [rowErr, setRowErr] = useState<string | null>(null);
  // IMP-6: which rule is open for editing, and its staged values.
  const [editing, setEditing] = useState<string | null>(null);
  const [draftText, setDraftText] = useState("");
  const [draftStatus, setDraftStatus] = useState("");
  const [busyId, setBusyId] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    // guard r.ok AND the shape: a server started before this endpoint existed
    // 404s, and treating {error} as rows rendered pure darkness (owner report
    // 2026-07-21). A surface must never be empty void - worst case it says
    // exactly what is wrong.
    try {
      const r = await fetch("/api/memory");
      const d: unknown = await r.json();
      if (!r.ok || !Array.isArray(d)) {
        setLoadErr(r.status === 404
          ? "This server predates the memory API - restart Rigma to enable it."
          : `memory API: ${(d as { error?: string })?.error ?? r.status}`);
        setRows([]);
        return;
      }
      setLoadErr(null);
      setRows(d as MemoryRow[]);
    } catch (e) {
      setLoadErr((e as Error).message);
      setRows([]);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (rows === null)
    return <main className="flex-1 flex items-center justify-center text-muted font-mono text-[12px]">loading…</main>;

  // IMP-6: save an edit through PATCH /api/memory/{id}. Only the fields that
  // changed are sent (memoryPatch), and a refused edit keeps the row open with
  // the server's sentence rather than closing as if it had saved.
  const saveEdit = async (m: MemoryRow) => {
    const patch = memoryPatch(m, draftText, draftStatus);
    if (Object.keys(patch).length === 0) {
      setEditing(null);
      return;
    }
    setBusyId(m.id);
    setRowErr(null);
    try {
      const r = await fetch(`/api/memory/${encodeURIComponent(m.id)}`, {
        method: "PATCH",
        headers: { "content-type": "application/json" },
        body: JSON.stringify(patch),
      });
      if (!r.ok) {
        setRowErr(await responseError(r));
        return;
      }
      setEditing(null);
      void refresh();
    } catch (e) {
      setRowErr((e as Error).message);
    } finally {
      setBusyId(null);
    }
  };

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[860px] mx-auto">
        <p className="text-secondary text-[13px] mb-4">
          Rules the agent learned from its own runs. Verified rules earned a
          success in a run other than the one that wrote them; drafts are
          still on probation. Deleting is safe — a useful rule will be
          re-learned.
        </p>
        {loadErr && (
          <LoadError message={loadErr} onRetry={() => void refresh()} />
        )}
        {rowErr && <InlineError message={rowErr} />}
        {/* IMP-10: the empty state means the fetch SUCCEEDED and returned
            nothing. A failed fetch renders the error above instead — it must
            never read as "nothing learned yet". */}
        {!loadErr && rows.length === 0 && (
          <EmptyState
            title="nothing learned yet"
            body="Memories are written when autonomous runs fail and recover. Run a mission and check back."
            actionLabel="refresh"
            onAction={() => void refresh()} />
        )}
        {rows.length > 0 && (
          <ul className="flex flex-col gap-1.5">
            {rows.map((m) => {
              const isEditing = editing === m.id;
              return (
              <li key={m.id} className="group rounded-lg bg-panel px-4 py-2.5 flex items-start gap-3">
                {isEditing ? (
                  /* IMP-6: correcting a rule the agent got wrong. PATCH
                     /api/memory/{id} takes text and status; the id stays put
                     so a run that referenced this rule still resolves. */
                  <div className="flex-1 min-w-0 flex flex-col gap-1.5">
                    <textarea
                      value={draftText}
                      onChange={(e) => setDraftText(e.target.value)}
                      rows={2}
                      aria-label={`edit rule: ${m.text}`}
                      className="w-full rounded-md bg-surface px-2 py-1 text-[13px] outline-none resize-y"
                    />
                    <div className="flex items-center gap-2">
                      <select
                        value={draftStatus}
                        onChange={(e) => setDraftStatus(e.target.value)}
                        aria-label="Memory status"
                        className="rounded-md bg-surface px-2 py-0.5 font-mono text-[11.5px] outline-none"
                      >
                        {MEMORY_STATUSES.map((s) => (
                          <option key={s} value={s}>{s}</option>
                        ))}
                      </select>
                      <button
                        disabled={busyId === m.id}
                        onClick={() => void saveEdit(m)}
                        className="rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
                      >
                        save
                      </button>
                      <button
                        onClick={() => setEditing(null)}
                        className="rounded-md bg-surface hover:bg-float px-2.5 py-0.5 text-[12px]"
                      >
                        cancel
                      </button>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 min-w-0">
                    <div className="text-[13.5px]">{m.text}</div>
                    <div className="font-mono text-[11px] text-muted mt-0.5">
                      {m.kind} ·{" "}
                      <span className={m.status === "verified" ? "text-moss" : ""}>
                        {m.status}
                      </span>{" "}
                      · seen {m.seen_count} · score{" "}
                      <span className={m.outcome_score > 0 ? "text-moss" : m.outcome_score < 0 ? "text-red" : ""}>
                        {m.outcome_score > 0 ? "+" : ""}{m.outcome_score}
                      </span>
                    </div>
                  </div>
                )}
                {!isEditing && (
                  <>
                    <button
                      className="shrink-0 rounded-md px-2 py-1 text-[13px] text-muted hover:text-amber hover:bg-surface"
                      aria-label={`edit: ${m.text}`}
                      onClick={() => {
                        setRowErr(null);
                        setEditing(m.id);
                        setDraftText(m.text);
                        setDraftStatus(m.status);
                      }}
                    >
                      edit
                    </button>
                    <button
                      disabled={busyId === m.id}
                      className="shrink-0 rounded-md px-2 py-1 text-[13px] text-muted hover:text-red hover:bg-surface disabled:opacity-40"
                      aria-label={`forget: ${m.text}`}
                      onClick={async () => {
                        setRowErr(null);
                        setBusyId(m.id);
                        // AUDIT F11-4: this ignored r.ok and refreshed, so a refused
                        // forget looked exactly like one that worked — the rule just
                        // stayed put with no reason.
                        try {
                          const r = await fetch(
                            `/api/memory/${encodeURIComponent(m.id)}`,
                            { method: "DELETE" });
                          if (!r.ok) {
                            setRowErr(await responseError(r));
                            return;
                          }
                          void refresh();
                        } catch (e) {
                          setRowErr((e as Error).message);
                        } finally {
                          setBusyId(null);
                        }
                      }}
                    >
                      forget
                    </button>
                  </>
                )}
              </li>
              );
            })}
          </ul>
        )}
      </div>
    </main>
  );
}
