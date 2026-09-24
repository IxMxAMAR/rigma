// The memory trust surface — the first UI the agent-memory system has had.
// "Memory that acts on the model without being inspectable is a debugging
// nightmare" (agent-memory spec). Every learned rule, its evidence, delete.
import { useCallback, useEffect, useState } from "react";

import EmptyState from "../EmptyState";
import LoadError from "../LoadError";
import { responseError } from "../lib/listFetch";

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
  const [err, setErr] = useState<string | null>(null);
  const refresh = useCallback(async () => {
    // guard r.ok AND the shape: a server started before this endpoint existed
    // 404s, and treating {error} as rows rendered pure darkness (owner report
    // 2026-07-21). A surface must never be empty void - worst case it says
    // exactly what is wrong.
    try {
      const r = await fetch("/api/memory");
      const d: unknown = await r.json();
      if (!r.ok || !Array.isArray(d)) {
        setErr(r.status === 404
          ? "This server predates the memory API - restart Rigma to enable it."
          : `memory API: ${(d as { error?: string })?.error ?? r.status}`);
        setRows([]);
        return;
      }
      setErr(null);
      setRows(d as MemoryRow[]);
    } catch (e) {
      setErr((e as Error).message);
      setRows([]);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  if (rows === null)
    return <main className="flex-1 flex items-center justify-center text-muted font-mono text-[12px]">loading…</main>;

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[860px] mx-auto">
        <p className="text-secondary text-[13px] mb-4">
          Rules the agent learned from its own runs. Verified rules earned a
          success in a run other than the one that wrote them; drafts are
          still on probation. Deleting is safe — a useful rule will be
          re-learned.
        </p>
        {err && (
          <LoadError message={err} onRetry={() => void refresh()} />
        )}
        {/* IMP-10: the empty state means the fetch SUCCEEDED and returned
            nothing. A failed fetch renders the error above instead — it must
            never read as "nothing learned yet". */}
        {!err && rows.length === 0 && (
          <EmptyState
            title="nothing learned yet"
            body="Memories are written when autonomous runs fail and recover. Run a mission and check back."
            actionLabel="refresh"
            onAction={() => void refresh()} />
        )}
        {rows.length > 0 && (
          <ul className="flex flex-col gap-1.5">
            {rows.map((m) => (
              <li key={m.id} className="group rounded-lg bg-panel px-4 py-2.5 flex items-center gap-3">
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
                <button
                  className="shrink-0 rounded-md px-2 py-1 text-[13px] text-muted hover:text-red hover:bg-surface"
                  aria-label={`forget: ${m.text}`}
                  onClick={async () => {
                    setErr(null);
                    // AUDIT F11-4: this ignored r.ok and refreshed, so a refused
                    // forget looked exactly like one that worked — the rule just
                    // stayed put with no reason.
                    const r = await fetch(
                      `/api/memory/${encodeURIComponent(m.id)}`,
                      { method: "DELETE" });
                    if (!r.ok) {
                      setErr(await responseError(r));
                      return;
                    }
                    void refresh();
                  }}
                >
                  forget
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>
    </main>
  );
}
