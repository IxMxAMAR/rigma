// The memory trust surface — the first UI the agent-memory system has had.
// "Memory that acts on the model without being inspectable is a debugging
// nightmare" (agent-memory spec). Every learned rule, its evidence, delete.
import { useCallback, useEffect, useState } from "react";

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
  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/memory");
      setRows((await r.json()) as MemoryRow[]);
    } catch {
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
        {rows.length === 0 ? (
          <div className="text-center pt-16">
            <div className="font-mono text-[12px] text-muted uppercase tracking-[0.1em] mb-2">
              nothing learned yet
            </div>
            <p className="text-secondary text-[13.5px] max-w-[400px] mx-auto">
              Memories are written when autonomous runs fail and recover.
              Run a mission and check back.
            </p>
          </div>
        ) : (
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
                  className="opacity-0 group-hover:opacity-100 shrink-0 rounded-md px-2 py-1 text-[13px] text-muted hover:text-red hover:bg-surface"
                  aria-label={`forget: ${m.text}`}
                  onClick={async () => {
                    await fetch(`/api/memory/${m.id}`, { method: "DELETE" });
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
