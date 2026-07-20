// Autonomous dashboard: launch a mission, watch it live, steer or stop it,
// browse history. Polls the active run at 2s; history at rest.
import { useCallback, useEffect, useRef, useState } from "react";

interface PlanStep {
  id: number;
  text: string;
  status: string;
}

interface RunSummary {
  id: string;
  status: string;
  mission: string;
  iteration: number;
}

interface Run extends RunSummary {
  plan?: PlanStep[];
  halt_reason?: string;
  summary?: string;
  activity?: { kind: string; text: string }[];
  log_tail?: string;
}

const ACTIVE = new Set(["running", "paused"]);

function Launcher({ onLaunched }: { onLaunched: (id: string) => void }) {
  const [mission, setMission] = useState("");
  const [workspace, setWorkspace] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        new mission
      </h3>
      <textarea
        value={mission}
        onChange={(e) => setMission(e.target.value)}
        rows={3}
        placeholder="What should the agent do, start to finish? Name concrete deliverables."
        aria-label="Mission"
        className="w-full rounded-md bg-surface px-3 py-2 text-[13.5px] outline-none resize-y placeholder:text-muted"
      />
      <div className="flex gap-2 mt-2">
        <input
          value={workspace}
          onChange={(e) => setWorkspace(e.target.value)}
          placeholder="workspace folder (where files go)"
          aria-label="Workspace folder"
          className="flex-1 min-w-0 rounded-md bg-surface px-3 py-1.5 font-mono text-[12.5px] outline-none placeholder:text-muted"
        />
        <button
          disabled={busy || !mission.trim()}
          onClick={async () => {
            setBusy(true);
            setErr(null);
            try {
              const r = await fetch("/api/runs", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify({
                  mission: mission.trim(),
                  workspace: workspace.trim(),
                  budget_hours: 8,
                }),
              });
              const d = (await r.json()) as { id?: string; error?: string };
              if (!r.ok || !d.id) throw new Error(d.error ?? "launch failed");
              setMission("");
              onLaunched(d.id);
            } catch (e) {
              setErr((e as Error).message);
            } finally {
              setBusy(false);
            }
          }}
          className="shrink-0 rounded-md bg-amber/15 text-amber px-4 py-1.5 text-[13px] font-semibold disabled:opacity-40"
        >
          {busy ? "launching…" : "launch"}
        </button>
      </div>
      {err && <div className="text-red text-[12.5px] mt-2">{err}</div>}
    </section>
  );
}

function statusTone(s: string) {
  if (s === "running") return "text-amber";
  if (s === "done") return "text-moss";
  if (s === "paused") return "text-secondary";
  return "text-red";
}

function ActiveRun({ run, onAction }: { run: Run; onAction: () => void }) {
  const [note, setNote] = useState("");
  const act = async (path: string, body?: unknown) => {
    await fetch(`/api/runs/${run.id}/${path}`, {
      method: "POST",
      headers: body !== undefined ? { "content-type": "application/json" } : {},
      body: body !== undefined ? JSON.stringify(body) : undefined,
    }).catch(() => {});
    onAction();
  };
  const plan = run.plan ?? [];
  const done = plan.filter((s) => s.status === "done").length;
  return (
    <section className="rounded-lg bg-panel p-4">
      <div className="flex items-center gap-3 mb-1">
        <span className={`font-mono text-[11.5px] font-semibold uppercase ${statusTone(run.status)}`}>
          {run.status}
        </span>
        <span className="font-mono text-[11.5px] text-muted">
          iter {run.iteration}
          {plan.length > 0 && ` · ${done}/${plan.length} steps`}
        </span>
        <div className="ml-auto flex gap-1.5">
          {run.status === "running" ? (
            <button onClick={() => void act("pause")}
                    className="rounded-md bg-surface hover:bg-float px-2.5 py-1 text-[12px]">
              pause
            </button>
          ) : (
            <button onClick={() => void act("resume")}
                    className="rounded-md bg-surface hover:bg-float px-2.5 py-1 text-[12px]">
              resume
            </button>
          )}
          <button onClick={() => void act("stop")}
                  className="rounded-md bg-red/15 text-red px-2.5 py-1 text-[12px] font-semibold">
            stop
          </button>
        </div>
      </div>
      <p className="text-[13px] text-secondary mb-3">{run.mission}</p>
      {plan.length > 0 && (
        <ul className="flex flex-col gap-1 mb-3">
          {plan.map((s) => (
            <li key={s.id} className="flex items-start gap-2 text-[12.5px]">
              <span className={`font-mono mt-px ${s.status === "done" ? "text-moss" : "text-muted"}`}>
                {s.status === "done" ? "✓" : "○"}
              </span>
              <span className={s.status === "done" ? "text-muted" : "text-primary"}>
                {s.text}
              </span>
            </li>
          ))}
        </ul>
      )}
      {run.log_tail && (
        <pre className="font-mono text-[11.5px] text-secondary bg-canvas rounded-md p-3 max-h-44 overflow-y-auto whitespace-pre-wrap">
          {run.log_tail}
        </pre>
      )}
      <form
        className="flex gap-1.5 mt-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!note.trim()) return;
          void act("inject", { message: note.trim() });
          setNote("");
        }}
      >
        <input
          value={note}
          onChange={(e) => setNote(e.target.value)}
          placeholder="steer the agent… (delivered next turn)"
          aria-label="Steering note"
          className="flex-1 min-w-0 rounded-md bg-surface px-3 py-1.5 text-[13px] outline-none placeholder:text-muted"
        />
        <button className="shrink-0 rounded-md bg-surface hover:bg-float px-3 py-1.5 text-[12.5px]">
          send
        </button>
      </form>
    </section>
  );
}

export default function AutonomousSurface() {
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [active, setActive] = useState<Run | null>(null);
  const activeId = useRef<string | null>(null);

  const refreshHistory = useCallback(async () => {
    try {
      const r = await fetch("/api/runs");
      const d: unknown = await r.json();
      if (r.ok && Array.isArray(d)) setHistory(d as RunSummary[]);
    } catch { /* keep last */ }
  }, []);

  const pollActive = useCallback(async () => {
    const id = activeId.current;
    if (!id) return;
    try {
      const r = await fetch(`/api/runs/${id}`);
      const d = (await r.json()) as Run;
      setActive(d);
      if (!ACTIVE.has(d.status)) {
        activeId.current = null;
        void refreshHistory();
      }
    } catch { /* transient */ }
  }, [refreshHistory]);

  useEffect(() => {
    void refreshHistory();
    fetch("/api/runs/active")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: Run | null) => {
        if (d?.id) {
          activeId.current = d.id;
          setActive(d);
        }
      })
      .catch(() => {});
  }, [refreshHistory]);

  useEffect(() => {
    const t = setInterval(pollActive, 2000);
    return () => clearInterval(t);
  }, [pollActive]);

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[860px] mx-auto flex flex-col gap-4">
        {active && ACTIVE.has(active.status) ? (
          <ActiveRun run={active} onAction={pollActive} />
        ) : (
          <Launcher
            onLaunched={(id) => {
              activeId.current = id;
              void pollActive();
            }}
          />
        )}
        {active && !ACTIVE.has(active.status) && (
          <section className="rounded-lg bg-panel p-4">
            <div className={`font-mono text-[11.5px] font-semibold uppercase mb-1 ${statusTone(active.status)}`}>
              {active.status}
            </div>
            <p className="text-[13px] text-secondary">
              {active.summary || active.halt_reason || active.mission}
            </p>
          </section>
        )}
        {history.length > 0 && (
          <section className="rounded-lg bg-panel p-4">
            <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
              history
            </h3>
            <ul className="flex flex-col gap-1">
              {history.map((h) => (
                <li key={h.id} className="flex items-center gap-3 text-[12.5px] rounded-md hover:bg-surface px-2 py-1">
                  <span className={`font-mono text-[11px] w-16 shrink-0 ${statusTone(h.status)}`}>
                    {h.status}
                  </span>
                  <span className="flex-1 truncate text-secondary">{h.mission}</span>
                  <span className="font-mono text-[11px] text-muted shrink-0">{h.id.slice(0, 15)}</span>
                </li>
              ))}
            </ul>
          </section>
        )}
      </div>
    </main>
  );
}
