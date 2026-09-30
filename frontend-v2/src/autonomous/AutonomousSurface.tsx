// Autonomous dashboard: launch a mission, watch it live, steer or stop it,
// browse history. Polls the active run at 2s; history at rest.
import { useCallback, useEffect, useRef, useState,
         type RefObject } from "react";

import EmptyState from "../EmptyState";
import InlineError from "../InlineError";
import LoadError from "../LoadError";
import { readList, responseError } from "../lib/listFetch";
import { budget, remainingTime, stopSentence } from "./budget";
import { DEFAULT_PROFILE, PROFILES, profileSentence } from "./profiles";

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

interface Deliverable {
  path: string;
  description?: string;
}

/** The compiled mission, pinned into the system prompt every turn. It was
 *  always in the payload and never drawn, so a run's OBJECTIVE — the one
 *  thing that says what the run is for — was invisible in the surface named
 *  after it. Only the plan list was shown, which is the how, not the what. */
interface Spec {
  objective: string;
  deliverables?: Deliverable[];
  constraints?: string[];
  compiled?: boolean;
}

interface Run extends RunSummary {
  spec?: Spec | null;
  plan?: PlanStep[];
  halt_reason?: string;
  summary?: string;
  activity?: { kind: string; text: string }[];
  log_tail?: string;
  pending_question?: { q: string } | null;
  /** IMP-8: the budget the run is judged against. `iter_ceiling` is present
   *  only on a restarted run; `deadline` is an absolute epoch (runs.create). */
  iter_ceiling?: number;
  deadline?: number;
}

// A plan step has THREE states on the server (`pending`, `done`, `blocked`)
// and the surface drew two glyphs, so `blocked` was indistinguishable from
// `pending` — a run stuck on a step read as a run that had not reached it,
// which is the opposite diagnosis and the one that wastes a user's time.
function planGlyph(status: string): string {
  if (status === "done") return "✓";
  if (status === "blocked") return "✕";
  return "○";
}

function planTone(status: string): string {
  if (status === "done") return "text-moss";
  if (status === "blocked") return "text-red";
  return "text-muted";
}

const ACTIVE = new Set(["running", "paused"]);
// terminal states the server can reattach a loop to — mirrors runs.RESTARTABLE
const RESTARTABLE = new Set([
  "interrupted", "stopped", "stalled", "frozen", "budget_exhausted", "error",
]);

/** Exported so the OD-1 option 4 first-launch test can pin the safety-profile
 *  chooser without mounting the whole surface (which fetches on mount). */
export function Launcher({ onLaunched, missionRef }: {
  onLaunched: (id: string) => void;
  /** IMP-10: the empty history's primary action focuses the mission box. */
  missionRef?: RefObject<HTMLTextAreaElement>;
}) {
  const [mission, setMission] = useState("");
  const [workspace, setWorkspace] = useState("");
  // IMP-4: the run's safety profile. POST /api/runs has always accepted it and
  // the v2 launcher never sent one, so every run silently got "all" and the
  // main safety control was unreachable from this UI.
  const [profile, setProfile] = useState(DEFAULT_PROFILE);
  // R3-4: code execution for an unattended run. `POST /api/runs` sets
  // `allow_code=True` for every run, but execution ALSO needs its own explicit
  // confirmation (13-3) and the run's chat is filtered out of the session list
  // while it is active — so before this control existed there was NO surface
  // that could grant it, and run_shell/run_python/start_job were refused for
  // every run with advice pointing at a setting nobody could reach.
  //
  // Deliberately a separate, default-OFF switch rather than part of the profile:
  // a profile narrows a roster, this one GRANTS a capability, and conflating
  // them is how "all" came to mean something it did not do.
  const [exec, setExec] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        new mission
      </h3>
      <textarea
        ref={missionRef}
        value={mission}
        onChange={(e) => {
          setMission(e.target.value);
          // grow with the prompt (owner request): auto height up to a sane
          // cap, then scroll inside — long pasted missions stay readable
          const el = e.currentTarget;
          el.style.height = "auto";
          el.style.height = Math.min(el.scrollHeight, 420) + "px";
        }}
        rows={3}
        placeholder="What should the agent do, start to finish? Name concrete deliverables."
        aria-label="Mission"
        className="w-full rounded-md bg-surface px-3 py-2 text-[13.5px] outline-none resize-y placeholder:text-muted max-h-[420px] overflow-y-auto"
      />
      <label className="flex items-center gap-2 mt-2 text-[12.5px]">
        <span className="w-24 shrink-0 text-secondary">safety profile</span>
        <select
          value={profile}
          onChange={(e) => setProfile(e.target.value)}
          aria-label="Run safety profile"
          className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none"
        >
          {PROFILES.map((p) => (
            <option key={p.value} value={p.value}>{p.label}</option>
          ))}
        </select>
      </label>
      <p className="text-[11px] text-muted leading-snug pl-24 pr-1">
        {profileSentence(profile)}
      </p>
      <label className="flex items-start gap-2 mt-2 text-[12.5px]">
        <input
          type="checkbox"
          checked={exec}
          onChange={(e) => setExec(e.target.checked)}
          aria-label="Allow code execution in this run"
          className="mt-0.5 shrink-0"
        />
        <span className="text-secondary leading-snug">
          allow code execution
          <span className="block text-[11px] text-muted">
            {exec
              ? "run_shell, run_python and start_job may run commands on this "
                + "machine for the whole run. Only enable this for a mission you "
                + "would run yourself."
              : "off \u2014 the agent can still read and write files, but cannot "
                + "run commands. This is the default."}
          </span>
        </span>
      </label>
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
                  profile,
                  confirm_exec: exec,
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
  const [err, setErr] = useState<string | null>(null);
  const act = async (path: string, body?: unknown): Promise<boolean> => {
    setErr(null);
    try {
      const r = await fetch(`/api/runs/${run.id}/${path}`, {
        method: "POST",
        headers: body !== undefined ? { "content-type": "application/json" } : {},
        body: body !== undefined ? JSON.stringify(body) : undefined,
      });
      if (!r.ok) {
        // AUDIT F11-4: a refused pause/stop/resume used to look exactly like
        // one that worked — the poll just re-read the same old state. Say what
        // the server said and do not pretend the action landed.
        setErr(await responseError(r));
        return false;
      }
    } catch (e) {
      setErr((e as Error).message);
      return false;
    }
    onAction();
    return true;
  };
  const plan = run.plan ?? [];
  const done = plan.filter((s) => s.status === "done").length;
  const b = budget(run, Date.now() / 1000);
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
      {/* IMP-8: the step and time budget the run is judged against — both are
          already in the record. The token axis is deliberately absent: the
          server never writes `tokens_used` (finding 03-2), so a token number
          here would be a fiction. */}
      <div className="font-mono text-[11px] text-muted mb-2">
        budget · {b.remainingSteps.toLocaleString()} of{" "}
        {b.ceiling.toLocaleString()} steps left ·{" "}
        {remainingTime(b.remainingSeconds)} of run time left
      </div>
      {err && (
        <div role="alert"
             className="rounded-md bg-red/10 text-red px-2.5 py-1.5 text-[12px] mb-2">
          {err}
        </div>
      )}
      <p className="text-[13px] text-secondary mb-3">{run.mission}</p>
      {run.pending_question?.q && (
        <div className="rounded-md bg-amber/10 border border-amber/30 p-3 mb-3">
          <div className="font-mono text-[11px] text-amber uppercase tracking-[0.08em] mb-1">
            the agent is asking you
          </div>
          <p className="text-[13px] text-primary mb-2">{run.pending_question.q}</p>
          <p className="text-[11.5px] text-muted">
            Answer in the box below — sending resumes the run.
          </p>
        </div>
      )}
      {/* What the run is FOR. The plan below says how; this says what, and it
          was the missing half — a user watching an unattended run could see
          every step and not the objective they served. */}
      {run.spec?.objective && (
        <div className="rounded-md bg-panel px-3 py-2 mb-3">
          <div className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] mb-0.5">
            objective
            {run.spec.compiled === false && (
              <span className="text-amber normal-case tracking-normal">
                {" "}— compiled from the mission as given
              </span>
            )}
          </div>
          <p className="text-[13px] text-primary">{run.spec.objective}</p>
          {(run.spec.deliverables?.length ?? 0) > 0 && (
            <details className="mt-1">
              <summary className="cursor-pointer font-mono text-[11px] text-secondary">
                must exist when it finishes: {run.spec.deliverables!.length} file
                {run.spec.deliverables!.length === 1 ? "" : "s"}
              </summary>
              <ul className="mt-1 flex flex-col gap-0.5 pl-3 text-[11.5px] text-muted">
                {run.spec.deliverables!.map((d) => (
                  <li key={d.path} className="font-mono break-all">
                    — {d.path}
                    {d.description && (
                      <span className="font-sans text-muted"> {d.description}</span>
                    )}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {(run.spec.constraints?.length ?? 0) > 0 && (
            <details className="mt-1">
              <summary className="cursor-pointer font-mono text-[11px] text-secondary">
                {run.spec.constraints!.length} constraint
                {run.spec.constraints!.length === 1 ? "" : "s"}
              </summary>
              <ul className="mt-1 flex flex-col gap-0.5 pl-3 text-[11.5px] text-muted">
                {run.spec.constraints!.map((c, i) => (
                  <li key={i}>— {c}</li>
                ))}
              </ul>
            </details>
          )}
        </div>
      )}
      {plan.length > 0 && (
        <ul className="flex flex-col gap-1 mb-3">
          {plan.map((s) => (
            <li key={s.id} className="flex items-start gap-2 text-[12.5px]">
              {/* Three states, three glyphs. `blocked` used to render as the
                  same ○ as `pending`, so a run stuck on a step looked like a
                  run that had not reached it — the opposite diagnosis. */}
              <span className={`font-mono mt-px ${planTone(s.status)}`}>
                {planGlyph(s.status)}
              </span>
              <span className={s.status === "done" ? "text-muted" : "text-primary"}>
                {s.text}
                {s.status === "blocked" && (
                  <span className="text-red"> — blocked</span>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
      {(run.activity?.length ?? 0) > 0 && (
        <div className="bg-canvas rounded-md p-3 max-h-72 overflow-y-auto flex flex-col gap-1.5">
          {run.activity!.map((a, i) => (
            <div key={i} className={`text-[12px] whitespace-pre-wrap break-words ${
              a.kind === "tool" ? "font-mono text-amber"
              : a.kind === "result" ? "font-mono text-moss"
              : a.kind === "think" ? "text-muted italic"
              : "text-secondary"}`}>
              {a.kind === "tool" ? "→ " : a.kind === "result" ? "✓ " : ""}
              {a.text}
            </div>
          ))}
        </div>
      )}
      {run.log_tail && (
        <details className="mt-2">
          <summary className="font-mono text-[11px] text-muted cursor-pointer">
            progress log
          </summary>
          <pre className="font-mono text-[11.5px] text-secondary bg-canvas rounded-md p-3 max-h-44 overflow-y-auto whitespace-pre-wrap mt-1">
            {run.log_tail}
          </pre>
        </details>
      )}
      <form
        className="flex gap-1.5 mt-3"
        onSubmit={(e) => {
          e.preventDefault();
          if (!note.trim()) return;
          // keep the note when the server refused it — clearing it hid the
          // fact that nothing was delivered (AUDIT F11-4)
          void act("inject", { message: note.trim() })
            .then((ok) => { if (ok) setNote(""); });
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
  const [histErr, setHistErr] = useState<string | null>(null);
  // A refused restart is not a load failure: keep the two sentences apart, or
  // "run is not restartable" renders as "could not load runs".
  const [actErr, setActErr] = useState<string | null>(null);
  const activeId = useRef<string | null>(null);
  // IMP-10: the empty history's primary action focuses the mission box.
  const missionRef = useRef<HTMLTextAreaElement>(null);

  const refreshHistory = useCallback(async () => {
    // IMP-10: a failed fetch must not render as "no runs yet" — keep the last
    // list, but say the request failed.
    try {
      const res = await readList<RunSummary>(await fetch("/api/runs"));
      if (!res.ok) {
        setHistErr(res.error);
        return;
      }
      setHistErr(null);
      setHistory(res.rows);
    } catch (e) {
      setHistErr((e as Error).message);
    }
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
            missionRef={missionRef}
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
            {/* IMP-8: WHY it stopped, not only that it did — "finished" and
                "gave up" used to be the same screen. The server's own
                halt_reason wins when it wrote one. */}
            <p className="text-[13px] text-secondary">
              {stopSentence(active.status, active.halt_reason)}
            </p>
            {active.summary && (
              <p className="text-[12.5px] text-muted mt-1.5">{active.summary}</p>
            )}
            <p className="font-mono text-[11px] text-muted mt-2">
              {active.mission}
            </p>
          </section>
        )}
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
            history
          </h3>
          {actErr && <InlineError message={actErr} />}
          {histErr && (
            <LoadError message={`could not load runs: ${histErr}`}
                       onRetry={() => void refreshHistory()} />
          )}
          {!histErr && history.length === 0 && (
            <EmptyState
              title="no runs yet"
              body="A mission runs unattended and leaves its files in the workspace you name. Launch one above and its progress appears here."
              actionLabel="launch a mission"
              onAction={() => missionRef.current?.focus()} />
          )}
          <ul className="flex flex-col gap-1">
              {history.map((h) => (
                <li key={h.id} className="flex items-center gap-3 text-[12.5px] rounded-md hover:bg-surface px-2 py-1">
                  <span className={`font-mono text-[11px] w-16 shrink-0 ${statusTone(h.status)}`}
                        title={stopSentence(h.status)}>
                    {h.status}
                  </span>
                  <span className="flex-1 truncate text-secondary">{h.mission}</span>
                  {RESTARTABLE.has(h.status) && !active && (
                    <button
                      onClick={async () => {
                        setActErr(null);
                        try {
                          const r = await fetch(
                            `/api/runs/${h.id}/restart`, { method: "POST" });
                          if (!r.ok) {
                            // AUDIT F11-4: a 409 used to set activeId and poll
                            // anyway, which re-read the old terminal state and
                            // made the resume button vanish with no reason.
                            setActErr(await responseError(r));
                            return;
                          }
                          activeId.current = h.id;
                          void pollActive();
                        } catch (e) {
                          setActErr((e as Error).message);
                        }
                      }}
                      className="shrink-0 rounded-md bg-amber/15 text-amber px-2 py-0.5 text-[11.5px] font-semibold"
                    >
                      resume
                    </button>
                  )}
                  <span className="font-mono text-[11px] text-muted shrink-0">{h.id.slice(0, 15)}</span>
                </li>
              ))}
            </ul>
          </section>
      </div>
    </main>
  );
}
