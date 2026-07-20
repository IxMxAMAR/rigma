// Engine room: what's loaded, how it's doing, and the levers.
// Telemetry is mono + instant (CONSTITUTION §6: never animate data values).
import { useCallback, useEffect, useState } from "react";
import { engineApi, type ServerInfo, type SwitchOption } from "../lib/engineApi";

function uptime(startedAt: number): string {
  const s = Math.max(0, Math.floor(Date.now() / 1000 - startedAt));
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
  return `${Math.floor(s / 86400)}d ${Math.floor((s % 86400) / 3600)}h`;
}

function Stat({ label, value, tone }: { label: string; value: string;
                                        tone?: "amber" | "moss" | "red" }) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="font-mono text-[11px] text-muted uppercase tracking-[0.08em]">
        {label}
      </span>
      <span
        className={`font-mono text-[15px] ${
          tone === "amber" ? "text-amber"
          : tone === "moss" ? "text-moss"
          : tone === "red" ? "text-red" : "text-primary"
        }`}
      >
        {value}
      </span>
    </div>
  );
}

export default function EngineSurface() {
  const [info, setInfo] = useState<ServerInfo | null>(null);
  const [options, setOptions] = useState<SwitchOption[]>([]);
  const [log, setLog] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setInfo(await engineApi.server());
      setErr(null);
    } catch (e) {
      setInfo(null);
      setErr((e as Error).message);
    }
  }, []);

  useEffect(() => {
    void refresh();
    const t = setInterval(refresh, 4000);
    return () => clearInterval(t);
  }, [refresh]);

  useEffect(() => {
    engineApi.switchOptions().then(setOptions).catch(() => setOptions([]));
    engineApi.log().then(setLog).catch(() => {});
  }, [info?.model]);

  const act = async (name: string, fn: () => Promise<unknown>) => {
    setBusy(name);
    setErr(null);
    try {
      await fn();
      await refresh();
    } catch (e) {
      setErr((e as Error).message);
    } finally {
      setBusy(null);
    }
  };

  if (!info)
    return (
      <main className="flex-1 flex items-center justify-center">
        <div className="text-center">
          <div className="font-mono text-[12px] text-muted uppercase tracking-[0.1em] mb-2">
            engine
          </div>
          <p className="text-secondary text-[13.5px]">
            {err ? `Engine unreachable: ${err}` : "No engine running."}
          </p>
          <button
            onClick={() => void act("load", engineApi.load)}
            className="mt-4 rounded-md bg-amber/15 text-amber px-4 py-1.5 text-[13px] font-semibold"
          >
            {busy === "load" ? "loading…" : "load last model"}
          </button>
        </div>
      </main>
    );

  const tg = info.last_tg ?? null;
  const verdictTone =
    info.verdict === "healthy" ? "moss" : info.verdict ? "red" : undefined;

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[1200px] mx-auto flex flex-col gap-5">
        {err && (
          <div className="rounded-md bg-red/10 text-red px-3 py-2 text-[13px]">{err}</div>
        )}

        <section className="rounded-lg bg-panel p-5">
          <div className="flex items-baseline justify-between mb-4">
            <h2 className="text-[15px] font-semibold">{info.model || "—"}</h2>
            <span className="font-mono text-[12px] text-secondary">
              {info.engine_version as string}
            </span>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-4 xl:grid-cols-8 gap-4">
            <Stat label="quant" value={String(info.quant ?? "—")} />
            <Stat label="backend" value={String(info.backend ?? "—")} />
            <Stat label="context" value={info.ctx ? `${Math.round((info.ctx as number) / 1024)}K` : "—"} />
            <Stat
              label="tok/s"
              value={tg != null ? tg.toFixed(1) : "—"}
              tone={verdictTone as "moss" | "red" | undefined}
            />
            <Stat label="expected" value={info.expected_tg != null ? `${(info.expected_tg as number).toFixed(0)}` : "—"} />
            <Stat label="uptime" value={info.started_at ? uptime(info.started_at as number) : "—"} />
            <Stat label="port" value={String((info as Record<string, unknown>).public_port ?? "—")} />
            <Stat label="verdict" value={String(info.verdict ?? "—")}
                  tone={verdictTone as "moss" | "red" | undefined} />
          </div>
          {info.openai_base != null && (
            <div className="font-mono text-[11.5px] text-muted mt-3">
              OpenAI API: <span className="text-secondary">{String(info.openai_base)}</span>
            </div>
          )}
          <div className="flex gap-2 mt-5 items-center">
            <label className="flex items-center gap-1.5 font-mono text-[12px] text-secondary">
              ctx
              <select
                value={String(info.ctx ?? "")}
                onChange={(e) => {
                  const v = Number(e.target.value);
                  if (v && v !== info.ctx &&
                      window.confirm(`Relaunch at ${Math.round(v / 1024)}K context?`))
                    void act("ctx", () => engineApi.setCtx(v));
                }}
                aria-label="Context size"
                className="rounded-md bg-surface px-2 py-1 outline-none"
              >
                {[8192, 16384, 32768, 65536, 131072]
                  .concat(info.ctx && ![8192, 16384, 32768, 65536, 131072]
                          .includes(info.ctx as number) ? [info.ctx as number] : [])
                  .sort((a, b) => a - b)
                  .map((v) => (
                    <option key={v} value={v}>{Math.round(v / 1024)}K</option>
                  ))}
              </select>
            </label>
            <button
              onClick={() => void act("unload", engineApi.unload)}
              className="rounded-md bg-surface hover:bg-float px-3 py-1.5 text-[13px]"
            >
              {busy === "unload" ? "unloading…" : "unload"}
            </button>
            <button
              onClick={() => void act("recal", engineApi.recalibrate)}
              className="rounded-md bg-surface hover:bg-float px-3 py-1.5 text-[13px]"
            >
              {busy === "recal" ? "tuning…" : "recalibrate"}
            </button>
          </div>
        </section>

        {options.length > 0 && (
          <section className="rounded-lg bg-panel p-5">
            <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-3">
              switch to (on disk, no downloads)
            </h3>
            <ul className="flex flex-col gap-1">
              {options.map((o) => (
                <li key={o.model} className="flex items-center gap-3 rounded-md hover:bg-surface px-3 py-2">
                  <div className="flex-1 min-w-0">
                    <div className="text-[13.5px] truncate">{o.model}</div>
                    <div className="font-mono text-[11.5px] text-muted">{o.reason}</div>
                  </div>
                  <button
                    onClick={() => void act(`sw-${o.model}`, () => engineApi.switchTo(o.model))}
                    className="shrink-0 rounded-md bg-amber/15 text-amber px-3 py-1 text-[12.5px] font-semibold"
                  >
                    {busy === `sw-${o.model}` ? "switching…" : "switch"}
                  </button>
                </li>
              ))}
            </ul>
          </section>
        )}

        <section className="rounded-lg bg-panel p-5">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-3">
            engine log
          </h3>
          <pre className="font-mono text-[11.5px] text-secondary whitespace-pre-wrap max-h-64 overflow-y-auto">
            {log || "(empty)"}
          </pre>
        </section>
      </div>
    </main>
  );
}
