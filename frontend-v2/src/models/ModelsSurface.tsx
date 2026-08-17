// The Hangar: installed models as cards, quants with live download progress,
// HF search-and-add. Polls fast only while a download is actually running.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_FIT, engineApi, eta, gb,
  type FitConfig, type HfHit, type ModelCard, type QuantRow,
} from "../lib/engineApi";

function PullBar({ q }: { q: QuantRow }) {
  const p = q.pull;
  if (!p || p.status !== "downloading") return null;
  const pct = p.done != null ? Math.min(100, (p.done / q.bytes) * 100) : 0;
  return (
    <div className="flex items-center gap-2 flex-1 min-w-0">
      <div className="flex-1 h-1.5 rounded-full bg-canvas overflow-hidden">
        <div className="h-full bg-amber" style={{ width: `${pct}%` }} />
      </div>
      <span className="font-mono text-[11px] text-amber shrink-0">
        {pct.toFixed(0)}%{p.eta != null ? ` · ${eta(p.eta)}` : ""}
      </span>
    </div>
  );
}

const K = (n: number) =>
  n >= 1024 * 1024 ? `${Math.round(n / 1024 / 1024)}M` : `${Math.round(n / 1024)}K`;

/** How much of the quant sits on the GPU — the thing size alone can't tell you.
 *  A 22GB Q6 that "fits" by spilling half to RAM is slower than a 13GB Q3 that
 *  doesn't, so the tier is ranked by SPEED, not by file size. */
const SPEED: Record<string, { dot: string; text: string; label: string; hint: string }> = {
  gpu:     { dot: "bg-moss",  text: "text-moss",  label: "gpu",     hint: "fully on the GPU — fastest" },
  light:   { dot: "bg-moss/60", text: "text-secondary", label: "light", hint: "mostly on the GPU — still quick" },
  offload: { dot: "bg-amber", text: "text-amber", label: "offload", hint: "spills to system RAM — runs, but slow" },
  no:      { dot: "bg-red/70", text: "text-red",  label: "too big", hint: "does not fit this machine, even at minimum context" },
};

/** The arithmetic behind a single "8K". A bare number cannot show that a
 *  doubling was missed by 46MB, or that 888MB of it went to a vision projector
 *  you may not want. */
function budgetHint(fit?: QuantRow["fit"]): string {
  const b = fit?.budget;
  if (!b) {
    return fit?.ok ? "fits this machine" : "does not fit this machine";
  }
  const L = [
    `weights        ${b.file_mb.toLocaleString()} MB`,
    ...(b.mmproj_mb ? [`vision proj    ${b.mmproj_mb.toLocaleString()} MB  (always resident)`] : []),
    `KV @ ${K(b.ctx)} ${b.kv_type}   ${b.kv_mb.toLocaleString()} MB`,
    `──`,
    `VRAM budget    ${b.budget_mb.toLocaleString()} MB`,
    b.over_mb > 0
      ? `OVER by        ${b.over_mb.toLocaleString()} MB → that much spills to RAM`
      : `headroom       ${(-b.over_mb).toLocaleString()} MB`,
  ];
  return L.join("\n");
}

/** Context this quant can actually hold here — its own column, because it is
 *  the number people compare across quants and it must line up to be read. */
function CtxCell({ fit }: { fit?: QuantRow["fit"] }) {
  const has = fit?.ok && fit.ctx;
  return (
    <span
      className={`w-[46px] shrink-0 text-right font-mono text-[11px] whitespace-nowrap ${
        has ? "text-secondary" : "text-muted/50"}`}
      title={budgetHint(fit)}
    >
      {has ? K(fit!.ctx!) : "—"}
    </span>
  );
}

/** Where the weights end up. Driven by the resolver's own plan (ngl /
 *  n_cpu_moe), so it cannot claim "gpu" for a quant it decided to offload. */
function RunsCell({ fit }: { fit?: QuantRow["fit"] }) {
  if (!fit || fit.speed === undefined) return <span className="w-[74px] shrink-0" />;
  const s = SPEED[fit.speed] ?? SPEED.no;
  const off = fit.offload_pct ?? 0;
  return (
    <span
      className="w-[74px] shrink-0 flex items-center gap-1 font-mono text-[11px] whitespace-nowrap"
      title={`${s.hint}${off > 0 && fit.ok ? ` — ${off}% of the weights sit in system RAM` : ""}` +
             (fit.n_cpu_moe ? `; ${fit.n_cpu_moe} layers' experts on CPU` : "")}
    >
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${s.dot}`} />
      <span className={s.text}>{s.label}</span>
    </span>
  );
}

const TIER: Record<string, string> = {
  lossless: "text-moss", excellent: "text-moss", great: "text-secondary",
  good: "text-secondary", fair: "text-amber", poor: "text-amber",
  damaged: "text-red",
};

/** Quality given up vs BF16. Reference figures for the FORMAT, never a
 *  measurement of this model — hence the "≈" and the tooltip. A blank cell is
 *  the honest rendering for a repo whose own naming has no published data. */
function QualityCell({ q }: { q: QuantRow }) {
  const t = q.total;
  const k = q.quality;
  // Prefer the TOTAL: weights alone understates what you are running. A Q3 with
  // a q4_0 cache and a Q3 with an f16 cache are two different numbers, and the
  // weight column is identical for both.
  if (t) {
    const pct = t.total_pct < 0.1 ? "<0.1" : t.total_pct.toFixed(t.total_pct < 10 ? 1 : 0);
    return (
      <span
        className="w-[96px] shrink-0 font-mono text-[11px] flex items-center gap-1 whitespace-nowrap"
        title={`≈${t.total_pct}% worse than BF16 weights + f16 cache.\n` +
               `  weights (${q.quant}): ≈${t.weights_pct}%\n` +
               `  KV cache: ≈${t.kv_pct}%\n` +
               `  composed as ratios, not added.\n\n` +
               (k ? `${k.note}. ${k.bpw} bits/weight.\n\n` : "") +
               "Reference figures for the FORMATS, mostly from 7B-13B " +
               "LLaMA-family evals — NOT measured on this model. Larger models " +
               "lose less, so treat it as a pessimistic upper bound. The K:V " +
               "split of the cache term is weighted 2:1, an approximation."}
      >
        <span className={TIER[t.tier] ?? "text-secondary"}>≈{pct}%</span>
        <span className="text-muted">{t.tier}</span>
      </span>
    );
  }
  if (!k) {
    return (
      <span className="w-[96px] shrink-0 font-mono text-[11px] text-muted/50"
            title={"No published quality figure for this file's naming — it " +
                   "does not use a standard llama.cpp quant name, so any " +
                   "percentage here would be made up."}>
        —
      </span>
    );
  }
  const pct = k.ppl_pct < 0.1 ? "<0.1" : k.ppl_pct.toFixed(k.ppl_pct < 10 ? 1 : 0);
  return (
    <span
      className="w-[96px] shrink-0 font-mono text-[11px] flex items-center gap-1 whitespace-nowrap"
      title={`${k.note}. Typical +${k.ppl_pct}% perplexity vs BF16 at ${k.bpw} ` +
             `bits/weight (~${Math.round((k.bpw / 16) * 100)}% of BF16 size).\n\n` +
             "Reference figure for the quant FORMAT, mostly measured on 7B-13B " +
             "LLaMA-family models — NOT measured on this model. Larger models " +
             "lose less, so on a 27B treat it as a pessimistic upper bound."}
    >
      <span className={TIER[k.tier] ?? "text-secondary"}>≈{pct}%</span>
      <span className="text-muted">{k.tier}</span>
    </span>
  );
}

function QuantLine({ card, q, onAction, best }: {
  card: ModelCard; q: QuantRow; onAction: () => void; best?: string;
}) {
  const [busy, setBusy] = useState(false);
  const downloading = q.pull?.status === "downloading";
  const run = async (fn: () => Promise<unknown>) => {
    setBusy(true);
    try { await fn(); } catch { /* surfaced via next poll */ }
    setBusy(false);
    onAction();
  };
  return (
    <li className="flex items-center gap-2 px-3 py-1.5 rounded-md hover:bg-surface/70">
      <span
        className={`w-1.5 h-1.5 rounded-full shrink-0 ${q.on_disk ? "bg-moss" : "bg-float"}`}
        title={q.on_disk ? "on disk" : "not downloaded"}
      />
      <span className={`font-mono text-[12.5px] w-24 shrink-0 truncate ${
        q.fit?.ok === false ? "text-muted" : ""}`} title={q.file}>
        {q.quant}
      </span>
      <span className="font-mono text-[12px] text-muted w-[52px] shrink-0 text-right">{gb(q.bytes)}</span>
      {downloading ? <PullBar q={q} /> : (
        <>
          <CtxCell fit={q.fit} />
          <RunsCell fit={q.fit} />
          <QualityCell q={q} />
        </>
      )}
      {!downloading && best === q.quant && (
        <span className="shrink-0 font-mono text-[10.5px] text-amber bg-amber/10 rounded px-1.5 py-0.5"
              title="best quality that still runs at GPU speed here">
          best here
        </span>
      )}
      {!downloading && <span className="flex-1" />}
      {!q.on_disk && q.pullable && !downloading && (
        <button
          disabled={busy}
          onClick={() => void run(() => engineApi.pull(card.slug, q.file))}
          className="shrink-0 rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
        >
          pull
        </button>
      )}
      {q.on_disk && !card.running && (
        <button
          disabled={busy}
          onClick={() => {
            if (window.confirm(`Delete ${q.file} from disk?`))
              void run(() => engineApi.deleteFile(card.slug, q.file));
          }}
          className="shrink-0 rounded-md px-2 py-0.5 text-[12px] text-muted hover:text-red hover:bg-surface"
          aria-label={`delete ${q.file}`}
        >
          ×
        </button>
      )}
    </li>
  );
}

function Card({ card, onAction }: { card: ModelCard; onAction: () => void }) {
  const [busy, setBusy] = useState(false);
  const anyOnDisk = card.quants.some((q) => q.on_disk);
  return (
    <section className="rounded-lg bg-panel p-4">
      <div className="flex items-center gap-2 mb-1">
        <h3 className="text-[14px] font-semibold truncate">{card.slug}</h3>
        {card.running && (
          <span className="font-mono text-[10.5px] text-moss bg-moss/10 rounded px-1.5 py-0.5">
            RUNNING
          </span>
        )}
        {card.custom && (
          <span className="font-mono text-[10.5px] text-secondary bg-surface rounded px-1.5 py-0.5">
            custom
          </span>
        )}
        {anyOnDisk && !card.running && (
          <button
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              try { await engineApi.switchTo(card.slug); } catch { /* poll */ }
              setBusy(false);
              onAction();
            }}
            className="ml-auto rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
          >
            {busy ? "switching…" : "run"}
          </button>
        )}
      </div>
      <div className="font-mono text-[11.5px] text-muted mb-2">
        {card.kind} · {Math.round(card.native_ctx / 1024)}K native
        {card.capabilities.length > 0 && ` · ${card.capabilities.join(" ")}`}
      </div>
      <div className="flex items-center gap-2 px-3 pb-1 font-mono text-[10px]
                      text-muted uppercase tracking-[0.06em]">
        <span className="w-1.5 shrink-0" />
        <span className="w-24 shrink-0">quant</span>
        <span className="w-[52px] shrink-0 text-right">size</span>
        <span className="w-[46px] shrink-0 text-right" title="context window this quant can hold on this machine">
          ctx
        </span>
        <span className="w-[74px] shrink-0" title="where the weights end up: fully on the GPU, or partly in system RAM">
          runs
        </span>
        <span className="w-[96px] shrink-0" title="typical quality given up vs BF16 — reference figure for the format, not measured on this model">
          vs bf16
        </span>
      </div>
      <ul className="flex flex-col">
        {card.quants.map((q) => (
          <QuantLine key={q.file} card={card} q={q} onAction={onAction}
                     best={card.recommended ?? undefined} />
        ))}
        {card.mmproj && (
          <QuantLine
            key={card.mmproj.file}
            card={card}
            q={{ ...card.mmproj, quant: "mmproj" } as QuantRow}
            onAction={onAction}
          />
        )}
      </ul>
    </section>
  );
}

function HfSearch({ onAdded }: { onAdded: () => void }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<HfHit[]>([]);
  const [state, setState] = useState<"idle" | "busy" | "err">("idle");
  const [adding, setAdding] = useState<string | null>(null);
  const [addErr, setAddErr] = useState<string | null>(null);
  const timer = useRef<number | null>(null);

  const search = (text: string) => {
    setQ(text);
    if (timer.current) window.clearTimeout(timer.current);
    if (!text.trim()) { setHits([]); return; }
    timer.current = window.setTimeout(async () => {
      setState("busy");
      try {
        setHits(await engineApi.hfSearch(text));
        setState("idle");
      } catch { setState("err"); }
    }, 350);
  };

  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        add from hugging face
      </h3>
      <input
        value={q}
        onChange={(e) => search(e.target.value)}
        placeholder="search GGUF repos…"
        aria-label="Search Hugging Face"
        className="w-full rounded-md bg-surface px-3 py-2 text-[13.5px] outline-none placeholder:text-muted focus:bg-float"
      />
      {state === "busy" && <div className="font-mono text-[11.5px] text-muted mt-2">searching…</div>}
      {state === "err" && <div className="font-mono text-[11.5px] text-red mt-2">search failed — offline?</div>}
      {q.trim() !== "" && state === "idle" && hits.length === 0 && (
        <div className="font-mono text-[11.5px] text-muted mt-2">
          no GGUF repos match that — check the spelling, or paste the exact
          owner/name from the model's Hugging Face page
        </div>
      )}
      {addErr && (
        <div className="rounded-md bg-red/10 text-red px-2.5 py-1.5 font-mono text-[11.5px] mt-2">
          {addErr}
        </div>
      )}
      <ul className="mt-2 flex flex-col gap-1">
        {hits.slice(0, 8).map((h) => (
          <li key={h.repo} className="flex items-center gap-2 rounded-md hover:bg-surface px-2 py-1.5">
            <span className="font-mono text-[12.5px] flex-1 truncate" title={h.repo}>{h.repo}</span>
            <button
              disabled={adding === h.repo}
              onClick={async () => {
                setAdding(h.repo);
                setAddErr(null);
                // Say so when it fails. Swallowing the error left the button
                // looking inert and gave no way to tell a bad repo from a
                // dead network.
                try {
                  await engineApi.hfAdd(h.repo);
                  onAdded();
                } catch (e) {
                  setAddErr(`could not add ${h.repo}: ${(e as Error).message}`);
                }
                setAdding(null);
              }}
              className="shrink-0 rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
            >
              {adding === h.repo ? "adding…" : "add"}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

/** The knobs the fit math used to hide. Each one is a real choice with a real
 *  cost, and the page previously picked one silently and showed the result as
 *  if it were the only answer. Nothing here launches or saves anything — it
 *  changes what the arithmetic ASSUMES, so it is safe to play with. */
function FitControls({ cfg, onChange }: {
  cfg: FitConfig; onChange: (c: FitConfig) => void;
}) {
  const sel = "rounded bg-surface px-1.5 py-0.5 font-mono text-[11px] outline-none";
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1.5 rounded-lg bg-panel px-4 py-2">
      <span className="font-mono text-[10px] text-muted uppercase tracking-[0.08em]">
        assume
      </span>

      <label className="flex items-center gap-1.5 font-mono text-[11px] text-secondary"
             title={"KV cache precision. Halving it roughly doubles the context " +
                    "that fits — but unlike weight error, cache error is applied " +
                    "per token as it is written and every later token attends " +
                    "over the degraded history, so it compounds with sequence " +
                    "length.\n\nK and V are always the same type here: " +
                    "llama.cpp's fused flash-attention kernel only fires when " +
                    "they match, and a mismatch silently drops to a much slower " +
                    "path (measured on RDNA4)."}>
        kv cache
        <select className={sel} value={cfg.kv}
                onChange={(e) => onChange({ ...cfg, kv: e.target.value })}
                aria-label="KV cache precision">
          <option value="">model default</option>
          <option value="f16">f16 (no loss)</option>
          <option value="q8_0">q8_0</option>
          <option value="q5_1">q5_1</option>
          <option value="q4_0">q4_0</option>
        </select>
      </label>

      <label className="flex items-center gap-1.5 font-mono text-[11px] text-secondary"
             title={"The vision projector is permanently resident and counted " +
                    "against VRAM whether or not you ever send an image. " +
                    "Un-tick to see the text-only budget."}>
        <input type="checkbox" checked={cfg.vision} className="accent-amber"
               onChange={(e) => onChange({ ...cfg, vision: e.target.checked })} />
        vision projector
      </label>

      <label className="flex items-center gap-1.5 font-mono text-[11px] text-secondary"
             title={"speed: never move a GPU layer to RAM to gain context (the " +
                    "default). context: spend up to 15% of the layers for a " +
                    "bigger window — every offloaded layer costs time on EVERY " +
                    "token of a dense model."}>
        prefer
        <select className={sel} value={cfg.grow}
                onChange={(e) => onChange({ ...cfg,
                  grow: e.target.value as FitConfig["grow"] })}
                aria-label="Growth policy">
          <option value="speed">speed</option>
          <option value="context">context</option>
        </select>
      </label>

      {(cfg.kv || !cfg.vision || cfg.grow !== "speed") && (
        <button onClick={() => onChange(DEFAULT_FIT)}
                className="font-mono text-[11px] text-amber hover:underline">
          reset
        </button>
      )}
    </div>
  );
}

export default function ModelsSurface() {
  const [cards, setCards] = useState<ModelCard[]>([]);
  const [cfg, setCfg] = useState<FitConfig>(() => {
    try {
      const raw = localStorage.getItem("rigma.fitConfig");
      return raw ? { ...DEFAULT_FIT, ...JSON.parse(raw) } : DEFAULT_FIT;
    } catch { return DEFAULT_FIT; }
  });
  const setCfgPersist = (c: FitConfig) => {
    setCfg(c);
    localStorage.setItem("rigma.fitConfig", JSON.stringify(c));
  };
  const [view, setView] = useState<"grid" | "list">(
    () => (localStorage.getItem("rigma.modelsView") === "list" ? "list" : "grid"));
  const setViewPersist = (v: "grid" | "list") => {
    setView(v);
    localStorage.setItem("rigma.modelsView", v);
  };
  const refresh = useCallback(async () => {
    try {
      setCards((await engineApi.models(cfg)).models);
    } catch { /* keep last */ }
  }, [cfg]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // fast poll ONLY while something is downloading; slow heartbeat otherwise
  const downloading = cards.some((c) =>
    [...c.quants, ...(c.mmproj ? [c.mmproj] : [])].some(
      (qq) => qq.pull?.status === "downloading"));
  useEffect(() => {
    const t = setInterval(refresh, downloading ? 1500 : 8000);
    return () => clearInterval(t);
  }, [refresh, downloading]);

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[1200px] mx-auto flex flex-col gap-4">
        <div className="flex items-start gap-3">
          <div className="flex-1 min-w-0">
            <HfSearch onAdded={refresh} />
          </div>
          <div className="shrink-0 flex rounded-md bg-panel p-0.5 font-mono text-[12px]"
               role="group" aria-label="View">
            {(["grid", "list"] as const).map((v) => (
              <button
                key={v}
                onClick={() => setViewPersist(v)}
                aria-pressed={view === v}
                className={`px-2.5 py-1 rounded ${view === v ? "bg-surface text-primary" : "text-muted hover:text-secondary"}`}
              >
                {v}
              </button>
            ))}
          </div>
        </div>
        <FitControls cfg={cfg} onChange={setCfgPersist} />
        {cards.length === 0 && (
          <p className="text-secondary text-[13.5px] text-center pt-12">
            No models yet — search Hugging Face above, or drop a GGUF into
            ~/.rigma/models.
          </p>
        )}
        {/* grid = CSS column flow (masonry-ish): short cards pack under each
            other instead of leaving row-aligned holes next to tall ones */}
        <div className={view === "grid"
          ? "columns-1 lg:columns-2 gap-4"
          : "flex flex-col gap-4"}>
          {cards.map((c) => (
            <div key={c.slug}
                 className={view === "grid" ? "break-inside-avoid mb-4" : ""}>
              <Card card={c} onAction={refresh} />
            </div>
          ))}
        </div>
      </div>
    </main>
  );
}
