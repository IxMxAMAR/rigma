// The Hangar: installed models as cards, quants with live download progress,
// HF search-and-add. Polls fast only while a download is actually running.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  DEFAULT_FIT, engineApi, eta, gb,
  type FitConfig, type HfHit, type HfRepoDetail, type ModelCard,
  type QuantRow,
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
  if (!fit || fit.speed === undefined) return <span className="w-[66px] shrink-0" />;
  const s = SPEED[fit.speed] ?? SPEED.no;
  const off = fit.offload_pct ?? 0;
  return (
    <span
      className="w-[66px] shrink-0 flex items-center gap-1 font-mono text-[11px] whitespace-nowrap"
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
        className="w-[78px] shrink-0 font-mono text-[11px] flex items-center gap-1 whitespace-nowrap"
        title={`≈${t.total_pct}% worse than BF16 weights + f16 cache.\n` +
               `  weights (${q.quant}): ≈${t.weights_pct}%\n` +
               `  KV cache: ≈${t.kv_pct}%\n` +
               `  composed as ratios, not added.\n\n` +
               (k ? `${k.note}. ${k.bpw} bits/weight.\n\n` : "") +
               "Reference figures for the FORMATS, mostly from 7B-13B " +
               "LLaMA-family evals — NOT measured on this model. Larger models " +
               "lose less, so treat it as a pessimistic upper bound."}
      >
        <span className={TIER[t.tier] ?? "text-secondary"}>≈{pct}%</span>
        <span className="text-muted">{t.tier}</span>
      </span>
    );
  }
  if (!k) {
    // No published figure for this naming — but the file's own bits-per-weight
    // is still countable, and an em dash on every row of a repo that names its
    // quants I-Balanced / I-Quality / I-Compact left nothing to choose between
    // them. bpw is measured, not a quality percentage, and is labelled as such.
    return (
      <span className="w-[78px] shrink-0 font-mono text-[11px] text-muted/50"
            title={"No published quality figure for this file's naming — it " +
                   "does not use a standard llama.cpp quant name, so any " +
                   "percentage here would be made up.\n\n" +
                   (q.bpw
                     ? `What it does spend is ${q.bpw} bits per weight, ` +
                       "measured: this file's bytes divided by the model's " +
                       "parameter count. That is the size of the budget, not " +
                       "the quality it buys — those need a perplexity run."
                     : "")}>
        {q.bpw ? <span className="text-secondary">{q.bpw} bpw</span> : "—"}
      </span>
    );
  }
  const pct = k.ppl_pct < 0.1 ? "<0.1" : k.ppl_pct.toFixed(k.ppl_pct < 10 ? 1 : 0);
  return (
    <span
      className="w-[78px] shrink-0 font-mono text-[11px] flex items-center gap-1 whitespace-nowrap"
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

/** Whether THIS file carries the speculative-decoding draft head.
 *
 *  Per file, never per model: MTP is kept or dropped by the quantiser per
 *  artefact, so a card that wears one "mtp" chip over twenty quants is making a
 *  claim it cannot support. Only a positive, verified answer earns a mark —
 *  silence covers both "no head" and "not read yet", and the tooltip says
 *  which. Asking llama.cpp for draft-mtp against a file without the tensors
 *  resets the GPU driver, so an optimistic guess here is expensive. */
function MtpMark({ q }: { q: QuantRow }) {
  if (q.mtp !== true) return <span className="w-8 shrink-0" />;
  return (
    <span className="w-8 shrink-0 font-mono text-[10px] text-moss/90"
          title={"This file carries the MTP tensors (blk.N.nextn.*), read from " +
                 "its tensor table — so speculative decoding (--spec draft-mtp) " +
                 "can actually run on it.\n\nVerified in the file, not inferred " +
                 "from the repo name or the header."}>
      mtp
    </span>
  );
}

/** A gguf with no chat template at all.
 *
 *  The capability line above renders `capabilities.join(" ")`, so a quantiser
 *  that dropped tokenizer.chat_template produces a card that silently says
 *  nothing — indistinguishable from a model that genuinely has no tools and no
 *  thinking. It is worse than cosmetic: with no template in the file,
 *  llama-server falls back to a generic one, so the model is being prompted in
 *  a format it was not trained on, and nothing anywhere says so. */
function NoTemplateNotice({ card }: { card: ModelCard }) {
  if (card.has_template !== false) return null;
  // A repaired template standing in for the missing one changes the story
  // completely: the model is NOT on a fallback format. Saying otherwise was a
  // warning that had quietly become false (owner, 2026-08-19).
  const fixed = card.template_override === true;
  return (
    <div className={`rounded-md px-2.5 py-1.5 font-mono text-[11.5px] mb-2 ${
      fixed ? "bg-moss/10 text-moss" : "bg-amber/10 text-amber"}`}
         title={"llama.cpp falls back to a generic chat template when the gguf " +
                "carries none. Rigma passes ~/.rigma/templates/" + card.slug +
                ".jinja to the engine with --chat-template-file when that file " +
                "exists."}>
      {fixed
        ? <>no chat template in this gguf, so a replacement is installed and in
            use. The capability list below is still blank because it is read
            from the gguf — that says nothing about what the model can do.</>
        : <>no chat template in this gguf — the capability list below is blank
            because there is nothing to read, not because the model lacks the
            features. It is running on the engine's fallback format.</>}
    </div>
  );
}

function QuantLine({ card, q, onAction, best, showSpec }: {
  card: ModelCard; q: QuantRow; onAction: () => void; best?: string;
  showSpec?: boolean;
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
        className={`w-1.5 h-1.5 rounded-full shrink-0 ${
          q.running ? "bg-amber ring-2 ring-amber/30"
                    : q.on_disk ? "bg-moss" : "bg-float"}`}
        title={q.running ? "LOADED — this is the quant the engine is running"
                         : q.on_disk ? "on disk" : "not downloaded"}
      />
      {/* The one cell allowed to give way. Everything else in the row is a
          fixed column, so without something shrinkable a long quant name plus
          the "best here" badge pushed the action button clean outside the card
          (owner, 2026-08-23). A truncated name is the right thing to lose. */}
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
          {showSpec && <MtpMark q={q} />}
        </>
      )}
      {/* Badges and buttons travel together, pinned to the right edge. They
          used to be separate children after a flex-1 spacer, which let a badge
          displace the button past the edge of the card instead of squeezing
          the row. `ml-auto` does the pinning; `shrink-0` keeps the button
          reachable at any width. */}
      <div className="ml-auto flex min-w-0 items-center justify-end gap-2
                      overflow-hidden">
        {!downloading && best === q.quant && (
          <span className="font-mono text-[10.5px] text-amber bg-amber/10 rounded px-1.5 py-0.5"
                title="best quality that still runs at GPU speed here">
            best here
          </span>
        )}
        {!downloading && q.running && (
          <span className="font-mono text-[10.5px] text-amber bg-amber/15 rounded px-1.5 py-0.5"
                title="the engine has this exact quant loaded right now">
            loaded
          </span>
        )}
      </div>
      <div className="flex shrink-0 items-center gap-2">
        {q.on_disk && !q.running && !downloading && (
          <button
            disabled={busy}
            onClick={() => void run(() => engineApi.switchTo(card.slug, q.quant))}
            className="rounded-md bg-moss/15 text-moss px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
            title={"Stop the engine and load THIS quant. Two quants on disk had "
                   + "no way to choose between them."}
          >
            load
          </button>
        )}
        {!q.on_disk && q.pullable && !downloading && (
          <button
            disabled={busy}
            onClick={() => void run(() => engineApi.pull(card.slug, q.file))}
            className="rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
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
            className="rounded-md px-2 py-0.5 text-[12px] text-muted hover:text-red hover:bg-surface"
            aria-label={`delete ${q.file}`}
          >
            ×
          </button>
        )}
      </div>
    </li>
  );
}

/** How many rows a card shows before it asks permission to keep going. Below
 *  this a list is scannable; above it the card stops being a card. */
const COLLAPSE_AT = 12;

/** The variant axes present across a set of rows, in the order a reader meets
 *  them. Most repos have none and get no chips at all. */
function axesOf(rows: QuantRow[]): string[] {
  const seen: string[] = [];
  for (const r of rows)
    for (const v of r.variants ?? [])
      if (!seen.includes(v)) seen.push(v);
  return seen;
}

/** One chip per variant the repo actually ships. Off means "hide those rows",
 *  which is what makes 103 files readable as 26. */
function VariantChips({ axes, on, toggle, counts }: {
  axes: string[]; on: Set<string>; toggle: (a: string) => void;
  counts: Record<string, number>;
}) {
  return (
    <div className="flex items-center gap-1.5 px-3 pb-2 flex-wrap">
      <span className="font-mono text-[10px] text-muted uppercase tracking-[0.06em] mr-0.5">
        variants
      </span>
      {axes.map((a) => (
        <button
          key={a}
          onClick={() => toggle(a)}
          aria-pressed={on.has(a)}
          title={on.has(a)
            ? `hide the ${counts[a]} ${a} builds`
            : `this repo also ships ${counts[a]} ${a} builds — show them`}
          className={`rounded-md px-2 py-0.5 font-mono text-[11px] border transition-colors ${
            on.has(a)
              ? "bg-amber/15 text-amber border-amber/40"
              : "bg-surface text-muted border-transparent hover:text-secondary"}`}
        >
          {a}
          <span className="ml-1.5 opacity-60">{counts[a]}</span>
        </button>
      ))}
    </div>
  );
}

/** The quant list: variant chips, the column header, and the rows themselves.
 *  Shared by the library card and the pre-download repo view so the two cannot
 *  disagree about what a row means. */
/** Variant filtering + collapse, shared by the library card and the
 *  pre-download repo view. Both render a quant list; only the row markup
 *  differs, so the rule for WHICH rows appear lives here once. */
function useQuantView(rows: QuantRow[]) {
  const axes = axesOf(rows);
  const [on, setOn] = useState<Set<string>>(new Set());
  const [all, setAll] = useState(false);
  const toggle = (a: string) => setOn((prev) => {
    const next = new Set(prev);
    if (next.has(a)) next.delete(a); else next.add(a);
    return next;
  });
  const counts: Record<string, number> = {};
  for (const a of axes) counts[a] = rows.filter(
    (r) => (r.variants ?? []).includes(a)).length;

  // A row survives the filter when every variant it carries is switched on.
  // Anything ON DISK or RUNNING ignores the filter completely — hiding a file
  // the user already downloaded, or the one the engine is holding, would be a
  // filter that lies about what they have.
  const visible = rows.filter((r) => {
    if (r.on_disk || r.running) return true;
    return (r.variants ?? []).every((v) => on.has(v));
  });
  const shown = all || visible.length <= COLLAPSE_AT
    ? visible : visible.slice(0, COLLAPSE_AT);
  return { axes, on, toggle, counts, visible, shown, all, setAll,
           hidden: visible.length - shown.length };
}

/** "show 78 more" / "collapse" — only when there is something to hide. */
function MoreRows({ v }: { v: ReturnType<typeof useQuantView> }) {
  if (v.hidden > 0)
    return (
      <button
        onClick={() => v.setAll(true)}
        className="mt-1 mx-3 self-start rounded-md bg-surface px-2.5 py-1 font-mono text-[11.5px] text-secondary hover:text-amber"
      >
        show {v.hidden} more {v.hidden === 1 ? "quant" : "quants"}
      </button>
    );
  if (v.all && v.visible.length > COLLAPSE_AT)
    return (
      <button
        onClick={() => v.setAll(false)}
        className="mt-1 mx-3 self-start rounded-md bg-surface px-2.5 py-1 font-mono text-[11.5px] text-muted hover:text-secondary"
      >
        collapse
      </button>
    );
  return null;
}

function QuantTable({ card, rows, onAction, best, extra }: {
  card: ModelCard; rows: QuantRow[]; onAction: () => void;
  best?: string; extra?: React.ReactNode;
}) {
  const v = useQuantView(rows);
  const { axes, on, toggle, counts, shown } = v;
  const anySpec = rows.some((q) => q.mtp === true);

  return (
    <>
      {axes.length > 0 && (
        <VariantChips axes={axes} on={on} toggle={toggle} counts={counts} />
      )}
      {/* These widths are the table. Every one is repeated on the row cells
          below, so changing one here without the other silently un-aligns the
          column — the cells are the components CtxCell/RunsCell/QualityCell. */}
      <div className="flex items-center gap-2 px-3 pb-1 font-mono text-[10px]
                      text-muted uppercase tracking-[0.06em]">
        <span className="w-1.5 shrink-0" />
        <span className="w-24 shrink-0">quant</span>
        <span className="w-[52px] shrink-0 text-right">size</span>
        <span className="w-[46px] shrink-0 text-right" title="context window this quant can hold on this machine">
          ctx
        </span>
        <span className="w-[66px] shrink-0" title="where the weights end up: fully on the GPU, or partly in system RAM">
          runs
        </span>
        <span className="w-[78px] shrink-0" title="typical quality given up vs BF16 — reference figure for the format, not measured on this model">
          vs bf16
        </span>
        {anySpec && (
          <span className="w-8 shrink-0" title="carries the MTP draft head, so speculative decoding can run on it">
            spec
          </span>
        )}
      </div>
      <ul className="flex flex-col">
        {shown.map((q) => (
          <QuantLine key={q.file} card={card} q={q} onAction={onAction}
                     best={best} showSpec={anySpec} />
        ))}
        {extra}
      </ul>
      <MoreRows v={v} />
    </>
  );
}

function Card({ card, onAction }: { card: ModelCard; onAction: () => void }) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const anyOnDisk = card.quants.some((q) => q.on_disk);
  const onDiskGb = card.quants.filter((q) => q.on_disk)
    .reduce((n, q) => n + q.bytes, 0)
    + (card.mmproj?.on_disk ? card.mmproj.bytes : 0);

  // Only custom (Hangar-added) models can be removed outright — a registry
  // model would just reappear on the next `rigma update`, so for those the
  // per-file × is the whole story and delete_model refuses with a 409.
  const remove = async () => {
    const files = anyOnDisk
      ? `\n\nThis also deletes ${gb(onDiskGb)} of downloaded files.`
      : "";
    if (!window.confirm(`Remove ${card.slug} from your library?${files}`)) return;
    setBusy(true);
    setErr(null);
    try {
      await engineApi.deleteModel(card.slug);
    } catch (e) {
      setErr((e as Error).message);
    }
    setBusy(false);
    onAction();
  };
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
        {card.custom && !card.running && (
          <button
            disabled={busy}
            onClick={() => void remove()}
            title={anyOnDisk
              ? `remove from library and delete ${gb(onDiskGb)} on disk`
              : "remove from library"}
            aria-label={`remove ${card.slug}`}
            className={`${anyOnDisk ? "" : "ml-auto "}shrink-0 rounded-md px-2 py-0.5 font-mono text-[11.5px] text-muted hover:text-red hover:bg-surface disabled:opacity-40`}
          >
            remove
          </button>
        )}
      </div>
      {err && (
        <div className="rounded-md bg-red/10 text-red px-2.5 py-1.5 font-mono text-[11.5px] mb-2">
          {err}
        </div>
      )}
      <NoTemplateNotice card={card} />
      <div className="font-mono text-[11.5px] text-muted mb-2">
        {card.kind} · {Math.round(card.native_ctx / 1024)}K native
        {card.capabilities.length > 0 && ` · ${card.capabilities.join(" ")}`}
        {card.source && (
          <>
            {" · "}
            <a href={`https://huggingface.co/${card.source}`} target="_blank"
               rel="noreferrer" className="hover:text-amber underline"
               title={"where this came from. The name above is the gguf's own "
                      + "general.name, which is often nothing like the repo."}>
              {card.source}
            </a>
          </>
        )}
      </div>
      <QuantTable
        card={card}
        rows={card.quants}
        onAction={onAction}
        best={card.recommended ?? undefined}
        extra={card.mmproj ? (
          <QuantLine
            key={card.mmproj.file}
            card={card}
            q={{ ...card.mmproj, quant: "mmproj" } as QuantRow}
            onAction={onAction}
          />
        ) : undefined}
      />
    </section>
  );
}

/** The same columns as an installed card, for a repo nothing has been
 *  downloaded from yet. Answering "is this worth 15GB" BEFORE spending the
 *  15GB is the entire point of the ranged header read. */
function RepoPreview({ id, cfg }: { id: string; cfg: FitConfig }) {
  const [d, setD] = useState<HfRepoDetail | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setD(null);
    setErr(null);
    engineApi.hfRepo(id, cfg)
      .then((r) => { if (live) setD(r); })
      .catch((e) => { if (live) setErr((e as Error).message); });
    return () => { live = false; };
  }, [id, cfg]);

  // BEFORE the early returns — hook order has to be identical on every render,
  // and this component bails out early on both error and loading.
  const v = useQuantView(d?.ggufs ?? []);

  if (err) {
    return <div className="px-3 py-2 font-mono text-[11.5px] text-red">{err}</div>;
  }
  if (!d) {
    return (
      <div className="px-3 py-2 font-mono text-[11.5px] text-muted">
        reading gguf headers…
      </div>
    );
  }
  return (
    <div className="rounded-md bg-canvas/60 px-2 py-2 mt-1">
      <div className="font-mono text-[11px] text-muted px-1 pb-1.5">
        {d.kind} · {Math.round(d.native_ctx / 1024)}K native
        {d.capabilities.length > 0 && ` · ${d.capabilities.join(" ")}`}
        {d.has_template === false && (
          <span className="text-amber"
                title={"This repo's ggufs carry no tokenizer.chat_template, so " +
                       "there is nothing to read capabilities from and the " +
                       "engine would prompt it with a generic fallback format."}>
            {" · no chat template"}
          </span>
        )}
        {d.mmproj && " · vision projector"}
        {d.already && <span className="text-amber"> · already in your library</span>}
        {d.split_skipped > 0 &&
          ` · ${d.split_skipped} split file(s) skipped (not supported)`}
      </div>
      {v.axes.length > 0 && (
        <VariantChips axes={v.axes} on={v.on} toggle={v.toggle}
                      counts={v.counts} />
      )}
      <div className="flex items-center gap-2 px-3 pb-1 font-mono text-[10px]
                      text-muted uppercase tracking-[0.06em]">
        <span className="w-24 shrink-0">quant</span>
        <span className="w-[52px] shrink-0 text-right">size</span>
        <span className="w-[46px] shrink-0 text-right">ctx</span>
        <span className="w-[66px] shrink-0">runs</span>
        <span className="w-[78px] shrink-0">vs bf16</span>
      </div>
      <ul className="flex flex-col">
        {v.shown.map((q) => (
          <li key={q.file}
              className="flex items-center gap-2 px-3 py-1 rounded hover:bg-surface/50">
            <span className={`font-mono text-[12px] w-24 shrink-0 truncate ${
              q.fit?.ok === false ? "text-muted" : ""}`} title={q.file}>
              {q.quant}
            </span>
            <span className="font-mono text-[11.5px] text-muted w-[52px] shrink-0 text-right">
              {gb(q.bytes)}
            </span>
            <CtxCell fit={q.fit} />
            <RunsCell fit={q.fit} />
            <QualityCell q={q} />
            <MtpMark q={q} />
            {d.recommended === q.quant && (
              <span className="shrink-0 font-mono text-[10px] text-amber bg-amber/10 rounded px-1.5"
                    title="best quality that still runs at GPU speed here">
                best here
              </span>
            )}
          </li>
        ))}
      </ul>
      <MoreRows v={v} />
    </div>
  );
}

function HfSearch({ onAdded, cfg }: { onAdded: () => void; cfg: FitConfig }) {
  const [q, setQ] = useState("");
  const [hits, setHits] = useState<HfHit[]>([]);
  const [state, setState] = useState<"idle" | "busy" | "err">("idle");
  const [adding, setAdding] = useState<string | null>(null);
  const [addErr, setAddErr] = useState<string | null>(null);
  const [open, setOpen] = useState<string | null>(null);
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
          <li key={h.repo} className="rounded-md hover:bg-surface/60 px-2 py-1.5">
          <div className="flex items-center gap-2">
            <button
              onClick={() => setOpen(open === h.repo ? null : h.repo)}
              aria-expanded={open === h.repo}
              aria-label={`inspect ${h.repo} before adding`}
              title="see every quant's fit and quality before downloading"
              className="shrink-0 w-4 font-mono text-[11px] text-muted hover:text-amber"
            >
              {open === h.repo ? "▾" : "▸"}
            </button>
            <span className="font-mono text-[12.5px] flex-1 truncate" title={h.repo}>{h.repo}</span>
            {typeof h.downloads === "number" && (
              <span className="shrink-0 font-mono text-[10.5px] text-muted"
                    title={`${h.downloads.toLocaleString()} downloads`}>
                {h.downloads > 1e6 ? `${(h.downloads / 1e6).toFixed(1)}M`
                  : `${Math.round(h.downloads / 1e3)}K`}
              </span>
            )}
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
          </div>
          {open === h.repo && <RepoPreview id={h.repo} cfg={cfg} />}
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
            <HfSearch onAdded={refresh} cfg={cfg} />
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
