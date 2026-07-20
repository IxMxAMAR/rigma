// The Hangar: installed models as cards, quants with live download progress,
// HF search-and-add. Polls fast only while a download is actually running.
import { useCallback, useEffect, useRef, useState } from "react";
import {
  engineApi, eta, gb, type HfHit, type ModelCard, type QuantRow,
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

function QuantLine({ card, q, onAction }: {
  card: ModelCard; q: QuantRow; onAction: () => void;
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
    <li className="flex items-center gap-3 px-3 py-1.5 rounded-md hover:bg-surface/70">
      <span className={`w-1.5 h-1.5 rounded-full shrink-0 ${q.on_disk ? "bg-moss" : "bg-float"}`} />
      <span className="font-mono text-[12.5px] w-20 shrink-0">{q.quant}</span>
      <span className="font-mono text-[12px] text-muted w-16 shrink-0">{gb(q.bytes)}</span>
      {downloading ? (
        <PullBar q={q} />
      ) : (
        <span className="flex-1" />
      )}
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
      <ul className="flex flex-col">
        {card.quants.map((q) => (
          <QuantLine key={q.file} card={card} q={q} onAction={onAction} />
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
      <ul className="mt-2 flex flex-col gap-1">
        {hits.slice(0, 8).map((h) => (
          <li key={h.id} className="flex items-center gap-2 rounded-md hover:bg-surface px-2 py-1.5">
            <span className="font-mono text-[12.5px] flex-1 truncate">{h.id}</span>
            <button
              disabled={adding === h.id}
              onClick={async () => {
                setAdding(h.id);
                try { await engineApi.hfAdd(h.id); onAdded(); } catch { /* row stays */ }
                setAdding(null);
              }}
              className="shrink-0 rounded-md bg-amber/15 text-amber px-2.5 py-0.5 text-[12px] font-semibold disabled:opacity-40"
            >
              {adding === h.id ? "adding…" : "add"}
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}

export default function ModelsSurface() {
  const [cards, setCards] = useState<ModelCard[]>([]);
  const refresh = useCallback(async () => {
    try {
      setCards((await engineApi.models()).models);
    } catch { /* keep last */ }
  }, []);

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
      <div className="max-w-[860px] mx-auto flex flex-col gap-4">
        <HfSearch onAdded={refresh} />
        {cards.length === 0 && (
          <p className="text-secondary text-[13.5px] text-center pt-12">
            No models yet — search Hugging Face above, or drop a GGUF into
            ~/.rigma/models.
          </p>
        )}
        {cards.map((c) => (
          <Card key={c.slug} card={c} onAction={refresh} />
        ))}
      </div>
    </main>
  );
}
