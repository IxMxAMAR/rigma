// D2: the first-load configuration dialog.
//
// The route pair it drives is `GET`/`POST /api/models/{slug}/defaults`. Before
// D2 the only route was the POST, so a dialog could WRITE a default and had no
// way to show the one already stored; and the POST's allowlist had no `backend`,
// so a per-model compute backend was unsettable over HTTP. Both are fixed on the
// server side; this is the UI.
//
// `LaunchDefaultsForm` is pure props — no store, no fetch, no effects — so the
// two sentences that matter (the registry-model refusal, and the server's own
// error) are assertable with `renderToStaticMarkup`. The default export is the
// stateful container that loads the model's facts and posts the diff.
//
// TWO HARD RULES, both enforced in `launchDefaults.ts` and asserted here:
//   * a registry model's save must show the sentence the SERVER returns, not a
//     local guess — so the button is live and the 400's `{error}` is rendered;
//   * a cleared field is sent as `null` (which the route reads as "clear"),
//     never as the empty string the stored sentinel would round-trip.
import { useEffect, useRef, useState } from "react";

import { engineApi, type LaunchDefaults } from "../lib/engineApi";
import {
  CTX_STEPS,
  KV_TYPES,
  defaultsChanged,
  defaultsPayload,
  draftFromLaunch,
  type DefaultsDraft,
} from "./launchDefaults";

/** Prefill for fields the stored defaults leave unset. The Sidecar's
 *  "remember this configuration" button passes the RUNNING settings here, so
 *  the dialog opens on what is actually serving rather than on nothing. */
export type DefaultsSeed = Partial<DefaultsDraft>;

/** A seed only fills what the stored launch leaves unset — a pinned default is
 *  never overwritten by the running config. */
export function seedDraft(
  d: DefaultsDraft,
  initial: LaunchDefaults | null | undefined,
  seed: DefaultsSeed | undefined,
): DefaultsDraft {
  if (!seed) return d;
  const out = { ...d };
  if (seed.quant !== undefined && !(initial?.quant ?? "").trim()) out.quant = seed.quant;
  if (seed.ctx !== undefined && !(initial?.ctx ?? 0)) out.ctx = seed.ctx;
  if (seed.kv !== undefined && !(initial?.kv ?? "").trim()) out.kv = seed.kv;
  if (seed.vision !== undefined && (initial?.vision ?? null) === null) out.vision = seed.vision;
  if (seed.backend !== undefined && !(initial?.backend ?? "").trim()) out.backend = seed.backend;
  return out;
}

export interface LaunchDefaultsFormProps {
  slug: string;
  draft: DefaultsDraft;
  initial: LaunchDefaults | null;
  /** `false` = a registry model, whose launch settings are hand-authored. */
  custom: boolean;
  /** On-disk quants only — a default that cannot be loaded is not a default. */
  quants: string[];
  nativeCtx: number;
  hasMmproj: boolean;
  backends: string[];
  /** D2: opened because `first_load` was true, rather than from the Sidecar. */
  firstLoad?: boolean;
  busy: boolean;
  saved: boolean;
  error: string | null;
  onDraft: (d: DefaultsDraft) => void;
  onSave: () => void;
  onClose: () => void;
}

export function LaunchDefaultsForm({
  slug, draft, initial, custom, quants, nativeCtx, hasMmproj, backends,
  firstLoad, busy, saved, error, onDraft, onSave, onClose,
}: LaunchDefaultsFormProps) {
  const row = "flex items-center gap-2 text-[12.5px]";
  const label = "w-28 shrink-0 text-secondary";
  const sel =
    "flex-1 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none disabled:opacity-40";

  const ctxs = CTX_STEPS.filter((c) => c <= nativeCtx);
  if (draft.ctx && !ctxs.includes(Number(draft.ctx))) ctxs.push(Number(draft.ctx));
  const qs = quants.slice();
  if (draft.quant && !qs.includes(draft.quant)) qs.push(draft.quant);
  const changed = defaultsChanged(draft, initial);

  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center"
         role="dialog" aria-modal="true"
         aria-label={`Launch defaults for ${slug}`}>
      <div className="w-[520px] max-w-[92vw] rounded-lg bg-float shadow-2xl p-4 flex flex-col gap-2">
        <h2 className="text-[14px] font-semibold">
          {firstLoad ? `First load of ${slug}` : `How ${slug} comes up`}
        </h2>
        <p className="text-[12px] text-secondary -mt-1">
          {firstLoad
            ? "Nothing is pinned for this model and no turn has ever finished on "
              + "it. Pick how it should come up — or skip and let the resolver "
              + "decide."
            : "These are stored on the model and used whenever it is loaded "
              + "without an explicit setting. Every field can be left at the "
              + "model's own default."}
        </p>

        <label className={row} title="Which gguf to prefer, by the label the Models page shows.">
          <span className={label}>quant</span>
          <select className={sel} value={draft.quant} disabled={busy}
                  aria-label="Default quant"
                  onChange={(e) => onDraft({ ...draft, quant: e.target.value })}>
            <option value="">model default</option>
            {qs.map((q) => <option key={q} value={q}>{q}</option>)}
          </select>
        </label>

        <label className={row} title="The context window to come up at.">
          <span className={label}>context</span>
          <select className={sel} value={draft.ctx} disabled={busy}
                  aria-label="Default context"
                  onChange={(e) => onDraft({ ...draft, ctx: e.target.value })}>
            <option value="">model default</option>
            {ctxs.map((c) => (
              <option key={c} value={c}>{Math.round(c / 1024)}K</option>
            ))}
          </select>
        </label>

        <label className={row} title="KV cache precision. Halving it roughly doubles the
context that fits, but cache error is written per token and compounds.">
          <span className={label}>kv cache</span>
          <select className={sel} value={draft.kv} disabled={busy}
                  aria-label="Default KV cache"
                  onChange={(e) => onDraft({ ...draft, kv: e.target.value })}>
            <option value="">model default</option>
            {KV_TYPES.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>

        {hasMmproj && (
          <label className={row} title="Tri-state on purpose: 'last launch' keeps
whatever the engine used, which is not the same as 'off'.">
            <span className={label}>vision</span>
            <select className={sel} value={draft.vision} disabled={busy}
                    aria-label="Default vision"
                    onChange={(e) => onDraft({
                      ...draft, vision: e.target.value as DefaultsDraft["vision"],
                    })}>
              <option value="keep">last launch</option>
              <option value="on">on — load the projector</option>
              <option value="off">off — text only, frees VRAM</option>
            </select>
          </label>
        )}

        {backends.length > 1 && (
          <label className={row} title="Which compute backend to launch on. A model
whose quantised weights need one fork's kernels only runs on that backend.">
            <span className={label}>backend</span>
            <select className={sel} value={draft.backend} disabled={busy}
                    aria-label="Default backend"
                    onChange={(e) => onDraft({ ...draft, backend: e.target.value })}>
              <option value="">model default</option>
              {backends.map((b) => <option key={b} value={b}>{b}</option>)}
            </select>
          </label>
        )}

        {!custom && (
          // OD-11: `hangar.set_launch_defaults` refuses a `custom: false` spec,
          // because a registry model's launch settings are hand-authored. The
          // sentence below is Rigma's own explanation; the SERVER's sentence
          // arrives separately in `error` and is never rewritten.
          <p role="note" className="text-[12px] text-amber">
            {slug} is a registry model. Its launch settings are hand-authored,
            so saving here will be refused.
          </p>
        )}

        {error !== null && (
          <div role="alert"
               className="rounded-md bg-red/10 text-red px-2.5 py-1.5 font-mono text-[11.5px] break-words">
            {error}
          </div>
        )}

        <div className="flex items-center gap-2 pt-1">
          <button
            type="button"
            disabled={busy || !changed}
            onClick={onSave}
            className="rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold disabled:opacity-40"
          >
            {busy ? "saving…" : saved ? "saved" : "save as default"}
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={onClose}
            className="rounded-md bg-surface hover:bg-float px-3 py-1 text-[13px]"
          >
            not now
          </button>
          {!changed && !busy && (
            <span className="font-mono text-[11px] text-muted">
              nothing changed yet
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

export default function LaunchDefaultsDialog({
  slug, open, seed, firstLoad, onClose, onSaved,
}: {
  slug: string;
  open: boolean;
  seed?: DefaultsSeed;
  firstLoad?: boolean;
  onClose: () => void;
  onSaved?: () => void;
}) {
  const [draft, setDraft] = useState<DefaultsDraft | null>(null);
  const [initial, setInitial] = useState<LaunchDefaults | null>(null);
  const [custom, setCustom] = useState(true);
  const [quants, setQuants] = useState<string[]>([]);
  const [nativeCtx, setNativeCtx] = useState(262144);
  const [hasMmproj, setHasMmproj] = useState(false);
  const [backends, setBackends] = useState<string[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // The seed is an inline object at the call site, so it must not be a
  // dependency — a new identity every render would re-run the load forever.
  const seedRef = useRef(seed);
  seedRef.current = seed;

  useEffect(() => {
    if (!open) return;
    let live = true;
    setLoading(true);
    setLoadError(null);
    setError(null);
    setSaved(false);
    void (async () => {
      try {
        const md = await engineApi.modelDefaults(slug);
        const [models, srv] = await Promise.all([
          engineApi.models().catch(() => null),
          engineApi.server().catch(() => null),
        ]);
        if (!live) return;
        const card = models?.models.find((m) => m.slug === slug) ?? null;
        setInitial(md.launch);
        setCustom(md.custom);
        setQuants(card ? card.quants.filter((q) => q.on_disk).map((q) => q.quant)
                       : []);
        setNativeCtx(card?.native_ctx || srv?.native_ctx || 262144);
        setHasMmproj(!!card?.mmproj || srv?.has_mmproj === true);
        setBackends((srv?.backends ?? []).filter((b) => b.buildable)
          .map((b) => b.name));
        setDraft(seedDraft(draftFromLaunch(md.launch), md.launch, seedRef.current));
        setLoading(false);
      } catch (e) {
        if (!live) return;
        setLoadError((e as Error).message);
        setLoading(false);
      }
    })();
    return () => { live = false; };
  }, [open, slug]);

  if (!open) return null;

  const save = async () => {
    if (!draft) return;
    const payload = defaultsPayload(draft, initial);
    if (Object.keys(payload).length === 0) {
      setSaved(true);
      onSaved?.();
      return;
    }
    setBusy(true);
    setError(null);
    try {
      await engineApi.setDefaults(slug, payload);
      setSaved(true);
      onSaved?.();
    } catch (e) {
      // The server's own sentence, verbatim — the registry refusal and the
      // kv-validation message both arrive here and neither is rewritten.
      setError((e as Error).message);
    }
    setBusy(false);
  };

  if (loading || loadError !== null || !draft) {
    return (
      <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center"
           role="dialog" aria-modal="true"
           aria-label={`Launch defaults for ${slug}`}>
        <div className="w-[420px] max-w-[92vw] rounded-lg bg-float shadow-2xl p-4">
          {loadError !== null ? (
            <div role="alert" className="text-red font-mono text-[12px] break-words">
              could not read this model's defaults: {loadError}
            </div>
          ) : (
            <p className="font-mono text-[12px] text-muted">loading defaults…</p>
          )}
          <button type="button" onClick={onClose}
                  className="mt-3 rounded-md bg-surface hover:bg-float px-3 py-1 text-[13px]">
            close
          </button>
        </div>
      </div>
    );
  }

  return (
    <LaunchDefaultsForm
      slug={slug}
      draft={draft}
      initial={initial}
      custom={custom}
      quants={quants}
      nativeCtx={nativeCtx}
      hasMmproj={hasMmproj}
      backends={backends}
      firstLoad={firstLoad}
      busy={busy}
      saved={saved}
      error={error}
      onDraft={(d) => { setDraft(d); setError(null); setSaved(false); }}
      onSave={() => void save()}
      onClose={onClose}
    />
  );
}
