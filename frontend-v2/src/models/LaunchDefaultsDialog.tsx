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

import {
  engineApi,
  type Budget,
  type Fit,
  type LaunchDefaults,
  type ModelCard,
} from "../lib/engineApi";
import {
  CTX_STEPS,
  KV_TYPES,
  defaultsChanged,
  defaultsPayload,
  draftFromLaunch,
  fitAnswer,
  fitForQuant,
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
  /** The model's real native window, or `null` when NEITHER `/api/models` nor
   *  `/api/server` could say. `null` is not 0 and must not be replaced with a
   *  plausible-looking ceiling: the pre-DR2-4 fallback was `262144`, so a
   *  32K-native model was offered 64K/128K/256K, the value was stored, and the
   *  launcher clamped it silently. An unknown window is "no opinion", not
   *  "every window". */
  nativeCtx: number | null;
  hasMmproj: boolean;
  backends: string[];
  /** C10: the fit the Models page already computed for the selected quant. The
   *  dialog must SHOW it — `ngl` is the fit's output and an override is a cap,
   *  and a larger `ubatch` costs VRAM the fit pays. `undefined` when the
   *  hardware probe failed, in which case the form says nothing rather than
   *  inventing a verdict. */
  fit?: Fit & { budget?: Budget };
  /** `spec.n_layers`, so "59 of 64 layers" can be said out loud. */
  nLayers?: number;
  /** D2: opened because `first_load` was true, rather than from the Sidecar. */
  firstLoad?: boolean;
  busy: boolean;
  saved: boolean;
  error: string | null;
  onDraft: (d: DefaultsDraft) => void;
  onSave: () => void;
  onClose: () => void;
}

/** `null` — or a non-positive number — is an UNKNOWN window, never a real one.
 *  Typed as a predicate so the arithmetic below cannot read it as `0`. */
function ctxKnown(n: number | null): n is number {
  return n !== null && n > 0;
}

export function LaunchDefaultsForm({
  slug, draft, initial, custom, quants, nativeCtx, hasMmproj, backends,
  fit, nLayers, firstLoad, busy, saved, error, onDraft, onSave, onClose,
}: LaunchDefaultsFormProps) {
  const row = "flex items-center gap-2 text-[12.5px]";
  const label = "w-28 shrink-0 text-secondary";
  const sel =
    "flex-1 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none disabled:opacity-40";

  const ctxs = ctxKnown(nativeCtx) ? CTX_STEPS.filter((c) => c <= nativeCtx) : [];
  // A STORED value is kept visible so it can be CLEARED; it is not a new offer.
  // Without this an unknown window would hide the pinned value and the user
  // could not remove it from here.
  if (draft.ctx && !ctxs.includes(Number(draft.ctx))) ctxs.push(Number(draft.ctx));
  const currentCtx = Number(draft.ctx) || 0;
  // The launch path (`server_ops.perform_switch`) does
  // `want = max(2048, min(int(ctx), spec.native_ctx))` — so a stored value above
  // the window is CLAMPED, not honoured. Say it here, where the value was
  // chosen, instead of letting the resolver do it silently.
  const ctxClampedTo = ctxKnown(nativeCtx) && currentCtx > nativeCtx
    ? nativeCtx : null;
  // C6: the same launch path FLOORS the value: `want = max(2048, min(int(ctx),
  // spec.native_ctx))` (server_ops.py:754). A stored value below 2048 is
  // therefore raised at launch as silently as one above the window is lowered,
  // so it gets the same kind of note. The select cannot create one (its steps
  // start at 8K), but the API and a hand-edited spec can store one.
  const CTX_FLOOR = 2048;
  const ctxBelowFloor = currentCtx > 0 && currentCtx < CTX_FLOOR
    ? currentCtx : null;
  const qs = quants.slice();
  if (draft.quant && !qs.includes(draft.quant)) qs.push(draft.quant);
  const changed = defaultsChanged(draft, initial);
  const fitLine = fitAnswer(fit, nLayers);
  // C2: the fit shown is the STORED one (the server priced it before this
  // draft existed), so the note must name the stored ubatch — `draft.ubatch`
  // made it say "stored ubatch (8192)" the moment the user typed, while the fit
  // on screen was still priced at the stored 2048.
  const storedUbatch = Number(initial?.ubatch) || 0;

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
        {!ctxKnown(nativeCtx) && (
          // DR2-4: the pre-fix fallback invented 262144, so a 32K model was
          // offered 64K/128K/256K. An unknown window gets NO steps — only
          // "model default", which is the resolver's own safe choice — and the
          // control says so. A value already in the draft (a stored default, or
          // the Sidecar's running-config seed) stays visible so it can be
          // cleared; it is not a new offer.
          <p role="note" className="text-[11.5px] text-amber pl-28 -mt-1">
            {slug}'s native context window is unknown right now — the models
            list and the engine both failed to answer. Rigma will not offer a
            context it cannot justify, so only "model default" (the resolver's
            own safe choice) is offered here
            {currentCtx
              ? `; the current value ${currentCtx} is kept visible so you can `
                + "clear it"
              : ""}.
          </p>
        )}
        {ctxClampedTo !== null && (
          <p role="note" className="text-[11.5px] text-amber pl-28 -mt-1">
            the current context {currentCtx} is above {slug}'s native window
            {" "}{ctxClampedTo}, so a launch clamps it to {ctxClampedTo} — the
            pinned value is not what runs. Pick {ctxClampedTo} or lower, or
            clear it.
          </p>
        )}
        {ctxBelowFloor !== null && (
          // C6: mirrors the above-window note for the other direction. The
          // launch floors at 2048, so a stored value below it does not run
          // either — and before this it did so with nothing said.
          <p role="note" className="text-[11.5px] text-amber pl-28 -mt-1">
            the current context {currentCtx} is below the launch floor
            {" "}{CTX_FLOOR}, so a launch raises it to {CTX_FLOOR} — the pinned
            value is not what runs. Pick {CTX_FLOOR} or higher, or clear it.
          </p>
        )}

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

        {/* C10: the two batch sizes and the GPU-layer cap. Rigma could already
            EMIT -b/-ub/-ngl; until this route accepted them no caller could ask
            for them. The two sizes are a REQUEST (the fit cannot contradict
            them), `ngl` is a CAP on the fit's own output. */}
        <label className={row} title="The logical batch: how many tokens are processed per prompt step (-b). Blank = the engine's 2048.">
          <span className={label}>batch</span>
          <input className={sel} type="number" min={0} inputMode="numeric"
                 value={draft.batch} disabled={busy}
                 placeholder="engine default (2048)"
                 aria-label="Default batch"
                 onChange={(e) => onDraft({ ...draft, batch: e.target.value })} />
        </label>

        <label className={row} title="The physical batch: the slice llama.cpp sizes its compute buffer for (-ub). Blank = the engine's 512.">
          <span className={label}>ubatch</span>
          <input className={sel} type="number" min={0} inputMode="numeric"
                 value={draft.ubatch} disabled={busy}
                 placeholder="engine default (512)"
                 aria-label="Default ubatch"
                 onChange={(e) => onDraft({ ...draft, ubatch: e.target.value })} />
        </label>
        <p role="note" className="text-[11.5px] text-amber pl-28 -mt-1">
          llama.cpp sizes its compute buffer from the physical batch, so a
          larger ubatch costs VRAM the fit must pay — it is not free. The fit
          below is priced at the stored ubatch
          {storedUbatch ? ` (${storedUbatch})` : " (the engine default 512)"};
          saving a new value re-prices it. A physical batch larger than the
          logical one is refused by the SERVER with a 400 — llama.cpp itself
          only clamps n_ubatch to n_batch (b9867), so Rigma rejects the pair at
          write time rather than letting it load.
        </p>

        <label className={row} title="A CAP on the fit's own placement (-ngl). Blank = the fit decides. 0 = every layer on the CPU.">
          <span className={label}>ngl</span>
          <input className={sel} type="number" min={0} inputMode="numeric"
                 value={draft.ngl} disabled={busy}
                 placeholder="the fit's choice"
                 aria-label="Default ngl"
                 onChange={(e) => onDraft({ ...draft, ngl: e.target.value })} />
        </label>
        {fitLine && (
          // The fit's OWN answer, straight from /api/models. Shown because ngl
          // is its output and an override is only a cap: without this the field
          // reads as a free choice, which is the thing C10 must not present.
          <p role="note" className="text-[11.5px] text-muted pl-28 -mt-1">
            {fitLine}
          </p>
        )}

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
  const [modelCard, setModelCard] = useState<ModelCard | null>(null);
  // `null` = the window is UNKNOWN. It used to default to 262144, which is how
  // a failed model API turned into an invented native window (DR2-4).
  const [nativeCtx, setNativeCtx] = useState<number | null>(null);
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
        // DR3-1: `/api/server`'s `native_ctx`/`has_mmproj` describe the model that
        // is RUNNING (`serve.py:4527` reads the live state's own slug), not the one
        // this dialog is for. Falling back to them unconditionally re-introduced
        // the invented-ceiling bug DR2-4 removed: when `/api/models` fails or the
        // slug is absent from it, the dialog offered another model's context steps
        // and the "unknown window" note never rendered, because the window looked
        // known. Use the server's answer only when the server says it is about
        // THIS model — which is the Sidecar case, where the dialog is opened for
        // the model already loaded.
        const srvHere = srv?.model === slug ? srv : null;
        setInitial(md.launch);
        setCustom(md.custom);
        setModelCard(card);
        setNativeCtx(card?.native_ctx || srvHere?.native_ctx || null);
        setHasMmproj(!!card?.mmproj || srvHere?.has_mmproj === true);
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
    // No empty-payload guard: the button is disabled on `!changed`, and
    // `changed` IS `Object.keys(defaultsPayload(...)).length > 0`, so this is
    // only reachable with something to send.
    const payload = defaultsPayload(draft, initial);
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

  // Derived from the card so the fit follows the SELECTED quant as the user
  // changes it. The card is the only place the fit is computed; the dialog must
  // not re-derive the arithmetic (`resolve.py` owns it).
  const quants = modelCard
    ? modelCard.quants.filter((q) => q.on_disk).map((q) => q.quant) : [];
  const fit = fitForQuant(modelCard?.quants, draft.quant);

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
      fit={fit}
      nLayers={modelCard?.n_layers}
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
