// D2: the decisions behind the launch-defaults dialog, kept apart from the
// markup for the reason the rest of this project states — what a payload must
// contain, and what "cleared" means, is what a test can reach.
//
// Two facts drive every rule here, and both come from serve.py/hangar.py:
//
//   1. `POST /api/models/{slug}/defaults` treats `null` as "clear this field"
//      (hangar.set_launch_defaults maps it to the field's own sentinel). The
//      stored shape uses falsy SENTINELS (`""`, `0`, `null`), so a UI that sent
//      the empty string it read back would be sending a value, not a clear —
//      the field would appear cleared on screen and stay pinned on disk.
//   2. Only the fields the user actually changed may be sent. The route merges
//      what it is given over the stored defaults, so a payload that named every
//      field would silently clear the ones the dialog does not show.
import type {
  Budget,
  Fit,
  LaunchDefaults,
  QuantRow,
} from "../lib/engineApi";

/** Every cache type `server_ops.KV_CACHE_TYPES` accepts — `tuple(CACHE_BYTES)`
 *  in models.py:20-28. The Sidecar's own picker was a stale 4-value subset
 *  (f16/q8_0/q5_1/q4_0), which is one of the things D2 is about. */
export const KV_TYPES = [
  "bf16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0",
];

/** The context steps the engine settings already offer. */
export const CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 262144];

/** The dialog's editable state. Everything is a string because a `<select>`
 *  value is a string; `vision` keeps its own tri-state because "keep whatever
 *  the last launch used" is a real answer, not the same as "off". */
export interface DefaultsDraft {
  quant: string;
  ctx: string;
  kv: string;
  vision: "keep" | "on" | "off";
  backend: string;
  /** C10. Strings because a text input's value is a string; "" is "no opinion"
   *  for the two batch sizes, and for `ngl` — whose sentinel is -1, NOT 0. */
  batch: string;
  ubatch: string;
  ngl: string;
}

/** The stored defaults as the form's state. A missing launch is all-unset. */
export function draftFromLaunch(l: LaunchDefaults | null | undefined): DefaultsDraft {
  return {
    quant: (l?.quant ?? "").trim(),
    // `0` is the sentinel, not a context window, so it reads as unset.
    ctx: l?.ctx ? String(l.ctx) : "",
    kv: (l?.kv ?? "").trim(),
    vision: l?.vision === true ? "on" : l?.vision === false ? "off" : "keep",
    backend: (l?.backend ?? "").trim(),
    batch: l?.batch ? String(l.batch) : "",
    ubatch: l?.ubatch ? String(l.ubatch) : "",
    // -1 is the sentinel and 0 is a REAL request (`-ngl 0` = every layer on the
    // CPU), so this cannot use the falsy test the two sizes use.
    ngl: l?.ngl !== undefined && l?.ngl !== null && l.ngl >= 0
      ? String(l.ngl) : "",
  };
}

/** A size field as the route wants it: `null` when cleared, the integer when
 *  it parses, and `undefined` when it does not — an unparseable entry must be
 *  OMITTED, never sent as a clear, or typing "abc" would delete a default. */
function sizeField(v: string): number | null | undefined {
  const s = v.trim();
  if (s === "") return null;
  if (!/^\d+$/.test(s)) return undefined;
  return Number(s);
}

/** Only what changed, with `null` for a field the user cleared.
 *
 *  Pure and total: an unchanged field is omitted, a cleared field is `null`
 *  (never `""`/`0`), and a value the form cannot parse is omitted rather than
 *  sent as something the server would reject. */
export function defaultsPayload(
  d: DefaultsDraft,
  initial: LaunchDefaults | null | undefined,
): Record<string, unknown> {
  const p: Record<string, unknown> = {};
  const init = initial ?? {};

  const q = d.quant.trim();
  if (q !== (init.quant ?? "").trim()) p.quant = q || null;

  const c = d.ctx.trim();
  const c0 = init.ctx ?? 0;
  if (c === "") {
    if (c0) p.ctx = null;              // cleared → clear, not "0"
  } else {
    const n = Number(c);
    if (Number.isFinite(n) && n > 0 && n !== c0) p.ctx = n;
  }

  const kv = d.kv.trim();
  if (kv !== (init.kv ?? "").trim()) p.kv = kv || null;

  // tri-state: "keep" is the stored `null`, and only a real change is sent.
  const v: boolean | null =
    d.vision === "keep" ? null : d.vision === "on";
  if (v !== (init.vision ?? null)) p.vision = v;

  const b = d.backend.trim();
  if (b !== (init.backend ?? "").trim()) p.backend = b || null;

  // C10: the two batch sizes and the ngl cap. `0` (sizes) and `-1` (ngl) are
  // the stored sentinels, so a cleared field is `null` — the route maps it back
  // to the sentinel — and an unchanged field is omitted entirely. A stored
  // launch that predates C10 has no key at all, which is the same as the
  // sentinel, so a blank input on it must send NOTHING (not a clear).
  for (const f of ["batch", "ubatch"] as const) {
    const n = sizeField(d[f]);
    if (n === undefined) continue;
    const stored = init[f] ?? 0;
    if (n === null ? stored !== 0 : n !== stored) p[f] = n;
  }
  const g = sizeField(d.ngl);
  if (g !== undefined) {
    const stored = init.ngl ?? -1;
    if (g === null ? stored !== -1 : g !== stored) p.ngl = g;
  }

  return p;
}

/** The fit the Models page already computed for the quant this dialog will
 *  default to. `draft.quant` names it when set; otherwise the first on-disk
 *  row — the same one `hangar.list_models` marks `on_disk`. */
export function fitForQuant(
  rows: QuantRow[] | undefined,
  quant: string,
): (Fit & { budget?: Budget }) | undefined {
  if (!rows || rows.length === 0) return undefined;
  const named = quant ? rows.find((r) => r.quant === quant) : undefined;
  return (named ?? rows.find((r) => r.on_disk) ?? rows[0])?.fit;
}

/** One honest sentence for the fit's OWN answer, or "" when there is no verdict
 *  to show (no hardware probe, or the fit threw). `ngl` is the fit's OUTPUT:
 *  this is the number a stored override is priced against, and the ubatch note
 *  in the form is why a larger physical batch is not free. */
export function fitAnswer(
  fit: (Fit & { budget?: Budget }) | undefined,
  nLayers?: number,
): string {
  if (!fit) return "";
  const where = `at ctx ${fit.ctx ?? fit.budget?.ctx ?? "?"} with a `
    + `${fit.kv ?? fit.budget?.kv_type ?? "?"} cache`;
  if (fit.ok === false) {
    // The refused shape carries no `ngl` at all, so this must be answered
    // BEFORE the `ngl === undefined` test below.
    return `the fit cannot place this model on this machine ${where} — an ngl `
      + "override cannot make it fit.";
  }
  if (fit.ngl === undefined) return "";
  // C1: `ngl` is NOT always a count. `ComboFlags.ngl` defaults to 99
  // (models.py:415) and that value is the "all layers" SENTINEL, not a number
  // of layers: the fully-resident dense branch returns `ComboFlags(ctx=…)`
  // with no `ngl` (resolve.py:768), and `resolve.py:825` reads 99 as "all".
  // Rendering it literally made the best case — a model that fits entirely on
  // the GPU — read "99 of 64 layers" (verify-w5f5 C1). The same sentinel is
  // what `engineApi.Fit.ngl` documents as "99 = all" (engineApi.ts:129).
  const all = fit.ngl === 99;
  const placed = all
    ? (nLayers ? `all ${nLayers} layers` : "every layer")
    : `${fit.ngl}${nLayers ? ` of ${nLayers}` : ""} layers`;
  const b = fit.budget;
  const vram = b
    ? (b.over_mb > 0
        ? `; ${b.over_mb} MB OVER the ${b.budget_mb} MB budget`
        : `; ${-b.over_mb} MB of headroom under the ${b.budget_mb} MB budget`)
    : "";
  return `the fit places ${placed} on the GPU ${where}`
    + ` (${fit.offload_pct ?? 0}% of the weights offloaded)${vram}. ngl is the`
    + " fit's output: a lower value is honoured, a higher one is clamped to"
    + " what the fit allows at launch, and the plan says which.";
}

/** True when the draft says something the stored defaults do not. The Save
 *  button is disabled on this rather than on a bare "is anything filled in" —
 *  a form whose every field equals the stored value has nothing to save. */
export function defaultsChanged(
  d: DefaultsDraft,
  initial: LaunchDefaults | null | undefined,
): boolean {
  return Object.keys(defaultsPayload(d, initial)).length > 0;
}
