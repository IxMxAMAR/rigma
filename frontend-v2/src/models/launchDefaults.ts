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
import type { LaunchDefaults } from "../lib/engineApi";

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
  };
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

  return p;
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
