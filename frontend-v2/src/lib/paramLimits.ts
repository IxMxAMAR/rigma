// Sampler bounds, owned by the SERVER.
//
// `sessions.PARAM_RANGES` is the one definition the 400 comes from, and
// /api/server publishes it as `param_ranges`. The panel used to re-declare its
// own table and drifted: it let `repeat_penalty` 0.2 through (server floor 0.5)
// and capped `temperature` at 2 where the server accepts 4. Everything here
// reads the server's copy; the fallback exists only for the window before the
// first /api/server reply lands (and for a server too old to send the field).
export type ParamRange = readonly [number, number];
export type ParamRanges = Record<string, ParamRange>;

/** Must stay equal to sessions.PARAM_RANGES for the fields the panel renders.
 *  paramLimits.test.ts pins the two that had drifted. */
export const PARAM_RANGE_FALLBACK: ParamRanges = {
  temperature: [0, 4],
  top_p: [0, 1],
  min_p: [0, 1],
  repeat_penalty: [0.5, 2],
  max_tokens: [1, 262144],
  dry_multiplier: [0, 2],
  dry_base: [1, 4],
};

/** `param_ranges` from /api/server, keeping only well-formed pairs — a
 *  malformed payload must not widen a bound past what the server enforces. */
export function parseParamRanges(raw: unknown): ParamRanges | undefined {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return undefined;
  const out: ParamRanges = {};
  for (const [k, v] of Object.entries(raw as Record<string, unknown>)) {
    if (!Array.isArray(v) || v.length !== 2) continue;
    const [lo, hi] = v as unknown[];
    if (typeof lo === "number" && typeof hi === "number" &&
        Number.isFinite(lo) && Number.isFinite(hi) && lo <= hi)
      out[k] = [lo, hi];
  }
  return Object.keys(out).length > 0 ? out : undefined;
}

/** The bounds for one field. `cap` (the engine's context) may only TIGHTEN
 *  max_tokens, never loosen it past the server's ceiling. */
export function rangeFor(key: string, ranges: ParamRanges | undefined,
                         cap?: number): { min: number; max: number } | null {
  const r = ranges?.[key] ?? PARAM_RANGE_FALLBACK[key];
  if (!r) return null;
  const [lo, hi] = r;
  if (key === "max_tokens" && cap != null && Number.isFinite(cap))
    return { min: lo, max: Math.max(lo, Math.min(hi, cap)) };
  return { min: lo, max: hi };
}

/** Clamp a value into its server range. An unknown key, or a non-finite
 *  value, is returned untouched so the server's own 400 names it. */
export function clampParam(key: string, value: number,
                           ranges: ParamRanges | undefined,
                           cap?: number): number {
  const r = rangeFor(key, ranges, cap);
  if (!r || !Number.isFinite(value)) return value;
  return Math.min(r.max, Math.max(r.min, value));
}
