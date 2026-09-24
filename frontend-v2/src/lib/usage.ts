// Formatting for the lifetime usage odometer (IMP-7).
//
// `~/.rigma/stats.json` has counted total tokens, total replies and a per-model
// token split since the beginning, and serve publishes it at
// `/api/server/stats`; nothing in v2 read it. These are the pure pieces.

/** A token count as a compact string: 999, 12.5K, 1.5M, 2.0B. */
export function tokens(n: number): string {
  if (!Number.isFinite(n) || n <= 0) return "0";
  if (n < 1e3) return String(Math.round(n));
  if (n < 1e4) return `${(n / 1e3).toFixed(1)}K`;
  if (n < 1e6) return `${Math.round(n / 1e3)}K`;
  if (n < 1e7) return `${(n / 1e6).toFixed(1)}M`;
  if (n < 1e9) return `${Math.round(n / 1e6)}M`;
  return `${(n / 1e9).toFixed(1)}B`;
}

export interface ModelUsage {
  model: string;
  tokens: number;
}

/** The per-model split, biggest first, zero rows dropped and the tail capped.
 *  A `by_model` entry with no usable number is skipped rather than rendered as
 *  0 — an unreadable count is not a model that did nothing. */
export function topModels(
  by: Record<string, number> | null | undefined, cap = 5,
): ModelUsage[] {
  return Object.entries(by ?? {})
    .map(([model, t]) => ({ model, tokens: Number(t) || 0 }))
    .filter((r) => r.tokens > 0)
    .sort((a, b) => b.tokens - a.tokens)
    .slice(0, cap);
}
