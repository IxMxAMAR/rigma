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

export interface ModelUsageRow extends ModelUsage {
  /** Turns attributed to this model, when the server counts them. */
  turns: number;
  /** Epoch seconds of the last turn, when the server stamps it. */
  last_used: number | null;
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

/** `/api/server/stats` has two shapes in the wild: the original flat `by_model`
 *  map, and the IMP-7 enrichment (`models[]`, pre-sorted, each with a turn
 *  count and a last-used stamp). Prefer the enrichment when it is there and
 *  fall back to the flat map, so the panel works against either server. */
export function usageRows(
  s: {
    by_model?: Record<string, number> | null;
    by_model_turns?: Record<string, number> | null;
    last_used?: Record<string, number> | null;
    models?: { model?: unknown; tokens?: unknown; turns?: unknown;
               last_used?: unknown }[] | null;
  } | null | undefined,
  cap = 5,
): ModelUsageRow[] {
  if (!s) return [];
  if (Array.isArray(s.models) && s.models.length > 0) {
    return s.models
      .map((m) => ({
        model: String(m.model ?? ""),
        tokens: Number(m.tokens) || 0,
        turns: Number(m.turns) || 0,
        last_used: typeof m.last_used === "number" ? m.last_used : null,
      }))
      .filter((m) => m.tokens > 0)
      .sort((a, b) => b.tokens - a.tokens)
      .slice(0, cap);
  }
  const turns = s.by_model_turns ?? {};
  const last = s.last_used ?? {};
  return topModels(s.by_model, cap).map((r) => ({
    ...r,
    turns: Number(turns[r.model]) || 0,
    last_used: typeof last[r.model] === "number" ? last[r.model] : null,
  }));
}

/** "2h ago" for a last-used stamp, or "" when the server has not stamped one. */
export function sinceLabel(ts: number | null | undefined, now: number): string {
  if (ts == null || !Number.isFinite(ts) || ts <= 0) return "";
  const s = Math.max(0, Math.floor(now - ts));
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}
