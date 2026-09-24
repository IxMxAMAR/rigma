// Per-chat composer drafts, mirrored to localStorage.
//
// The map itself lives in chatStore, keyed by session id (AUDIT F11-1). Storing
// it is best-effort: a reload must not lose a half-written paragraph, but a
// draft is user text in a small shared quota, so an oversized draft is kept in
// memory only and the map is trimmed oldest-first when it would not fit
// (IMP-1). Nothing here may throw — a full or unavailable localStorage
// (private mode) must not stop a message from being sent.

export const DRAFT_KEY = "rigma.drafts";
/** A single draft longer than this is not persisted. */
export const DRAFT_MAX_CHARS = 20000;
/** Serialized cap for the whole map, well inside a 5MB quota shared with the
 *  other `rigma.*` keys. */
export const DRAFTS_MAX_SERIALIZED = 65536;

/** The subset worth persisting, oldest entries dropped first while the whole
 *  map would exceed the cap. Never mutates its input. */
export function persistableDrafts(
  drafts: Record<string, string>,
  maxOne = DRAFT_MAX_CHARS,
  maxTotal = DRAFTS_MAX_SERIALIZED,
): Record<string, string> {
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(drafts))
    if (typeof v === "string" && v.length > 0 && v.length <= maxOne) out[k] = v;
  const keys = Object.keys(out);
  for (const k of keys) {
    if (JSON.stringify(out).length <= maxTotal) break;
    delete out[k];
  }
  return out;
}

/** Read stored drafts. A malformed, foreign or non-string value is ignored
 *  rather than thrown: a corrupted key must not stop the app from starting. */
export function parseDrafts(raw: string | null): Record<string, string> {
  if (!raw) return {};
  try {
    const d: unknown = JSON.parse(raw);
    if (!d || typeof d !== "object" || Array.isArray(d)) return {};
    const out: Record<string, string> = {};
    for (const [k, v] of Object.entries(d as Record<string, unknown>))
      if (typeof v === "string" && v) out[k] = v;
    return persistableDrafts(out);
  } catch {
    return {};
  }
}

/** Best-effort write; an empty map removes the key instead of storing "{}". */
export function saveDrafts(drafts: Record<string, string>): void {
  try {
    const keep = persistableDrafts(drafts);
    if (Object.keys(keep).length === 0) localStorage.removeItem(DRAFT_KEY);
    else localStorage.setItem(DRAFT_KEY, JSON.stringify(keep));
  } catch { /* best effort — storage may be full or unavailable */ }
}
