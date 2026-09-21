/** How a tool call reads in the transcript.
 *
 * Pure, and separate from the component, because this is the part with rules
 * worth testing: what identifies a call, what may be shown, and what is noise.
 */

// identity, never payload — same rule the legacy chips learned the hard way.
// The order IS the order of identity: the first key present is the thing that
// tells you WHICH call this is. It carries an external agent's vocabulary too,
// which is not Rigma's — mcode's `bash` says `command`, not `cmd`, so a chip
// reading "bash" and nothing else was the one call a reader most needs to see.
export const IDENTITY_KEYS = [
  "command", "cmd", "path", "paths", "pattern", "url", "query", "question",
  "description", "objective", "name", "todos", "task", "task_id", "id",
  "action",
];

/** The one line a closed chip shows after the tool's name. */
export function previewArgs(args: unknown): string {
  if (!args || typeof args !== "object") return "";
  const a = args as Record<string, unknown>;
  for (const k of IDENTITY_KEYS) {
    const raw = a[k];
    if (raw === undefined || raw === null || raw === "") continue;
    let v: string;
    if (Array.isArray(raw)) {
      // a list of paths is a list worth reading; a list of OBJECTS is a count,
      // because `[object Object]` tells the reader nothing at all
      v = raw.some((x) => x && typeof x === "object")
        ? `${raw.length} items`
        : raw.join(", ");
    } else if (typeof raw === "object") {
      v = `${Object.keys(raw).length} fields`;
    } else {
      v = String(raw);
    }
    const size = typeof a.content === "string" ? ` · ${a.content.length} chars` : "";
    return v.slice(0, 56) + (v.length > 56 ? "…" : "") + size;
  }
  return Object.keys(a).join(", ").slice(0, 56);
}

/** What a chip shows when it is opened.
 *
 * The summary line is truncated on purpose — a chip is one line. But for
 * `bash` the command IS the call, and a reader deciding whether to trust it
 * needs all of it, so the identifying argument gets its own untruncated line
 * and the rest follows as JSON.
 */
export function formatArgs(args: unknown): string {
  if (args === undefined || args === null) return "";
  if (typeof args === "string") return args;
  if (typeof args !== "object") return String(args);
  const a = args as Record<string, unknown>;
  const key = IDENTITY_KEYS.find(
    (k) => a[k] !== undefined && a[k] !== null && a[k] !== "");
  if (!key) {
    try { return JSON.stringify(a, null, 2); } catch { return String(a); }
  }
  const rest: Record<string, unknown> = { ...a };
  delete rest[key];
  const tail = Object.keys(rest).length ? JSON.stringify(rest, null, 2) : "";
  return [String(a[key]), tail].filter(Boolean).join("\n");
}
