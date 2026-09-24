// Filtering a log tail for the engine-log panel (IMP-11).
//
// `/api/server/log?lines=N` already bounds the tail server-side; the filter is
// the one thing the server does not do, and it is pure.

/** Keep only lines containing `q`, case-insensitively. An empty filter keeps
 *  every line. A trailing newline does not produce a phantom empty line. */
export function filterLines(text: string, q: string): string[] {
  const all = text.split("\n");
  if (all.length > 0 && all[all.length - 1] === "") all.pop();
  const needle = q.trim().toLowerCase();
  if (!needle) return all;
  return all.filter((l) => l.toLowerCase().includes(needle));
}
