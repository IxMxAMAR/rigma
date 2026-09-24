// Formatting for a download in flight (IMP-2).
//
// The server already measures everything: `/api/models` returns each pull's
// `done`, a rolling `bps` (bytes/sec, computed in serve from the previous
// poll) and an `eta` in seconds. The Models page rendered only `done / bytes`
// as a percentage, so a stalled pull looked exactly like a slow one and the
// rate the server had already measured was thrown away. These are the pure
// pieces; the component passes in `gb` and `eta` so this file depends on
// nothing.

/** Bytes per second as a human string, in the same binary units as `gb`. */
export function rate(bps: number | null | undefined): string {
  if (bps == null || !Number.isFinite(bps) || bps <= 0) return "";
  const mb = bps / 1024 / 1024;
  if (mb >= 1024) return `${(mb / 1024).toFixed(1)} GB/s`;
  if (mb >= 1) return `${mb.toFixed(1)} MB/s`;
  return `${Math.max(1, Math.round(bps / 1024))} KB/s`;
}

/** 0..100. 0 when the total is unknown — never NaN, which rendered as a
 *  zero-width bar with nothing to explain it. */
export function pct(done: number | null | undefined, total: number): number {
  if (done == null || !Number.isFinite(total) || total <= 0) return 0;
  return Math.min(100, Math.max(0, (done / total) * 100));
}

/** One line: percentage, bytes of total, rate and ETA. Every piece the server
 *  has not measured yet is omitted rather than shown as a zero. */
export function progressLine(
  p: { done?: number | null; bps?: number | null; eta?: number | null },
  total: number,
  fmtBytes: (n: number) => string,
  fmtEta: (s: number) => string,
): string {
  const bits: string[] = [`${pct(p.done, total).toFixed(0)}%`];
  if (p.done != null && total > 0)
    bits.push(`${fmtBytes(p.done)} / ${fmtBytes(total)}`);
  const r = rate(p.bps);
  if (r) bits.push(r);
  if (p.eta != null && Number.isFinite(p.eta)) bits.push(`${fmtEta(p.eta)} left`);
  return bits.join(" · ");
}
