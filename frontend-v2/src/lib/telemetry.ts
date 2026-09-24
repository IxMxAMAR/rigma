// The sentence behind the header's tok/s.
//
// "Is 38 tok/s good here?" is the question the number raises, and the Engine
// page already answers it (expected_tg from the calibration). The always-visible
// header should carry that comparison rather than leave a bare number.
export function tpsHint(
  tps: number | null | undefined,
  expected: number | null | undefined,
): string {
  if (tps == null || !Number.isFinite(tps)) return "";
  const base = `${tps.toFixed(1)} tok/s generated`;
  if (expected == null || !Number.isFinite(expected) || expected <= 0)
    return `${base}. No calibrated expectation for this model on this machine yet.`;
  const pct = Math.round((tps / expected) * 100);
  return `${base} — about ${pct}% of the ${expected.toFixed(1)} tok/s this `
    + `model is expected to reach on this machine.`;
}
