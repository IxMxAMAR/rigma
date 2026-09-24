import { describe, expect, it } from "vitest";

import { pct, progressLine, rate } from "./download";

const b = (n: number) => `${n} B`;

describe("download formatting", () => {
  it("never yields NaN for an unknown total", () => {
    expect(pct(null, 0)).toBe(0);
    expect(pct(10, 0)).toBe(0);
    expect(pct(undefined, 100)).toBe(0);
  });

  it("clamps to 0..100", () => {
    expect(pct(50, 100)).toBe(50);
    expect(pct(200, 100)).toBe(100);
    expect(pct(-5, 100)).toBe(0);
  });

  it("omits a rate it cannot measure", () => {
    expect(rate(null)).toBe("");
    expect(rate(0)).toBe("");
    expect(rate(Number.NaN)).toBe("");
    expect(rate(1536)).toBe("2 KB/s");
    expect(rate(12.4 * 1024 * 1024)).toBe("12.4 MB/s");
  });

  it("builds the full line when the server measured everything", () => {
    expect(progressLine(
      { done: 50, bps: 1024 * 1024, eta: 30 }, 100, b, (s) => `${s}s`))
      .toBe("50% · 50 B / 100 B · 1.0 MB/s · 30s left");
  });

  it("omits the ETA and rate until the second poll", () => {
    expect(progressLine({ done: 25 }, 100, b, (s) => `${s}s`))
      .toBe("25% · 25 B / 100 B");
  });

  it("still reports a percentage when the size is unknown", () => {
    expect(progressLine({ done: 25, eta: 5 }, 0, b, (s) => `${s}s`))
      .toBe("0% · 5s left");
  });
});
