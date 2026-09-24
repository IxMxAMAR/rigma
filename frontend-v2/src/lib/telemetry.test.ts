import { describe, expect, it } from "vitest";

import { tpsHint } from "./telemetry";

describe("the tok/s tooltip", () => {
  it("compares against the calibrated expectation", () => {
    expect(tpsHint(38, 50)).toContain("about 76%");
    expect(tpsHint(38, 50)).toContain("50.0 tok/s");
  });

  it("says so when there is no expectation yet", () => {
    expect(tpsHint(38, null)).toContain("No calibrated expectation");
    expect(tpsHint(38, 0)).toContain("No calibrated expectation");
  });

  it("is empty when there is no rate to describe", () => {
    expect(tpsHint(null, 50)).toBe("");
    expect(tpsHint(undefined, 50)).toBe("");
    expect(tpsHint(NaN, 50)).toBe("");
  });
});
