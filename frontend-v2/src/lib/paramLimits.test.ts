import { describe, expect, it } from "vitest";

import {
  clampParam,
  PARAM_RANGE_FALLBACK,
  parseParamRanges,
  rangeFor,
} from "./paramLimits";

// AUDIT F11-2: the sampler panel's limits had drifted from the server's own
// PARAM_RANGES, so the UI offered values the server 400s. These pin the two
// that had drifted and the server-override path that stops it recurring.
describe("server-owned sampler limits", () => {
  it("uses the server's range when one arrives", () => {
    expect(rangeFor("repeat_penalty", { repeat_penalty: [0.5, 2] }))
      .toEqual({ min: 0.5, max: 2 });
    expect(rangeFor("temperature", { temperature: [0, 4] }))
      .toEqual({ min: 0, max: 4 });
  });

  it("clamps a repeat_penalty the server would refuse", () => {
    // the old panel let 0.2 through; the server answered
    // "repeat_penalty: must be between 0.5 and 2.0" and the error was dropped
    expect(clampParam("repeat_penalty", 0.2, undefined)).toBe(0.5);
  });

  it("does not cap temperature below what the server accepts", () => {
    // the old panel capped it at 2.0; the server accepts 4.0
    expect(clampParam("temperature", 2.5, undefined)).toBe(2.5);
    expect(clampParam("temperature", 5, undefined)).toBe(4);
  });

  it("lets the engine context tighten max_tokens, never loosen it", () => {
    expect(rangeFor("max_tokens", undefined, 8192)).toEqual({ min: 1, max: 8192 });
    expect(rangeFor("max_tokens", { max_tokens: [1, 4096] }, 8192))
      .toEqual({ min: 1, max: 4096 });
    expect(clampParam("max_tokens", 99_999, undefined, 8192)).toBe(8192);
  });

  it("keeps the fallback equal to the server's drifted fields", () => {
    expect(PARAM_RANGE_FALLBACK.repeat_penalty).toEqual([0.5, 2]);
    expect(PARAM_RANGE_FALLBACK.temperature).toEqual([0, 4]);
  });

  it("drops a malformed param_ranges payload", () => {
    expect(parseParamRanges({ temperature: [0, 4] }))
      .toEqual({ temperature: [0, 4] });
    expect(parseParamRanges({ temperature: [4, 0] })).toBeUndefined();
    expect(parseParamRanges({ temperature: [0, "4"] })).toBeUndefined();
    expect(parseParamRanges({ temperature: "0..4" })).toBeUndefined();
    expect(parseParamRanges("nope")).toBeUndefined();
    expect(parseParamRanges(null)).toBeUndefined();
  });

  it("passes an unknown key through so the server names it", () => {
    expect(clampParam("seed", 999_999_999_999, undefined)).toBe(999_999_999_999);
    expect(rangeFor("seed", undefined)).toBeNull();
  });
});
