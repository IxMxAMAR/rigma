import { describe, expect, it } from "vitest";

import { usageLine } from "./AgentState";

// The usage panel used to be a generic `key value` join filtered to numbers. It
// therefore rendered `durationMs 4200` with no unit, and dropped `model` and
// `usageIncomplete` entirely — so two real facts a backend reported never
// appeared, with no error anywhere. These tests pin the shapes of BOTH backends:
//
//   DSH    inputTokens + outputTokens required; totalTokens, cacheReadTokens,
//          cacheWriteTokens, reasoningTokens optional
//          (session-format-v0-to-v1/src/payload-validation.ts:415-421)
//   mcode  the same token keys, plus durationMs, model and usageIncomplete,
//          spread in as siblings of `usage` by its projector

describe("the usage line names what the backend actually reported", () => {
  it("labels the required DSH pair", () => {
    expect(usageLine({ inputTokens: 10, outputTokens: 4 })).toBe("in 10 · out 4");
  });

  it("includes the optional token keys when present", () => {
    const line = usageLine({
      inputTokens: 10,
      outputTokens: 4,
      totalTokens: 14,
      cacheReadTokens: 8,
      cacheWriteTokens: 2,
      reasoningTokens: 3,
    });
    expect(line).toBe(
      "in 10 · out 4 · total 14 · cache read 8 · cache write 2 · reasoning 3",
    );
  });

  it("omits optional keys that are absent rather than printing them empty", () => {
    const line = usageLine({ inputTokens: 1, outputTokens: 2, totalTokens: 3 });
    expect(line).not.toContain("cache");
    expect(line).not.toContain("reasoning");
  });

  it("renders a duration in SECONDS, with a unit", () => {
    // The old renderer printed `durationMs 4200`: a bare number, no unit.
    expect(usageLine({ durationMs: 4200 })).toBe("4.2s");
    expect(usageLine({ durationMs: 359 })).toBe("0.4s");
    expect(usageLine({ durationMs: 0 })).toBe("0.0s");
  });

  it("unwraps mcode's model object instead of dropping it", () => {
    // mcode sends {modelId: "..."} — the projector's own fixture agrees.
    expect(usageLine({ model: { modelId: "MiniMax-M2" } })).toBe("MiniMax-M2");
  });

  it("accepts a bare string model too", () => {
    expect(usageLine({ model: "MiniMax-M2" })).toBe("MiniMax-M2");
  });

  it("tries the other model field names before giving up", () => {
    expect(usageLine({ model: { id: "a" } })).toBe("a");
    expect(usageLine({ model: { name: "b" } })).toBe("b");
    expect(usageLine({ model: { somethingElse: "c" } })).toBe("");
  });

  it("combines tokens, duration and model in a stable order", () => {
    const line = usageLine({
      inputTokens: 10,
      outputTokens: 4,
      durationMs: 4200,
      model: { modelId: "MiniMax-M2" },
    });
    expect(line).toBe("in 10 · out 4 · 4.2s · MiniMax-M2");
  });

  it("does NOT print usageIncomplete here — the caller explains it in a sentence", () => {
    const line = usageLine({ inputTokens: 1, usageIncomplete: true });
    expect(line).toBe("in 1");
    expect(line).not.toContain("usageIncomplete");
  });

  it("still shows an unrecognised numeric key rather than dropping it", () => {
    // A future backend key must not vanish the way `model` did.
    expect(usageLine({ inputTokens: 1, futureCounter: 7 })).toBe(
      "in 1 · futureCounter 7",
    );
  });

  it("ignores non-numeric junk instead of printing it", () => {
    expect(usageLine({ inputTokens: 1, note: "hello", flag: true })).toBe("in 1");
  });

  it("returns an empty line for an empty payload", () => {
    expect(usageLine({})).toBe("");
  });

  it("does not treat a zero count as absent", () => {
    // 0 is a real number and a real answer; only `typeof` decides.
    expect(usageLine({ inputTokens: 0, outputTokens: 0 })).toBe("in 0 · out 0");
  });
});
