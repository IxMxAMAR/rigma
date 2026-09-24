import { describe, expect, it } from "vitest";

import { sinceLabel, tokens, topModels, usageRows } from "./usage";

describe("usage formatting", () => {
  it("formats token counts compactly", () => {
    expect(tokens(0)).toBe("0");
    expect(tokens(-3)).toBe("0");
    expect(tokens(Number.NaN)).toBe("0");
    expect(tokens(999)).toBe("999");
    expect(tokens(1500)).toBe("1.5K");
    expect(tokens(15000)).toBe("15K");
    expect(tokens(1_500_000)).toBe("1.5M");
    expect(tokens(15_000_000)).toBe("15M");
    expect(tokens(2_000_000_000)).toBe("2.0B");
  });

  it("sorts models by tokens and drops empty ones", () => {
    expect(topModels({ a: 5, b: 100, c: 0 })).toEqual([
      { model: "b", tokens: 100 }, { model: "a", tokens: 5 }]);
  });

  it("survives a missing or malformed split", () => {
    expect(topModels(undefined)).toEqual([]);
    expect(topModels({})).toEqual([]);
    expect(topModels({ a: Number.NaN as unknown as number })).toEqual([]);
  });

  it("caps the list", () => {
    const by: Record<string, number> = {};
    for (let i = 0; i < 10; i++) by[`m${i}`] = i + 1;
    expect(topModels(by, 3)).toHaveLength(3);
  });
});

describe("usage rows across both server shapes", () => {
  it("prefers the enriched models[] when present", () => {
    expect(usageRows({
      by_model: { a: 10 },
      models: [
        { model: "b", tokens: 300, turns: 3, last_used: 100 },
        { model: "a", tokens: 100, turns: 1, last_used: 50 },
      ],
    })).toEqual([
      { model: "b", tokens: 300, turns: 3, last_used: 100 },
      { model: "a", tokens: 100, turns: 1, last_used: 50 },
    ]);
  });

  it("falls back to the flat map on an older server", () => {
    expect(usageRows({ by_model: { a: 10, b: 30 } })).toEqual([
      { model: "b", tokens: 30, turns: 0, last_used: null },
      { model: "a", tokens: 10, turns: 0, last_used: null },
    ]);
  });

  it("joins turns and last-used onto the flat map when only those are sent", () => {
    expect(usageRows({
      by_model: { a: 10 },
      by_model_turns: { a: 4 },
      last_used: { a: 1000 },
    })).toEqual([{ model: "a", tokens: 10, turns: 4, last_used: 1000 }]);
  });

  it("survives a null stats body", () => {
    expect(usageRows(null)).toEqual([]);
    expect(usageRows(undefined)).toEqual([]);
  });

  it("labels a last-used stamp, or says nothing", () => {
    // stamps are epoch seconds; a tiny `now` would push "2h ago" below zero,
    // which sinceLabel rightly reads as "never stamped"
    const now = 1_790_000_000;
    expect(sinceLabel(null, now)).toBe("");
    expect(sinceLabel(0, now)).toBe("");
    expect(sinceLabel(now - 30, now)).toBe("just now");
    expect(sinceLabel(now - 120, now)).toBe("2m ago");
    expect(sinceLabel(now - 7200, now)).toBe("2h ago");
    expect(sinceLabel(now - 3 * 86400, now)).toBe("3d ago");
  });
});
