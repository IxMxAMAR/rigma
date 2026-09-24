import { describe, expect, it } from "vitest";

import { tokens, topModels } from "./usage";

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
