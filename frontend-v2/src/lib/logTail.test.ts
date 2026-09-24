import { describe, expect, it } from "vitest";

import { filterLines } from "./logTail";

describe("engine log filter", () => {
  const text = "load_model: ok\nwarn: cache_reuse refused\nload_model: done\n";

  it("returns every line for an empty filter", () => {
    expect(filterLines(text, "")).toEqual([
      "load_model: ok", "warn: cache_reuse refused", "load_model: done"]);
    expect(filterLines(text, "   ")).toHaveLength(3);
  });

  it("matches case-insensitively", () => {
    expect(filterLines(text, "WARN")).toEqual(["warn: cache_reuse refused"]);
    expect(filterLines(text, "load_model")).toHaveLength(2);
  });

  it("returns nothing when no line matches", () => {
    expect(filterLines(text, "nope")).toEqual([]);
  });

  it("handles empty text without inventing a line", () => {
    expect(filterLines("", "")).toEqual([]);
    expect(filterLines("", "x")).toEqual([]);
  });
});
