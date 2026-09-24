import { describe, expect, it } from "vitest";

import { MEMORY_STATUSES, memoryPatch } from "./edit";

const row = { text: "always quote long filenames", status: "draft" };

describe("memory edit patch", () => {
  it("sends nothing when nothing changed", () => {
    expect(memoryPatch(row, row.text, row.status)).toEqual({});
    expect(memoryPatch(row, `  ${row.text}  `, row.status)).toEqual({});
  });

  it("sends only the changed text, trimmed", () => {
    expect(memoryPatch(row, "  new rule  ", "draft"))
      .toEqual({ text: "new rule" });
  });

  it("sends only the changed status", () => {
    expect(memoryPatch(row, row.text, "retired"))
      .toEqual({ status: "retired" });
  });

  it("never erases the rule with a blank box", () => {
    expect(memoryPatch(row, "   ", "draft")).toEqual({});
  });

  it("offers exactly the statuses the store validates", () => {
    expect([...MEMORY_STATUSES]).toEqual(["draft", "verified", "retired"]);
  });
});
