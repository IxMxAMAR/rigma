import { describe, expect, it } from "vitest";

import { argHint, delegateSentence, summariseDelegate } from "./delegate";

const call = (name: string, extra: Record<string, unknown> = {}) => ({ name, ...extra });

describe("summariseDelegate", () => {
  it("counts what ran, what failed and what was refused", () => {
    const s = summariseDelegate([
      call("read_file", { ok: true }),
      call("grep", { ok: true }),
      call("run_shell", { ok: false }),
      call("write_file", { ok: false, blocked: true }),
    ]);
    expect(s).toMatchObject({ ran: 3, failed: 1, blocked: 1 });
  });

  // A blocked call never ran, so counting it as a run would overstate the work
  // and understate the refusal.
  it("does not count a refused tool as a run", () => {
    const s = summariseDelegate([call("write_file", { blocked: true })]);
    expect(s.ran).toBe(0);
    expect(s.blocked).toBe(1);
  });

  // The helper's roster is read-only by design, so a refusal is normal and
  // expected — it must still be reported, because it explains a thin answer.
  it("reports a refusal in the sentence", () => {
    const s = summariseDelegate([call("write_file", { blocked: true })]);
    expect(delegateSentence(s)).toContain("refused");
  });

  it("lists distinct tools in first-use order, without repeats", () => {
    const s = summariseDelegate([
      call("grep", { ok: true }),
      call("read_file", { ok: true }),
      call("grep", { ok: true }),
    ]);
    expect(s.tools).toEqual(["grep", "read_file"]);
  });

  it("handles an empty trace without claiming work happened", () => {
    const s = summariseDelegate([]);
    expect(s).toMatchObject({ ran: 0, blocked: 0, failed: 0, tools: [] });
    expect(delegateSentence(s)).toContain("used no tools");
  });

  it("never mutates the rows it was given", () => {
    const rows = [call("grep", { ok: true })];
    const copy = JSON.parse(JSON.stringify(rows));
    summariseDelegate(rows);
    expect(rows).toEqual(copy);
  });

  it("survives a malformed row rather than throwing", () => {
    const s = summariseDelegate([{} as never, { name: "" } as never]);
    expect(s.ran).toBe(2);
    expect(s.tools).toEqual([]);
  });
});

describe("delegateSentence", () => {
  it("singularises one call", () => {
    expect(delegateSentence(summariseDelegate([call("grep", { ok: true })])))
      .toContain("1 tool call —");
  });

  it("summarises a long tool list instead of listing it all", () => {
    const rows = ["a", "b", "c", "d", "e", "f", "g", "h"].map((n) => call(n, { ok: true }));
    expect(delegateSentence(summariseDelegate(rows))).toContain("+2 more");
  });
});

describe("argHint", () => {
  it("renders the first few arguments on one line", () => {
    expect(argHint({ path: "a.py", limit: 10 })).toBe("path=a.py limit=10");
  });

  it("clips a long value rather than wrapping the row", () => {
    const hint = argHint({ q: "x".repeat(200) });
    expect(hint.length).toBeLessThan(90);
    expect(hint).toContain("…");
  });

  it("says nothing for an argument set it cannot render", () => {
    expect(argHint(null)).toBe("");
    expect(argHint("nope")).toBe("");
    expect(argHint({})).toBe("");
  });
});
