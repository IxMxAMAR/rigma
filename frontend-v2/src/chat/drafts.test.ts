import { describe, expect, it } from "vitest";

import { parseDrafts, persistableDrafts } from "./drafts";

// IMP-1: drafts survive a reload. The cap is the only interesting logic — a
// draft is user text, so it is never truncated, only declined; and the map is
// trimmed oldest-first so a reload cannot blow the shared localStorage quota.
describe("persisting per-chat drafts", () => {
  it("keeps ordinary drafts, in order", () => {
    expect(persistableDrafts({ a: "one", b: "two" }))
      .toEqual({ a: "one", b: "two" });
  });

  it("drops an oversized draft instead of truncating the user's text", () => {
    const out = persistableDrafts({ big: "x".repeat(11), ok: "keep" }, 10, 1000);
    expect(out).toEqual({ ok: "keep" });
  });

  it("drops the oldest entries when the map does not fit", () => {
    const out = persistableDrafts({ first: "a".repeat(40), second: "b".repeat(40),
                                    third: "c".repeat(40) }, 100, 70);
    expect(Object.keys(out)).toEqual(["third"]);
  });

  it("never mutates its input", () => {
    const d = { a: "x".repeat(11) };
    persistableDrafts(d, 10, 1000);
    expect(d).toEqual({ a: "x".repeat(11) });
  });

  it("ignores an empty value, malformed JSON and a foreign shape", () => {
    expect(persistableDrafts({ a: "", b: "keep" })).toEqual({ b: "keep" });
    expect(parseDrafts(null)).toEqual({});
    expect(parseDrafts("not json")).toEqual({});
    expect(parseDrafts("[1,2]")).toEqual({});
    expect(parseDrafts('"a string"')).toEqual({});
  });

  it("keeps only string values out of a hand-edited store", () => {
    expect(parseDrafts('{"a":"keep","b":5,"c":null,"d":""}'))
      .toEqual({ a: "keep" });
  });

  it("round-trips through JSON", () => {
    const d = { s1: "half-written thought\nwith a newline" };
    expect(parseDrafts(JSON.stringify(persistableDrafts(d)))).toEqual(d);
  });
});
