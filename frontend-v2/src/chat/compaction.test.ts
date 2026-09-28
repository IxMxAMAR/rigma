import { describe, expect, it } from "vitest";

import {
  EMPTY_COMPACTIONS,
  compactionLine,
  foldCompaction,
  running,
  summaryText,
  type Compactions,
} from "./compaction";

const start = (id: string) => ({ event: "compaction/start", data: { compactionId: id, turn: 1 } });
const end = (id: string, error = "") => ({
  event: "compaction/end",
  data: { compactionId: id, turn: 1, ...(error ? { error } : {}) },
});
const summary = (id: string, tokens: number, seqs: number[], text = "the summary") => ({
  event: "compaction/summary",
  data: {
    compactionId: id,
    summary: [{ type: "text", text }],
    shadowedRange: { start: 0, end: seqs.length },
    shadowedSeqs: Array.from({ length: seqs.length }, (_, i) => i),
    shadowedTokenCount: tokens,
    provider: "deepseek-official",
    model: "DeepSeek-V4.1-Flash",
  },
});
const prune = (tokens: number, n: number) => ({
  event: "compaction/prune",
  data: {
    shadowedRange: { start: 0, end: n },
    shadowedSeqs: Array.from({ length: n }, (_, i) => i),
    shadowedTokenCount: tokens,
  },
});

// DSH reports compaction as a BRACKET: start opens it, end closes it, paired by
// compactionId. Rigma's native path already reports its own compaction, but a DSH
// turn reported none of it, so a long turn busy summarising its own context
// looked exactly like a turn that had hung.
describe("foldCompaction", () => {
  it("opens a bracket on start", () => {
    const c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    expect(c.items).toHaveLength(1);
    expect(c.items[0]).toMatchObject({ id: "c1", open: true });
    expect(c.openId).toBe("c1");
    expect(running(c)?.id).toBe("c1");
  });

  it("closes the bracket on end", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, end("c1"));
    expect(c.items[0].open).toBe(false);
    expect(c.openId).toBe("");
    // Nothing is running any more, which is what stops the UI claiming it is.
    expect(running(c)).toBeNull();
  });

  it("records the summary's token count, which is the number worth showing", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, summary("c1", 12345, [1, 2, 3]));
    expect(c.items[0].shadowedTokens).toBe(12345);
    expect(c.items[0].shadowedCount).toBe(3);
    expect(c.items[0].summary).toBe("the summary");
    expect(c.items[0].model).toBe("DeepSeek-V4.1-Flash");
  });

  // prune has no id of its own, so it must be attributed to the open compaction.
  it("attributes a prune to the open compaction", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, prune(500, 4));
    expect(c.items[0].shadowedTokens).toBe(500);
    expect(c.items[0].shadowedCount).toBe(4);
  });

  it("adds a prune onto a summary's count rather than replacing it", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, summary("c1", 1000, [1, 2]));
    c = foldCompaction(c, prune(250, 3));
    expect(c.items[0].shadowedTokens).toBe(1250);
    expect(c.items[0].shadowedCount).toBe(5);
  });

  it("keeps two compactions apart", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, end("c1"));
    c = foldCompaction(c, start("c2"));
    c = foldCompaction(c, summary("c2", 900, [1]));
    expect(c.items).toHaveLength(2);
    expect(c.items[0].shadowedTokens).toBe(0);
    expect(c.items[1].shadowedTokens).toBe(900);
    expect(running(c)?.id).toBe("c2");
  });

  it("records a failure, which is the one thing a reader must not miss", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, end("c1", "summariser returned nothing"));
    expect(c.items[0].error).toBe("summariser returned nothing");
    expect(c.items[0].open).toBe(false);
  });

  // A replay must not double the row.
  it("does not duplicate a repeated start for one id", () => {
    let c = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    c = foldCompaction(c, start("c1"));
    expect(c.items).toHaveLength(1);
  });

  // The fold knows real work happened even when the opening bracket is missing.
  it("keeps a summary whose start was never seen", () => {
    const c = foldCompaction(EMPTY_COMPACTIONS, summary("ghost", 700, [1, 2]));
    expect(c.items).toHaveLength(1);
    expect(c.items[0].shadowedTokens).toBe(700);
  });

  it("ignores an unknown compaction kind rather than inventing one", () => {
    const c = foldCompaction(EMPTY_COMPACTIONS, {
      event: "compaction/something-new", data: { compactionId: "c1" },
    });
    expect(c).toBe(EMPTY_COMPACTIONS);
  });

  // Same bug already fixed once in governance.ts: stripping a prefix that is not
  // there is a no-op, so a non-compaction payload would become a compaction kind.
  it("refuses a payload that is not a compaction event", () => {
    expect(foldCompaction(EMPTY_COMPACTIONS, { event: "goal", data: {} })).toBe(EMPTY_COMPACTIONS);
    expect(foldCompaction(EMPTY_COMPACTIONS, {})).toBe(EMPTY_COMPACTIONS);
    expect(foldCompaction(EMPTY_COMPACTIONS, null)).toBe(EMPTY_COMPACTIONS);
  });

  it("never mutates the state it was given", () => {
    const before: Compactions = foldCompaction(EMPTY_COMPACTIONS, start("c1"));
    const snapshot = JSON.parse(JSON.stringify(before)) as Compactions;
    foldCompaction(before, summary("c1", 999, [1]));
    expect(before).toEqual(snapshot);
  });
});

describe("summaryText", () => {
  it("joins content blocks with no separator", () => {
    // A space between blocks would insert a gap inside a word wherever the
    // provider split a block mid-token — the same bug already fixed once in the
    // DSH subagent rendering.
    expect(summaryText([{ type: "text", text: "hel" }, { type: "text", text: "lo" }]))
      .toBe("hello");
  });

  it("accepts a bare string block", () => {
    expect(summaryText(["plain"])).toBe("plain");
  });

  it("drops a non-text block rather than stringifying it", () => {
    // "[object Object]" in a status line is worse than nothing.
    expect(summaryText([{ type: "image", data: "x" }, { type: "text", text: "kept" }]))
      .toBe("kept");
  });

  it("is empty for anything that is not a list", () => {
    expect(summaryText(undefined)).toBe("");
    expect(summaryText("nope")).toBe("");
  });
});

describe("compactionLine", () => {
  it("reports the messages and tokens folded away", () => {
    const line = compactionLine({
      id: "c1", open: false, shadowedTokens: 12345, shadowedCount: 7,
      summary: "", provider: "", model: "", error: "",
    });
    expect(line).toContain("7 messages");
    expect(line).toContain("12,345 tokens");
  });

  it("says nothing when it knows nothing, rather than guessing", () => {
    expect(compactionLine({
      id: "c1", open: false, shadowedTokens: 0, shadowedCount: 0,
      summary: "", provider: "", model: "", error: "",
    })).toBeNull();
  });

  it("leads with the failure", () => {
    expect(compactionLine({
      id: "c1", open: false, shadowedTokens: 0, shadowedCount: 0,
      summary: "", provider: "", model: "", error: "out of memory",
    })).toContain("out of memory");
  });

  it("singularises one message", () => {
    const line = compactionLine({
      id: "c1", open: false, shadowedTokens: 0, shadowedCount: 1,
      summary: "", provider: "", model: "", error: "",
    });
    expect(line).toContain("1 message ");
  });
});
