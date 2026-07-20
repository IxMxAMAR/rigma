// The regression suite for the wrong-row bug class (caught live twice on
// 2026-07-21). These tests pin the id-keyed contract of the pure reducer.
import { describe, expect, it } from "vitest";
import { applyEvent, emptyTurn, type StreamingTurn } from "./chatStore";
import { makeSseParser } from "../lib/sse";

const feed = (turn: StreamingTurn, evs: [string, unknown][]) =>
  evs.reduce((t, [event, data]) => applyEvent(t, { event, data }), turn);

describe("applyEvent chips", () => {
  it("keys results by id, not arrival order", () => {
    // the live failure: chip A, chip B, result A — order-matching hung
    // A's result on B. Id-matching cannot.
    const t = feed(emptyTurn(), [
      ["tool", { id: "a", name: "read_file", args: { path: "1.txt" } }],
      ["tool", { id: "b", name: "read_file", args: { path: "2.txt" } }],
      ["tool_result", { id: "b", name: "read_file", result: "two" }],
    ]);
    expect(t.chips.find((c) => c.id === "a")?.state).toBe("running");
    expect(t.chips.find((c) => c.id === "b")?.state).toBe("done");
    expect(t.chips.find((c) => c.id === "b")?.result).toBe("two");
  });

  it("same-name parallel calls resolve independently", () => {
    const t = feed(emptyTurn(), [
      ["tool", { id: "a", name: "write_file" }],
      ["tool", { id: "b", name: "write_file" }],
      ["tool_result", { id: "a", result: "one" }],
      ["tool_result", { id: "b", result: "two" }],
    ]);
    expect(t.chips.map((c) => c.result)).toEqual(["one", "two"]);
  });

  it("id-less legacy results fall back to first open chip", () => {
    const t = feed(emptyTurn(), [
      ["tool", { name: "read_file" }],
      ["tool", { name: "read_file" }],
      ["tool_result", { result: "first" }],
    ]);
    expect(t.chips[0].state).toBe("done");
    expect(t.chips[0].result).toBe("first");
    expect(t.chips[1].state).toBe("running");
  });

  it("a result never resolves an already-done chip", () => {
    const t = feed(emptyTurn(), [
      ["tool", { id: "a", name: "x" }],
      ["tool_result", { id: "a", result: "one" }],
      ["tool_result", { id: "a", result: "dupe" }],
    ]);
    expect(t.chips[0].result).toBe("one");
  });

  it("is pure: input turn is never mutated", () => {
    const before = emptyTurn();
    applyEvent(before, { event: "tool", data: { id: "a", name: "x" } });
    expect(before.chips).toEqual([]);
  });
});

describe("applyEvent text/thinking/errors", () => {
  it("accumulates deltas and thinking separately", () => {
    const t = feed(emptyTurn(), [
      ["think", { delta: "hm " }],
      ["think", { delta: "ok" }],
      ["message", { delta: "Hel" }],
      ["message", { delta: "lo" }],
    ]);
    expect(t.thinking).toBe("hm ok");
    expect(t.text).toBe("Hello");
  });

  it("captures errors without losing partial text", () => {
    const t = feed(emptyTurn(), [
      ["message", { delta: "partial" }],
      ["error", { message: "engine died" }],
    ]);
    expect(t.text).toBe("partial");
    expect(t.error).toBe("engine died");
  });
});

describe("sse parser", () => {
  it("reassembles events across chunk boundaries", () => {
    const parse = makeSseParser();
    const a = parse('event: tool\ndata: {"id": "a", "na');
    const b = parse('me": "read_file"}\n\ndata: {"delta": "hi"}\n\n');
    expect(a).toEqual([]);
    expect(b).toHaveLength(2);
    expect(b[0].event).toBe("tool");
    expect((b[1].data as { delta: string }).delta).toBe("hi");
  });

  it("drops [DONE] and malformed frames without crashing", () => {
    const parse = makeSseParser();
    const evs = parse("data: [DONE]\n\ndata: {broken\n\ndata: {\"delta\": \"x\"}\n\n");
    expect(evs).toHaveLength(1);
  });
});
