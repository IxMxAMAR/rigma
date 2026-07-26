// The regression suite for the wrong-row bug class (caught live twice on
// 2026-07-21). These tests pin the id-keyed contract of the pure reducer.
import { describe, expect, it } from "vitest";
import {
  alreadySaved, applyEvent, emptyTurn, STOPPED_SUFFIX, type StreamingTurn,
} from "./chatStore";
import type { ChatMessage } from "../lib/api";
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

describe("macro turns", () => {
  it("folds macro_step into the turn", () => {
    const t = applyEvent(emptyTurn(), {
      event: "macro_step",
      data: { index: 0, total: 3, kind: "tool", label: "Finish chapter" },
    });
    expect(t.macro).toEqual({ label: "Finish chapter", index: 0, total: 3 });
  });

  it("keeps chips while stepping, and tracks the current step", () => {
    let t = applyEvent(emptyTurn(), {
      event: "macro_step",
      data: { index: 0, total: 2, kind: "tool", label: "M" },
    });
    t = applyEvent(t, { event: "tool", data: { id: "m0", name: "read_file" } });
    t = applyEvent(t, {
      event: "macro_step",
      data: { index: 1, total: 2, kind: "prompt", label: "M" },
    });
    expect(t.macro?.index).toBe(1);
    expect(t.chips).toHaveLength(1);
  });

  it("clears the macro banner when the macro is done", () => {
    let t = applyEvent(emptyTurn(), {
      event: "macro_step",
      data: { index: 0, total: 1, kind: "note", label: "M" },
    });
    t = applyEvent(t, { event: "macro_done", data: { steps: 1 } });
    expect(t.macro).toBeNull();
  });

  it("a plain chat turn never grows a macro banner", () => {
    const t = feed(emptyTurn(), [["message", { delta: "hi" }]]);
    expect(t.macro).toBeNull();
  });
});

// Stop used to throw the partial reply away: the turn never reaches the
// server's persist step, so what was on screen was the only copy. send() now
// appends it — exactly once, which is what alreadySaved() decides.
describe("stopped-turn persistence", () => {
  const asst = (content: string): ChatMessage =>
    ({ role: "assistant", content }) as ChatMessage;
  const user = (content: string): ChatMessage =>
    ({ role: "user", content }) as ChatMessage;

  it("appends when the server saved nothing", () => {
    expect(alreadySaved([user("go")], "The answer begins here")).toBe(false);
  });

  it("does not append twice when the abort landed after the save", () => {
    // the race: stop() fires while the turn is already persisting
    const partial = "The answer begins here and keeps going";
    const saved = [user("go"), asst(partial + STOPPED_SUFFIX)];
    expect(alreadySaved(saved, partial)).toBe(true);
  });

  it("still matches when the saved copy was reworked around the text", () => {
    // <think> stripping and the suffix mean the two are never identical
    const partial = "Here is the plan you asked for, step one";
    const saved = [asst("Here is the plan you asked for, step one — more")];
    expect(alreadySaved(saved, partial)).toBe(true);
  });

  it("a trailing USER message never counts as the saved reply", () => {
    expect(alreadySaved([asst("older"), user("go")], "anything")).toBe(false);
  });

  it("empty or whitespace-only partials are never appended", () => {
    expect(alreadySaved([], "")).toBe(false);
    expect(alreadySaved([asst("x")], "   ")).toBe(false);
  });

  it("an empty transcript is safe to check", () => {
    expect(alreadySaved([], "text")).toBe(false);
  });

  it("the marker says the user stopped it, not the model", () => {
    expect(STOPPED_SUFFIX).toMatch(/stopped/i);
    expect(STOPPED_SUFFIX).toMatch(/partial/i);
  });
});
