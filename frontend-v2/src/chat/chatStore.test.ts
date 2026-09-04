// The regression suite for the wrong-row bug class (caught live twice on
// 2026-07-21). These tests pin the id-keyed contract of the pure reducer.
//
// The second half drives the store itself against a fake server, because the
// worst failure this file guards is not a wrong row but a wrong CHAT: a turn
// that finishes after you clicked another session used to write its
// transcript into that session's pane, and one regenerate then saved it over
// the other chapter.
import { beforeEach, describe, expect, it, vi } from "vitest";
import {
  alreadySaved, applyEvent, emptyTurn, errText, promptSurvived,
  selectAnyStreaming, selectStreaming, STOPPED_SUFFIX, useChat,
  type StreamingTurn,
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

// ---------------------------------------------------------------------------
// Store-level regressions. A fake server stands in for the REST surface: the
// store's whole job here is deciding WHICH chat a finished turn belongs to,
// and that decision is invisible to a pure-function test.

const h = vi.hoisted(() => ({
  streamChat: vi.fn(),
  runMacroStream: vi.fn(),
  api: {
    listSessions: vi.fn(),
    getSession: vi.fn(),
    createSession: vi.fn(),
    updateSession: vi.fn(),
    deleteSession: vi.fn(),
  },
}));

vi.mock("../lib/api", () => ({ api: h.api }));
vi.mock("../lib/methods", () => ({ runMacroStream: h.runMacroStream }));
vi.mock("../lib/sse", async (importOriginal) => ({
  ...(await importOriginal<typeof import("../lib/sse")>()),
  streamChat: h.streamChat,
}));

interface FakeSession {
  id: string;
  title: string;
  messages: ChatMessage[];
}

const PRISTINE = useChat.getState();
const tick = () => new Promise<void>((r) => setTimeout(r, 0));
const asUser = (t: string) => ({ role: "user", content: t }) as ChatMessage;
const asAsst = (t: string) => ({ role: "assistant", content: t }) as ChatMessage;

describe("store: one chat's turn never lands in another chat", () => {
  let server: Record<string, FakeSession>;
  /** resolve the turn the fake engine is holding open, per session */
  let release: Record<string, () => void>;

  /** streamChat that emits one delta and then waits to be released. */
  const holdOpen = () =>
    h.streamChat.mockImplementation(
      async (sid: string, _body: unknown,
             onEvent: (ev: { event: string; data: unknown }) => void) => {
        onEvent({ event: "message", data: { delta: `reply for ${sid}` } });
        await new Promise<void>((r) => { release[sid] = r; });
      });

  beforeEach(() => {
    vi.clearAllMocks();
    useChat.setState(PRISTINE, true);
    release = {};
    server = {
      A: { id: "A", title: "chapter one", messages: [asUser("open A")] },
      B: { id: "B", title: "chapter two",
           messages: [asUser("open B"), asAsst("B's own reply")] },
    };
    h.api.listSessions.mockResolvedValue([]);
    h.api.getSession.mockImplementation(async (id: string) => {
      const s = server[id];
      if (!s) throw new Error("no such session");
      return { ...s, messages: [...s.messages] };
    });
    h.api.updateSession.mockImplementation(
      async (id: string, patch: Record<string, unknown>) => {
        Object.assign(server[id], patch);
        return server[id];
      });
    h.api.createSession.mockImplementation(async () => {
      server.C = { id: "C", title: "New chat", messages: [] };
      return { ...server.C };
    });
    h.api.deleteSession.mockResolvedValue({});
  });

  it("a reply finishing after you switch chats stays out of the other pane",
     async () => {
    holdOpen();
    await useChat.getState().open("A");
    const turn = useChat.getState().send("write more");
    await tick();
    await useChat.getState().open("B");
    // the engine finishes A while B is the chat on screen
    server.A.messages = [...server.A.messages, asUser("write more"),
                         asAsst("reply for A")];
    release.A();
    await turn;
    expect(useChat.getState().currentId).toBe("B");
    expect(useChat.getState().messages).toEqual(server.B.messages);
    // and nothing of A's was written over B's session
    expect(h.api.updateSession).not.toHaveBeenCalled();
    expect(server.B.messages).toHaveLength(2);
  });

  it("stop cancels the chat you are looking at, not the last turn started",
     async () => {
    const signals: Record<string, AbortSignal> = {};
    h.streamChat.mockImplementation(
      async (sid: string, _body: unknown, _on: unknown, signal: AbortSignal) => {
        signals[sid] = signal;
        await new Promise<void>((_res, rej) => {
          signal.addEventListener("abort", () =>
            rej(Object.assign(new Error("aborted"), { name: "AbortError" })));
        });
      });
    await useChat.getState().open("A");
    const a = useChat.getState().send("in A");
    await tick();
    await useChat.getState().open("B");
    const b = useChat.getState().send("in B");
    await tick();
    await useChat.getState().open("A");
    useChat.getState().stop();
    expect(signals.A.aborted).toBe(true);
    expect(signals.B.aborted).toBe(false);
    await useChat.getState().open("B");
    useChat.getState().stop();
    await Promise.all([a, b]);
  });

  it("re-opening a chat mid-reply does not let a second turn start in it",
     async () => {
    holdOpen();
    await useChat.getState().open("A");
    const turn = useChat.getState().send("first");
    await tick();
    await useChat.getState().open("B");
    await useChat.getState().open("A");
    void useChat.getState().send("second");
    await tick();
    expect(h.streamChat).toHaveBeenCalledTimes(1);
    release.A();
    await turn;
  });

  it("a regenerate left running in one chat cannot fold its take into another",
     async () => {
    server.A.messages = [asUser("go"), asAsst("A's discarded take")];
    h.streamChat.mockImplementation(async (sid: string) => {
      if (sid === "A") await new Promise<void>((r) => { release.A = r; });
    });
    await useChat.getState().open("A");
    const regen = useChat.getState().regenerate();
    await tick();
    await useChat.getState().open("B");
    await useChat.getState().send("carry on");
    // B ends in an assistant message, so a stray pendingVariant would be
    // folded into it and PUT to B's session
    const wrote = h.api.updateSession.mock.calls.filter((c) => c[0] === "B");
    expect(wrote).toEqual([]);
    expect(JSON.stringify(server.B.messages))
      .not.toContain("A's discarded take");
    release.A();
    await regen;
  });

  it("a macro finishing after you switch chats stays out of the other pane",
     async () => {
    h.runMacroStream.mockImplementation(
      async (sid: string, _body: unknown,
             onEvent: (ev: { event: string; data: unknown }) => void) => {
        onEvent({ event: "message", data: { delta: "macro output" } });
        await new Promise<void>((r) => { release[sid] = r; });
      });
    await useChat.getState().open("A");
    const macro = useChat.getState().runMacro("finish-chapter");
    await tick();
    await useChat.getState().open("B");
    server.A.messages = [...server.A.messages, asAsst("macro output")];
    release.A();
    await macro;
    expect(useChat.getState().currentId).toBe("B");
    expect(useChat.getState().messages).toEqual(server.B.messages);
  });
});

// A failed turn used to leave the screen exactly as if the model had chosen
// silence: the error lived on the streaming object, and the reload that ends
// every turn threw that object away tens of milliseconds after it appeared.
describe("store: a failed turn says so", () => {
  let server: Record<string, FakeSession>;

  beforeEach(() => {
    vi.clearAllMocks();
    useChat.setState(PRISTINE, true);
    server = { A: { id: "A", title: "t", messages: [asUser("hi")] } };
    h.api.listSessions.mockResolvedValue([]);
    h.api.getSession.mockImplementation(async (id: string) => {
      const s = server[id];
      if (!s) throw new Error("no such session");
      return { ...s, messages: [...s.messages] };
    });
    h.api.updateSession.mockResolvedValue({});
    h.api.createSession.mockResolvedValue({ id: "C", title: "New chat",
                                            messages: [] });
  });

  it("keeps the engine's reason on screen after the transcript reloads",
     async () => {
    h.streamChat.mockImplementation(
      async (_sid: string, _b: unknown,
             onEvent: (ev: { event: string; data: unknown }) => void) => {
        onEvent({ event: "error",
                  data: { message: "the engine is unloaded" } });
      });
    await useChat.getState().open("A");
    await useChat.getState().send("write");
    expect(useChat.getState().lastError).toMatch(/engine is unloaded/);
  });

  it("keeps a rejected turn's reason, and the message the server never saw",
     async () => {
    // the vision guard rejects with 400 BEFORE the user message is stored
    h.streamChat.mockRejectedValue(new Error("this model can't see images"));
    await useChat.getState().open("A");
    await useChat.getState().send("look at this");
    expect(useChat.getState().lastError).toMatch(/can't see images/);
    const last = useChat.getState().messages.slice(-1)[0];
    expect(last?.role).toBe("user");
    expect(last?.content).toBe("look at this");
  });

  it("a new turn clears the previous failure", async () => {
    h.streamChat.mockRejectedValueOnce(new Error("engine died"));
    await useChat.getState().open("A");
    await useChat.getState().send("one");
    expect(useChat.getState().lastError).toBe("engine died");
    h.streamChat.mockImplementation(async () => {});
    await useChat.getState().send("two");
    expect(useChat.getState().lastError).toBeNull();
  });

  it("a first send that cannot even create the chat reports and gives back "
     + "the message", async () => {
    h.api.createSession.mockRejectedValue(new Error("server replied 500"));
    const ok = await useChat.getState().send("a paragraph I typed");
    expect(ok).toBe(false);
    expect(useChat.getState().lastError).toMatch(/500/);
    expect(useChat.getState().currentId).toBeNull();
    expect(useChat.getState().messages).toEqual([]);
  });

  it("a session that will not open reports instead of failing silently",
     async () => {
    await useChat.getState().open("missing");
    expect(useChat.getState().lastError).toMatch(/no such session/);
    expect(useChat.getState().currentId).toBeNull();
  });
});

// The other half of the same defect: the reply you walked away from used to
// be discarded on arrival — every event dropped, nothing persisted server-side
// until the turn ends — so coming back showed a prompt with no answer.
describe("store: a turn you walk away from keeps going", () => {
  let server: Record<string, FakeSession>;
  let release: Record<string, () => void>;

  beforeEach(() => {
    vi.clearAllMocks();
    useChat.setState(PRISTINE, true);
    release = {};
    server = {
      A: { id: "A", title: "chapter one", messages: [asUser("open A")] },
      B: { id: "B", title: "chapter two", messages: [asUser("open B")] },
    };
    h.api.listSessions.mockResolvedValue([]);
    h.api.getSession.mockImplementation(async (id: string) => {
      const s = server[id];
      if (!s) throw new Error("no such session");
      return { ...s, messages: [...s.messages] };
    });
    h.api.updateSession.mockResolvedValue({});
    h.streamChat.mockImplementation(
      async (sid: string, _body: unknown,
             onEvent: (ev: { event: string; data: unknown }) => void) => {
        onEvent({ event: "message", data: { delta: `reply for ${sid}` } });
        await new Promise<void>((r) => { release[sid] = r; });
      });
  });

  it("renders in its own chat and nowhere else", async () => {
    await useChat.getState().open("A");
    const turn = useChat.getState().send("write more");
    await tick();
    expect(selectStreaming(useChat.getState())?.text).toBe("reply for A");
    await useChat.getState().open("B");
    expect(selectStreaming(useChat.getState())).toBeNull();
    await useChat.getState().open("A");
    // still generating, still visible, still A's
    expect(selectStreaming(useChat.getState())?.text).toBe("reply for A");
    release.A();
    await turn;
  });

  it("still counts as generating for the engine controls", async () => {
    await useChat.getState().open("A");
    const turn = useChat.getState().send("write more");
    await tick();
    await useChat.getState().open("B");
    expect(selectStreaming(useChat.getState())).toBeNull();
    expect(selectAnyStreaming(useChat.getState())).toBe(true);
    release.A();
    await turn;
    expect(selectAnyStreaming(useChat.getState())).toBe(false);
  });

  it("leaves no key behind when it ends", async () => {
    await useChat.getState().open("A");
    const turn = useChat.getState().send("write more");
    await tick();
    release.A();
    await turn;
    expect(useChat.getState().streams).toEqual({});
    expect(useChat.getState().aborts).toEqual({});
    expect(useChat.getState().pendingVariants).toEqual({});
  });

  it("is cancelled when the chat it belongs to is deleted", async () => {
    const signals: Record<string, AbortSignal> = {};
    h.streamChat.mockImplementation(
      async (sid: string, _b: unknown, _on: unknown, signal: AbortSignal) => {
        signals[sid] = signal;
        await new Promise<void>((_res, rej) => {
          signal.addEventListener("abort", () =>
            rej(Object.assign(new Error("aborted"), { name: "AbortError" })));
        });
      });
    h.api.deleteSession.mockResolvedValue({});
    await useChat.getState().open("A");
    const turn = useChat.getState().send("write more");
    await tick();
    await useChat.getState().open("B");
    await useChat.getState().deleteChat("A");
    expect(signals.A.aborted).toBe(true);
    await turn;
    expect(useChat.getState().currentId).toBe("B");
  });
});

describe("promptSurvived", () => {
  it("is false when the server never stored the prompt", () => {
    // the vision guard's 400 lands before the message is appended
    expect(promptSurvived([asUser("older")], "look at this")).toBe(false);
    expect(promptSurvived([], "look at this")).toBe(false);
  });

  it("is true when the reload came back with it", () => {
    expect(promptSurvived([asUser("older"), asUser("go")], "go")).toBe(true);
  });

  it("compares image messages by shape, not by reference", () => {
    const parts = [{ type: "text", text: "what is this" },
                   { type: "image_url", image_url: { url: "data:,x" } }];
    const stored = { role: "user", content: JSON.parse(JSON.stringify(parts)) };
    expect(promptSurvived([stored as ChatMessage],
                          parts as ChatMessage["content"])).toBe(true);
  });

  it("never counts a reply as the prompt", () => {
    expect(promptSurvived([asAsst("go")], "go")).toBe(false);
  });
});

describe("errText", () => {
  it("uses the message a thrown Error carries", () => {
    expect(errText(new Error("engine is unloaded"))).toBe("engine is unloaded");
  });

  it("never renders the word undefined", () => {
    expect(errText(undefined)).toBe("the request failed");
    expect(errText(new Error("   "))).toBe("the request failed");
  });

  it("keeps a plain string rejection", () => {
    expect(errText("network down")).toBe("network down");
  });
});
