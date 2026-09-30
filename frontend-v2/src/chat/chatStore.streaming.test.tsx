// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "../lib/api";
import { useChat } from "./chatStore";
import Transcript from "./Transcript";

// D3a/D3b, the client half. `chatStore.ts` reads the server's `streaming` key
// on three paths (`loadSessions`, `refreshRemote`, `open`) and the other vitest
// files hardcode the word in their fixtures, so a rename on ONE side alone left
// every client test green while the "interrupted" notice came back over a turn
// that was still running. The Python half (tests/test_sessions_streaming.py)
// reads the key from the LIVE payload; this half feeds the REAL payload shape
// through the REAL client paths and asserts the flag it produces.
//
// The no-key case is the one that matters: `s.streaming === true` on an absent
// key is `false`, and `liveTail(messages, false)` (reattach.ts) passes the
// server's own "this reply was interrupted" notice through unchanged — the
// pre-D3b state, never a wrong "still generating".

const NOTICE =
  "_(this reply was interrupted — the text above is what had been generated. "
  + "Say **continue** to resume it.)_";

const partial = (): ChatMessage => ({
  role: "assistant",
  content: "half a reply",
  partial: true,
  ckpt_id: "ck1",
  notice: NOTICE,
});

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

/** The session detail route the store's `open`/`refreshRemote` both call. */
function serveSession(body: unknown) {
  vi.stubGlobal("fetch", vi.fn(async () => reply(200, body)));
}

describe("the client's read of the server's streaming flag", () => {
  beforeEach(() => {
    useChat.setState({
      currentId: null, messages: [], remoteStreaming: {}, streams: {},
      sessions: [], lastError: null,
    });
  });

  afterEach(() => {
    vi.unstubAllGlobals();
  });

  it("open() records true from a payload that carries the key", async () => {
    serveSession({ id: "s1", messages: [], streaming: true });
    await useChat.getState().open("s1");
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
  });

  it("open() degrades to false when the payload omits the key", async () => {
    serveSession({ id: "s2", messages: [] });
    await useChat.getState().open("s2");
    expect(useChat.getState().remoteStreaming.s2).toBe(false);
  });

  it("refreshRemote() records true from a payload that carries the key", async () => {
    serveSession({ id: "s3", messages: [], streaming: true });
    await useChat.getState().refreshRemote("s3");
    expect(useChat.getState().remoteStreaming.s3).toBe(true);
  });

  it("refreshRemote() degrades to false when the payload omits the key", async () => {
    serveSession({ id: "s4", messages: [] });
    await useChat.getState().refreshRemote("s4");
    expect(useChat.getState().remoteStreaming.s4).toBe(false);
  });

  it("loadSessions() keeps the open chat's flag in step with the row", async () => {
    useChat.setState({ currentId: "s1" });
    vi.stubGlobal("fetch", vi.fn(async () => reply(200, [
      { id: "s1", title: "chat one", streaming: true },
      { id: "s2", title: "chat two" },        // no key at all
    ])));
    await useChat.getState().loadSessions();
    expect(useChat.getState().remoteStreaming.s1).toBe(true);

    // and the same row without the key is a plain `false`, not a stale `true`
    vi.stubGlobal("fetch", vi.fn(async () => reply(200, [
      { id: "s1", title: "chat one" },
    ])));
    await useChat.getState().loadSessions();
    expect(useChat.getState().remoteStreaming.s1).toBe(false);
  });
});

describe("a payload with no streaming key never reads as still generating", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    Element.prototype.scrollIntoView = vi.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    useChat.setState({
      currentId: null, messages: [], remoteStreaming: {}, streams: {},
      sessions: [], lastError: null,
    });
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  it("shows the server's interrupted notice, not 'still generating'", async () => {
    // the reload payload the client actually receives when the server omits the
    // flag (a backend that stopped sending it): the flag degrades to false and
    // the transcript must fall back to the server's own sentence
    serveSession({ id: "s1", messages: [partial()] });
    await act(async () => {
      await useChat.getState().open("s1");
    });
    expect(useChat.getState().remoteStreaming.s1).toBe(false);

    await act(async () => { root.render(<Transcript />); });
    expect(container.textContent).toContain("interrupted");
    expect(container.textContent).not.toContain("still generating on the server");
  });

  it("says still generating when the key really is true", async () => {
    serveSession({ id: "s1", messages: [partial()], streaming: true });
    await act(async () => {
      await useChat.getState().open("s1");
    });
    await act(async () => { root.render(<Transcript />); });
    expect(container.textContent).toContain("still generating on the server");
  });
});
