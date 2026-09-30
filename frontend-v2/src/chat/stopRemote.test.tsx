// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Composer } from "./ChatSurface";
import { emptyTurn, useChat, type ChatState } from "./chatStore";

// OD-13. D3b makes a reloaded chat READ the server's `streaming` state, but the
// client's stop control only existed for a turn THIS tab started — a reloaded
// chat could watch a generation and not cancel it. Option 1 adds a control that
// POSTs the EXISTING stop route (`serve.py:5528 POST /api/sessions/{sid}/stop`).
//
// The care OD-13 names is that two tabs must not both believe they own the stop.
// So the store records a REQUEST and the server's own `streaming` flag is the
// truth: a `stopped: true` answer is not a finished stop, and a `stopped: false`
// answer ("nothing was running") is said out loud rather than claimed.

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

/** The routes a mounted `Composer` touches, plus the two `stopRemote` uses. */
function route(
  stop: () => Response,
  session: () => Response,
  seen?: (url: string, method: string) => void,
) {
  return vi.fn(async (url: string, init?: RequestInit) => {
    const u = String(url);
    const m = String(init?.method ?? "GET");
    seen?.(u, m);
    if (u === "/api/sessions/s1/stop") return stop();
    if (u === "/api/methods") return reply(200, { methods: [] });
    if (u === "/api/sessions/s1") return session();
    return reply(200, {});
  });
}

function seed(extra: Partial<ChatState> = {}) {
  useChat.setState({
    currentId: "s1", messages: [], streams: {}, aborts: {},
    remoteStreaming: { s1: true }, remoteStopRequested: {},
    sessions: [], savedAgent: {}, drafts: {}, lastError: null, notice: null,
    ...extra,
  });
}

describe("stopRemote asks the server, and lets the server decide", () => {
  beforeEach(() => seed());
  afterEach(() => vi.unstubAllGlobals());

  it("POSTs the existing stop route for that session id", async () => {
    const seen: [string, string][] = [];
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
      (u, m) => seen.push([u, m]),
    ));
    await useChat.getState().stopRemote("s1");
    expect(seen).toContainEqual(["/api/sessions/s1/stop", "POST"]);
  });

  it("records a REQUEST while the server still streams — not a finished stop", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBe(true);
    // nothing was claimed as finished, and the server still says it runs
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
    expect(useChat.getState().notice).toBeNull();
  });

  it("says the turn was ALREADY OVER when the server reports nothing running", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: false }),
      () => reply(200, { id: "s1", messages: [], streaming: false }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().notice).toContain("already finished");
    expect(useChat.getState().notice).toContain("nothing was running");
    // and the request is spent, not left armed for the next remote turn
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(false);
  });

  it("does not treat a REFUSED request as a stop", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(409, { error: "this chat is being driven by a run" }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().lastError).toContain("this chat is being driven");
    // the request is cleared (the control must not stick on "stop requested…")
    // and the server's own flag is untouched
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
  });

  it("fires ONE POST for a fast double-click", async () => {
    let posts = 0;
    vi.stubGlobal("fetch", route(
      () => { posts += 1; return reply(200, { ok: true, stopped: true }); },
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await Promise.all([
      useChat.getState().stopRemote("s1"),
      useChat.getState().stopRemote("s1"),
    ]);
    expect(posts).toBe(1);
  });

  it("drops a spent request the moment the server says the chat is done", async () => {
    // The other-tab case: this tab never clicked, or clicked and the turn ended.
    // The server's flag is the only truth, so the request cannot outlive it.
    useChat.setState({ remoteStopRequested: { s1: true } });
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: false }),
    ));
    await useChat.getState().refreshRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(false);
  });
});

describe("the composer's remote stop control", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    Element.prototype.scrollIntoView = vi.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
    seed();
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  it("offers a stop for a remotely-owned turn, and records the request", async () => {
    const seen: string[] = [];
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
      (u) => seen.push(u),
    ));
    await act(async () => { root.render(<Composer />); });
    const btn = container.querySelector('[aria-label="stop the remote turn"]');
    expect(btn).not.toBeNull();

    await act(async () => {
      btn!.dispatchEvent(new MouseEvent("click", { bubbles: true }));
    });
    expect(seen).toContain("/api/sessions/s1/stop");
    // honest label: a request, not a stop
    expect(container.querySelector('[aria-label="stop requested"]')).not.toBeNull();
    expect(container.textContent).toContain("stop requested…");
  });

  it("draws the remote control only while the server says it is generating", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await act(async () => { root.render(<Composer />); });
    expect(container.querySelector('[aria-label="stop the remote turn"]'))
      .not.toBeNull();

    // the server's own state ends the turn (this tab, or another one, or just
    // the reply finishing) — the control goes away with it
    await act(async () => {
      useChat.setState({ remoteStreaming: { s1: false } });
    });
    expect(container.querySelector('[aria-label="stop the remote turn"]'))
      .toBeNull();
  });

  it("keeps the local stop for a turn THIS tab owns", async () => {
    seed({ streams: { s1: emptyTurn() } });
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await act(async () => { root.render(<Composer />); });
    expect(container.querySelector('[aria-label="stop the remote turn"]'))
      .toBeNull();
    expect(container.textContent).toContain("stop");
  });
});
