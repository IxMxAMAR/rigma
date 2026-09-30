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
    remoteStreaming: { s1: true }, remoteStopRequested: {}, remoteStopTurn: {},
    sessions: [], savedAgent: {}, drafts: {}, lastError: null, notice: null,
    ...extra,
  });
}

/** A transcript carrying the server's per-turn checkpoint id (api.ts:48-50). */
function partial(ckpt: string) {
  return { role: "assistant" as const, content: "…", partial: true, ckpt_id: ckpt };
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

  it("does not let a request outlive its turn across a FAILED refresh", async () => {
    // P9 (verify-w14a): the click's follow-up read fails, so `remoteStreaming`
    // stays stale `true`; the network recovers only after a NEW remote turn has
    // begun. The request was made before any checkpoint existed, so the tab
    // cannot name its turn — and a lost confirmation means it cannot prove the
    // turn it now hears about is that one. It must not keep a disabled "stop
    // requested…" on a turn this tab never asked to stop.
    let fail = true;
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => {
        if (fail) throw new Error("network down");
        return reply(200, { id: "s1", messages: [], streaming: true });
      },
    ));
    await useChat.getState().stopRemote("s1");
    // the failed read left the request armed and the flag stale
    expect(useChat.getState().remoteStopRequested.s1).toBe(true);
    expect(useChat.getState().remoteStreaming.s1).toBe(true);

    // the network recovers and a new remote turn is already running
    fail = false;
    await useChat.getState().refreshRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
  });

  it("drops a request the server answers for a DIFFERENT turn", async () => {
    // The turn the request belongs to is named by the checkpoint on screen
    // (`ckpt_id`, api.ts:48-50). When the server reports a different turn, the
    // request belongs to a turn that has ended — even though the chat is still
    // generating.
    seed({ messages: [partial("ckA")] });
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [partial("ckB")], streaming: true }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
  });

  it("keeps the request while the server still reports the SAME turn", async () => {
    // The other side of P9: the fix must not drop every request. A failed read
    // that recovers to the SAME turn leaves the request live, so the control
    // stays honest about having asked.
    seed({ messages: [partial("ckA")] });
    let fail = true;
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => {
        if (fail) throw new Error("network down");
        return reply(200, { id: "s1", messages: [partial("ckA")], streaming: true });
      },
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBe(true);

    fail = false;
    await useChat.getState().refreshRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBe(true);
  });

  it("drops an UNNAMEABLE request once the server names a turn (DR3-3)", async () => {
    // The control is drawn from the server's `streaming` flag alone, so it can be
    // clicked before the turn's first checkpoint exists — `liveTurnId` is then
    // null and the pin is `{ turn: null, lost: false }`. The old rule required a
    // name neither side had, so it returned false forever: "stop requested…"
    // stayed disabled across every later turn and `stopRemote`'s early return
    // made the chat unstoppable. The tab cannot prove a nameable turn is its own,
    // so it must spend the request instead.
    seed({ messages: [] });
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [partial("ckA")], streaming: true }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBeUndefined();
    expect(useChat.getState().remoteStreaming.s1).toBe(true);
  });

  it("keeps an UNNAMEABLE request while the server cannot name a turn either", async () => {
    // The other side of DR3-3: nothing has changed, so there is nothing to spend.
    // Keeping it armed is what stops a fast double-click from POSTing twice.
    seed({ messages: [] });
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true, stopped: true }),
      () => reply(200, { id: "s1", messages: [], streaming: true }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().remoteStopRequested.s1).toBe(true);
  });

  it("does not invent an answer when the server omits `stopped`", async () => {
    // An absent key is UNKNOWN, not false. Reading it as false would print
    // "already finished — nothing was running to stop" against a server that
    // simply did not say — a fact nobody stated.
    vi.stubGlobal("fetch", route(
      () => reply(200, { ok: true }),
      () => reply(200, { id: "s1", messages: [], streaming: false }),
    ));
    await useChat.getState().stopRemote("s1");
    expect(useChat.getState().notice).not.toContain("already finished");
    expect(useChat.getState().notice).not.toContain("nothing was running");
    expect(useChat.getState().notice).toContain("did not say");
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
