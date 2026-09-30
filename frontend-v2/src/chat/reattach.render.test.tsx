// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "../lib/api";
import { SessionRail } from "./ChatSurface";
import { useChat } from "./chatStore";
import Transcript from "./Transcript";

// D3b, the wiring half. `reattach.test.ts` pins the rule; this mounts the REAL
// `Transcript` and the REAL rail over a store seeded the way a reload leaves it,
// because the wave-2 verifier's lesson (W5F1a) is that a small render line can be
// deleted with the whole suite green. The notice string is `serve.py`'s own.
//
// No `@testing-library/react` — `react-dom/client`, jsdom and `act` are already
// dependencies (the harness App.test.tsx uses).

const NOTICE =
  "_(this reply was interrupted — the text above is what had been generated. "
  + "Say **continue** to resume it.)_";

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

const partial = (): ChatMessage => ({
  role: "assistant",
  content: "half a reply",
  partial: true,
  ckpt_id: "ck1",
  notice: NOTICE,
});

function seed(messages: ChatMessage[], streaming: boolean) {
  useChat.setState({
    currentId: "s1",
    messages,
    remoteStreaming: { s1: streaming },
    sessions: [
      { id: "s1", title: "chat one", streaming },
      { id: "s2", title: "chat two", streaming: false },
    ],
    savedAgent: {},
    lastError: null,
  });
}

describe("a reloaded chat and its live tail", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    // jsdom has no layout, and the transcript's stick-to-bottom effect calls this.
    Element.prototype.scrollIntoView = vi.fn();
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
    vi.unstubAllGlobals();
    vi.useRealTimers();
  });

  it("does not call the checkpoint an interruption while the turn runs", async () => {
    seed([partial()], true);
    await act(async () => {
      root.render(<Transcript />);
    });
    expect(container.textContent).toContain("still generating on the server");
    expect(container.textContent).not.toContain("interrupted");
  });

  it("keeps the server's own sentence when the turn really did stop", async () => {
    seed([partial()], false);
    await act(async () => {
      root.render(<Transcript />);
    });
    expect(container.textContent).toContain("interrupted");
    expect(container.textContent).not.toContain("still generating on the server");
  });

  it("polls while the server is generating, and stops when it says it is done", async () => {
    vi.useFakeTimers();
    seed([partial()], true);
    const calls: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      calls.push(String(url));
      return reply(200, {
        id: "s1",
        title: "chat one",
        messages: [{ role: "assistant", content: "a whole reply" }],
        streaming: false,
      });
    }));
    await act(async () => {
      root.render(<Transcript />);
    });
    expect(container.textContent).toContain("still generating on the server");

    await act(async () => {
      await vi.advanceTimersByTimeAsync(10000);
    });

    expect(calls).toContain("/api/sessions/s1");
    expect(container.textContent).toContain("a whole reply");
    expect(container.textContent).not.toContain("still generating on the server");
  });

  it("marks a live chat in the rail, and only a live one", async () => {
    seed([], true);
    await act(async () => {
      root.render(<SessionRail />);
    });
    expect(container.querySelector('[aria-label="generating in chat one"]'))
      .not.toBeNull();
    expect(container.querySelector('[aria-label="generating in chat two"]'))
      .toBeNull();
  });

  it("draws no rail dot when nothing is generating", async () => {
    seed([], false);
    await act(async () => {
      root.render(<SessionRail />);
    });
    expect(container.querySelector('[role="img"][aria-label^="generating"]'))
      .toBeNull();
  });
});
