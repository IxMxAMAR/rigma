// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ChatMessage } from "../lib/api";
import { emptyTurn, useChat } from "./chatStore";
import Transcript from "./Transcript";

// Owner report 2026-10-23: "the chat was progressing when the model's thinking
// block just vanished, so I don't know if it's processing or still working."
//
// Two separate defects produced that, and both are render-level, so both are
// pinned by mounting the REAL transcript:
//
//   1. a LIVE turn's thinking collapsed the moment the first answer token
//      arrived, because `live` was derived from `turn.text === ""`. On a
//      multi-round turn (text, tool, more reasoning) that hid the block while
//      the turn was still running.
//   2. a FINISHED turn's thinking was never rendered at all — the server
//      persists it and the transcript dropped it.

const THINKING = "the reader is asking why the block disappeared";

function seed(messages: ChatMessage[], turn?: ReturnType<typeof emptyTurn>) {
  useChat.setState({
    currentId: "s1",
    messages,
    streams: turn ? { s1: turn } : {},
    remoteStreaming: {},
    sessions: [{ id: "s1", title: "chat one", streaming: Boolean(turn) }],
    savedAgent: {},
    lastError: null,
  });
}

function thinkingButton(container: HTMLElement): HTMLButtonElement | null {
  return container.querySelector<HTMLButtonElement>("button[aria-expanded]");
}

describe("the thinking block", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
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
  });

  it("stays open while the turn is still running, even after text arrives", async () => {
    seed([], { ...emptyTurn(), thinking: THINKING, text: "an answer so far" });
    await act(async () => {
      root.render(<Transcript />);
    });
    // The bug: this text used to disappear the instant `text` was non-empty.
    expect(container.textContent).toContain(THINKING);
    expect(thinkingButton(container)?.getAttribute("aria-expanded"))
      .toBe("true");
    expect(container.querySelector('[title="still reasoning"]')).not.toBeNull();
  });

  it("shows a finished reply's thinking, collapsed, and opens on click", async () => {
    seed([{ role: "assistant", content: "the answer", thinking: THINKING }]);
    await act(async () => {
      root.render(<Transcript />);
    });
    expect(container.textContent).toContain("thinking");
    expect(container.textContent).not.toContain(THINKING);

    await act(async () => {
      thinkingButton(container)!.dispatchEvent(
        new MouseEvent("click", { bubbles: true }));
    });
    expect(container.textContent).toContain(THINKING);
  });

  it("renders nothing for a turn that never reasoned", async () => {
    seed([{ role: "assistant", content: "a direct answer" }]);
    await act(async () => {
      root.render(<Transcript />);
    });
    expect(thinkingButton(container)).toBeNull();
  });
});
