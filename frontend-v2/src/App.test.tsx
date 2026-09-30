// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { Header, WsAlert } from "./App";
import { useChat } from "./chat/chatStore";
import { useApp } from "./store";

// W5F1a. A10 made the header surface a REFUSED workspace save instead of
// swallowing it, but the only assertion was on the pure `saveWorkspace`
// transport: reverting App.tsx to the pre-fix version left the whole suite
// green, so the alert could be deleted silently.
//
// Two assertions here, and they cover different halves:
//
//   1. `renderToStaticMarkup(<WsAlert/>)` pins the markup — `role="alert"`,
//      the server's own sentence, not a generic one.
//   2. The interaction test drives the REAL Header in jsdom (already installed,
//      no new dependency): type a new path, press Enter, let the mocked fetch
//      refuse, and read the DOM. This is what dies if the
//      `{wsErr && <WsAlert error={wsErr} />}` line is removed, because the
//      refusal never reaches the screen.
//
// `act` is exported by react@18.3 itself; `react-dom/client` + jsdom are
// already dependencies. No `@testing-library/react`.

/** A duck-typed `Response`: only the members the code reads. Keeps the test
 *  independent of whether the jsdom environment exposes the global `Response`. */
function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

function text(): string {
  return document.body.textContent ?? "";
}

describe("the refused-save alert", () => {
  it("names the server's reason, as an alert", () => {
    const markup = renderToStaticMarkup(<WsAlert error="no such folder" />);
    expect(markup).toContain('role="alert"');
    expect(markup).toContain("no such folder");
  });
});

describe("saving the workspace path from the header", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    // React 18 requires this flag for `act` to flush without a warning.
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useApp.setState({ surface: "chat", workspacePath: "C:\\old", server: null });
    useChat.setState({ currentId: "s1" });
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
  });

  /** Open the path editor, type `next`, and commit it with Enter. */
  async function save(next: string) {
    await act(async () => {
      root.render(<Header />);
    });
    const edit = container.querySelector<HTMLButtonElement>(
      '[aria-label="Edit workspace path"]',
    );
    expect(edit).not.toBeNull();
    await act(async () => {
      edit!.click();
    });
    const input = container.querySelector<HTMLInputElement>(
      '[aria-label="Workspace folder path"]',
    );
    expect(input).not.toBeNull();
    await act(async () => {
      input!.value = next;
      input!.dispatchEvent(
        new KeyboardEvent("keydown", { key: "Enter", bubbles: true }),
      );
      // Let the save's fetch/parse chain settle INSIDE act, so the resulting
      // state update is flushed before the assertions read the DOM.
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("shows the refusal and leaves the store on the old path", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      reply(500, { error: "no such folder" })));

    await save("D:\\missing");

    expect(text()).toContain("no such folder");
    expect(useApp.getState().workspacePath).toBe("C:\\old");
  });

  it("shows no alert when the server accepted the path", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => reply(200, { ok: true })));

    await save("D:\\writing");

    expect(text()).not.toContain("no such folder");
    expect(container.querySelector('[role="alert"]')).toBeNull();
    expect(useApp.getState().workspacePath).toBe("D:\\writing");
  });
});
