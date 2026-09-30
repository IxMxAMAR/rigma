// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { emptyTurn, useChat } from "./chatStore";
import ControlPanel from "./ControlPanel";

// B6d. The wave-2 verifier's finding was that `mode_set` had no UI at all, and the
// residual risk in every control-plane fix is the same one W5F1a chased: the pure
// helper can be tested while the render/wiring line is deleted with the suite
// green. So this mounts the REAL panel in jsdom, selects a mode the session
// advertised, and asserts the request body and the confirmation.
//
// It also pins the two product rules: the options are the SESSION's own list (a
// fabricated `plan`/`default` pair would fail the option assertion), and an
// unknown/empty list renders as "unknown" with no select at all.
//
// No `@testing-library/react` — `react-dom/client`, jsdom and `act` are already
// dependencies (the harness App.test.tsx and ModelPicker.test.tsx use).

/** A duck-typed `Response`, as in App.test.tsx / EngineSurface.test.tsx. */
function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

/** React tracks a controlled select's value, so setting `.value` directly leaves
 *  its tracker in step and the change event is swallowed. The native prototype
 *  setter bypasses the tracker — the documented workaround, and the reason this
 *  file uses jsdom rather than a static render. */
function setSelect(sel: HTMLSelectElement, value: string) {
  const setter = Object.getOwnPropertyDescriptor(
    HTMLSelectElement.prototype, "value",
  )!.set!;
  setter.call(sel, value);
  sel.dispatchEvent(new Event("change", { bubbles: true }));
}

const MODES = {
  availableModes: [{ id: "default" }, { id: "plan", name: "Plan" }],
  currentModeId: "default",
  known: true,
};

describe("the control plane's mode select", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useChat.setState({
      currentId: "s1",
      streams: { s1: { ...emptyTurn(), acpModes: MODES } },
    });
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

  async function mount() {
    await act(async () => {
      root.render(
        <ControlPanel sessionId="s1" harness="mcode" transport="acp" hasSession />,
      );
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("offers the session's own modes and sends the one chosen", async () => {
    const calls: { url: string; body: unknown }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      calls.push({
        url: String(url),
        body: init?.body ? JSON.parse(String(init.body)) : null,
      });
      return reply(200, { ok: true, op: "mode_set", result: { modeId: "plan" } });
    }));

    await mount();
    const sel = container.querySelector<HTMLSelectElement>(
      '[aria-label="mcode session mode"]',
    );
    expect(sel).not.toBeNull();
    // The options are the SESSION's, in its order — never a fabricated pair.
    expect(Array.from(sel!.options).map((o) => o.value)).toEqual(["default", "plan"]);
    expect(Array.from(sel!.options).map((o) => o.textContent))
      .toEqual(["default", "Plan"]);

    await act(async () => {
      setSelect(sel!, "plan");
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    const call = calls.find((c) => c.url === "/api/sessions/s1/control");
    expect(call, "no control request was sent").toBeTruthy();
    expect(call!.body).toEqual({ op: "mode_set", params: { modeId: "plan" } });
    expect(container.textContent).toContain("mode is now plan");
    expect(container.textContent).not.toContain("done");
    // The accepted change is mirrored, so the select does not snap back.
    expect(useChat.getState().streams.s1.acpModes?.currentModeId).toBe("plan");
  });

  it("renders unknown — with no select — when no list was advertised", async () => {
    useChat.setState({ currentId: "s1", streams: { s1: emptyTurn() } });
    await mount();
    expect(container.textContent).toContain("unknown");
    expect(container.querySelector('[aria-label="mcode session mode"]')).toBeNull();
    // No fabricated value leaked into the DOM.
    expect(container.textContent).not.toContain("acceptEdits");
  });

  it("renders unknown for an empty advertised list too", async () => {
    useChat.setState({
      currentId: "s1",
      streams: {
        s1: { ...emptyTurn(), acpModes: { availableModes: [], currentModeId: "" } },
      },
    });
    await mount();
    expect(container.textContent).toContain("unknown");
    expect(container.querySelector('[aria-label="mcode session mode"]')).toBeNull();
  });
});
