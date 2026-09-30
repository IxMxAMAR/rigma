// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import SwitchNote from "../lib/SwitchNote";
import { useApp } from "../store";
import ModelPicker from "./ModelPicker";

// W5F1a. A8c made a switch's `notice` visible on the model picker (a KV-cache
// restore it refused, a cache it stepped down). Only the pure `switchNotice`
// had an assertion, so deleting the modal's note line left the suite green.
//
//  1. `renderToStaticMarkup(<SwitchNote/>)` pins the markup and, importantly,
//     that an empty notice draws NOTHING.
//  2. The interaction test mounts the REAL picker in jsdom, clicks a model with
//     the switch mocked to answer with a notice, and reads the DOM. Deleting
//     the `<SwitchNote .../>` line fails it.
//
// No `@testing-library/react`: `react-dom/client` and jsdom are already
// dependencies, and `act` ships with react@18.3.

/** Duck-typed `Response` — only the members the code reads. */
function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

const MODELS = {
  models: [{ slug: "m", quants: [{ quant: "q4_0", on_disk: true }] }],
};

describe("SwitchNote", () => {
  it("draws the switch's sentence", () => {
    const markup = renderToStaticMarkup(
      <SwitchNote note="KV cache restored as f16" />,
    );
    expect(markup).toContain("KV cache restored as f16");
  });

  it("draws nothing when the switch had nothing to say", () => {
    expect(renderToStaticMarkup(<SwitchNote note="" />)).toBe("");
  });
});

describe("the model picker and a switch that answers with a notice", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    // No healthy engine: the picker is open, which is when a switch happens.
    useApp.setState({ surface: "chat", server: null });
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

  async function pick(switchBody: unknown) {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (String(url).startsWith("/api/models")) return reply(200, MODELS);
      if (String(url) === "/api/server/switch") return reply(200, switchBody);
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<ModelPicker />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const button = container.querySelector<HTMLButtonElement>("ul button");
    expect(button).not.toBeNull();
    await act(async () => {
      button!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("shows the notice the switch returned instead of closing silently", async () => {
    await pick({ model: "m", notice: "KV cache restored as f16" });
    expect(container.textContent).toContain("Model loaded — with a note");
    expect(container.textContent).toContain("KV cache restored as f16");
  });

  it("shows no note when the switch returned none", async () => {
    await pick({ model: "m" });
    expect(container.textContent).not.toContain("KV cache restored as f16");
    expect(container.textContent).not.toContain("with a note");
  });
});
