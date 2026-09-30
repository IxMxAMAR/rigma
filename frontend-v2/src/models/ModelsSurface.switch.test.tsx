// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCard } from "../lib/engineApi";
import ModelsSurface from "./ModelsSurface";

// W5F1a residual. A8c surfaces a switch's `notice` on the Models page at TWO
// call sites — the quant row (`ModelsSurface.tsx:390`, a KV-cache restore the
// load refused) and the card header (`:646`, the card-level "run"). The wave-2
// verifier deleted both and the full suite stayed green: no test imports
// `ModelsSurface` except for `RunsCell`, and both notes are local `useState`, so
// `renderToStaticMarkup` cannot reach either.
//
// These mount the REAL page in jsdom, click the REAL button, and read the DOM at
// the row and at the card. Deleting either `<SwitchNote .../>` line fails its
// test, and the two are pinned apart: the row's note must be inside the `<li>`,
// the card's must not be.
//
// No `@testing-library/react` — `react-dom/client` and jsdom are dependencies.

/** A duck-typed `Response`, as in App.test.tsx / EngineSurface.test.tsx. */
function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

const CARD: ModelCard = {
  slug: "m",
  family: "qwen",
  kind: "gguf",
  custom: false,
  capabilities: [],
  native_ctx: 32768,
  running: false,
  quants: [
    { file: "m-q4.gguf", quant: "q4_0", bytes: 1, on_disk: true, pullable: false },
  ],
};

describe("the Models page and a switch that answers with a notice", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear();
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

  /** Mount the page, wait for `/api/models`, then click the button named `label`. */
  async function click(label: string, switchBody: unknown) {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const u = String(url);
      if (u.startsWith("/api/models")) return reply(200, { models: [CARD] });
      if (u === "/api/server/switch") return reply(200, switchBody);
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<ModelsSurface />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const button = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((b) => b.textContent === label);
    expect(button, `no "${label}" button rendered`).not.toBeNull();
    await act(async () => {
      button!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("draws the quant row's notice inside that row", async () => {
    await click("load", { notice: "KV cache restored as f16" });
    const row = container.querySelector("li");
    expect(row).not.toBeNull();
    expect(row!.textContent).toContain("KV cache restored as f16");
  });

  it("draws no row note when the switch had nothing to say", async () => {
    await click("load", { model: "m" });
    expect(container.textContent).not.toContain("KV cache restored as f16");
  });

  it("draws the card's notice at card level, not on a row", async () => {
    await click("run", { notice: "KV cache restored as f16" });
    // The card-level note is a sibling of the quant table, so it must NOT be
    // inside the row's <li> — that is what keeps the two call sites distinct.
    expect(container.textContent).toContain("KV cache restored as f16");
    const row = container.querySelector("li");
    expect(row).not.toBeNull();
    expect(row!.textContent).not.toContain("KV cache restored as f16");
  });

  it("draws no card note when the switch had nothing to say", async () => {
    await click("run", { model: "m" });
    expect(container.textContent).not.toContain("KV cache restored as f16");
  });
});
