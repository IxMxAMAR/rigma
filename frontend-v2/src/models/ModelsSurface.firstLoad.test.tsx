// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCard } from "../lib/engineApi";
import ModelsSurface from "./ModelsSurface";

// D2's trigger. `GET /api/models/{slug}/defaults` answers `first_load`, and the
// server means by it exactly what it says: nothing is pinned AND no turn has
// ever finished. That is the one honest moment to ask how the model should come
// up, and the page must ask BEFORE it switches — the engine starts on the
// resolver's guess otherwise, which is the state D2 exists to remove.
//
// There are TWO trigger sites and both must be pinned: the card's own `run`
// (which switches the model, no quant) and the per-quant `load` on a row
// (`ModelsSurface.tsx` `QuantLine`, which switches that exact quant). The wave-4
// verifier found only the first one tested — disabling the row's check left the
// suite green.
//
// The negative half matters just as much: a model that is not a first load must
// switch immediately, with no extra prompt in the way.

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

describe("the first-load trigger", () => {
  let container: HTMLDivElement;
  let root: Root;
  let switches: unknown[];

  function serve(firstLoad: boolean) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u === "/api/models/m/defaults") {
        return reply(200, {
          slug: "m", launch: null, custom: false, last_used: null, first_load: firstLoad,
        });
      }
      if (u.startsWith("/api/models")) return reply(200, { models: [CARD] });
      if (u === "/api/server/switch") {
        switches.push(JSON.parse(String(init?.body)));
        return reply(200, { model: "m" });
      }
      return reply(200, {});
    }));
  }

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear();
    switches = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  async function clickRun() {
    await act(async () => {
      root.render(<ModelsSurface />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const run = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "run");
    expect(run, "no run button").not.toBeUndefined();
    await act(async () => {
      run!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  /** The row's own `load` — a different button from the card's `run`, and the
   *  second place `first_load` gates the switch. */
  async function clickLoad() {
    await act(async () => {
      root.render(<ModelsSurface />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const load = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "load");
    expect(load, "no per-quant load button").not.toBeUndefined();
    await act(async () => {
      load!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("asks before a first load instead of switching straight away", async () => {
    serve(true);
    await clickRun();
    expect(container.textContent).toContain("First load of m");
    expect(switches).toEqual([]);
  });

  it("switches a model that has been loaded before, with no dialog", async () => {
    serve(false);
    await clickRun();
    expect(container.textContent).not.toContain("First load of m");
    expect(switches).toEqual([{ model: "m" }]);
  });

  it("still loads when the dialog is dismissed with 'not now'", async () => {
    serve(true);
    await clickRun();
    const skip = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "not now");
    expect(skip).not.toBeUndefined();
    await act(async () => {
      skip!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(switches).toEqual([{ model: "m" }]);
  });

  it("asks before a per-quant load too, not only before the card's run", async () => {
    serve(true);
    await clickLoad();
    expect(container.textContent).toContain("First load of m");
    expect(switches).toEqual([]);          // the row did not switch behind the dialog
  });

  it("switches straight to that quant when it is not a first load", async () => {
    serve(false);
    await clickLoad();
    expect(container.textContent).not.toContain("First load of m");
    expect(switches).toEqual([{ model: "m", quant: "q4_0" }]);
  });

  it("loads the quant when the per-quant dialog is dismissed with 'not now'", async () => {
    serve(true);
    await clickLoad();
    const skip = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "not now");
    expect(skip).not.toBeUndefined();
    await act(async () => {
      skip!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(switches).toEqual([{ model: "m", quant: "q4_0" }]);
  });
});
