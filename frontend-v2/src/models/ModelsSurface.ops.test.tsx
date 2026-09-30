// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCard } from "../lib/engineApi";
import ModelsSurface from "./ModelsSurface";

// D4b. `/api/models/{slug}/reprobe` and `/rename` had no caller anywhere. Both
// refuse a registry spec with a 409 and a sentence, and both are worth a UI for
// a custom model: reprobe corrects a stale geometry read, and rename carries
// the repaired template and calibration rows that a hand-rename orphans.
//
// These mount the real page and click the real controls, because the failure
// this replaces is a control that draws and does nothing.

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
  custom: true,
  capabilities: [],
  native_ctx: 32768,
  running: false,
  quants: [
    { file: "m-q4.gguf", quant: "q4_0", bytes: 1, on_disk: true, pullable: false },
  ],
};

describe("the Models page's reprobe and rename", () => {
  let container: HTMLDivElement;
  let root: Root;
  let calls: { url: string; body: unknown }[];

  function serve(reprobe: () => Response, rename: () => Response) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u === "/api/models/m/reprobe") {
        calls.push({ url: u, body: null });
        return reprobe();
      }
      if (u === "/api/models/m/rename") {
        calls.push({ url: u, body: JSON.parse(String(init?.body)) });
        return rename();
      }
      if (u.startsWith("/api/models")) return reply(200, { models: [CARD] });
      return reply(200, {});
    }));
  }

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    localStorage.clear();
    calls = [];
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  async function mount() {
    await act(async () => {
      root.render(<ModelsSurface />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  const button = (label: string) =>
    [...container.querySelectorAll("button")]
      .find((b) => b.textContent === label);

  it("reprobes the model and shows the server's refusal", async () => {
    serve(
      () => reply(409, {
        error: "nothing of m is downloaded, so there is no file to read",
      }),
      () => reply(200, {}),
    );
    await mount();
    await act(async () => {
      button("reprobe")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(calls).toEqual([{ url: "/api/models/m/reprobe", body: null }]);
    expect(container.textContent)
      .toContain("nothing of m is downloaded, so there is no file to read");
  });

  it("renames the model and posts the new name", async () => {
    serve(
      () => reply(200, {}),
      () => reply(409, { error: "a model named other already exists" }),
    );
    await mount();
    await act(async () => { button("rename")!.click(); });

    const input = container.querySelector<HTMLInputElement>(
      '[aria-label="new name for m"]');
    expect(input).not.toBeNull();
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(input, "other");
      input!.dispatchEvent(new Event("input", { bubbles: true }));
    });
    await act(async () => {
      button("save name")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(calls).toEqual([
      { url: "/api/models/m/rename", body: { slug: "other" } },
    ]);
    expect(container.textContent).toContain("a model named other already exists");
  });
});
