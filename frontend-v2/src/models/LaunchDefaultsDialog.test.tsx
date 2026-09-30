// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { ModelCard } from "../lib/engineApi";
import LaunchDefaultsDialog, { LaunchDefaultsForm } from "./LaunchDefaultsDialog";
import { draftFromLaunch } from "./launchDefaults";

// D2. Two hard requirements live here:
//
//   1. A registry model's save must show the sentence the SERVER returns
//      (OD-11: `hangar.set_launch_defaults` refuses a `custom: false` spec).
//      The failure mode this replaces is the silent one — a dialog that catches
//      the 400 and closes as if it had saved.
//   2. A cleared field is a `null` in the body, not the `""` the stored
//      sentinel round-trips as.
//
// The pure rules are pinned in `launchDefaults.test.ts`; these prove the form
// and the container are actually wired to them.

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

const REFUSAL = "m is a registry model — its launch settings are hand-authored "
  + "and are not overwritten here";

const STORED = {
  slug: "m",
  launch: { quant: "q4_0", ctx: 8192, kv: "f16", vision: null,
            spec_type: "", spec_n_max: 0, backend: "" },
  custom: false,
  last_used: null,
  first_load: true,
};

describe("LaunchDefaultsForm markup", () => {
  const base = {
    slug: "m",
    draft: draftFromLaunch(STORED.launch),
    initial: STORED.launch,
    custom: true,
    quants: ["q4_0"],
    nativeCtx: 32768,
    hasMmproj: false,
    backends: [],
    busy: false,
    saved: false,
    error: null,
    onDraft: () => {},
    onSave: () => {},
    onClose: () => {},
  };

  it("draws the server's own refusal sentence", () => {
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm {...base} error={REFUSAL} />);
    expect(markup).toContain(REFUSAL);
    expect(markup).toContain('role="alert"');
  });

  it("says why a registry model will be refused, before the attempt", () => {
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm {...base} custom={false} />);
    expect(markup).toContain("registry model");
    expect(markup).toContain("hand-authored");
  });

  it("keeps a custom model free of the registry warning", () => {
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm {...base} custom />);
    expect(markup).not.toContain("registry model");
  });

  it("labels the first load and offers the full kv list", () => {
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm {...base} firstLoad />);
    expect(markup).toContain("First load of m");
    expect(markup).toContain("q5_0");
    expect(markup).toContain("bf16");
  });
});

describe("the launch-defaults dialog against the real route", () => {
  let container: HTMLDivElement;
  let root: Root;
  let posts: unknown[];

  function serve(post: () => Response) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u === "/api/models/m/defaults" && init?.method === "POST") {
        posts.push(JSON.parse(String(init.body)));
        return post();
      }
      if (u === "/api/models/m/defaults") return reply(200, STORED);
      if (u.startsWith("/api/models")) return reply(200, { models: [CARD] });
      if (u === "/api/server") {
        return reply(200, {
          model: "m", native_ctx: 32768,
          backends: [
            { name: "rocm", ready: true, buildable: true },
            { name: "vulkan", ready: true, buildable: true },
          ],
        });
      }
      return reply(200, {});
    }));
  }

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    posts = [];
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
      root.render(<LaunchDefaultsDialog slug="m" open firstLoad onClose={() => {}} />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  /** A controlled `<select>` only sees a change through the native setter. */
  async function choose(label: string, value: string) {
    const sel = container.querySelector<HTMLSelectElement>(
      `[aria-label="${label}"]`);
    expect(sel, `no select labelled ${label}`).not.toBeNull();
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLSelectElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(sel, value);
      sel!.dispatchEvent(new Event("change", { bubbles: true }));
    });
  }

  async function clickSave() {
    const save = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "save as default");
    expect(save, "no save button").not.toBeUndefined();
    await act(async () => {
      save!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("shows the refusal the server returns instead of closing", async () => {
    serve(() => reply(400, { error: REFUSAL }));
    await mount();
    await choose("Default context", "");
    await clickSave();
    expect(posts).toEqual([{ ctx: null }]);
    // BOTH sentences, and they are different sentences: the local note explains
    // that a registry model will be refused, the alert carries the server's own
    // wording. Asserting only "registry model" would pass off the local note.
    expect(container.textContent).toContain(REFUSAL);
    expect(container.textContent).toContain("saving here will be refused");
  });

  it("clears a field with null, not the empty string", async () => {
    serve(() => reply(200, { slug: "m", launch: null }));
    await mount();
    await choose("Default context", "");
    await choose("Default KV cache", "");
    await clickSave();
    expect(posts).toEqual([{ ctx: null, kv: null }]);
    for (const body of posts as Record<string, unknown>[]) {
      for (const v of Object.values(body)) expect(v).not.toBe("");
    }
  });

  it("offers the backend the route can now accept", async () => {
    serve(() => reply(200, { slug: "m", launch: null }));
    await mount();
    expect(container.querySelector('[aria-label="Default backend"]'))
      .not.toBeNull();
    await choose("Default backend", "vulkan");
    await clickSave();
    expect(posts).toEqual([{ backend: "vulkan" }]);
  });
});
