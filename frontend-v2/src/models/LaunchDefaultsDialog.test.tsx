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

  // --- C10: batch, ubatch, ngl ---------------------------------------------

  it("offers the three C10 levers and says a larger ubatch is not free", () => {
    const markup = renderToStaticMarkup(<LaunchDefaultsForm {...base} />);
    expect(markup).toContain('aria-label="Default batch"');
    expect(markup).toContain('aria-label="Default ubatch"');
    expect(markup).toContain('aria-label="Default ngl"');
    // The whole point of the note: the physical batch buys the compute buffer,
    // so an override is a VRAM cost, not a free knob.
    expect(markup).toContain("costs VRAM");
    expect(markup).toContain("not free");
  });

  it("shows the fit's own answer, because ngl is the fit's output", () => {
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm
        {...base}
        nLayers={64}
        fit={{
          ok: true, ctx: 16384, ngl: 59, kv: "q4_0", offload_pct: 0,
          budget: { file_mb: 14200, mmproj_mb: 0, kv_mb: 800,
                    budget_mb: 15018, over_mb: -18, ctx: 16384,
                    kv_type: "q4_0" },
        }} />);
    expect(markup).toContain("places 59 of 64 layers");
    expect(markup).toContain("18 MB of headroom");
    expect(markup).toContain("clamped");
  });

  it("draws a fully-resident fit as all 64 layers, never '99 of 64'", () => {
    // C1: the fully-resident case the only committed fixture (ngl 59) never
    // reached. `ngl: 99` is the sentinel, so the markup must not carry it.
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm
        {...base}
        nLayers={64}
        fit={{
          ok: true, ctx: 16384, ngl: 99, kv: "q8_0", offload_pct: 0,
          budget: { file_mb: 14200, mmproj_mb: 0, kv_mb: 800,
                    budget_mb: 15018, over_mb: 5518, ctx: 16384,
                    kv_type: "q8_0" },
        }} />);
    expect(markup).toContain("places all 64 layers on the GPU");
    expect(markup).not.toContain("99 of 64");
    expect(markup).not.toContain("places 99");
  });

  it("prices the fit at the STORED ubatch, not the one being typed", () => {
    // C2: the fit on screen is the stored one, so a draft ubatch must not
    // relabel it. Before the fix this said "stored ubatch (8192)" the moment
    // the user typed, while the fit was still priced at the stored 2048.
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm
        {...base}
        initial={{ ...STORED.launch, ubatch: 2048 }}
        draft={{ ...draftFromLaunch(STORED.launch), ubatch: "8192" }} />);
    expect(markup).toContain("stored ubatch (2048)");
    expect(markup).not.toContain("stored ubatch (8192)");
  });

  it("says the SERVER refuses the batch pair, not the engine", () => {
    // C3: b9867 clamps `n_ubatch` to `n_batch`; the 400 comes from Rigma's own
    // route (hangar.set_launch_defaults -> HangarError). The old note claimed
    // the engine refuses, which is the backend's own wrong provenance.
    const markup = renderToStaticMarkup(<LaunchDefaultsForm {...base} />);
    expect(markup).toContain("refused by the SERVER with a 400");
    expect(markup).not.toContain("refused by the engine");
  });

  it("says a stored context below the 2048 floor is raised at launch", () => {
    // C6: `server_ops.perform_switch` does `max(2048, min(int(ctx), native))`,
    // so a stored value below the floor does not run either — and said nothing
    // before this note. Mirrors the above-window note.
    const markup = renderToStaticMarkup(
      <LaunchDefaultsForm
        {...base}
        draft={{ ...draftFromLaunch(STORED.launch), ctx: "1024" }} />);
    expect(markup).toContain("below the launch floor 2048");
    expect(markup).toContain("a launch raises it to 2048");
    expect(markup).not.toContain("above m's native window");
  });

  it("says nothing about a fit it does not have", () => {
    const markup = renderToStaticMarkup(<LaunchDefaultsForm {...base} />);
    expect(markup).not.toContain("layers on the GPU");
  });
});

describe("the launch-defaults dialog against the real route", () => {
  let container: HTMLDivElement;
  let root: Root;
  let posts: unknown[];

  function serve(post: () => Response, stored: unknown = STORED) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u === "/api/models/m/defaults" && init?.method === "POST") {
        posts.push(JSON.parse(String(init.body)));
        return post();
      }
      if (u === "/api/models/m/defaults") return reply(200, stored);
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

  /** A controlled number input only sees a change through the native setter. */
  async function type(label: string, value: string) {
    const inp = container.querySelector<HTMLInputElement>(
      `[aria-label="${label}"]`);
    expect(inp, `no input labelled ${label}`).not.toBeNull();
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(inp, value);
      inp!.dispatchEvent(new Event("input", { bubbles: true }));
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

  it("cannot save an unchanged draft, so the empty payload is unreachable", async () => {
    // The reason `save()` carries no empty-payload branch: the button is
    // disabled on `!changed`, and `changed` is exactly "the payload is
    // non-empty". Pin that guard so the removed branch cannot come back to life
    // as a silent no-op save.
    serve(() => reply(200, {}));
    await mount();
    const save = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "save as default") as HTMLButtonElement;
    expect(save).not.toBeUndefined();
    expect(save.disabled).toBe(true);
    expect(container.textContent).toContain("nothing changed yet");
    await act(async () => {
      save.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(posts).toEqual([]);
  });

  // --- DR2-4: never invent a native context --------------------------------
  //
  // The state the pre-fix tests never reached: `models()` AND `server()` both
  // fail, so the dialog cannot know the real window. `nativeCtx` then fell back
  // to 262144 and a 32K-native model was offered 64K/128K/256K — the value was
  // stored, and `server_ops.perform_switch` clamped it at the next launch with
  // nothing said. `modelDefaults()` must still succeed: the dialog is openable.

  it("offers no context step when the model's native window is unknown", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (String(url) === "/api/models/m/defaults") {
        return reply(200, { ...STORED, launch: null, first_load: true });
      }
      throw new Error("down");     // both /api/models and /api/server
    }));
    await mount();
    const sel = container.querySelector<HTMLSelectElement>(
      '[aria-label="Default context"]');
    expect(sel, "the context control must still be drawn").not.toBeNull();
    expect([...sel!.querySelectorAll("option")].map((o) => o.textContent))
      .toEqual(["model default"]);
    for (const invented of ["64K", "128K", "256K"]) {
      expect(container.textContent).not.toContain(invented);
    }
    expect(container.textContent).toContain("native context window is unknown");
    // …and it says WHICH choice it made: no steps, only the resolver's default.
    expect(container.textContent).toContain('only "model default"');
  });

  it("keeps a stored context visible when the window is unknown, so it can be cleared", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u === "/api/models/m/defaults" && init?.method === "POST") {
        posts.push(JSON.parse(String(init.body)));
        return reply(200, { slug: "m", launch: null });
      }
      if (u === "/api/models/m/defaults") return reply(200, STORED);
      throw new Error("down");
    }));
    await mount();
    const sel = container.querySelector<HTMLSelectElement>(
      '[aria-label="Default context"]')!;
    expect([...sel.querySelectorAll("option")].map((o) => o.textContent))
      .toEqual(["model default", "8K"]);
    await choose("Default context", "");
    await clickSave();
    expect(posts).toEqual([{ ctx: null }]);
  });

  it("offers no context step above a KNOWN native window", async () => {
    serve(() => reply(200, {}));
    await mount();
    expect(container.textContent).toContain("32K");
    expect(container.textContent).not.toContain("64K");
    expect(container.textContent).not.toContain("256K");
    expect(container.textContent)
      .not.toContain("native context window is unknown");
  });

  it("says a stored context above the model's window will be clamped", async () => {
    // The value is legal to store (the route checks `kv` only) but the launch
    // path does `min(int(ctx), native_ctx)`. Surface that where the value is
    // chosen rather than letting the clamp happen silently.
    serve(() => reply(200, {}), {
      ...STORED,
      launch: { ...STORED.launch, ctx: 262144 },
    });
    await mount();
    expect(container.textContent).toContain("262144");
    expect(container.textContent).toMatch(/clamps it to 32768/);
  });

  // --- C10: the server's own 400, verbatim ---------------------------------

  it("draws the server's own 400 for a batch pair the server refuses", async () => {
    // C5: this is the REAL body, verbatim — pydantic's `"Value error, "` prefix
    // included (verified against `LaunchDefaults(batch=4096, ubatch=8192)`,
    // whose validator message `hangar.py:1166` joins into the 400). The old
    // fixture was a hand-written paraphrase without the prefix, so the test
    // would have passed even if the dialog stripped the prefix on the way in.
    const REFUSAL_400 = "Value error, ubatch 8192 exceeds batch 4096: "
      + "llama.cpp refuses to start when the physical batch is larger than "
      + "the logical batch";
    serve(() => reply(400, { error: REFUSAL_400 }));
    await mount();
    await type("Default batch", "4096");
    await type("Default ubatch", "8192");
    await clickSave();
    expect(posts).toEqual([{ batch: 4096, ubatch: 8192 }]);
    expect(container.textContent).toContain(REFUSAL_400);
  });

  it("sends the C10 levers the user set and nothing else", async () => {
    serve(() => reply(200, { slug: "m", launch: null }));
    await mount();
    await type("Default ubatch", "2048");
    await type("Default ngl", "40");
    await clickSave();
    expect(posts).toEqual([{ ubatch: 2048, ngl: 40 }]);
  });
});
