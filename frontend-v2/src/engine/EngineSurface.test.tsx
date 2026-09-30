// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import EngineSurface from "./EngineSurface";

// W5F1a residual. A8c made a switch's `notice` visible on the Engine page, and the
// wave-2 verifier measured that deleting the render line
// (`EngineSurface.tsx:166 {note && <SwitchNote note={note} />}`) left the whole
// suite green — because `info` is local `useState(null)` and
// `renderToStaticMarkup(<EngineSurface/>)` only ever reaches the `if (!info)`
// branch, so a static render cannot see the note at all.
//
// The jsdom pattern the wave-2 Header test used is the answer: mount the REAL
// component, let its `/api/server` poll answer, click a real "switch" button whose
// route answers with a `notice`, and read the DOM. Deleting the render line fails
// the first test and nothing else.
//
// No `@testing-library/react`: `react-dom/client`, jsdom and `act` are already
// dependencies (the same harness as App.test.tsx / ModelPicker.test.tsx).

/** A duck-typed `Response`: `json` for the JSON routes, `text` for the log tail. */
function reply(status: number, body: unknown, text = ""): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => text,
  } as unknown as Response;
}

const INFO = {
  model: "qwen-test",
  quant: "Q4_0",
  backend: "rocm",
  ctx: 8192,
  kv_cache: "f16",
  native_ctx: 32768,
  engine_version: "b9867",
  last_tg: 42.5,
  expected_tg: 40,
  started_at: 1_700_000_000,
  public_port: 8080,
  verdict: "healthy",
};

const OPTIONS = [
  { model: "other", quant: "Q4_0", ctx: 8192, backend: "rocm", reason: "on disk" },
];

/** Every route EngineSurface touches, with the switch's answer overridable. */
function serve(switchBody: unknown) {
  return vi.fn(async (url: string) => {
    const u = String(url);
    // `switch-options` also starts with `/api/server/switch`, so it is checked first.
    if (u.startsWith("/api/server/switch-options")) return reply(200, OPTIONS);
    if (u === "/api/server/switch") return reply(200, switchBody);
    if (u.startsWith("/api/server/findings")) return reply(200, { findings: [] });
    if (u.startsWith("/api/server/stats")) {
      return reply(200, { total_tokens: 0, total_turns: 0, by_model: {} });
    }
    if (u.startsWith("/api/server/log")) return reply(200, null, "");
    if (u.startsWith("/api/server")) return reply(200, INFO);
    return reply(200, {});
  });
}

describe("the Engine page and a switch that answers with a notice", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
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

  /** Mount the page, wait for its poll, then click the option's "switch" button. */
  async function switchTo(switchBody: unknown) {
    vi.stubGlobal("fetch", serve(switchBody));
    await act(async () => {
      root.render(<EngineSurface />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const button = Array.from(
      container.querySelectorAll<HTMLButtonElement>("button"),
    ).find((b) => b.textContent === "switch");
    expect(button).not.toBeNull();
    await act(async () => {
      button!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("draws the sentence the switch returned", async () => {
    await switchTo({ model: "other", notice: "KV cache restored as f16" });
    expect(container.textContent).toContain("KV cache restored as f16");
  });

  it("draws no note when the switch had nothing to say", async () => {
    await switchTo({ model: "other" });
    expect(container.textContent).not.toContain("KV cache restored as f16");
  });
});
