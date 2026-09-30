// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import SettingsSurface from "./SettingsSurface";

// D4a, end to end on the page. The card's markup and the transport are covered
// separately; this proves the two are WIRED — the value on screen comes from
// `GET /api/settings`, and pressing save sends `POST /api/settings` and then
// shows the value the server stored. Without this, the card could be correct and
// never mounted, which is exactly the state D4a found the route in.

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
  } as unknown as Response;
}

const SETTINGS = {
  settings: { idle_unload_minutes: 15 },
  idle_unload_minutes: 15,
  env_override: false,
};

describe("the settings page and /api/settings", () => {
  let container: HTMLDivElement;
  let root: Root;
  let posts: { url: string; body: unknown }[];

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    posts = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const path = String(url);
      if (path === "/api/settings" && init?.method === "POST") {
        const body = JSON.parse(String(init.body)) as { idle_unload_minutes: number };
        posts.push({ url: path, body });
        return reply(200, {
          settings: { idle_unload_minutes: body.idle_unload_minutes },
          idle_unload_minutes: body.idle_unload_minutes,
          env_override: false,
        });
      }
      if (path === "/api/settings") return reply(200, SETTINGS);
      if (path === "/api/presets") return reply(200, []);
      if (path === "/api/server") return reply(200, { model: "m" });
      return reply(200, {});
    }));
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

  async function settle() {
    await act(async () => {
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("loads the stored timeout and saves a new one", async () => {
    await act(async () => {
      root.render(<SettingsSurface />);
    });
    await settle();

    const input = container.querySelector<HTMLInputElement>(
      '[aria-label="Idle unload minutes"]',
    );
    expect(input).not.toBeNull();
    expect(input!.value).toBe("15");
    expect(container.textContent).toContain("in effect: 15 min");

    // A controlled React input only sees a change through the native setter.
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(input, "30");
      input!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const save = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "save");
    expect(save).toBeDefined();
    await act(async () => {
      save!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(posts).toEqual([
      { url: "/api/settings", body: { idle_unload_minutes: 30 } },
    ]);
    expect(container.textContent).toContain("in effect: 30 min");
  });

  it("shows the server's refusal and keeps the old value", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const path = String(url);
      if (path === "/api/settings" && init?.method === "POST")
        return reply(400, { error: "idle_unload_minutes: must be between 0 and "
                                  + "1440 (0 = never unload)" });
      if (path === "/api/settings") return reply(200, SETTINGS);
      if (path === "/api/presets") return reply(200, []);
      return reply(200, { model: "m" });
    }));

    await act(async () => {
      root.render(<SettingsSurface />);
    });
    await settle();

    const input = container.querySelector<HTMLInputElement>(
      '[aria-label="Idle unload minutes"]',
    )!;
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      // A value the CARD accepts, so the request is actually made and the
      // refusal comes from the server rather than from the input's own bound.
      setValue.call(input, "30");
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const save = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "save")!;
    await act(async () => {
      save.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(container.textContent).toContain("must be between 0 and 1440");
    expect(container.textContent).toContain("in effect: 15 min");
  });
});
