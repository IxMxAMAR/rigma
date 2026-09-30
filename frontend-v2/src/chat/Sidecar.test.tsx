// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "./chatStore";
import { EngineCard, GroundingCard, SamplingCard } from "./Sidecar";

// W5F1a residual — the wave-1 D5 call site. D5 extracted the harness disclosure
// into the pure `HarnessFacts`, but its single call site (`Sidecar.tsx:903`) had
// no assertion: the wave-2 verifier deleted the line (and its import) and the full
// suite stayed green at 39 files / 509 tests. `HarnessFacts` is well covered in
// isolation; what was never covered is that the Sidecar actually draws it.
//
// `SamplingCard` owns that call and is not the default export, so it is exported
// for this test. The harness menu is mocked and the chat store is seeded with a
// backend that advertises capabilities; deleting the `<HarnessFacts .../>` line
// removes "+ goals" from the DOM and fails this test.
//
// No `@testing-library/react`: `react-dom/client`, jsdom and `act` are already
// dependencies (same harness as App.test.tsx).

/** A duck-typed `Response`, as in App.test.tsx / EngineSurface.test.tsx. */
function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

const MENU = {
  built_in: "native",
  endpoint: "",
  harnesses: [
    {
      name: "dsh",
      label: "DSH",
      drives: "dsh",
      runnable: true,
      installed: true,
      needs: "",
      wire: "stdio",
      verified: "1.0.0",
      unsupported: [],
      pending: "",
      capabilities: ["goals", "subagents"],
    },
  ],
};

describe("the Sidecar's harness card", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useChat.setState({ harness: "dsh", currentId: null });
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
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const u = String(url);
      if (u.startsWith("/api/harnesses")) return reply(200, MENU);
      if (u.startsWith("/api/presets")) return reply(200, []);
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<SamplingCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("draws the selected backend's disclosure where the picker is", async () => {
    await mount();
    expect(container.textContent).toContain("what Rigma gives DSH");
    expect(container.textContent).toContain("+ goals");
    expect(container.textContent).toContain("+ subagents");
  });

  it("keeps the card intact when the menu never arrives", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      const u = String(url);
      if (u.startsWith("/api/harnesses")) return reply(500, { error: "no menu" });
      if (u.startsWith("/api/presets")) return reply(200, []);
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<SamplingCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    // The card still renders its own heading; the disclosure has nothing to say.
    expect(container.textContent).toContain("this chat");
    expect(container.textContent).not.toContain("what Rigma gives");
  });
});

describe("the engine card's launch-defaults tooltip", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useChat.setState({ streams: {}, currentId: null });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  it("says the seed only fills fields the stored launch leaves unset", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (String(url) === "/api/server") {
        return reply(200, {
          model: "m", ctx: 8192, kv_cache: "f16", backend: "rocm",
          native_ctx: 32768, no_vision: false, has_mmproj: false,
          backends: [], unloaded: false,
        });
      }
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<EngineCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    const btn = [...container.querySelectorAll("button")]
      .find((b) => b.textContent === "launch defaults…");
    expect(btn, "no launch defaults button").not.toBeUndefined();
    const title = btn!.getAttribute("title") ?? "";
    expect(title).toContain("prefills fields the stored launch leaves UNSET");
    expect(title).toContain("never overwritten by the running config");
    // the old claim was false whenever a field was already pinned
    expect(title).not.toContain("opens on the RUNNING configuration");
  });
});

// OD-3 option 2 (accepted). The credential-`exclude` denylist applies to new
// indexing only, so an index built before it can still hold credentials. The
// owner accepted a "rebuild index" button and the decision stays theirs — the
// rebuild must never fire on its own. These pin the three things that make the
// control honest: one POST per click, and the server's 409 / 400 sentence
// reaching the screen instead of being swallowed (the `r.ok` defect this repo
// has already fixed twice).
describe("the grounding card's rebuild control", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useChat.setState({ currentId: null });
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => { root.unmount(); });
    container.remove();
    vi.unstubAllGlobals();
  });

  /** The routes the card touches on mount, plus the reindex POST. */
  function route(reindex: () => Response,
                 seen?: (url: string, method: string) => void) {
    return vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      const m = String(init?.method ?? "GET");
      seen?.(u, m);
      if (u === "/api/rag/status") {
        return reply(200, { running: true, sources: [], indexing: false, error: "" });
      }
      if (u === "/api/rag/discover") return reply(200, { candidates: [] });
      if (u === "/api/rag/reindex") return reindex();
      return reply(200, {});
    });
  }

  function button(): HTMLButtonElement {
    const b = [...container.querySelectorAll("button")]
      .find((el) => el.textContent === "rebuild index");
    expect(b, "no rebuild index button").not.toBeUndefined();
    return b as HTMLButtonElement;
  }

  async function mount() {
    await act(async () => {
      root.render(<GroundingCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    return button();
  }

  async function click(b: HTMLButtonElement) {
    await act(async () => {
      b.dispatchEvent(new MouseEvent("click", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  it("POSTs /api/rag/reindex exactly once and acknowledges the 202", async () => {
    const seen: [string, string][] = [];
    vi.stubGlobal("fetch", route(
      () => reply(202, { indexing: true }),
      (u, m) => seen.push([u, m]),
    ));
    await click(await mount());
    expect(seen.filter(([u]) => u === "/api/rag/reindex"))
      .toEqual([["/api/rag/reindex", "POST"]]);
    expect(container.textContent).toContain("rebuild started");
  });

  it("says the server's 409 sentence out loud instead of swallowing it", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(409, { error: "an ingest is already running" })));
    await click(await mount());
    const alert = container.querySelector('[role="alert"]');
    expect(alert, "the 409 was silently dropped").not.toBeNull();
    expect(alert!.textContent).toContain("an ingest is already running");
  });

  it("says the server's 400 sentence out loud too", async () => {
    vi.stubGlobal("fetch", route(
      () => reply(400, { error: "raggity is not installed" })));
    await click(await mount());
    const alert = container.querySelector('[role="alert"]');
    expect(alert, "the 400 was silently dropped").not.toBeNull();
    expect(alert!.textContent).toContain("raggity is not installed");
  });

  it("is disabled while an ingest is already running", async () => {
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      if (String(url) === "/api/rag/status") {
        return reply(200, { running: true, sources: [], indexing: true, error: "" });
      }
      return reply(200, {});
    }));
    await act(async () => {
      root.render(<GroundingCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(button().disabled).toBe(true);
  });
});
