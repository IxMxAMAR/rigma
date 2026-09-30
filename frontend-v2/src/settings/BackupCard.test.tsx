// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import BackupCard from "./BackupCard";

// D4c. `/api/restore` replaces MEMORY and MERGES settings and methods, so the
// requirement is not "there is a button" — it is that the destructive step
// cannot happen in one click, and that the confirmation NAMES what is replaced
// and what is only overwritten entry by entry.
//
// These mount the real card: choosing a file shows the confirmation with the
// document's own counts, the replace button is disabled until the confirmation
// is ticked, and only then does the POST happen.

const DOC = {
  rigma_backup: 1,
  app_version: "0.1.0",
  created_at: "2026-09-30T00:00:00Z",
  settings: { idle_unload_minutes: 15 },
  methods: [{ id: "a" }, { id: "b" }],
  memory: [{ id: "1" }, { id: "2" }, { id: "3" }],
};

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

describe("the restore card", () => {
  let container: HTMLDivElement;
  let root: Root;
  let posts: unknown[];

  function serve(post: () => Response) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      if (String(url) === "/api/restore" && init?.method === "POST") {
        posts.push(JSON.parse(String(init.body)));
        return post();
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
      root.render(<BackupCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  /** A chosen file, as the card reads it: only `.text()` is used. */
  async function choose(doc: unknown) {
    const input = container.querySelector<HTMLInputElement>(
      'input[type="file"]');
    expect(input).not.toBeNull();
    Object.defineProperty(input, "files", {
      configurable: true,
      value: [{ text: async () => JSON.stringify(doc) }],
    });
    await act(async () => {
      input!.dispatchEvent(new Event("change", { bubbles: true }));
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  const byText = (label: string) =>
    [...container.querySelectorAll("button")]
      .find((b) => b.textContent === label);

  it("names what is replaced and what is merged, and will not replace it in one click", async () => {
    serve(() => reply(200, { restored: true, methods: 2,
                             memory: { before: 3, after: 3 } }));
    await mount();
    expect(byText("replace memory and merge the rest")).toBeUndefined();

    await choose(DOC);
    expect(container.textContent).toContain("1 settings key, 2 methods, 3 memory rows");
    expect(container.textContent).toContain("replaces all memory");
    // the two operations are stated separately: memory is replaced, the rest is
    // merged — the false "anything not in the file is gone" is gone with them
    expect(container.textContent).toContain("everything else is kept");
    expect(container.textContent).not.toContain("Anything not in the file is gone");
    expect(container.textContent).not.toContain("replaces the whole store");

    const replace = byText("replace memory and merge the rest") as HTMLButtonElement;
    expect(replace.disabled).toBe(true);       // one click is not enough

    await act(async () => {
      replace.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(posts).toEqual([]);                 // and it really did not post
  });

  it("posts only after the confirmation is ticked", async () => {
    serve(() => reply(200, { restored: true, methods: 2,
                             memory: { before: 3, after: 3 } }));
    await mount();
    await choose(DOC);

    const box = container.querySelector<HTMLInputElement>(
      '[aria-label="Confirm replacing all memory"]')!;
    await act(async () => { box.click(); });

    const replace = byText("replace memory and merge the rest") as HTMLButtonElement;
    expect(replace.disabled).toBe(false);
    await act(async () => {
      replace.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(posts).toEqual([DOC]);
    expect(container.textContent).toContain("restored 2 method(s)");
  });

  it("shows the server's refusal and keeps the file on screen", async () => {
    serve(() => reply(400, {
      error: "unknown backup version 9; this build restores version 1",
    }));
    await mount();
    await choose({ ...DOC, rigma_backup: 9 });

    const box = container.querySelector<HTMLInputElement>(
      '[aria-label="Confirm replacing all memory"]')!;
    await act(async () => { box.click(); });
    await act(async () => {
      (byText("replace memory and merge the rest") as HTMLButtonElement).click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });

    expect(container.textContent)
      .toContain("unknown backup version 9; this build restores version 1");
    // the confirmation is still up, so the user can pick another file
    expect(byText("replace memory and merge the rest")).not.toBeUndefined();
  });

  it("refuses a file that is not a backup before anything is sent", async () => {
    serve(() => reply(200, {}));
    await mount();
    await choose({ hello: "world" });
    expect(container.textContent).toContain("carries no `rigma_backup` version");
    expect(byText("replace memory and merge the rest")).toBeUndefined();
    expect(posts).toEqual([]);
  });
});
