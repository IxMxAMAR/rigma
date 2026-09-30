// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useChat } from "./chatStore";
import { MethodCard } from "./Sidecar";

// D4d. `DELETE /api/methods/{mid}` had no caller: a method the user built could
// be applied, exported and imported, but never removed. The route answers 400
// with its own sentence for a built-in, so the UI offers no button for one and
// still shows that sentence if the server ever returns it.
//
// `GET /api/methods/{mid}` has no separate UI on purpose: the catalog GET
// already carries every method's full body. `POST /api/methods` is redundant
// with draft→promote and stays unwired (removing a route is an owner decision).

function reply(status: number, body: unknown): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    json: async () => body,
    text: async () => "",
  } as unknown as Response;
}

const USER = {
  id: "user1", name: "My method", tagline: "mine", guide: ["step one"],
  builtin: false,
};
const BUILTIN = {
  id: "usecase:writing", name: "Writing", tagline: "built-in", guide: [],
  builtin: true,
};

describe("deleting a user method", () => {
  let container: HTMLDivElement;
  let root: Root;
  let deletes: string[];

  function serve(del: () => Response) {
    vi.stubGlobal("fetch", vi.fn(async (url: string, init?: RequestInit) => {
      const u = String(url);
      if (u.startsWith("/api/methods/") && init?.method === "DELETE") {
        deletes.push(u);
        return del();
      }
      if (u === "/api/methods") return reply(200, { methods: [USER, BUILTIN] });
      return reply(200, {});
    }));
  }

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    useChat.setState({ currentId: null });
    vi.stubGlobal("confirm", vi.fn(() => true));
    deletes = [];
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
      root.render(<MethodCard />);
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
  }

  async function expand(name: string) {
    const row = [...container.querySelectorAll("button")]
      .find((b) => (b.textContent ?? "").includes(name));
    expect(row, `no row for ${name}`).not.toBeUndefined();
    await act(async () => { row!.click(); });
  }

  it("offers Delete on a user method and not on a built-in", async () => {
    serve(() => reply(200, { deleted: "user1" }));
    await mount();
    await expand("My method");
    expect([...container.querySelectorAll("button")]
      .find((b) => b.textContent === "Delete")).not.toBeUndefined();

    await expand("Writing");
    const builtinRow = [...container.querySelectorAll("li")]
      .find((li) => (li.textContent ?? "").includes("Writing"));
    expect(builtinRow!.textContent).not.toContain("Delete");
  });

  it("deletes through the route and refreshes the list", async () => {
    serve(() => reply(200, { deleted: "user1" }));
    await mount();
    await expand("My method");
    await act(async () => {
      [...container.querySelectorAll("button")]
        .find((b) => b.textContent === "Delete")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(deletes).toEqual(["/api/methods/user1"]);
  });

  it("shows the server's refusal sentence", async () => {
    serve(() => reply(400, {
      error: "built-in methods cannot be deleted -- save a user method with "
        + "the same id to override it instead",
    }));
    await mount();
    await expand("My method");
    await act(async () => {
      [...container.querySelectorAll("button")]
        .find((b) => b.textContent === "Delete")!.click();
      await new Promise((resolve) => setTimeout(resolve, 0));
    });
    expect(container.textContent)
      .toContain("built-in methods cannot be deleted");
  });
});
