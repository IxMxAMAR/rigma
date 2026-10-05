import { afterEach, describe, expect, it, vi } from "vitest";

import { pickWorkspace, saveWorkspace } from "./workspace";

const json = (body: unknown, status: number) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

describe("saveWorkspace", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("POSTs the path to the session and reports success", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return json({ id: "s1" }, 200);
    }));

    expect(await saveWorkspace("s1", "D:\\Writing")).toEqual({ ok: true });
    expect(seen).toHaveLength(1);
    expect(seen[0].url).toBe("/api/sessions/s1");
    expect(seen[0].init.method).toBe("POST");
    expect(JSON.parse(String(seen[0].init.body)))
      .toEqual({ workspace: "D:\\Writing" });
  });

  it("returns the server's own sentence for a refused save", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      json({ error: "workspace is not a directory" }, 500)));

    expect(await saveWorkspace("s1", "D:\\nope")).toEqual({
      ok: false,
      error: "workspace is not a directory",
    });
  });

  it("reads a bodiless refusal rather than resolving", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(null, { status: 500 })));

    expect(await saveWorkspace("s1", "D:\\nope")).toEqual({
      ok: false,
      error: "server replied 500",
    });
  });

  it("reports an unreachable server instead of throwing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }));

    expect(await saveWorkspace("s1", "D:\\Writing")).toEqual({
      ok: false,
      error: "could not reach the server",
    });
  });
});

describe("pickWorkspace", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("asks the picker route and returns the chosen folder", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return json({ path: "D:\\Writing" }, 200);
    }));

    expect(await pickWorkspace("s1", "C:\\start")).toEqual({
      ok: true,
      path: "D:\\Writing",
    });
    expect(seen[0].url).toBe("/api/sessions/s1/workspace/pick");
    expect(JSON.parse(String(seen[0].init.body)))
      .toEqual({ initial: "C:\\start" });
  });

  it("treats a 204 as a cancel, not as an error", async () => {
    // Closing the dialog is a normal outcome: the caller must be able to tell
    // it apart from "no dialog could be shown", which is the 503 below.
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(null, { status: 204 })));

    expect(await pickWorkspace("s1")).toEqual({ ok: true, path: "" });
  });

  it("surfaces the server's reason when no dialog could be shown", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      json({ error: "the server has no interactive desktop" }, 503)));

    expect(await pickWorkspace("s1")).toEqual({
      ok: false,
      error: "the server has no interactive desktop",
    });
  });

  it("reports an unreachable server instead of throwing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new TypeError("Failed to fetch");
    }));

    expect(await pickWorkspace("s1")).toEqual({
      ok: false,
      error: "could not reach the server",
    });
  });
});
