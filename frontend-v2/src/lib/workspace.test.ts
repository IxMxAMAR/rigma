import { afterEach, describe, expect, it, vi } from "vitest";

import { saveWorkspace } from "./workspace";

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
