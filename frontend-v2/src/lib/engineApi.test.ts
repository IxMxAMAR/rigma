import { afterEach, describe, expect, it, vi } from "vitest";

import { engineApi, switchNotice } from "./engineApi";

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

describe("switchNotice", () => {
  it("returns the server's sentence, trimmed", () => {
    expect(switchNotice({ notice: "  KV cache restored as f16  " }))
      .toBe("KV cache restored as f16");
  });

  it("says nothing when there is no usable notice", () => {
    expect(switchNotice({})).toBe("");
    expect(switchNotice(null)).toBe("");
    expect(switchNotice(undefined)).toBe("");
    expect(switchNotice("notice")).toBe("");
    expect(switchNotice({ notice: 5 })).toBe("");
    expect(switchNotice({ notice: "   " })).toBe("");
  });
});

describe("engineApi switch results", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("switchTo resolves the whole body, notice included", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return json({ model: "m", ctx: 8192, notice: "KV cache restored as f16" });
    }));

    const r = await engineApi.switchTo("m");
    expect(seen).toHaveLength(1);
    expect(seen[0].url).toBe("/api/server/switch");
    expect(seen[0].init.method).toBe("POST");
    expect(JSON.parse(String(seen[0].init.body))).toEqual({ model: "m" });
    expect(r.model).toBe("m");
    expect(switchNotice(r)).toBe("KV cache restored as f16");
  });

  it("switchTo carries the quant when one is named", async () => {
    const seen: RequestInit[] = [];
    vi.stubGlobal("fetch", vi.fn(async (_u: string, init: RequestInit) => {
      seen.push(init);
      return json({ model: "m" });
    }));

    await engineApi.switchTo("m", "q5_1");
    expect(JSON.parse(String(seen[0].body)))
      .toEqual({ model: "m", quant: "q5_1" });
  });

  it("relaunchWith keeps the notice the ctx route answers with", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return json({ ctx: 8192, notice: "asked for 32768, launched 8192" });
    }));

    const r = await engineApi.relaunchWith({ ctx: 32768, kv: "q5_1" });
    expect(seen[0].url).toBe("/api/server/ctx");
    expect(JSON.parse(String(seen[0].init.body)))
      .toEqual({ ctx: 32768, kv: "q5_1" });
    expect(switchNotice(r)).toBe("asked for 32768, launched 8192");
  });

  it("throws the server's sentence when the switch is refused", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      json({ error: "no such model" }, 502)));

    await expect(engineApi.switchTo("gone")).rejects.toThrow("no such model");
  });

  it("falls back to the status when a refusal carries no sentence", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      new Response(null, { status: 502 })));

    await expect(engineApi.switchTo("gone")).rejects.toThrow("server replied 502");
  });
});
