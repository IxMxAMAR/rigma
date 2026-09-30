import { afterEach, describe, expect, it, vi } from "vitest";

import {
  MAX_IDLE_MINUTES,
  loadSettings,
  minutesFromInput,
  saveSettings,
  settingsBody,
} from "./appSettings";

// D4a. `/api/settings` had no UI and no client at all, so the shape guards and
// the two requests are new code and get the same treatment as every other
// transport in this tree: a mocked fetch, and the server's own sentence kept.

const json = (body: unknown, status = 200) =>
  new Response(JSON.stringify(body), {
    status,
    headers: { "content-type": "application/json" },
  });

const GOOD = {
  settings: { idle_unload_minutes: 15 },
  idle_unload_minutes: 15,
  env_override: false,
};

describe("settingsBody", () => {
  it("reads the route's body", () => {
    expect(settingsBody(GOOD)).toEqual({
      settings: { idle_unload_minutes: 15 },
      idle_unload_minutes: 15,
      env_override: false,
    });
  });

  it("keeps the EFFECTIVE value separate from the stored one", () => {
    // RIGMA_KEEP_ALIVE_MIN in force: the input shows 15, the timeout is 30.
    const b = settingsBody({ ...GOOD, idle_unload_minutes: 30, env_override: true });
    expect(b?.settings.idle_unload_minutes).toBe(15);
    expect(b?.idle_unload_minutes).toBe(30);
    expect(b?.env_override).toBe(true);
  });

  it("treats only a real `true` as an override", () => {
    // A malformed flag must not disable the input and claim an override that is
    // not there.
    expect(settingsBody({ ...GOOD, env_override: "yes" })?.env_override).toBe(false);
    expect(settingsBody({ ...GOOD, env_override: undefined })?.env_override).toBe(false);
  });

  it("refuses a body that is not the promised shape", () => {
    expect(settingsBody(null)).toBeNull();
    expect(settingsBody([])).toBeNull();
    expect(settingsBody({ detail: "Internal Server Error" })).toBeNull();
    expect(settingsBody({ settings: {}, idle_unload_minutes: 1 })).toBeNull();
    expect(settingsBody({ settings: { idle_unload_minutes: "15" },
                          idle_unload_minutes: 15 })).toBeNull();
    expect(settingsBody({ settings: { idle_unload_minutes: 15 } })).toBeNull();
    expect(settingsBody({ settings: { idle_unload_minutes: Number.NaN },
                          idle_unload_minutes: Number.NaN })).toBeNull();
  });
});

describe("minutesFromInput", () => {
  it("accepts a whole number of minutes, including zero", () => {
    expect(minutesFromInput("15")).toBe(15);
    expect(minutesFromInput(" 0 ")).toBe(0);
    expect(minutesFromInput("1440")).toBe(MAX_IDLE_MINUTES);
    expect(minutesFromInput("0.5")).toBe(0.5);
  });

  it("refuses anything the server would reject, before the round trip", () => {
    expect(minutesFromInput("")).toBeNull();
    expect(minutesFromInput("   ")).toBeNull();
    expect(minutesFromInput("abc")).toBeNull();
    expect(minutesFromInput("-1")).toBeNull();
    expect(minutesFromInput("1441")).toBeNull();
  });
});

describe("the two settings requests", () => {
  afterEach(() => vi.unstubAllGlobals());

  it("GETs the settings and resolves the parsed body", async () => {
    const seen: string[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string) => {
      seen.push(url);
      return json(GOOD);
    }));

    const res = await loadSettings();
    expect(seen).toEqual(["/api/settings"]);
    expect(res.ok && res.body.settings.idle_unload_minutes).toBe(15);
  });

  it("POSTs only the minutes, and resolves the saved body", async () => {
    const seen: { url: string; init: RequestInit }[] = [];
    vi.stubGlobal("fetch", vi.fn(async (url: string, init: RequestInit) => {
      seen.push({ url, init });
      return json({ ...GOOD, settings: { idle_unload_minutes: 30 },
                    idle_unload_minutes: 30 });
    }));

    const res = await saveSettings(30);
    expect(seen[0].url).toBe("/api/settings");
    expect(seen[0].init.method).toBe("POST");
    expect(JSON.parse(String(seen[0].init.body)))
      .toEqual({ idle_unload_minutes: 30 });
    expect(res.ok && res.body.settings.idle_unload_minutes).toBe(30);
  });

  it("keeps the server's own refusal sentence", async () => {
    vi.stubGlobal("fetch", vi.fn(async () =>
      json({ error: "idle_unload_minutes: must be between 0 and 1440 "
                    + "(0 = never unload)" }, 400)));

    const res = await saveSettings(-1);
    expect(res.ok).toBe(false);
    expect(res.ok ? "" : res.error)
      .toBe("idle_unload_minutes: must be between 0 and 1440 (0 = never unload)");
  });

  it("says the body was unexpected rather than rendering a blank input", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => json({ detail: "boom" })));
    const res = await loadSettings();
    expect(res.ok).toBe(false);
    expect(res.ok ? "" : res.error)
      .toBe("the settings API returned an unexpected body");
  });

  it("reports an unreachable server instead of throwing", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => {
      throw new Error("network down");
    }));
    const res = await loadSettings();
    expect(res.ok).toBe(false);
    expect(res.ok ? "" : res.error).toBe("could not reach the server");
  });
});
