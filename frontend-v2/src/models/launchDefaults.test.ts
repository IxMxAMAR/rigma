import { describe, expect, it } from "vitest";

import {
  defaultsChanged,
  defaultsPayload,
  draftFromLaunch,
  KV_TYPES,
} from "./launchDefaults";

// D2. The two rules that make this dialog safe to ship are pure decisions, so
// they are pinned here rather than through the DOM:
//
//   1. `POST /api/models/{slug}/defaults` reads `null` as "clear this field".
//      The stored shape round-trips a cleared field as `""`/`0`, so a payload
//      built by echoing the form's state would send a VALUE where the user
//      asked to remove one — the field would look cleared and stay pinned.
//   2. Only changed fields may be sent. The route merges the body over the
//      stored defaults, so naming a field the user never touched clears it.

const STORED = {
  quant: "q5_1", ctx: 8192, kv: "f16", vision: null as boolean | null,
  spec_type: "", spec_n_max: 0, backend: "rocm",
};

describe("the launch-defaults payload", () => {
  it("sends a cleared field as null, never as the stored sentinel", () => {
    const draft = { ...draftFromLaunch(STORED), ctx: "", kv: "", quant: "" };
    expect(defaultsPayload(draft, STORED)).toEqual({
      quant: null, ctx: null, kv: null,
    });
    // Explicitly: no empty strings and no zeroes anywhere in the body.
    const body = defaultsPayload(draft, STORED);
    for (const v of Object.values(body)) expect(v).toBeNull();
  });

  it("sends nothing when the draft equals what is stored", () => {
    const draft = draftFromLaunch(STORED);
    expect(defaultsPayload(draft, STORED)).toEqual({});
    expect(defaultsChanged(draft, STORED)).toBe(false);
  });

  it("sends only the field that changed", () => {
    const draft = { ...draftFromLaunch(STORED), ctx: "32768" };
    expect(defaultsPayload(draft, STORED)).toEqual({ ctx: 32768 });
  });

  it("carries backend, which the old one-click write silently dropped", () => {
    const draft = { ...draftFromLaunch(STORED), backend: "vulkan" };
    expect(defaultsPayload(draft, STORED)).toEqual({ backend: "vulkan" });
  });

  it("clears backend too, as null", () => {
    const draft = { ...draftFromLaunch(STORED), backend: "" };
    expect(defaultsPayload(draft, STORED)).toEqual({ backend: null });
  });

  it("keeps vision tri-state: keep / on / off are three different writes", () => {
    const keep = { ...draftFromLaunch(STORED), vision: "keep" as const };
    const on = { ...draftFromLaunch(STORED), vision: "on" as const };
    const off = { ...draftFromLaunch(STORED), vision: "off" as const };
    expect(defaultsPayload(keep, STORED)).toEqual({});
    expect(defaultsPayload(on, STORED)).toEqual({ vision: true });
    expect(defaultsPayload(off, STORED)).toEqual({ vision: false });
  });

  it("treats a missing launch as all-unset, so a first load can set anything", () => {
    const draft = { ...draftFromLaunch(null), ctx: "16384", kv: "q4_0" };
    expect(defaultsPayload(draft, null)).toEqual({ ctx: 16384, kv: "q4_0" });
  });

  it("omits a context the form cannot parse rather than sending nonsense", () => {
    const draft = { ...draftFromLaunch(STORED), ctx: "abc" };
    expect(defaultsPayload(draft, STORED)).toEqual({});
  });
});

describe("draftFromLaunch", () => {
  it("reads the stored sentinels as unset, not as values", () => {
    expect(draftFromLaunch({ quant: "", ctx: 0, kv: "", vision: null,
                             backend: "" }))
      .toEqual({ quant: "", ctx: "", kv: "", vision: "keep", backend: "" });
  });

  it("keeps an explicit vision off distinct from the tri-state null", () => {
    expect(draftFromLaunch({ vision: false }).vision).toBe("off");
    expect(draftFromLaunch({ vision: true }).vision).toBe("on");
    expect(draftFromLaunch({}).vision).toBe("keep");
  });
});

describe("the kv list is the server's own", () => {
  it("offers every type server_ops.KV_CACHE_TYPES accepts", () => {
    // models.py:20-28 — tuple(CACHE_BYTES). The Sidecar's old picker was a
    // 4-value subset, so a model pinned to q5_0 or bf16 could not be named.
    expect(KV_TYPES).toEqual(
      ["bf16", "f16", "q8_0", "q5_1", "q5_0", "q4_1", "q4_0"]);
  });
});
