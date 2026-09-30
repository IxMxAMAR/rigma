import { describe, expect, it } from "vitest";

import {
  defaultsChanged,
  defaultsPayload,
  draftFromLaunch,
  fitAnswer,
  fitForQuant,
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
      .toEqual({ quant: "", ctx: "", kv: "", vision: "keep", backend: "",
                 batch: "", ubatch: "", ngl: "" });
  });

  it("keeps an explicit vision off distinct from the tri-state null", () => {
    expect(draftFromLaunch({ vision: false }).vision).toBe("off");
    expect(draftFromLaunch({ vision: true }).vision).toBe("on");
    expect(draftFromLaunch({}).vision).toBe("keep");
  });

  it("reads ngl 0 as a real request and -1 as the sentinel", () => {
    // `-ngl 0` is every layer on the CPU — a real answer. -1 is "no opinion".
    expect(draftFromLaunch({ ngl: 0 }).ngl).toBe("0");
    expect(draftFromLaunch({ ngl: -1 }).ngl).toBe("");
    expect(draftFromLaunch({ ngl: 40 }).ngl).toBe("40");
  });
});

// C10. The three new levers have the same two rules as every other field —
// a cleared value is `null`, an unchanged one is omitted — plus one of their
// own: a stored spec that predates C10 has NO key, which is the sentinel, so a
// blank input must send nothing rather than a clear.
describe("the C10 payload: batch, ubatch and ngl", () => {
  it("sends nothing for a pre-C10 stored spec with the fields left blank", () => {
    const draft = draftFromLaunch(STORED);      // STORED has no batch keys
    expect(defaultsPayload(draft, STORED)).toEqual({});
  });

  it("sends the changed sizes as numbers, and the untouched one not at all", () => {
    const draft = { ...draftFromLaunch(STORED), batch: "8192", ubatch: "2048" };
    expect(defaultsPayload(draft, STORED)).toEqual({ batch: 8192, ubatch: 2048 });
  });

  it("clears a stored size with null, not with 0", () => {
    const stored = { ...STORED, batch: 8192, ubatch: 2048, ngl: 40 };
    const draft = { ...draftFromLaunch(stored), batch: "", ubatch: "", ngl: "" };
    expect(defaultsPayload(draft, stored)).toEqual({
      batch: null, ubatch: null, ngl: null,
    });
  });

  it("omits an unparseable size instead of sending it as a clear", () => {
    const stored = { ...STORED, batch: 8192 };
    const draft = { ...draftFromLaunch(stored), batch: "abc" };
    expect(defaultsPayload(draft, stored)).toEqual({});
  });

  it("omits a negative ngl rather than sending it", () => {
    // -1 is the sentinel, so the way to express it is a blank field.
    const draft = { ...draftFromLaunch(STORED), ngl: "-1" };
    expect(defaultsPayload(draft, STORED)).toEqual({});
  });
});

describe("the fit the dialog must show", () => {
  const FIT = {
    ok: true, ctx: 16384, ngl: 59, kv: "q4_0", offload_pct: 0,
    budget: { file_mb: 14200, mmproj_mb: 0, kv_mb: 800, budget_mb: 15018,
              over_mb: -18, ctx: 16384, kv_type: "q4_0" },
  };

  it("names the fit's own ngl and says an override is a cap, not a pin", () => {
    const line = fitAnswer(FIT, 64);
    expect(line).toContain("places 59 of 64 layers");
    expect(line).toContain("ngl is the fit's output");
    expect(line).toContain("clamped");
  });

  it("reads a fully-resident fit's ngl 99 as every layer, not 99 of 64", () => {
    // C1. `ComboFlags.ngl` defaults to 99 (models.py:415) and 99 is the
    // "all layers" SENTINEL (resolve.py:825; `Fit.ngl`'s own doc says
    // "99 = all"): the dense branch that fits fully returns `ComboFlags(ctx=…)`
    // with no `ngl` at all (resolve.py:768). The only committed fit fixture
    // was the partial 59, so the best case — the model fits entirely — rendered
    // "the fit places 99 of 64 layers on the GPU".
    const line = fitAnswer({ ...FIT, ngl: 99, offload_pct: 0 }, 64);
    expect(line).toContain("places all 64 layers on the GPU");
    expect(line).not.toContain("99");
  });

  it("says 'every layer' for the sentinel when the layer count is unknown", () => {
    // The dialog always passes `nLayers`, but the helper is total: with no
    // count there is still no literal 99 to print.
    const line = fitAnswer({ ...FIT, ngl: 99 }, undefined);
    expect(line).toContain("places every layer on the GPU");
    expect(line).not.toContain("99");
  });

  it("says nothing at all when there is no verdict to show", () => {
    expect(fitAnswer(undefined, 64)).toBe("");
    expect(fitAnswer({}, 64)).toBe("");
  });

  it("says a fit that cannot place the model cannot be overridden into one", () => {
    expect(fitAnswer({ ok: false, speed: "no" }, 64))
      .toContain("cannot make it fit");
  });

  it("picks the fit for the named quant, else the first on-disk row", () => {
    const rows = [
      { file: "a", quant: "q8_0", bytes: 1, on_disk: false, pullable: true,
        fit: { ok: true, ngl: 1 } },
      { file: "b", quant: "q4_0", bytes: 1, on_disk: true, pullable: true,
        fit: { ok: true, ngl: 2 } },
    ];
    expect(fitForQuant(rows, "q8_0")?.ngl).toBe(1);
    expect(fitForQuant(rows, "")?.ngl).toBe(2);
    expect(fitForQuant([], "q4_0")).toBeUndefined();
    expect(fitForQuant(undefined, "q4_0")).toBeUndefined();
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
