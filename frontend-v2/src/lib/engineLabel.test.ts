// R3-VLLM-4: the engine badge's decision, which was the last unwired end.
//
// Two silent failure modes, which is why this is a pure function with tests
// rather than an inline expression:
//
//   1. ASSUMING. Records written before the `engine` field existed have none, and
//      a badge that says "llamacpp" for them is asserting something nobody
//      recorded — the same class of lie as the silent vLLM fallback this finding
//      is about.
//   2. NOT FLAGGING THE CONTRADICTION. A running engine that this machine's own
//      report calls unavailable is a real inconsistency worth looking at, and
//      rendering it as ordinary hides exactly the thing that matters.
import { describe, expect, it } from "vitest";
import { engineLabel } from "./engineApi";

const avail = (engine: string, available: boolean) => ({ engine, available });

describe("engineLabel", () => {
  it("says nothing when the record does not name an engine", () => {
    expect(engineLabel({})).toBeNull();
    expect(engineLabel({ engine: null })).toBeNull();
    expect(engineLabel({ engine: "" })).toBeNull();
    expect(engineLabel({ engine: "   " })).toBeNull();
  });

  it("names the engine that was recorded", () => {
    expect(engineLabel({ engine: "vllm" })).toEqual({ name: "vllm", warn: false });
  });

  it("does not warn when the running engine is the available one", () => {
    const got = engineLabel({
      engine: "llamacpp",
      engine_runtimes: [avail("llamacpp", true), avail("vllm", false)],
    });
    expect(got).toEqual({ name: "llamacpp", warn: false });
  });

  it("warns when the running engine is reported unavailable here", () => {
    // The contradiction: something IS serving vLLM on a host whose verdict says
    // it cannot. Worth surfacing, not smoothing over.
    const got = engineLabel({
      engine: "vllm",
      engine_runtimes: [avail("llamacpp", true), avail("vllm", false)],
    });
    expect(got).toEqual({ name: "vllm", warn: true });
  });

  it("does not warn merely because the runtime list is missing", () => {
    // No list is not evidence of unavailability. Warning on absence would train
    // the user to ignore the warning.
    expect(engineLabel({ engine: "vllm" })).toEqual({ name: "vllm", warn: false });
  });

  it("does not warn when the list simply omits the running engine", () => {
    const got = engineLabel({
      engine: "vllm",
      engine_runtimes: [avail("llamacpp", true)],
    });
    expect(got).toEqual({ name: "vllm", warn: false });
  });

  it("never assumes llama.cpp", () => {
    // stated as its own case because it is the whole point
    for (const rec of [{}, { engine: null }, { engine_runtimes: [avail("vllm", true)] }]) {
      expect(engineLabel(rec)).toBeNull();
    }
  });
});
