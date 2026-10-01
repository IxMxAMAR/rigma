import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { Budget, QuantRow } from "../lib/engineApi";
import { CtxCell, RunsCell, budgetHint } from "./ModelsSurface";

// A16b. A16 made a BROKEN fit distinguishable in the payload — only a fit whose
// arithmetic threw carries `error`; a quant the resolver genuinely refused is
// `{ok:false, speed:"no"}` and a machine with no profile at all is `{}`. The
// Models page keyed on `ok`/`speed`, so it drew the broken one exactly like a
// genuine no-fit, under a "does not fit this machine" hint. These render the
// actual cell, because the defect was in the markup and nowhere else.

function render(fit: QuantRow["fit"]): string {
  return renderToStaticMarkup(<RunsCell fit={fit} />);
}

describe("the runs cell tells a broken fit from a real one", () => {
  it("says the fit is broken, not that the model is too big", () => {
    const markup = render({
      ok: false, speed: "no", offload_pct: 100, error: "fit failed",
    });
    expect(markup).toContain("fit error");
    expect(markup).toContain("could not be computed");
    expect(markup).not.toContain("too big");
    expect(markup).not.toContain("does not fit this machine");
  });

  it("still says a genuine no-fit is too big", () => {
    const markup = render({ ok: false, speed: "no", offload_pct: 100 });
    expect(markup).toContain("too big");
    expect(markup).not.toContain("fit error");
  });

  it("keeps the existing verdicts for a fit that was computed", () => {
    expect(render({ ok: true, speed: "gpu" })).toContain("gpu");
    expect(render({ ok: true, speed: "offload", offload_pct: 60 }))
      .toContain("offload");
  });

  it("draws an empty cell when there is no fit at all (no profile)", () => {
    const markup = render({});
    expect(markup).not.toContain("too big");
    expect(markup).not.toContain("fit error");
    expect(markup).not.toContain("<span class=\"w-1.5");
  });
});

// A2b / A2d-budget. `rs_mb` (A2b) is the recurrent-state term the fit charges
// for a hybrid's SSM buffers, and `rs_unknown` (A2d-budget, merged as d22173a)
// says the geometry could not be read, so the charge is an ESTIMATE. The Models
// page printed `over_mb`/headroom with the estimate unlabelled — a bare number
// reads as a measured one, which is the silent confidence the A2 family exists
// to remove.

const budget = (extra: Partial<Budget> = {}): Budget => ({
  file_mb: 10_000, mmproj_mb: 0, kv_mb: 2_000, budget_mb: 16_000,
  over_mb: -4_000, ctx: 32768, kv_type: "f16", ...extra,
});

describe("the budget tooltip names the recurrent-state estimate", () => {
  it("labels the term an estimate when the geometry is unknown", () => {
    const hint = budgetHint({ ok: true, budget: budget({ rs_mb: 512, rs_unknown: true }) });
    expect(hint).toContain("recurrent state 512 MB");
    expect(hint).toContain("ESTIMATE, not a measurement");
    // the verdict line carries the caveat too — that is the line the page shows
    expect(hint).toContain("headroom       4,000 MB");
    expect(hint).toContain("the recurrent-state term is an estimate");
  });

  it("says an un-sizeable unknown charges 0 for lack of evidence", () => {
    const hint = budgetHint({ ok: true, budget: budget({ rs_mb: 0, rs_unknown: true }) });
    expect(hint).toContain("unknown geometry");
    expect(hint).toContain("0 is charged for lack of evidence");
    expect(hint).toContain("NOT because there is nothing to allocate");
  });

  it("leaves a measured term unlabelled", () => {
    const hint = budgetHint({ ok: true, budget: budget({ rs_mb: 512, rs_unknown: false }) });
    expect(hint).toContain("recurrent state 512 MB");
    expect(hint).not.toContain("ESTIMATE");
  });

  it("draws no recurrent line for a dense model", () => {
    const hint = budgetHint({ ok: true, budget: budget({ rs_mb: 0 }) });
    expect(hint).not.toContain("recurrent state");
    expect(hint).not.toContain("ESTIMATE");
  });

  // Wave-4 verifier's risk (verify-w5f4): `rs_unknown?` is optional, so a
  // backend that stops sending the marker must not have a charged estimate
  // silently read as a measurement. An absent key is a THIRD state, not `false`.
  it("labels a charged term whose provenance was NOT stated", () => {
    const hint = budgetHint({ ok: true, budget: budget({ rs_mb: 512 }) });
    expect(hint).toContain("recurrent state 512 MB");
    expect(hint).toContain(
      "the server did not say whether this is a measurement or an estimate");
    // it is NOT asserted to be a measurement, and NOT asserted to be an
    // estimate either — the server said nothing
    expect(hint).not.toContain("ESTIMATE, not a measurement");
    // the verdict line the page shows carries the caveat too
    expect(hint).toContain("headroom       4,000 MB");
    expect(hint).toContain("the recurrent-state term's provenance was not stated");
  });

  it("keeps the OVER line honest for an unstated provenance too", () => {
    const hint = budgetHint({
      ok: false, budget: budget({ over_mb: 1_200, rs_mb: 512 }),
    });
    expect(hint).toContain("OVER by        1,200 MB");
    expect(hint).toContain("the recurrent-state term's provenance was not stated");
    expect(hint).not.toContain("ESTIMATE, not a measurement");
  });

  it("carries the caveat on the OVER line as well", () => {
    const hint = budgetHint({
      ok: false,
      budget: budget({ over_mb: 1_200, rs_mb: 512, rs_unknown: true }),
    });
    expect(hint).toContain("OVER by        1,200 MB");
    expect(hint).toContain("the recurrent-state term is an estimate");
  });

  it("checks a broken fit FIRST, before any budget it carries", () => {
    // The wave-2 A16b nit: only `RunsCell` covered `fit.error`, so the
    // `budgetHint` branch itself could be deleted and the whole suite stayed
    // green. Assert it directly, with a budget present as well — without the
    // error-first check the function would fall through and render that
    // arithmetic as a real verdict.
    const hint = budgetHint({
      ok: false, speed: "no", offload_pct: 100, error: "fit failed",
      budget: budget({ over_mb: -4_000 }),
    });
    expect(hint).toContain("the fit could not be computed — fit failed");
    expect(hint).toContain(
      "This is NOT a verdict that the model does not fit this machine.");
    expect(hint).not.toContain("VRAM budget");     // no arithmetic was rendered
    expect(hint).not.toContain("headroom");
    expect(hint).not.toBe("does not fit this machine");
  });

  it("puts the estimate sentence in the cell's tooltip, not only in a helper", () => {
    const markup = renderToStaticMarkup(
      <CtxCell fit={{ ok: true, ctx: 32768,
                      budget: budget({ rs_mb: 512, rs_unknown: true }) }} />);
    expect(markup).toContain("ESTIMATE");
    expect(markup).toContain("32K");
  });

  it("puts the ABSENT-provenance sentence in the cell's tooltip too", () => {
    // verify-w14a nit: the committed suite asserted the absent-key wording only
    // against `budgetHint`, while the `CtxCell` render covered only the ESTIMATE
    // case. Pin it where the user reads it — the cell's own `title`.
    const markup = renderToStaticMarkup(
      <CtxCell fit={{ ok: true, ctx: 32768,
                      budget: budget({ rs_mb: 512 }) }} />);
    expect(markup).toContain(
      "the server did not say whether this is a measurement or an estimate");
    expect(markup).not.toContain("ESTIMATE");
  });
});
