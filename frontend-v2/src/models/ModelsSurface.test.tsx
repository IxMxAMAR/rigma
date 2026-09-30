import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import type { QuantRow } from "../lib/engineApi";
import { RunsCell } from "./ModelsSurface";

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
