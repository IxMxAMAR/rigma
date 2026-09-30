import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import type { HarnessInfo } from "../lib/api";
import HarnessFacts, { type HarnessFactsProps } from "./HarnessFacts";

/** A complete `HarnessInfo`, so each case only states the field under test. */
function info(over: Partial<HarnessInfo> = {}): HarnessInfo {
  return {
    name: "dsh",
    label: "DSH",
    drives: "dsh",
    runnable: true,
    installed: true,
    needs: "",
    wire: "stdio",
    verified: "1.0.0",
    unsupported: [],
    pending: "",
    ...over,
  };
}

/** `renderToStaticMarkup` writes `<!-- -->` between adjacent text nodes; strip
 *  it so the assertions read the sentence a user would actually see. */
function text(markup: string): string {
  return markup.replace(/<!-- -->/g, "");
}

function render(props: Partial<HarnessFactsProps>): string {
  return text(
    renderToStaticMarkup(
      <HarnessFacts
        selectedHarness={
          "selectedHarness" in props ? props.selectedHarness : info()
        }
        harness={props.harness ?? "dsh"}
        drifted={"drifted" in props ? props.drifted : null}
      />,
    ),
  );
}

describe("HarnessFacts", () => {
  it("lists what Rigma gives the selected backend, with a + prefix", () => {
    const markup = render({
      selectedHarness: info({ capabilities: ["goals", "subagents"] }),
    });
    expect(markup).toContain("what Rigma gives DSH");
    expect(markup).toContain("+ goals");
    expect(markup).toContain("+ subagents");
  });

  it("renders no capabilities block when the backend advertises none", () => {
    const markup = render({ selectedHarness: info({ capabilities: [] }) });
    expect(markup).not.toContain("what Rigma gives");
  });

  it("lists what the backend does not get, with an em-dash prefix", () => {
    const markup = render({
      selectedHarness: info({ unsupported: ["plan mode"] }),
    });
    expect(markup).toContain("what DSH does not get from Rigma");
    expect(markup).toContain("— plan mode");
  });

  it("hides the unsupported list for a backend that is not runnable here", () => {
    const markup = render({
      selectedHarness: info({ installed: false, unsupported: ["plan mode"] }),
    });
    expect(markup).not.toContain("does not get from Rigma");
  });

  it("says nothing about drift when nothing drifted", () => {
    expect(render({ drifted: null })).not.toContain("adapter was measured");
    expect(render({ drifted: undefined })).not.toContain("adapter was measured");
  });

  it("names both builds when a backend drifted", () => {
    const markup = render({
      drifted: info({ version: "2.0.0", verified: "1.0.0" }),
    });
    expect(markup).toContain(
      "DSH is 2.0.0 — its adapter was measured against 1.0.0.",
    );
  });

  it("falls back to honest wording when neither build is known", () => {
    const markup = render({ drifted: info({ version: "", verified: "" }) });
    expect(markup).toContain(
      "DSH is a different build — its adapter was measured against nothing.",
    );
  });

  it("states the permission override when the backend ignores the setting", () => {
    const markup = render({
      selectedHarness: info({ honours_permission: false }),
    });
    expect(markup).toContain("sets its own permission policy");
  });

  it("keeps the permission selector claim silent when the backend honours it", () => {
    const markup = render({
      selectedHarness: info({ honours_permission: true }),
    });
    expect(markup).not.toContain("sets its own permission policy");
  });

  it("keeps the claim silent for Rigma's own loop", () => {
    const markup = render({
      harness: "native",
      selectedHarness: info({ name: "native", label: "native", honours_permission: false }),
    });
    expect(markup).not.toContain("sets its own permission policy");
  });

  it("does not crash when the stored backend is not on the menu", () => {
    const markup = render({ selectedHarness: null, harness: "gone" });
    expect(markup).not.toContain("sets its own permission policy");
  });
});
