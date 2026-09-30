import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { IdleUnloadForm } from "./AppSettingsCard";

// D4a. The card's markup is where the decisions become visible: the value in
// force, the disabled input when the environment overrides it, and the server's
// own 400. Rendered directly so no store or event is needed — `renderToStatic-
// Markup` is already a dependency.

function render(over: Partial<Parameters<typeof IdleUnloadForm>[0]> = {}): string {
  return renderToStaticMarkup(
    <IdleUnloadForm
      draft="15"
      effective={15}
      envOverride={false}
      loading={false}
      loadError={null}
      busy={false}
      canSave
      saved={false}
      saveError={null}
      onDraft={() => {}}
      onSave={() => {}}
      onRetry={() => {}}
      {...over}
    />,
  ).replace(/<!-- -->/g, "");
}

describe("the app settings card", () => {
  it("shows the stored minutes and the value actually in force", () => {
    const markup = render();
    expect(markup).toContain('aria-label="Idle unload minutes"');
    expect(markup).toContain('value="15"');
    expect(markup).toContain("in effect: 15 min");
    expect(markup).toContain("save");
  });

  it("shows the effective value when the environment overrides the setting", () => {
    const markup = render({ draft: "15", effective: 30, envOverride: true });
    expect(markup).toContain("in effect: 30 min");
    // The reason is stated, because a control that changes nothing reads as broken.
    expect(markup).toContain("RIGMA_KEEP_ALIVE_MIN is set");
    expect(markup).toContain("disabled");
  });

  it("keeps the input enabled when nothing overrides it", () => {
    expect(render()).not.toContain("RIGMA_KEEP_ALIVE_MIN");
    expect(render()).not.toContain('disabled=""');
  });

  it("shows the server's refusal as an alert", () => {
    const markup = render({
      saveError: "idle_unload_minutes: must be between 0 and 1440 (0 = never unload)",
    });
    expect(markup).toContain('role="alert"');
    expect(markup).toContain("must be between 0 and 1440");
  });

  it("does not draw the form while it is still loading", () => {
    const markup = render({ loading: true, draft: "" });
    expect(markup).toContain("loading…");
    expect(markup).not.toContain("Idle unload minutes");
  });

  it("offers a retry when the load failed, instead of an empty form", () => {
    const markup = render({ loadError: "server replied 500", loading: false });
    expect(markup).toContain("could not load settings: server replied 500");
    expect(markup).toContain("retry");
    expect(markup).not.toContain("Idle unload minutes");
  });

  it("disables save when the value cannot be sent", () => {
    expect(render({ canSave: false })).toContain("disabled");
  });
});
