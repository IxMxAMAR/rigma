import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import AgentState from "./AgentState";
import { EMPTY_GOVERNANCE } from "./governance";

// W5F1a. The wave-1 fix B6b made ACP's `usage_update.cost` render, but only the
// PURE formatter (`costLine`, `usageLine`) had an assertion: deleting the
// component's `<p>{usageLine(usage!)}</p>` line left the whole suite green, so
// the fix could have been removed without anything noticing. This file renders
// the ACTUAL component and reads the markup, which is the only assertion that
// dies with that line.
//
// `AgentState` is a pure props component (no store, no effects), so
// `renderToStaticMarkup` — already a dependency, no jsdom needed — is enough.
// `<!-- -->` separates adjacent text nodes in the output; strip it so the
// assertions read what a user would see.
function render(usage: Record<string, unknown>): string {
  return renderToStaticMarkup(
    <AgentState
      goal={null}
      todos={[]}
      planMode={false}
      subagents={[]}
      usage={usage}
      governance={EMPTY_GOVERNANCE}
    />,
  ).replace(/<!-- -->/g, "");
}

describe("the usage panel renders what a turn cost", () => {
  it("draws the cost the store holds, currency and amount", () => {
    const markup = render({
      usedTokens: 1200,
      contextWindowTokens: 8192,
      cost: { amount: 0.0123, currency: "USD" },
    });
    expect(markup).toContain("usedTokens 1200");
    expect(markup).toContain("USD 0.0123");
  });

  it("draws no money when the backend reported no cost", () => {
    const markup = render({ usedTokens: 1200, contextWindowTokens: 8192 });
    expect(markup).toContain("usedTokens 1200");
    expect(markup).not.toContain("USD");
  });

  it("says nothing rather than guessing at a cost it cannot read", () => {
    const markup = render({ inputTokens: 1, cost: "free" });
    expect(markup).toContain("in 1");
    expect(markup).not.toContain("free");
  });
});

// B6c. `acp_commands` was a dead menu: the backend advertised commands and the
// panel drew `/name description` with no control, so a capability mcode had
// announced could not be invoked from the UI at all. The list must stay
// information on a durable turn, which has no live session to send a prompt to.
function renderCommands(onRunCommand?: (name: string) => void): string {
  return renderToStaticMarkup(
    <AgentState
      goal={null}
      todos={[]}
      planMode={false}
      subagents={[]}
      usage={null}
      governance={EMPTY_GOVERNANCE}
      acpCommands={[{ name: "compact", description: "summarise the session" }]}
      onRunCommand={onRunCommand}
    />,
  ).replace(/<!-- -->/g, "");
}

describe("an advertised ACP command", () => {
  it("is a button that runs it when the live turn supplies a handler", () => {
    const markup = renderCommands(() => {});
    expect(markup).toContain("<button");
    expect(markup).toContain("/compact");
    expect(markup).toContain("run /compact on the backend");
  });

  it("stays plain text on a turn that cannot run it", () => {
    const markup = renderCommands(undefined);
    expect(markup).toContain("/compact");
    expect(markup).not.toContain("<button");
  });

  it("draws no button for a command the server sent without a name", () => {
    // The only thing a nameless command could send is a bare "/".
    const markup = renderToStaticMarkup(
      <AgentState
        goal={null}
        todos={[]}
        planMode={false}
        subagents={[]}
        usage={null}
        governance={EMPTY_GOVERNANCE}
        acpCommands={[{ description: "no name at all" }]}
        onRunCommand={() => {}}
      />,
    );
    expect(markup).toContain("no name at all");
    expect(markup).not.toContain("<button");
  });
});
