// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentState from "./AgentState";
import { EMPTY_GOVERNANCE, foldApproval, type Governance } from "./governance";

// B4b. The wave-2 verifier's finding: `foldApproval` ignored `data.kind`, so
// mcode's `ask_user` fell through to `GovernanceBlock`'s Allow-once/Refuse card.
// Clicking either POSTs `{allow}` and the route answers 409 — dead controls on a
// turn that is genuinely blocked. This mounts the REAL `AgentState` (the same
// component the live transcript draws) over a governance trail folded from the
// server's own event, and pins both the markup and the submitted object.
//
// No `@testing-library/react`: `react-dom/client`, jsdom and `act` are already
// dependencies (the harness App.test.tsx uses).

/** The event `serve.py`'s `_answer_question` publishes, folded as the store does. */
function questionGov(over: Record<string, unknown> = {}): Governance {
  return foldApproval(EMPTY_GOVERNANCE, {
    event: "approval/asked",
    data: {
      id: "q-abc",
      kind: "question",
      question: "Which directory?",
      schema: {
        type: "object",
        required: ["path"],
        properties: {
          path: { type: "string", title: "Directory" },
          deep: { type: "boolean" },
        },
      },
      reason: "mcode is asking a question; this turn is waiting for your answer",
      awaiting: true,
      ...over,
    },
  });
}

function permissionGov(): Governance {
  return foldApproval(EMPTY_GOVERNANCE, {
    event: "approval/asked",
    data: { id: "call_1", toolName: "bash", awaiting: true },
  });
}

describe("an elicitation the turn is blocked on", () => {
  let container: HTMLDivElement;
  let root: Root;

  beforeEach(() => {
    (globalThis as unknown as { IS_REACT_ACT_ENVIRONMENT: boolean })
      .IS_REACT_ACT_ENVIRONMENT = true;
    container = document.createElement("div");
    document.body.appendChild(container);
    root = createRoot(container);
  });

  afterEach(async () => {
    await act(async () => {
      root.unmount();
    });
    container.remove();
  });

  async function mount(gov: Governance,
                      onAnswer?: (id: string, allow: boolean,
                                  answer?: Record<string, unknown>) => void) {
    await act(async () => {
      root.render(
        <AgentState
          goal={null}
          todos={[]}
          planMode={false}
          subagents={[]}
          usage={null}
          governance={gov}
          onAnswerApproval={onAnswer}
        />,
      );
    });
  }

  it("draws the question and its form, and no Allow/Refuse card", async () => {
    await mount(questionGov(), () => {});
    expect(container.textContent).toContain("Which directory?");
    expect(container.textContent).toContain("Directory");
    expect(container.textContent).toContain("send answer");
    // The dead controls must be gone: their POST is answered 409 for a question.
    expect(container.textContent).not.toContain("allow once");
    expect(container.textContent).not.toContain("refuse");
  });

  it("submits the answer object the route accepts, not an `allow`", async () => {
    const onAnswer = vi.fn();
    await mount(questionGov(), onAnswer);

    const path = container.querySelector<HTMLInputElement>(
      '[aria-label="Directory"]',
    );
    expect(path).not.toBeNull();
    // A controlled React input only sees a change through the native setter.
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(path, "C:/work");
      path!.dispatchEvent(new Event("input", { bubbles: true }));
    });

    const deep = container.querySelector<HTMLInputElement>('[aria-label="deep"]');
    expect(deep).not.toBeNull();
    await act(async () => {
      deep!.click();
    });

    const send = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent === "send answer");
    expect(send).toBeDefined();
    await act(async () => {
      send!.click();
    });

    expect(onAnswer).toHaveBeenCalledTimes(1);
    expect(onAnswer).toHaveBeenCalledWith("q-abc", false,
                                          { path: "C:/work", deep: true });
  });

  it("will not submit while a required field is empty", async () => {
    const onAnswer = vi.fn();
    await mount(questionGov(), onAnswer);
    const send = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent === "send answer");
    expect(send!.disabled).toBe(true);
    await act(async () => {
      send!.click();
    });
    expect(onAnswer).not.toHaveBeenCalled();
  });

  it("shows a question with no answer path as text, not as a dead button", async () => {
    // The durable panel passes no handler; a question there is history.
    await mount(questionGov(), undefined);
    expect(container.textContent).toContain("Which directory?");
    expect(container.textContent).not.toContain("send answer");
    expect(container.textContent).not.toContain("allow once");
  });

  it("still draws Allow/Refuse for a real permission request", async () => {
    // The regression the fix must not cause: only `kind: "question"` changes shape.
    const onAnswer = vi.fn();
    await mount(permissionGov(), onAnswer);
    expect(container.textContent).toContain("allow once");
    expect(container.textContent).toContain("refuse");
    const allow = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent === "allow once");
    await act(async () => {
      allow!.click();
    });
    expect(onAnswer).toHaveBeenCalledWith("call_1", true);
  });
});
