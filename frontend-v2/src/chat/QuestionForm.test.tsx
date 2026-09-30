// @vitest-environment jsdom
import { act } from "react";
import { createRoot, type Root } from "react-dom/client";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import AgentState from "./AgentState";
import { emptyTurn, useChat } from "./chatStore";
import {
  EMPTY_GOVERNANCE, foldApproval, isQuestion, type Governance,
} from "./governance";

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

/** B4b-schema: a schema that is NOT flat — a nested object with its own
 *  `required`, an array, and `default`s on both a top-level and a nested
 *  property. */
function nestedGov(): Governance {
  return foldApproval(EMPTY_GOVERNANCE, {
    event: "approval/asked",
    data: {
      id: "q-nest",
      kind: "question",
      question: "Configure the build?",
      schema: {
        type: "object",
        required: ["name", "config"],
        properties: {
          name: { type: "string", title: "Name", default: "release" },
          config: {
            type: "object",
            title: "Config",
            required: ["level"],
            properties: {
              level: { type: "integer", title: "Level", default: 3 },
              enabled: { type: "boolean", title: "Enabled" },
            },
          },
          tags: { type: "array", title: "Tags", items: { type: "string" } },
        },
      },
      awaiting: true,
    },
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

  /** A controlled React input only sees a change through the native setter. */
  async function type(input: HTMLInputElement, value: string) {
    const setValue = Object.getOwnPropertyDescriptor(
      HTMLInputElement.prototype, "value")!.set!;
    await act(async () => {
      setValue.call(input, value);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }

  function byLabel<T extends Element>(label: string): T {
    const el = container.querySelector<T>(`[aria-label="${label}"]`);
    expect(el).not.toBeNull();
    return el!;
  }

  function sendButton(): HTMLButtonElement {
    const b = Array.from(container.querySelectorAll("button"))
      .find((x) => x.textContent === "send answer");
    expect(b).toBeDefined();
    return b!;
  }

  it("still draws Allow/Refuse for a real permission request, and no form", async () => {
    // The regression the fix must not cause: only `kind: "question"` changes shape.
    const onAnswer = vi.fn();
    await mount(permissionGov(), onAnswer);
    expect(container.textContent).toContain("allow once");
    expect(container.textContent).toContain("refuse");
    // The permission path is untouched: no question form is drawn for it.
    expect(container.textContent).not.toContain("send answer");
    const allow = Array.from(container.querySelectorAll("button"))
      .find((b) => b.textContent === "allow once");
    await act(async () => {
      allow!.click();
    });
    expect(onAnswer).toHaveBeenCalledWith("call_1", true);
  });

  // --- B4b-schema: the form must express the WHOLE schema, not only its leaves.

  it("renders a nested object and an array, not only the top-level leaves", async () => {
    await mount(nestedGov(), () => {});
    expect(container.textContent).toContain("Configure the build?");
    expect(byLabel("Name")).toBeDefined();
    expect(byLabel("Level")).toBeDefined();
    expect(byLabel("Enabled")).toBeDefined();
    expect(byLabel("add Tags")).toBeDefined();
    // Still a question: no Allow/Refuse anywhere near it.
    expect(container.textContent).not.toContain("allow once");
    expect(container.textContent).not.toContain("refuse");
  });

  it("pre-fills the schema defaults and submits the nested object the route forwards", async () => {
    const onAnswer = vi.fn();
    await mount(nestedGov(), onAnswer);
    expect(byLabel<HTMLInputElement>("Name").value).toBe("release");
    expect(byLabel<HTMLInputElement>("Level").value).toBe("3");
    await act(async () => {
      byLabel<HTMLInputElement>("Enabled").click();
    });
    await act(async () => {
      sendButton().click();
    });
    expect(onAnswer).toHaveBeenCalledTimes(1);
    const [id, allow, answer] = onAnswer.mock.calls[0];
    // The EXACT JSON the route receives for a question. Transcript.tsx:305-306
    // POSTs `{requestId, answer}` verbatim for `answer !== undefined`; the
    // route validates it as an object against the slot's own `kind`
    // (serve.py:5657-5679) and hands it to mcode as the elicitation's content.
    expect(JSON.stringify({ requestId: id, answer })).toBe(
      '{"requestId":"q-nest","answer":{"name":"release","config":'
      + '{"level":3,"enabled":true}}}');
    expect(allow).toBe(false);
  });

  it("adds and removes array rows, and submits them as an array", async () => {
    const onAnswer = vi.fn();
    await mount(nestedGov(), onAnswer);
    await act(async () => {
      byLabel<HTMLButtonElement>("add Tags").click();
    });
    await act(async () => {
      byLabel<HTMLButtonElement>("add Tags").click();
    });
    await type(byLabel<HTMLInputElement>("Tags 1"), "alpha");
    await type(byLabel<HTMLInputElement>("Tags 2"), "beta");
    await act(async () => {
      byLabel<HTMLButtonElement>("remove Tags 2").click();
    });
    await act(async () => {
      sendButton().click();
    });
    const [, , answer] = onAnswer.mock.calls[0];
    expect(answer).toEqual({
      name: "release",
      config: { level: 3, enabled: false },
      tags: ["alpha"],
    });
  });

  it("offers no control at all once the question has expired (OD-12)", async () => {
    let g = questionGov();
    g = foldApproval(g, {
      event: "approval/decided", data: { id: "q-abc", decision: "expired" },
    });
    const onAnswer = vi.fn();
    await mount(g, onAnswer);
    expect(container.textContent).toContain("Which directory?");
    expect(container.textContent).toContain("expired");
    // A dead form is worse than none: no form, and no Allow/Refuse either.
    expect(container.textContent).not.toContain("send answer");
    expect(container.textContent).not.toContain("allow once");
  });

  it("draws an ANSWERED question as a tick, not as the failure cross (OD12-n1)", async () => {
    // The server's other question verdict. The glyph rule used to be "tick for
    // allowed-once, cross for everything else", so a question the user DID
    // answer read as a failure.
    let g = questionGov();
    g = foldApproval(g, {
      event: "approval/decided",
      data: { id: "q-abc", kind: "question", decision: "answered", outcome: "answered" },
    });
    await mount(g, vi.fn());
    expect(container.textContent).toContain("answered");
    expect(container.textContent).toContain("✓");
    expect(container.textContent).not.toContain("send answer");
  });
});

// B4b/OD-12. The route's 409 is the server's own word that it is no longer
// waiting on the request — the window closed, or it was already answered. The
// store folds it to the SAME terminal row an expiry EVENT produces, so the dead
// form stops being offered instead of merely reporting an error.
describe("a question the route refuses with 409", () => {
  afterEach(() => {
    useChat.setState({ currentId: null, streams: {}, lastError: null });
  });

  it("folds the live row terminal, so no form is offered afterwards", () => {
    const sid = "s1";
    useChat.setState({
      currentId: sid,
      streams: { [sid]: { ...emptyTurn(), governance: questionGov() } },
    });
    expect(useChat.getState().streams[sid].governance.approvals[0].outcome).toBe("");
    useChat.getState().expireQuestion("q-abc");
    const row = useChat.getState().streams[sid].governance.approvals[0];
    expect(row.outcome).toBe("expired");
    // Still the question, so its text is still the record — only the control is gone.
    expect(isQuestion(row)).toBe(true);
  });
});
