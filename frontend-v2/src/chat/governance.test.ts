import { describe, expect, it } from "vitest";

import {
  EMPTY_GOVERNANCE,
  foldApproval,
  isQuestion,
  outcomeLabel,
  outcomeTone,
  questionAnswer,
  questionFields,
  questionReady,
  sandboxLabel,
  sandboxTone,
} from "./governance";

const ask = (id: string, toolName = "bash", reason = "") => ({
  event: "approval/asked",
  data: { id, toolName, ...(reason ? { reason } : {}) },
});
const decided = (id: string, outcome: string) => ({
  event: "approval/decided",
  data: { id, outcome },
});

// DSH records its own governance as `log-only` events: durable and replayable,
// never part of the model transcript. Rigma could not see any of them.
describe("foldApproval", () => {
  it("records an ask as pending", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, ask("a1", "write", "outside the workspace"));
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0]).toMatchObject({
      kind: "asked",
      id: "a1",
      toolName: "write",
      reason: "outside the workspace",
      outcome: "",
    });
  });

  // The whole reason `decided` is folded rather than appended: two rows per
  // decision would double the trail and separate a question from its answer.
  it("folds a decision onto its own ask instead of adding a row", () => {
    let g = foldApproval(EMPTY_GOVERNANCE, ask("a1", "write"));
    g = foldApproval(g, decided("a1", "allowed-once"));
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0]).toMatchObject({ kind: "asked", toolName: "write", outcome: "allowed-once" });
  });

  it("pairs each decision with the RIGHT ask when several are open", () => {
    let g = foldApproval(EMPTY_GOVERNANCE, ask("a1", "first"));
    g = foldApproval(g, ask("a2", "second"));
    // Out of order on purpose: the second ask is answered first.
    g = foldApproval(g, decided("a2", "rejected"));
    g = foldApproval(g, decided("a1", "allowed-once"));
    expect(g.approvals.map((a) => [a.toolName, a.outcome])).toEqual([
      ["first", "allowed-once"],
      ["second", "rejected"],
    ]);
  });

  // An ask that predates this turn has no visible question. Dropping the
  // decision would hide that a tool was refused.
  it("keeps an orphaned decision rather than discarding it", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, decided("gone", "rejected"));
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0]).toMatchObject({ kind: "decided", id: "gone", outcome: "rejected" });
  });

  it("does not overwrite an ask that is already decided", () => {
    let g = foldApproval(EMPTY_GOVERNANCE, ask("a1", "write"));
    g = foldApproval(g, decided("a1", "allowed-once"));
    g = foldApproval(g, decided("a1", "rejected"));
    // A second decision for the same id is a NEW row, not a silent rewrite: the
    // first verdict is what actually happened.
    expect(g.approvals).toHaveLength(2);
    expect(g.approvals[0].outcome).toBe("allowed-once");
    expect(g.approvals[1].outcome).toBe("rejected");
  });

  it("records a policy switch", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/policy", data: { policy: "ask", source: "delegation" },
    });
    expect(g.approvals[0]).toMatchObject({ kind: "policy", policy: "ask" });
  });

  // An audit trail must not silently swallow an event it does not recognise.
  it("keeps an unknown approval kind", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, { event: "approval/something-new", data: { id: "x" } });
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0].kind).toBe("something-new");
  });

  it("ignores a payload that is not an approval", () => {
    expect(foldApproval(EMPTY_GOVERNANCE, null)).toBe(EMPTY_GOVERNANCE);
    expect(foldApproval(EMPTY_GOVERNANCE, {})).toBe(EMPTY_GOVERNANCE);
    expect(foldApproval(EMPTY_GOVERNANCE, { event: "goal", data: {} })).toBe(EMPTY_GOVERNANCE);
  });

  it("never mutates the governance it was given", () => {
    const before = foldApproval(EMPTY_GOVERNANCE, ask("a1", "write"));
    const snapshot = JSON.parse(JSON.stringify(before));
    foldApproval(before, decided("a1", "rejected"));
    expect(before).toEqual(snapshot);
  });
});

// B4. mcode's `ask_user` reaches the UI on the SAME `approval/asked` channel as a
// permission request, distinguished only by `data.kind`. `foldApproval` ignored it,
// so the row fell through to the Allow-once/Refuse card — whose POST the route
// answers 409 — and the turn stayed blocked until the 5 s decline.
describe("foldApproval: an elicitation is not a permission", () => {
  const question = (over: Record<string, unknown> = {}) => ({
    event: "approval/asked",
    data: {
      id: "q-abc",
      kind: "question",
      question: "Which directory?",
      schema: { type: "object", properties: { path: { type: "string" } } },
      reason: "mcode is asking a question; this turn is waiting for your answer",
      awaiting: true,
      ...over,
    },
  });

  it("carries kind, question and schema, not only the permission fields", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, question());
    expect(g.approvals).toHaveLength(1);
    expect(isQuestion(g.approvals[0])).toBe(true);
    expect(g.approvals[0].question).toBe("Which directory?");
    expect(g.approvals[0].schema).toEqual({
      type: "object", properties: { path: { type: "string" } },
    });
    expect(g.approvals[0].awaiting).toBe(true);
  });

  it("does not call a permission request a question", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, ask("call_1", "bash"));
    expect(isQuestion(g.approvals[0])).toBe(false);
    expect(g.approvals[0].question).toBe("");
    expect(g.approvals[0].schema).toEqual({});
  });

  it("treats a malformed schema as no form rather than crashing the fold", () => {
    for (const schema of [null, "nope", 7, []]) {
      const g = foldApproval(EMPTY_GOVERNANCE, question({ schema }));
      expect(g.approvals[0].schema).toEqual({});
    }
  });
});

describe("the elicitation form the schema describes", () => {
  it("reads the schema's own property names and titles", () => {
    const fields = questionFields({
      type: "object",
      required: ["path"],
      properties: {
        path: { type: "string", title: "Directory", description: "where to look" },
        deep: { type: "boolean" },
        mode: { enum: ["fast", "careful"] },
      },
    });
    expect(fields.map((f) => f.name)).toEqual(["path", "deep", "mode"]);
    expect(fields[0]).toMatchObject({
      label: "Directory", kind: "text", required: true, description: "where to look",
    });
    expect(fields[1].kind).toBe("boolean");
    expect(fields[2].kind).toBe("choice");
    expect(fields[2].choices).toEqual(["fast", "careful"]);
  });

  it("reads a closed set spelled as oneOf as well as enum", () => {
    const fields = questionFields({
      properties: { mode: { oneOf: [{ const: "a" }, { const: "b" }] } },
    });
    expect(fields[0].kind).toBe("choice");
    expect(fields[0].choices).toEqual(["a", "b"]);
  });

  it("invents no field when the server sent no schema", () => {
    for (const s of [null, {}, "nope", { properties: null }, { properties: [] }]) {
      expect(questionFields(s)).toEqual([]);
    }
  });

  it("builds the answer object the route expects, with no blank keys", () => {
    const fields = questionFields({
      required: ["path"],
      properties: {
        path: { type: "string" },
        deep: { type: "boolean" },
        note: { type: "string" },
      },
    });
    expect(questionAnswer(fields, { path: "  C:/work ", deep: true }))
      .toEqual({ path: "C:/work", deep: true });
    // A checkbox is a definite answer; a blank optional text is absent.
    expect(questionAnswer(fields, { path: "x", deep: false }))
      .toEqual({ path: "x", deep: false });
    expect(questionAnswer(fields, { deep: false })).toEqual({ deep: false });
  });

  it("blocks submit only on a required field that is empty", () => {
    const fields = questionFields({
      required: ["path"],
      properties: { path: { type: "string" }, note: { type: "string" } },
    });
    expect(questionReady(fields, {})).toBe(false);
    expect(questionReady(fields, { note: "hi" })).toBe(false);
    expect(questionReady(fields, { path: "C:/work" })).toBe(true);
    // No fields at all is ready: the answer is legitimately empty.
    expect(questionReady([], {})).toBe(true);
  });
});

describe("outcomeLabel", () => {
  it("spells out the verdict rather than showing the identifier", () => {
    expect(outcomeLabel("allowed-once")).toBe("allowed once");
    expect(outcomeLabel("rejected")).toBe("rejected");
  });

  // DSH's fail-closed outcome. "unavailable" alone reads like a network blip;
  // it actually means the tool was NOT permitted.
  it("makes the fail-closed outcome unmistakable", () => {
    expect(outcomeLabel("unavailable")).toContain("not permitted");
  });

  it("shows an unknown outcome rather than hiding it", () => {
    expect(outcomeLabel("new-verdict")).toBe("new-verdict");
  });
});

describe("outcomeTone", () => {
  it("marks only an allow as amber, and refusals as red", () => {
    expect(outcomeTone("allowed-once")).toContain("amber");
    expect(outcomeTone("rejected")).toContain("red");
    expect(outcomeTone("unavailable")).toContain("red");
  });
});

describe("sandboxLabel", () => {
  it("says what the mode MEANS, not just its identifier", () => {
    expect(sandboxLabel("workspace-write")).toContain("workspace");
    expect(sandboxLabel("read-only")).toBe("read-only");
  });

  // The one mode a reader must not skim past.
  it("does not let full access read as neutral", () => {
    expect(sandboxLabel("danger-full-access")).toContain("unrestricted");
  });

  it("shows an unknown mode rather than hiding it", () => {
    expect(sandboxLabel("something-new")).toBe("something-new");
  });
});

describe("sandboxTone", () => {
  it("grades the three real modes by how much they permit", () => {
    expect(sandboxTone("read-only")).toContain("moss");
    expect(sandboxTone("workspace-write")).toContain("amber");
    expect(sandboxTone("danger-full-access")).toContain("red");
  });

  it("is neutral for an unknown mode", () => {
    expect(sandboxTone("something-new")).toContain("muted");
  });
});

// R6-ACP: an mcode ACP permission decision reaches this SAME fold.
//
// `drive_turn_acp` translates ACP's permission answer into `approval/asked` +
// `approval/decided` rather than inventing a second vocabulary, which is why
// answering a permission prompt needed no new UI. These tests pin the payload shape
// the Python side actually emits — an `asked` with a title and no outcome, then a
// `decided` carrying only the id and the outcome — so a change on either side of the
// seam is caught here rather than showing as a blank row in the panel.
describe("R6-ACP: mcode's permission decision on the shared trail", () => {
  const acpAsked = (id: string, toolName: string) => ({
    event: "approval/asked",
    data: { id, toolName, reason: "decided by Rigma's policy", auto: false },
  });
  const acpDecided = (id: string, outcome: string) => ({
    event: "approval/decided",
    data: { id, outcome, optionId: "allow-once", policy: "auto" },
  });

  it("pairs an ACP ask with its decision instead of adding two rows", () => {
    let g = foldApproval(EMPTY_GOVERNANCE, acpAsked("call_1", "write file"));
    g = foldApproval(g, acpDecided("call_1", "allowed-once"));
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0]).toMatchObject({
      kind: "asked", toolName: "write file", outcome: "allowed-once",
    });
  });

  it("renders an ACP grant in the SAME words and colour as a DSH one", () => {
    // The point of translating rather than inventing: the panel's label and tone
    // functions are keyed on this exact word, so a third vocabulary would have shown
    // a granted permission as an unexplained neutral event.
    expect(outcomeLabel("allowed-once")).toBe("allowed once");
    expect(outcomeTone("allowed-once")).toBe("text-amber");
  });

  it("renders an ACP refusal as a refusal", () => {
    expect(outcomeLabel("rejected")).toBe("rejected");
    expect(outcomeTone("rejected")).toBe("text-red");
  });

  it("keeps an ACP decision with no visible question", () => {
    // An orphaned decision is still a decision: the ask can predate this turn.
    const g = foldApproval(EMPTY_GOVERNANCE, acpDecided("call_9", "rejected"));
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0].outcome).toBe("rejected");
  });

  it("does not treat an automatic ACP decision as a user decision", () => {
    // `auto: true` is what tells the reader nobody was asked. Dropping it would make
    // a policy decision look like a human click.
    const g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/asked",
      data: { id: "call_2", toolName: "rm -rf", auto: true },
    });
    expect(g.approvals[0].id).toBe("call_2");
  });
});

// R6-ACP-APPROVE: the flag that decides whether a button is DRAWN.
//
// `awaiting` is set only by mcode over ACP, where the server sends
// `session/request_permission` and blocks until answered. A DSH ask with no decision
// is pending in the audit trail but can never be answered over that wire, so the
// button must key on this flag and NOT on "no outcome yet" — otherwise Rigma promises
// something the connection cannot do.
describe("R6-ACP-APPROVE: awaiting is what arms the answer buttons", () => {
  it("keeps the flag on an ask that a transport is waiting on", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/asked",
      data: { id: "call_1", toolName: "write file", awaiting: true },
    });
    expect(g.approvals[0].awaiting).toBe(true);
  });

  it("leaves it false for a DSH ask, which cannot be answered", () => {
    const g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/asked",
      data: { id: "call_1", toolName: "bash" },
    });
    expect(g.approvals[0].awaiting).toBe(false);
  });

  it("refuses to arm a button from a truthy non-boolean", () => {
    // A string "false" is truthy in JavaScript, so a hand-written or older payload
    // carrying one would otherwise render a live "allow" button.
    const g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/asked",
      data: { id: "call_1", awaiting: "false" },
    });
    expect(g.approvals[0].awaiting).toBe(false);
  });

  it("clears the flag once the decision lands", () => {
    // The pair shares an id, so the decision folds onto the ask. `awaiting` stays on
    // the row — the BUTTON is gated on `awaiting && !outcome`, so a resolved request
    // stops being answerable without losing the record that it once was.
    let g = foldApproval(EMPTY_GOVERNANCE, {
      event: "approval/asked",
      data: { id: "call_1", toolName: "write file", awaiting: true },
    });
    g = foldApproval(g, {
      event: "approval/decided",
      data: { id: "call_1", outcome: "allowed-once" },
    });
    expect(g.approvals).toHaveLength(1);
    expect(g.approvals[0].outcome).toBe("allowed-once");
  });
});
