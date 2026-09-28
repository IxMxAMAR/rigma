/** R6-ACP-CONTROL: the control plane's decisions, tested without a renderer. */
import { describe, expect, it } from "vitest";

import {
  CONTROL_OPS,
  ROW_OPS,
  controlAvailability,
  controlOpError,
  controlParams,
  controlResultText,
} from "./controlPlane";

describe("R6-ACP-CONTROL: controlOpError", () => {
  it("requires the field the operation actually needs", () => {
    const goal = CONTROL_OPS.find((o) => o.op === "goal_create")!;
    const queue = CONTROL_OPS.find((o) => o.op === "queue_enqueue")!;
    expect(controlOpError(goal, "")).toContain("objective");
    expect(controlOpError(queue, "")).toContain("text");
    expect(controlOpError(goal, "ship it")).toBe("");
  });

  it("treats whitespace as empty", () => {
    // A goal of "   " is not a goal, and the server would store it as one.
    const goal = CONTROL_OPS.find((o) => o.op === "goal_create")!;
    expect(controlOpError(goal, "   ")).not.toBe("");
  });

  it("asks for nothing on an operation that takes no argument", () => {
    const clear = CONTROL_OPS.find((o) => o.op === "goal_clear")!;
    expect(controlOpError(clear, "")).toBe("");
  });
});

describe("R6-ACP-CONTROL: controlParams", () => {
  it("names the field the protocol expects, not a generic one", () => {
    // `objective` and `text` are different fields on different methods; sending `text`
    // to goal/create would create a goal with no objective.
    const goal = CONTROL_OPS.find((o) => o.op === "goal_create")!;
    const queue = CONTROL_OPS.find((o) => o.op === "queue_enqueue")!;
    expect(controlParams(goal, " ship it ")).toEqual({ objective: "ship it" });
    expect(controlParams(queue, " later ")).toEqual({ text: "later" });
  });

  it("sends nothing for an argument-less operation", () => {
    const clear = CONTROL_OPS.find((o) => o.op === "goal_clear")!;
    expect(controlParams(clear, "ignored")).toEqual({});
  });
});

describe("R6-ACP-CONTROL: controlResultText", () => {
  it("says what CHANGED, because a mutation with no feedback is indistinguishable "
     + "from a failure", () => {
    expect(controlResultText("goal_create", { goal: { objective: "ship it" } }))
      .toContain("ship it");
    expect(controlResultText("goal_patch", { goal: { status: "paused" } }))
      .toContain("paused");
    expect(controlResultText("queue_enqueue", { position: 3 })).toContain("3");
    expect(controlResultText("steer", {})).toContain("running turn");
  });

  it("does not print `[object Object]` or an empty string", () => {
    for (const spec of CONTROL_OPS) {
      const text = controlResultText(spec.op, { goal: null });
      expect(text).not.toBe("");
      expect(text).not.toContain("object Object");
      expect(text).not.toContain("undefined");
    }
  });

  it("reads a goal-less answer without inventing a goal", () => {
    // `goal_clear` answers with no goal, and a message naming one would be a lie.
    expect(controlResultText("goal_create", { goal: null })).not.toContain("null");
  });

  it("counts the queue rather than dumping it", () => {
    expect(controlResultText("queue_list", { items: [] })).toContain("empty");
    expect(controlResultText("queue_list", { items: [1] })).toContain("1 message");
    expect(controlResultText("queue_list", { items: [1, 2] })).toContain("2 messages");
  });
});

describe("R6-ACP-CONTROL: controlAvailability", () => {
  it("names the transport as the reason, because that is what the user can change", () => {
    // `exec` is the DEFAULT, so this is the state most users are in — and a panel that
    // simply did not appear would leave them with no idea a control plane exists.
    const why = controlAvailability("mcode", "exec", true);
    expect(why).toContain("acp");
    expect(why).toContain("exec");
  });

  it("says a session is required, and that the first turn creates one", () => {
    const why = controlAvailability("mcode", "acp", false);
    expect(why).toContain("first turn");
  });

  it("is available when the backend, the wire and the session all line up", () => {
    expect(controlAvailability("mcode", "acp", true)).toBe("");
  });

  it("does not offer the control plane on another backend", () => {
    expect(controlAvailability("dsh", "acp", true)).not.toBe("");
    expect(controlAvailability("native", "exec", true)).not.toBe("");
    expect(controlAvailability("", "acp", true)).not.toBe("");
  });
});

describe("R6-ACP-CONTROL: the operation table", () => {
  it("names operations the server's allowlist actually has", () => {
    // Pinned against the server's own list. A typo here is a button that 400s, and the
    // failure would only appear when a user pressed it.
    const server = [
      "goal_get", "goal_create", "goal_patch", "goal_clear",
      "queue_list", "queue_enqueue", "queue_update", "queue_delete", "queue_steer",
      "steer", "activate", "delegation_get", "delegation_stop",
      "mode_set", "config_set",
    ];
    for (const spec of CONTROL_OPS) {
      expect(server).toContain(spec.op);
    }
    for (const op of ROW_OPS) {
      expect(server).toContain(op);
    }
  });

  it("gives every operation a label and a hint", () => {
    // A button with no explanation is a button nobody presses.
    for (const spec of CONTROL_OPS) {
      expect(spec.label.length).toBeGreaterThan(0);
      expect(spec.hint.length).toBeGreaterThan(0);
    }
  });

  it("has no duplicate operations", () => {
    const names = CONTROL_OPS.map((o) => o.op);
    expect(new Set(names).size).toBe(names.length);
  });
});
