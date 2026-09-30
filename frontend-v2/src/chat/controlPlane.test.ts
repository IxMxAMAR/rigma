/** R6-ACP-CONTROL: the control plane's decisions, tested without a renderer. */
import { describe, expect, it } from "vitest";

import {
  CONTROL_OPS,
  QUEUE_ACTIONS,
  ROW_OPS,
  acpModeList,
  configOptionValues,
  controlAvailability,
  delegationDepths,
  planIsRenderable,
  controlOpError,
  controlParams,
  controlReady,
  controlResultText,
  rowActionParams,
  rowActionReady,
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

  it("names the mode a `mode_set` landed on, never a bare `done`", () => {
    // B6d. The verifier's note: `controlResultText` had no `mode_set` case, so the
    // operation fell through to the generic "done" — which cannot distinguish a
    // change from a no-op. ACP does not promise to echo the id, so the caller's
    // own value is the fallback.
    expect(controlResultText("mode_set", { modeId: "plan" })).toContain("plan");
    expect(controlResultText("mode_set", {}, "acceptEdits")).toContain("acceptEdits");
    const unknown = controlResultText("mode_set", {});
    expect(unknown).not.toBe("done");
    expect(unknown).not.toContain("undefined");
    expect(unknown).not.toContain("object Object");
  });
});

describe("R6-ACP-SETTINGS: the session's advertised modes", () => {
  it("lists exactly what the session advertised, in its own order", () => {
    const list = acpModeList({
      availableModes: [
        { id: "default" },
        { id: "plan", name: "Plan" },
        { id: "acceptEdits", name: "Accept edits" },
      ],
      currentModeId: "plan",
      known: true,
    });
    expect(list.unknown).toBe(false);
    expect(list.options.map((m) => m.id)).toEqual(["default", "plan", "acceptEdits"]);
    // The human name wins over the protocol id, and the id is the fallback.
    expect(list.options[1]).toEqual({ id: "plan", label: "Plan" });
    expect(list.options[0].label).toBe("default");
    expect(list.current).toBe("plan");
  });

  it("never invents a mode when the session advertised none", () => {
    // The whole reason `mode_set` had no UI: a hardcoded `plan`/`default` pair is
    // a control whose values stop matching the server on the next mcode version.
    for (const raw of [
      null, undefined, {}, { availableModes: [] }, { availableModes: null },
      { availableModes: "plan" }, { availableModes: ["plan"] },
      { availableModes: [{ name: "no id at all" }] }, "plan", 7,
    ]) {
      const list = acpModeList(raw);
      expect(list.options, JSON.stringify(raw)).toEqual([]);
      expect(list.unknown, JSON.stringify(raw)).toBe(true);
    }
  });

  it("drops an entry whose id cannot be sent, rather than offering a dead value", () => {
    const list = acpModeList({
      availableModes: [{ id: "  " }, { id: "plan" }, { id: 7 }, null],
    });
    expect(list.options.map((m) => m.id)).toEqual(["plan"]);
    expect(list.unknown).toBe(false);
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

describe("R6-ACP-CONTROL-ROW: per-row actions", () => {
  it("offers the queue's actions and withholds the one that needs an editor", () => {
    // `queue_update` edits a message's TEXT, which is a different interaction from a
    // button. Declared as absent rather than silently missing.
    const ops = QUEUE_ACTIONS.map((a) => a.op);
    expect(ops).toContain("queue_steer");
    expect(ops).toContain("queue_delete");
    expect(ops).not.toContain("queue_update");
  });

  it("withholds an action whose id the backend did not send", () => {
    // The id is what the protocol requires. Sending the operation without it would be
    // refused for a missing parameter — and a round trip to learn that costs a process,
    // so the button is not drawn at all.
    const steer = QUEUE_ACTIONS.find((a) => a.op === "queue_steer")!;
    expect(rowActionReady(steer, { itemId: "q1", status: "queued" })).toBe(true);
    expect(rowActionReady(steer, {})).toBe(false);
    expect(rowActionReady(steer, { itemId: "" })).toBe(false);
    // A non-string id is not an id.
    expect(rowActionReady(steer, { itemId: 7 })).toBe(false);
  });

  it("withholds a status-changing action on a row that is no longer in that status", () => {
    // A queue item that already FAILED or COMPLETED stays in the list — the type carries
    // `failedReason` precisely because those rows persist. mcode refuses to promote a
    // message that is not still queued, so the button could only ever produce an error the
    // user did not need to see.
    const steer = QUEUE_ACTIONS.find((a) => a.op === "queue_steer")!;
    expect(rowActionReady(steer, { itemId: "q1", status: "queued" })).toBe(true);
    expect(rowActionReady(steer, { itemId: "q1", status: "failed" })).toBe(false);
    expect(rowActionReady(steer, { itemId: "q1", status: "completed" })).toBe(false);
    // An UNREADABLE status is not ready either: a row whose state we cannot see is not one
    // to offer a state-changing control on.
    expect(rowActionReady(steer, { itemId: "q1" })).toBe(false);
    // And an action that declares no statuses is unaffected — dropping a message works
    // whatever state it is in.
    const drop = QUEUE_ACTIONS.find((a) => a.op === "queue_delete")!;
    expect(rowActionReady(drop, { itemId: "q1", status: "failed" })).toBe(true);
  });

  it("prints no empty bracket pair when an option list has no usable values", () => {
    // The list used to be decided in the JSX: it checked `options.length > 0` and then
    // FILTERED empty values out of what it printed — so entries that all lacked a `value`
    // rendered a literal " ()" with nothing inside the brackets.
    expect(configOptionValues([{ value: "a" }, { value: "b" }])).toEqual(["a", "b"]);
    expect(configOptionValues([{ value: "a" }, { name: "no value" }])).toEqual(["a"]);
    // Every one of these is empty, so the caller draws NOTHING.
    expect(configOptionValues([{ name: "x" }, { value: "" }, null])).toEqual([]);
    expect(configOptionValues([])).toEqual([]);
    // Not an array at all is the same as empty, not a crash.
    expect(configOptionValues(undefined)).toEqual([]);
    expect(configOptionValues("queued")).toEqual([]);
    // A bare string entry is a value, and an object is unwrapped rather than stringified
    // into "[object Object]".
    expect(configOptionValues(["low", { value: 7 }])).toEqual(["low", "7"]);
  });

  it("counts a plan as drawable only when the panel can actually draw it", () => {
    // TWO COPIES OF THIS RULE DISAGREED, and that is what produced an empty box: the child
    // panel required a string `content` or `uri`, the parent only required non-null. A plan
    // object with neither passed the parent and failed the child, so the parent drew its
    // container around nothing. There is now one function, and this pins it.
    expect(planIsRenderable({ content: "# Plan" })).toBe(true);
    expect(planIsRenderable({ uri: "file:///p.md" })).toBe(true);
    // ACP's `PlanUpdateContent` is a three-way union: `markdown` has `content`, `file` has
    // `uri`, and `items` has NEITHER — it has `entries`, which the backend routes to the
    // todos channel. So this is a CONFORMANT case, not only a malformed one.
    expect(planIsRenderable({ type: "items", planId: "p", entries: [] })).toBe(false);
    expect(planIsRenderable({})).toBe(false);
    expect(planIsRenderable(null)).toBe(false);
    expect(planIsRenderable(undefined)).toBe(false);
    // A non-string content is not drawable text.
    expect(planIsRenderable({ content: 7 })).toBe(false);
  });

  it("draws the delegation tree at the depth the members actually describe", () => {
    // Every member carries `parentSessionId` — mcode's `Se(e)` emits it unconditionally —
    // and the panel drew the list FLAT, so a child appeared as a sibling of its parent.
    const root = { sessionId: "root" };
    const a = { sessionId: "a", parentSessionId: "root" };
    const b = { sessionId: "b", parentSessionId: "a" };
    const c = { sessionId: "c", parentSessionId: "root" };
    expect(delegationDepths([root, a, b, c])).toEqual([0, 1, 2, 1]);
    // ORDER MUST NOT MATTER: a child may be listed before its parent.
    expect(delegationDepths([b, a, root])).toEqual([2, 1, 0]);
    // A parent that is NOT in the list is the root, so this is a top-level child.
    expect(delegationDepths([{ sessionId: "x", parentSessionId: "gone" }])).toEqual([0]);
    // No parent field at all — the pre-existing flat case — is all depth 0.
    expect(delegationDepths([{ sessionId: "a" }, { sessionId: "b" }])).toEqual([0, 0]);
    expect(delegationDepths([])).toEqual([]);
    // A MALFORMED CYCLE MUST TERMINATE, and that is the ONLY contract. A render that hangs
    // is a frozen window, so the walk stops at the first node it would revisit rather than
    // trusting the data. The exact depth of a member ON a cycle carries no information —
    // mcode's own root-resolution throws on a cycle, so no correct payload has one — and
    // pinning a particular number here would be pinning an accident.
    const cycle = delegationDepths([
      { sessionId: "a", parentSessionId: "b" },
      { sessionId: "b", parentSessionId: "a" },
    ]);
    expect(cycle).toHaveLength(2);
    expect(cycle.every((d) => d >= 0 && d <= 2)).toBe(true);
    const self = delegationDepths([{ sessionId: "a", parentSessionId: "a" }]);
    expect(self).toHaveLength(1);
    expect(self[0]).toBeGreaterThanOrEqual(0);
    // A LONG cycle terminates too, and stays within the bound. This is the case a naive
    // walk would spin on, and the one that would freeze the window.
    const long = delegationDepths([
      { sessionId: "a", parentSessionId: "b" },
      { sessionId: "b", parentSessionId: "c" },
      { sessionId: "c", parentSessionId: "a" },
    ]);
    expect(long).toHaveLength(3);
    expect(long.every((d) => d >= 0 && d <= 3)).toBe(true);
    // Non-string ids are not ids, so they cannot be linked.
    expect(delegationDepths([{ sessionId: 7, parentSessionId: "a" }])).toEqual([0]);
  });

  it("reads the delegation-stop receipt by the field names mcode actually sends", () => {
    // MEASURED from mcode's `stop()` in chunks/chunk-E2AN54L4.js:
    //   {schemaVersion, rootSessionId, rootStopped, stoppedSessionIds,
    //    activeSessionIds, failedSessionIds}
    //
    // This arm used to read `receipt.stopped`, which does not exist — so it fell through
    // to its own fallback and reported "all delegated work stopped" for EVERY outcome,
    // including the one where nothing was running. A guessed receipt shape is how a
    // destructive operation ends up lying about what it did.
    expect(controlResultText("delegation_stop",
      { receipt: { stoppedSessionIds: ["a"] } })).toBe("stopped 1 delegated task");
    expect(controlResultText("delegation_stop",
      { receipt: { stoppedSessionIds: ["a", "b"] } })).toBe("stopped 2 delegated tasks");
    expect(controlResultText("delegation_stop",
      { receipt: { stoppedSessionIds: [] } })).toBe("nothing was still running");
    // A partial failure has to be SAID: reporting a clean stop when some children refused
    // would be the same lie in a quieter form.
    expect(controlResultText("delegation_stop",
      { receipt: { stoppedSessionIds: ["a"], failedSessionIds: ["b", "c"] } }))
      .toBe("stopped 1 delegated task; 2 could not be stopped");
    expect(controlResultText("delegation_stop",
      { receipt: { stoppedSessionIds: [], failedSessionIds: ["b"] } }))
      .toBe("nothing was still running (1 could not be stopped)");
    // No receipt at all is the only case that falls back.
    expect(controlResultText("delegation_stop", {})).toBe("all delegated work stopped");
  });

  it("offers steer only in the statuses mcode treats as actionable", () => {
    // mcode's normaliser emits exactly queued/running/completed/failed/stopped/unknown —
    // "pending" can never occur, and this table said `["queued","pending"]` before. The
    // two mcode's own TUI filters for are queued and paused.
    const steer = QUEUE_ACTIONS.find((a) => a.op === "queue_steer")!;
    expect(steer.statuses).toEqual(["queued", "paused"]);
    expect(steer.statuses).not.toContain("pending");
  });

  it("gives every closed-set operation a value it can actually send", () => {
    // `goal_patch` used to carry no arg and no choices, so the panel sent it with NO
    // params: the server accepted the empty patch, returned the goal UNCHANGED, and the
    // result line printed the old status as the outcome. A button that does nothing and
    // reports success is the failure this module exists to prevent.
    const patchOp = CONTROL_OPS.find((o) => o.op === "goal_patch")!;
    expect(patchOp.choices?.param).toBe("status");
    expect(patchOp.choices!.values.length).toBeGreaterThan(0);
    // The default (first value) is what an untouched select sends, so it must be real.
    expect(controlParams(patchOp, "")).toEqual({ status: "active" });
    expect(controlParams(patchOp, "", "paused")).toEqual({ status: "paused" });
    // And readiness follows the value, not the text field it does not use.
    expect(controlReady(patchOp, "")).toBe(true);
  });

  it("returns null params rather than an operation without its id", () => {
    const drop = QUEUE_ACTIONS.find((a) => a.op === "queue_delete")!;
    expect(rowActionParams(drop, { itemId: "q1" })).toEqual({ itemId: "q1" });
    // Null, NOT `{}`: an empty body would be sent to the server and refused there.
    expect(rowActionParams(drop, {})).toBeNull();
  });

  it("has NO per-child delegation action, because the protocol has none", () => {
    // MEASURED from mcode's own handler:
    //   onRequest("mcode/session/delegation/stop", xt, async({params:i}) => {
    //       let s = r(i.sessionId), a = await mo(e.runtime, s);
    //       return {receipt: await e.runtime.stopDelegation(a)}})
    // `i.sessionId` is the ROOT session and the whole tree is stopped. So a button on a
    // member row would say "stop this child" and stop every child — worse than none.
    //
    // Asserted as an ABSENCE so that re-adding one has to confront this comment.
    expect(ROW_OPS as readonly string[]).not.toContain("delegation_stop");
    expect(CONTROL_OPS.map((o) => o.op)).toContain("delegation_stop");
  });

  it("says the delegation stop is session-wide, since that is what it does", () => {
    const stop = CONTROL_OPS.find((o) => o.op === "delegation_stop")!;
    expect(stop.hint.toLowerCase()).toMatch(/every child|session-wide|not\s+per-child/);
  });

  it("marks the destructive actions, because that is what decides how they draw", () => {
    expect(QUEUE_ACTIONS.find((a) => a.op === "queue_delete")!.danger).toBe(true);
    // Steering a queued message is not destructive: the message still runs.
    expect(QUEUE_ACTIONS.find((a) => a.op === "queue_steer")!.danger).toBeFalsy();
  });

  it("only names operations the server's allowlist has", () => {
    const server = ["queue_steer", "queue_delete", "queue_update"];
    for (const a of QUEUE_ACTIONS) {
      expect(server).toContain(a.op);
      expect(ROW_OPS).toContain(a.op);
    }
  });

  it("gives every row action a hint, since a bare verb is not enough", () => {
    for (const a of QUEUE_ACTIONS) {
      expect(a.hint.length).toBeGreaterThan(0);
      expect(a.label.length).toBeGreaterThan(0);
    }
  });
});
