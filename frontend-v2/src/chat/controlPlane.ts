/** R6-ACP-CONTROL: the mcode control plane, from the UI's side.
 *
 *  WHY THIS IS A SEPARATE MODULE. Everything here is a decision — is this operation
 *  available, what does this result mean, what should the button say — and decisions
 *  are what a test can reach. This project has no `*.test.tsx`, so render-layer
 *  behaviour is covered by pure helpers plus typecheck; putting the logic in the
 *  component would make it unreachable and therefore unchecked.
 *
 *  THE OPERATIONS ARE THE SERVER'S OWN NAMES. They are not re-spelled here, because a
 *  second vocabulary is a second thing to keep in step, and the server's allowlist is
 *  what actually decides.
 */

export interface ControlOp {
  /** The name the server's allowlist uses. */
  op: string;
  /** What the button says. */
  label: string;
  /** What the user has to type, if anything. */
  arg?: "text" | "objective";
  /** Shown when the user asks what it does. */
  hint: string;
}

/** The operations the panel offers, in the order a user is likely to want them.
 *
 *  DELIBERATELY NOT ALL FIFTEEN. The server's allowlist is the security boundary and it
 *  is larger than this; the panel offers the ones with a clear use from a chat window.
 *  `queue_update`, `queue_delete`, `queue_steer` and `delegation_stop` act on an id that
 *  the live panel has and a form does not, so they are offered per-row instead of here.
 */
export const CONTROL_OPS: ControlOp[] = [
  { op: "goal_create", label: "set a goal", arg: "objective",
    hint: "Give the agent a standing objective it works toward across turns. "
        + "mcode keeps it, and it survives this chat being reopened." },
  { op: "goal_patch", label: "pause or resume the goal",
    hint: "Change the goal's status without losing it." },
  { op: "goal_clear", label: "clear the goal",
    hint: "Remove the objective. The agent keeps its history; only the goal goes." },
  { op: "queue_enqueue", label: "queue a message", arg: "text",
    hint: "Add a message to run AFTER the current turn, rather than steering the turn "
        + "that is already running." },
  { op: "steer", label: "steer the running turn", arg: "text",
    hint: "Inject text into the turn that is running RIGHT NOW. It changes what the "
        + "agent is doing; it does not start a new turn." },
  { op: "activate", label: "make this the active session",
    hint: "Tell mcode this is the session to work on. Matters when several are open." },
];

/** The operations that need an id from a live row rather than a text field. */
export const ROW_OPS = ["queue_update", "queue_delete", "queue_steer",
                        "delegation_stop"] as const;

/** Why this operation cannot be sent, or "" if it can.
 *
 *  Mirrors the server's own `control_op_error`, and deliberately duplicates the rule
 *  rather than trusting the server alone: the point is to keep the button disabled, and
 *  a round trip to discover a missing field would be a button that looks ready and then
 *  fails.
 */
export function controlOpError(spec: ControlOp, arg: string): string {
  if (!spec.arg) return "";
  if (arg.trim() === "") {
    return spec.arg === "objective"
      ? "a goal needs an objective"
      : "this needs some text";
  }
  return "";
}

/** The params for one operation, from the panel's single input. */
export function controlParams(spec: ControlOp, arg: string): Record<string, unknown> {
  if (!spec.arg) return {};
  const value = arg.trim();
  return spec.arg === "objective" ? { objective: value } : { text: value };
}

/** What the server's answer means, as a sentence for the user.
 *
 *  A control operation is a MUTATION, so silence is not acceptable feedback: a button
 *  that appears to do nothing is indistinguishable from one that failed. Each result
 *  shape gets a sentence that says what changed.
 */
export function controlResultText(op: string, result: unknown): string {
  const r = (result ?? {}) as Record<string, unknown>;
  const goal = r.goal as Record<string, unknown> | null | undefined;
  switch (op) {
    case "goal_create":
      return goal ? `goal set: ${String(goal.objective ?? "")}` : "goal set";
    case "goal_patch":
      return goal ? `goal is now ${String(goal.status ?? "")}`
                  : "goal updated";
    case "goal_clear":
      return "goal cleared";
    case "queue_enqueue": {
      const pos = r.position;
      return typeof pos === "number"
        ? `queued, position ${pos}` : "queued";
    }
    case "queue_list": {
      const items = (r.items ?? []) as unknown[];
      return items.length === 0 ? "the queue is empty"
        : `${items.length} message${items.length === 1 ? "" : "s"} queued`;
    }
    case "steer":
      return "sent into the running turn";
    case "activate":
      return "this is now the active session";
    default:
      return "done";
  }
}

/** Whether the panel can be used at all, and why not when it cannot.
 *
 *  A control that is drawn but cannot work is worse than an absent one, so the panel
 *  states the reason rather than rendering disabled buttons with no explanation. The
 *  `exec` case is the one a user actually hits: it is the DEFAULT transport, so the
 *  control plane is unavailable in the default configuration and has to say so.
 */
export function controlAvailability(
  harness: string, transport: string, hasSession: boolean,
): string {
  if (harness !== "mcode") {
    return "the control plane belongs to mcode; this chat runs on "
      + (harness || "Rigma's own loop");
  }
  if (transport !== "acp") {
    return "the control plane needs the acp transport. On exec mcode runs one turn and "
      + "keeps no session, so there is nothing to steer and no queue to add to. "
      + "Switch transport above, then send a message.";
  }
  if (!hasSession) {
    return "no mcode session yet — it appears after the first turn, because the control "
      + "plane acts on a session rather than on the chat.";
  }
  return "";
}
