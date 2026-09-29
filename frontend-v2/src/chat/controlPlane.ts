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
  /** A CLOSED set of values, when the parameter is an enum rather than free text.
   *
   *  WHY THIS EXISTS. `goal_patch` used to carry no `arg`, so the panel sent it with NO
   *  params at all: the server accepted the empty patch, returned the goal UNCHANGED, and
   *  the result line printed "goal is now <old status>". The button did nothing and
   *  reported success — the exact failure the module header says this design prevents.
   *  A status is a closed set the protocol defines, so it is a select, not a text field.
   */
  choices?: { param: string; values: string[]; labels?: string[] };
  /** Shown when the user asks what it does. */
  hint: string;
}

/** The operations the panel offers, in the order a user is likely to want them.
 *
 *  DELIBERATELY NOT ALL FIFTEEN. The server's allowlist is the security boundary and it
 *  is larger than this; the panel offers the ones with a clear use from a chat window.
 *  `queue_update`, `queue_delete` and `queue_steer` act on an id that only the live
 *  panel has, so they are offered per-row instead of here.
 *
 *  `delegation_stop` is HERE rather than per-row, and that is a correction. It used to be
 *  a row action on the strength of the server table requiring a `sessionId` — but that
 *  id is the ROOT session, not a member: mcode's handler resolves `params.sessionId` and
 *  calls `stopDelegation` on the resolved session, stopping the whole tree. A per-child
 *  button would therefore have stopped every child, which is the kind of control that is
 *  worse than none.
 */
export const CONTROL_OPS: ControlOp[] = [
  { op: "goal_create", label: "set a goal", arg: "objective",
    hint: "Give the agent a standing objective it works toward across turns. "
        + "mcode keeps it, and it survives this chat being reopened." },
  { op: "goal_patch", label: "change the goal's status",
    // The statuses `mcode/session/goal/patch` accepts, measured from the extension
    // surface. `active` resumes; `paused` stops it being worked toward.
    choices: { param: "status", values: ["active", "paused"],
               labels: ["resume (active)", "pause (paused)"] },
    hint: "Pause the goal so the agent stops working toward it, or resume it. The goal "
        + "and its history are kept either way." },
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
  { op: "delegation_stop", label: "stop all delegated work",
    hint: "Abandon every child this session delegated to. This is session-wide, not "
        + "per-child: the protocol has no way to stop just one." },
];

/** The operations that need an id from a live row rather than a text field. */
export const ROW_OPS = ["queue_update", "queue_delete", "queue_steer"] as const;

/** One row action, as a button.
 *
 *  WHY THESE ARE NOT IN `CONTROL_OPS`. They act on an id that only a LIVE row has —
 *  a queue item's `itemId`. A form has no such id, so offering them there would mean
 *  asking the user to type one. They belong on the rows.
 *
 *  `needs` names the id, so the button can be withheld when the backend did not send it
 *  rather than sending an operation that would be refused for a missing parameter.
 */
export interface RowAction {
  op: (typeof ROW_OPS)[number];
  label: string;
  /** The field on the row this operation requires. */
  needs: "itemId";
  /** Destructive actions are drawn differently, and this is what decides. */
  danger?: boolean;
  /** The row statuses this action applies to. Absent means "any".
   *
   *  WHY THIS EXISTS. `rowActionReady` used to check only the id, so a queue item that had
   *  already FAILED or COMPLETED — states the type carries, because such rows are kept —
   *  still drew a "steer" button. mcode refuses to promote a message that is not queued,
   *  so the button could only ever produce an error the user did not have to see.
   */
  statuses?: string[];
  hint: string;
}

/** The queue's actions. `queue_update` is deliberately absent: it edits a message's
 *  text, which needs an editor, and an inline rename is a different interaction from a
 *  button. Declared rather than silently omitted. */
export const QUEUE_ACTIONS: RowAction[] = [
  { op: "queue_steer", label: "steer", needs: "itemId",
    // Only a message still WAITING can be promoted. A failed or completed one cannot, and
    // those rows stay in the list, so without this the button appears on rows where it
    // could only fail.
    statuses: ["queued", "pending"],
    hint: "Promote this queued message into the turn that is running NOW." },
  { op: "queue_delete", label: "drop", needs: "itemId", danger: true,
    hint: "Remove this message from the queue. It will never run." },
];

/* There are no per-child delegation actions, and their absence is deliberate rather
 * than unfinished. `delegation_stop` is session-wide (see `CONTROL_OPS` above), so a
 * button on a member row would misstate what it does. Nothing else in mcode's delegation
 * surface is per-member: `delegation/get` reads the tree and `delegation/stop` clears it.
 */

/** Whether a row action can be offered.
 *
 *  Two checks, because either one alone produces a button that cannot work: the backend
 *  has to have sent the id the operation needs, AND (when the action declares them) the
 *  row has to be in a status the operation applies to. An UNKNOWN status with a declared
 *  `statuses` list counts as not-ready: a row whose state we cannot read is not one to
 *  offer a state-changing control on.
 */
export function rowActionReady(a: RowAction, row: Record<string, unknown>): boolean {
  const v = row[a.needs];
  if (typeof v !== "string" || v === "") return false;
  if (!a.statuses) return true;
  const status = typeof row.status === "string" ? row.status : "";
  return a.statuses.includes(status);
}

/** The params for a row action, or null when the id is missing.
 *
 *  Null rather than `{}` so a caller cannot accidentally send an operation without its
 *  id — the server would refuse it, but a round trip to learn that is a wasted process. */
export function rowActionParams(
  a: RowAction, row: Record<string, unknown>,
): Record<string, unknown> | null {
  if (!rowActionReady(a, row)) return null;
  return { [a.needs]: String(row[a.needs]) };
}

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

/** The params for one operation, from the panel's inputs.
 *
 *  `choice` is a separate argument rather than overloading `arg`, because the two are
 *  different controls: `arg` is free text the user types, `choice` is one of a closed
 *  set the server defines. Passing an empty choice falls back to the first value, so a
 *  select that has never been touched still sends a real status rather than nothing.
 */
export function controlParams(spec: ControlOp, arg: string,
                              choice = ""): Record<string, unknown> {
  if (spec.choices) {
    const value = choice || spec.choices.values[0] || "";
    return { [spec.choices.param]: value };
  }
  if (!spec.arg) return {};
  const value = arg.trim();
  return spec.arg === "objective" ? { objective: value } : { text: value };
}

/** The printable values of an ACP configOption's `options`, in order.
 *
 *  WHY THIS IS NOT INLINE IN THE COMPONENT. It used to be: the JSX checked
 *  `c.options.length > 0` and then FILTERED empty values out of what it printed, so a list
 *  whose entries all lacked a `value` rendered a literal " ()" with nothing between the
 *  brackets. Deciding the list here makes the empty case reachable by a test.
 *
 *  An entry may be a bare string or an object with a `value`; anything else contributes
 *  nothing rather than the string "[object Object]".
 */
export function configOptionValues(options: unknown): string[] {
  if (!Array.isArray(options)) return [];
  return options
    .map((o) => {
      if (typeof o === "string") return o;
      if (o && typeof o === "object") {
        const v = (o as { value?: unknown }).value;
        return v == null ? "" : String(v);
      }
      return "";
    })
    .filter((v) => v !== "");
}

/** Whether an operation can be attempted, given both inputs. */
export function controlReady(spec: ControlOp, arg: string, choice = ""): boolean {
  if (spec.choices) {
    // The default is the first value, so this is satisfied unless the table is empty —
    // and an empty table is a bug in the table, not a user error, so it is reported as
    // an error rather than silently sending nothing.
    return Boolean(choice || spec.choices.values[0]);
  }
  return controlOpError(spec, arg) === "";
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
    case "config_set":
      // The panel's own select already shows the new value, so this only confirms.
      return "setting changed";
    case "delegation_stop": {
      // Session-wide, so the sentence has to say so: a user who expected one child to stop
      // needs to know the whole tree did.
      const stopped = (r.receipt as Record<string, unknown> | undefined)?.stopped;
      const n = Array.isArray(stopped) ? stopped.length : null;
      return n === null ? "all delegated work stopped"
                        : n === 0 ? "nothing was still running"
                                  : `stopped ${n} delegated ${n === 1 ? "task" : "tasks"}`;
    }
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
