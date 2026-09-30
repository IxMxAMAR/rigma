// What the harness was ALLOWED to do, and what it asked for.
//
// DSH records its own governance as session events, and every one of them is
// `log-only` in DSH's own words: durable and replayable, never part of the model
// transcript. That is precisely the shape of fact a reader wants and a
// conversation should not carry — so they are rendered beside the turn, not in
// it.
//
// **These are display-only on DSH, and that is a property of the transport rather
// than a limitation of this code.** DSH's SDK wire exposes exactly three methods —
// `initialize`, `session/prompt`, `shutdown` — so there is no way to answer an
// approval over it. Approval is decided by the policy engine. A clickable
// "Allow?" button on a DSH approval would misrepresent what the connection can do,
// so there is none.
//
// R6-ACP-APPROVE: that reasoning does NOT hold for mcode over ACP, which is why
// `GovernanceEvent.awaiting` exists. There the server sends
// `session/request_permission` and BLOCKS until it is answered, so a request marked
// `awaiting` is one a click can actually resolve — and one that, left unanswered,
// wedges the transport rather than failing a turn. The flag is what distinguishes
// the two cases, so the button is rendered on the transport's ability rather than on
// the reader's optimism.

/** One entry in the audit trail, in arrival order. */
export interface GovernanceEvent {
  /** `asked`, `decided` or `policy` — the DSH name with its prefix stripped.
   *
   *  B4 gives this a second meaning: an `asked` whose payload carries
   *  `kind: "question"` is mcode's `ask_user` elicitation, NOT a permission
   *  request. The server sends both on the same `approval/asked` channel, and
   *  `kind` is what tells them apart. */
  kind: "asked" | "decided" | "policy" | string;
  /** Pairs an `asked` with its `decided`; empty when the event has none. */
  id: string;
  /** The tool under question, on an `asked`. */
  toolName: string;
  /** The asker's own explanation, on an `asked`. */
  reason: string;
  /** The verdict, on a `decided`. */
  outcome: string;
  /** The policy in force, on a `policy`. */
  policy: string;
  /** B4: the wire's own `data.kind`. `"question"` for an elicitation; absent or
   *  `"permission"` for a permission request.
   *
   *  Without this the row fell through to the Allow-once/Refuse card, whose
   *  buttons POST `{allow}` — and the route answers 409 for a question, so the
   *  controls were dead and the turn stayed blocked. */
  requestKind?: string;
  /** B4: the question mcode asked (`data.question`), on a question. */
  question: string;
  /** B4: the form it wants back (`data.schema`, an ACP elicitation schema), on a
   *  question. `{}` when the server sent none. */
  schema: Record<string, unknown>;
  /** R6-ACP-APPROVE: this request is BLOCKING the turn and a click can answer it.
   *
   *  Set only for mcode over ACP, where the server sends
   *  `session/request_permission` and waits. A DSH approval never sets it, because
   *  there is no method on that wire to answer with — so this flag, not the presence
   *  of an outcome, is what decides whether a button is drawn. */
  awaiting?: boolean;
}

export interface Governance {
  /** The audit trail, oldest first. Capped by the caller's own turn length. */
  approvals: GovernanceEvent[];
  /** DSH's `sandbox/mode`: read-only | workspace-write | danger-full-access. */
  sandbox: string;
  /** DSH's `permission/preset`. */
  preset: string;
}

export const EMPTY_GOVERNANCE: Governance = {
  approvals: [],
  sandbox: "",
  preset: "",
};

const str = (v: unknown): string => (v == null ? "" : String(v));

/** Fold one `approval` SSE payload into the trail.
 *
 *  The server sends `{event: "approval/asked", data: {...}}` for all three
 *  approval events, so the DSH name is what distinguishes them. An event with an
 *  unknown suffix is kept rather than dropped — it is an audit trail, and an
 *  entry whose kind this build does not recognise is still evidence that
 *  something happened.
 */
export function foldApproval(
  gov: Governance, payload: unknown,
): Governance {
  if (!payload || typeof payload !== "object") return gov;
  const p = payload as Record<string, unknown>;
  const d = (p.data && typeof p.data === "object" ? p.data : {}) as Record<string, unknown>;
  // The prefix is REQUIRED, not merely stripped. Without this the store's
  // `approval` case would append any payload that reached it — `{event: "goal"}`
  // became a trail entry called "goal" — because stripping a prefix that is not
  // there is a no-op. The server only emits this on the `approval` SSE name, so
  // this is belt-and-braces; it is the kind of belt that catches a routing bug.
  const raw = str(p.event);
  if (!raw.startsWith("approval/")) return gov;
  const kind = raw.slice("approval/".length);
  if (!kind) return gov;

  const entry: GovernanceEvent = {
    kind,
    id: str(d.id),
    toolName: str(d.toolName),
    reason: str(d.reason),
    outcome: str(d.outcome),
    policy: str(d.policy),
    // B4: the elicitation's own fields. `requestKind` is a SEPARATE name from
    // `kind` on purpose: `kind` is the approval/ SUFFIX (`asked`/`decided`), and
    // overloading it would make a question indistinguishable from a permission
    // ask at the fold. `question` and `schema` are carried verbatim — the form
    // renders what the server asked for, never a shape invented here.
    requestKind: str(d.kind),
    question: str(d.question),
    schema: (d.schema && typeof d.schema === "object" && !Array.isArray(d.schema)
      ? d.schema : {}) as Record<string, unknown>,
    // Only ever true from a transport that can be answered. `=== true` rather than
    // truthiness so a string "false" from a hand-written payload cannot arm a button.
    awaiting: d.awaiting === true,
  };

  // `decided` carries only an id and an outcome. Folding it onto the matching
  // `asked` rather than appending a second row is what makes the trail readable:
  // two rows per decision would double the list and separate a question from its
  // answer. An ORPHANED decision — no matching ask, which happens when the ask
  // predates this turn — is appended rather than discarded, because a decision
  // with no visible question is still a decision.
  if (kind === "decided" && entry.id) {
    for (let i = gov.approvals.length - 1; i >= 0; i--) {
      const prev = gov.approvals[i];
      if (prev.id === entry.id && !prev.outcome) {
        const next = gov.approvals.slice();
        next[i] = { ...prev, outcome: entry.outcome };
        return { ...gov, approvals: next };
      }
    }
  }
  return { ...gov, approvals: [...gov.approvals, entry] };
}

/** B4: is this entry mcode's `ask_user` elicitation rather than a permission ask?
 *
 *  ONE PLACE STATES THE RULE. The server puts both on the `approval/asked`
 *  channel and distinguishes them only by `data.kind`, so every consumer has to
 *  read the same field; a second copy of the test is a second thing to keep in
 *  step, and the failure mode is a question drawn with Allow/Refuse buttons whose
 *  POST the route answers with 409 — dead controls on a blocked turn. */
export function isQuestion(a: GovernanceEvent): boolean {
  return a.requestKind === "question";
}

/** One field of an elicitation's form, as the schema asked for it. */
export interface QuestionField {
  /** The key the answer object must use — the schema's own property name. */
  name: string;
  /** What a human reads: the schema's `title`, or the key. */
  label: string;
  kind: "text" | "boolean" | "choice";
  /** The allowed values, when the schema enumerated them. */
  choices: string[];
  required: boolean;
  description: string;
}

/** The string values a property enumerates, from `enum`, `oneOf` or `anyOf`.
 *
 *  ACP's elicitation schema is JSON Schema, so a closed set may be spelled three
 *  ways. All three are read here rather than only `enum`, because a control that
 *  silently degrades to a free-text box for `oneOf` would let the user type a
 *  value the server refuses. */
function schemaChoices(prop: Record<string, unknown>): string[] {
  const out: string[] = [];
  const push = (v: unknown) => {
    if (typeof v === "string" && v !== "") out.push(v);
  };
  if (Array.isArray(prop.enum)) prop.enum.forEach(push);
  for (const key of ["oneOf", "anyOf"]) {
    const arr = prop[key];
    if (!Array.isArray(arr)) continue;
    for (const e of arr) {
      if (!e || typeof e !== "object") continue;
      const o = e as Record<string, unknown>;
      if (typeof o.const === "string") push(o.const);
      else if (Array.isArray(o.enum)) o.enum.forEach(push);
    }
  }
  return out;
}

/** The form an elicitation's `requestedSchema` describes, or none.
 *
 *  NEVER INVENTS A FIELD. A schema with no `properties` (the server is allowed to
 *  send `{}`) yields no fields, and the form says so rather than guessing a key —
 *  the answer object's keys are the server's contract. */
export function questionFields(schema: unknown): QuestionField[] {
  const s = (schema && typeof schema === "object" && !Array.isArray(schema)
    ? schema : {}) as Record<string, unknown>;
  const props = (s.properties && typeof s.properties === "object"
    && !Array.isArray(s.properties) ? s.properties : {}) as Record<string, unknown>;
  const required = Array.isArray(s.required) ? s.required.map((r) => String(r)) : [];
  const fields: QuestionField[] = [];
  for (const [name, raw] of Object.entries(props)) {
    if (!raw || typeof raw !== "object") continue;
    const p = raw as Record<string, unknown>;
    const choices = schemaChoices(p);
    const type = typeof p.type === "string" ? p.type : "";
    fields.push({
      name,
      label: typeof p.title === "string" && p.title !== "" ? p.title : name,
      kind: choices.length > 0 ? "choice" : type === "boolean" ? "boolean" : "text",
      choices,
      required: required.includes(name),
      description: typeof p.description === "string" ? p.description : "",
    });
  }
  return fields;
}

/** The answer object for `/approval`, built from what the user filled in.
 *
 *  Keys are the schema's own property names; a blank optional field is ABSENT
 *  rather than `""`, so mcode can tell "not answered" from "answered with
 *  nothing". A boolean is always present: a checkbox is a definite answer, and
 *  `false` is one. */
export function questionAnswer(
  fields: QuestionField[], values: Record<string, string | boolean>,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fields) {
    const v = values[f.name];
    if (f.kind === "boolean") out[f.name] = v === true;
    else if (typeof v === "string" && v.trim() !== "") out[f.name] = v.trim();
  }
  return out;
}

/** Whether every required field has an answer. Optional fields never block. */
export function questionReady(
  fields: QuestionField[], values: Record<string, string | boolean>,
): boolean {
  return fields.every((f) => {
    if (f.kind === "boolean" || !f.required) return true;
    const v = values[f.name];
    return typeof v === "string" && v.trim() !== "";
  });
}

/** `allowed-once` reads as permission granted; the other three do not. */export function outcomeTone(outcome: string): string {
  if (outcome === "allowed-once") return "text-amber";
  if (outcome === "rejected" || outcome === "unavailable") return "text-red";
  return "text-muted";
}

/** A decision in words, so `allowed-once` is not shown as a bare identifier. */
export function outcomeLabel(outcome: string): string {
  switch (outcome) {
    case "allowed-once": return "allowed once";
    case "rejected": return "rejected";
    case "cancelled": return "cancelled";
    // DSH's fail-closed outcome: nobody could answer, so it was NOT permitted.
    // Spelling that out matters — "unavailable" alone reads like a network blip.
    case "unavailable": return "not permitted (nobody could answer)";
    default: return outcome;
  }
}

/** The confinement in words. `danger-full-access` must not read as neutral. */
export function sandboxLabel(mode: string): string {
  switch (mode) {
    case "read-only": return "read-only";
    case "workspace-write": return "can write in the workspace";
    case "danger-full-access": return "unrestricted — no sandbox";
    default: return mode;
  }
}

export function sandboxTone(mode: string): string {
  if (mode === "read-only") return "text-moss";
  if (mode === "workspace-write") return "text-amber";
  if (mode === "danger-full-access") return "text-red";
  return "text-muted";
}
