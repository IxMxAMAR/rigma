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
    // OD-12: the server's own clock, never the client's. When a question's
    // window closes the server emits `approval/decided` for the SAME id with
    // `decision: "expired"` and NO `outcome` — a real allow/refuse still carries
    // `outcome`. Both land on the trail's own `outcome` field, so an expired
    // question becomes the same terminal, decided row every other verdict
    // already renders (glyph, tone, label, no control) instead of a second
    // visual language. `outcome` is read FIRST, so a real decision folds exactly
    // as it did; `decision` is only a fallback for the expiry word.
    outcome: str(d.outcome) || (str(d.decision) === "expired" ? "expired" : ""),
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

/** One field of an elicitation's form, as the schema asked for it.
 *
 *  B4b-schema: `requestedSchema` is JSON Schema, and nothing in the wire
 *  contract keeps it flat. A property whose `type` is `object` carries its own
 *  `properties` (and its own `required`), and an `array` carries an `items`
 *  schema. Before this, both fell to a text box that could not express the
 *  answer at all: a nested object was submitted as the string "[object Object]"
 *  and an array could not be built. */
export interface QuestionField {
  /** The key the answer object must use — the schema's own property name. */
  name: string;
  /** What a human reads: the schema's `title`, or the key. */
  label: string;
  /** `object` and `array` are the nested shapes; the rest are leaves. */
  kind: "text" | "number" | "boolean" | "choice" | "object" | "array";
  /** The allowed values, when the schema enumerated them. */
  choices: string[];
  required: boolean;
  description: string;
  /** The schema's own `default`, when it gave one. `hasDefault` is a SEPARATE
   *  fact because `default: false` and `default: ""` are real pre-fills while an
   *  absent default must leave the control empty. */
  hasDefault: boolean;
  default: unknown;
  /** `object`: its own properties, with the nested `required` already applied. */
  fields?: QuestionField[];
  /** `array`: the element's shape, parsed from `items`. An absent `items` is
   *  JSON Schema's "any element", read as free text — the same reading an
   *  untyped property already gets. */
  items?: QuestionField;
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

/** The element an unconstrained `items` (or an untyped property) gets. */
function textField(name: string): QuestionField {
  return {
    name, label: name, kind: "text", choices: [], required: false,
    description: "", hasDefault: false, default: undefined,
  };
}

/** One property of a schema's `properties`, or null when it is not an object.
 *
 *  `required` is the CONTAINING schema's list — a nested object's own `required`
 *  is applied when its `fields` are parsed, not here. */
function schemaField(
  name: string, raw: unknown, required: string[],
): QuestionField | null {
  if (!raw || typeof raw !== "object" || Array.isArray(raw)) return null;
  const p = raw as Record<string, unknown>;
  const choices = schemaChoices(p);
  const type = typeof p.type === "string" ? p.type : "";
  const field: QuestionField = {
    name,
    label: typeof p.title === "string" && p.title !== "" ? p.title : name,
    kind: "text",
    choices: [],
    required: required.includes(name),
    description: typeof p.description === "string" ? p.description : "",
    hasDefault: "default" in p,
    default: p.default,
  };
  // A closed set wins over the declared type: a control that silently degrades
  // to free text would let the user type a value the server refuses.
  if (choices.length > 0) return { ...field, kind: "choice", choices };
  if (type === "boolean") return { ...field, kind: "boolean" };
  // `integer` is a number to a form; the schema's own `multipleOf`/`minimum`
  // are not enforced here, and refusing to submit a non-numeric string is.
  if (type === "number" || type === "integer") return { ...field, kind: "number" };
  if (type === "array") {
    return { ...field, kind: "array", items: schemaField("", p.items, []) ?? textField("") };
  }
  if (type === "object" || (p.properties && typeof p.properties === "object"
      && !Array.isArray(p.properties))) {
    return { ...field, kind: "object", fields: schemaFields(p) };
  }
  return field;
}

/** The `properties` of one schema object, in the schema's own order. */
function schemaFields(s: Record<string, unknown>): QuestionField[] {
  const props = (s.properties && typeof s.properties === "object"
    && !Array.isArray(s.properties) ? s.properties : {}) as Record<string, unknown>;
  const required = Array.isArray(s.required) ? s.required.map((r) => String(r)) : [];
  const fields: QuestionField[] = [];
  for (const [name, raw] of Object.entries(props)) {
    const f = schemaField(name, raw, required);
    if (f) fields.push(f);
  }
  return fields;
}

/** The form an elicitation's `requestedSchema` describes, or none.
 *
 *  NEVER INVENTS A FIELD. A schema with no `properties` (the server is allowed to
 *  send `{}`) yields no fields, and the form says so rather than guessing a key —
 *  the answer object's keys are the server's contract. */
export function questionFields(schema: unknown): QuestionField[] {
  const s = (schema && typeof schema === "object" && !Array.isArray(schema)
    ? schema : {}) as Record<string, unknown>;
  return schemaFields(s);
}

/** A value the form holds for one field: a leaf, or a nested container for an
 *  `object`/`array` field. The shape mirrors the answer object's. */
export type QuestionInput =
  | string
  | boolean
  | QuestionInput[]
  | { [k: string]: QuestionInput };

/** The form's values, keyed by the top-level schema property names. */
export type QuestionValues = Record<string, QuestionInput>;

/** The form value a schema `default` should pre-fill, or undefined to leave the
 *  control empty. A default whose type does not match the field is IGNORED
 *  rather than coerced: a checkbox cannot be "true" because a string said so.
 *
 *  An `object` recurses even when it has NO default of its own, because a nested
 *  property may carry one — a parent default is not required for a child to
 *  pre-fill. */
function defaultFor(f: QuestionField, raw: unknown): QuestionInput | undefined {
  if (raw === null) return undefined;
  switch (f.kind) {
    case "boolean":
      return typeof raw === "boolean" ? raw : undefined;
    case "number":
      return typeof raw === "number" || typeof raw === "string" ? String(raw) : undefined;
    case "choice":
    case "text":
      return typeof raw === "string" ? raw
        : typeof raw === "number" ? String(raw) : undefined;
    case "object": {
      if (raw !== undefined && (typeof raw !== "object" || Array.isArray(raw))) {
        return undefined;
      }
      const src = (raw !== undefined ? raw : {}) as Record<string, unknown>;
      const out = defaultsFrom(f.fields ?? [], src);
      return Object.keys(out).length > 0 ? out : undefined;
    }
    case "array": {
      if (!Array.isArray(raw) || !f.items) return undefined;
      const out: QuestionInput[] = [];
      for (const e of raw) {
        const v = defaultFor(f.items, e);
        if (v !== undefined) out.push(v);
      }
      return out;
    }
  }
}

/** Pre-fill every field that names a default, from `src` when the parent's own
 *  default object supplies a value and from the field's `default` otherwise. */
function defaultsFrom(
  fields: QuestionField[], src: Record<string, unknown>,
): QuestionValues {
  const out: QuestionValues = {};
  for (const f of fields) {
    const raw = src[f.name] !== undefined ? src[f.name]
      : f.hasDefault ? f.default : undefined;
    const v = defaultFor(f, raw);
    if (v !== undefined) out[f.name] = v;
  }
  return out;
}

/** The form's starting values: the schema's own `default`s, so a question that
 *  names one can be answered by pressing send. */
export function questionDefaults(fields: QuestionField[]): QuestionValues {
  return defaultsFrom(fields, {});
}

/** One field's answer value, or undefined when it has none to contribute. */
function answerFor(f: QuestionField, raw: QuestionInput | undefined): unknown {
  switch (f.kind) {
    case "boolean":
      // Always present: a checkbox is a definite answer, and `false` is one.
      return raw === true;
    case "number": {
      if (typeof raw !== "string" || raw.trim() === "") return undefined;
      const n = Number(raw.trim());
      // A non-numeric string would be refused by mcode against its own schema,
      // so it is never sent; `questionProblem` blocks the submit first.
      return Number.isFinite(n) ? n : undefined;
    }
    case "choice":
    case "text":
      return typeof raw === "string" && raw.trim() !== "" ? raw.trim() : undefined;
    case "object": {
      const src = (raw && typeof raw === "object" && !Array.isArray(raw)
        ? raw : {}) as QuestionValues;
      const out = answerObject(f.fields ?? [], src);
      // An object with nothing in it is ABSENT, not `{}`: the same "no blank
      // keys" rule the flat form already used.
      return Object.keys(out).length > 0 ? out : undefined;
    }
    case "array": {
      if (!Array.isArray(raw) || !f.items) return undefined;
      const out: unknown[] = [];
      for (const e of raw) {
        const v = answerFor(f.items, e);
        // A row that expresses nothing is dropped rather than sent as null —
        // which is why an untouched trailing row is not an element.
        if (v !== undefined) out.push(v);
      }
      return out.length > 0 ? out : undefined;
    }
  }
}

function answerObject(
  fields: QuestionField[], values: QuestionValues,
): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  for (const f of fields) {
    const v = answerFor(f, values[f.name]);
    if (v !== undefined) out[f.name] = v;
  }
  return out;
}

/** The answer object for `/approval`, built from what the user filled in.
 *
 *  Keys are the schema's own property names — nested ones included, so the
 *  object mirrors the schema rather than flattening it. A blank optional field is
 *  ABSENT rather than `""`, so mcode can tell "not answered" from "answered with
 *  nothing". */
export function questionAnswer(
  fields: QuestionField[], values: QuestionValues,
): Record<string, unknown> {
  return answerObject(fields, values);
}

/** The first thing stopping submit, in words, or "" when the form is ready.
 *
 *  A sentence rather than a boolean so the form can say WHY. "fill the required
 *  fields first" is wrong for a number that will not parse, and a silent
 *  disabled button on a question with a 5-second life is a dead end. */
function problemFor(f: QuestionField, raw: QuestionInput | undefined): string {
  if (f.kind === "boolean") return "";
  if (f.kind === "object") {
    const src = (raw && typeof raw === "object" && !Array.isArray(raw)
      ? raw : {}) as QuestionValues;
    for (const sub of f.fields ?? []) {
      const p = problemFor(sub, src[sub.name]);
      if (p) return p;
    }
    return "";
  }
  if (f.kind === "array") {
    const rows = Array.isArray(raw) ? raw : [];
    // A required array needs at least one row; an optional one may be absent.
    if (rows.length === 0) return f.required ? `“${f.label}” is required` : "";
    for (const e of rows) {
      const p = f.items ? problemFor(f.items, e) : "";
      if (p) return p;
    }
    return "";
  }
  const s = typeof raw === "string" ? raw.trim() : "";
  if (s === "") return f.required ? `“${f.label}” is required` : "";
  if (f.kind === "number" && !Number.isFinite(Number(s))) {
    return `“${f.label}” must be a number`;
  }
  return "";
}

/** The first problem, in words, or "" when every field is answerable. */
export function questionProblem(
  fields: QuestionField[], values: QuestionValues,
): string {
  for (const f of fields) {
    const p = problemFor(f, values[f.name]);
    if (p) return p;
  }
  return "";
}

/** Whether every required field has an answer. Optional fields never block. */
export function questionReady(
  fields: QuestionField[], values: QuestionValues,
): boolean {
  return questionProblem(fields, values) === "";
}

/** What to say when the approval route refuses an answer with 409.
 *
 *  The route's own sentence — "that request is no longer the one being waited
 *  on" — describes the wire, not what happened to the question. A 409 is the
 *  server saying it is not waiting on this request any more: the window closed,
 *  or it was already answered. Until an expiry EVENT arrives it is the only
 *  signal the client gets, so it must read as the expired state rather than as a
 *  network error. */
export function questionRefusal(message: string, status?: number): string {
  if (status === 409) {
    return "this question is no longer available — it expired or was already answered";
  }
  return message;
}

/** `allowed-once` reads as permission granted; the other three do not. */export function outcomeTone(outcome: string): string {
  if (outcome === "allowed-once") return "text-amber";
  // OD12-n1: `answered` is a question the user DID answer — the turn got what it
  // was waiting for. It is a verdict in favour, not a refusal, and must not sit
  // in the muted default beside `cancelled`.
  if (outcome === "answered") return "text-moss";
  // `expired` is the same fail-closed family as `unavailable`: nobody answered,
  // so the question was NOT answered and the turn went on without one. It must
  // not read as a neutral `cancelled`, which was a decision the user made.
  if (outcome === "rejected" || outcome === "unavailable"
      || outcome === "expired") return "text-red";
  return "text-muted";
}

/** The glyph for a decision: a tick only for a verdict IN FAVOUR.
 *
 *  OD12-n1: the question channel's own two verdicts are `answered` and
 *  `expired`. `expired` belongs with the refusals; `answered` belongs with
 *  `allowed-once` — drawing it as ✕ said the question had failed when it had
 *  succeeded. An empty outcome is `?`: nothing has been decided yet. */
export function outcomeGlyph(outcome: string): string {
  if (!outcome) return "?";
  return outcome === "allowed-once" || outcome === "answered" ? "✓" : "✕";
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
    // OD-12: the word the server sends when a question's 5 s window closed with
    // no answer. "expired" alone reads like a cache miss; the row must say that
    // nobody answered in time and the turn went on without one.
    case "expired": return "expired — no answer in time";
    // OD12-n1: the server's other question verdict — the answer BEAT the window.
    case "answered": return "answered";
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
