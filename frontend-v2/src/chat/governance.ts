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
  /** `asked`, `decided` or `policy` — the DSH name with its prefix stripped. */
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

/** `allowed-once` reads as permission granted; the other three do not. */
export function outcomeTone(outcome: string): string {
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
