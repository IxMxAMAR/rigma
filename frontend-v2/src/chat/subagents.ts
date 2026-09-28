// Folding a backend's subagent lifecycle into rows a person can read.
//
// WHY THIS IS ITS OWN MODULE. The house convention is that pure logic lives in a
// plain `.ts` beside the surface and gets the tests, because nothing here renders
// components (`@testing-library/react` is not a dependency and there are no
// `*.test.tsx` files). Folding lifecycle events is exactly the kind of logic that
// is easy to get subtly wrong — an end arriving before a start, a child that is
// never ended, a repeated start — so it is separated to be testable.
//
// THE TWO SHAPES ARE DIFFERENT, which is the trap. DSH reports
// `subagent.started` as `{parentSessionId, childSessionId}` and
// `subagent.finished` as `{provider, agentId, parentSessionId, childSessionId,
// status, stopReason, lastAssistantMessage?}`. The child's session id is the only
// key present in both, so it is what a row is keyed by.

export interface Subagent {
  /** The child's session id — the only field both events carry. */
  id: string;
  /** "running" until a finish arrives for this id. */
  state: "running" | "done";
  /** The provider that ran it ("spawn", "fork"), when the finish event said. */
  provider?: string;
  /** Why it stopped, when the backend said. */
  stopReason?: string;
  /** "ok" | "error" — the finish event's own word, not inferred. */
  status?: string;
  /** The child's last message, clipped. The reason to show a row at all. */
  last?: string;
  /**
   * A human name for the child, when the backend has one.
   *
   * R4-MCODE-2. DSH's lifecycle pair carries no name — `subagent.started` is
   * `{parentSessionId, childSessionId}` and nothing else — so a row could only
   * ever say "subagent running", which is unreadable the moment there are two.
   * The name does arrive, just on OTHER events: `subagent/descriptor` carries
   * `label` and `subagent/catalog` carries `label` keyed by `childId`, which is
   * the same child session id used here. mcode sends `agent_name` directly.
   * Both were being forwarded by the server and then discarded by this fold.
   */
  name?: string;
}

/** The most a subagent's closing message may add to a row. */
const LAST_CLIP = 400;

/**
 * Reduce one subagent payload to `(event, data)`, whichever backend sent it.
 *
 * WHY THIS EXISTS. There are three shapes on the wire and only one of them used
 * to be handled:
 *
 *   1. DSH lifecycle: `{event: "subagent.started"|"subagent.finished", data}`
 *      where data carries `childSessionId`.
 *   2. mcode, wrapped by the server: `{event: "subagent", data: {taskId,
 *      subSessionId, name, status}}`. The server wraps EVERY subagent payload
 *      (serve.py, the `_ev.startswith("subagent")` arm), so this is what mcode
 *      actually arrives as.
 *   3. mcode, flat: `{taskId, subSessionId, name, status}` with no `data`.
 *
 * Shape 3 was the only mcode branch written, and shape 2 is what the server
 * produces — so mcode subagents rendered NOTHING, and the tests that covered the
 * flat shape exercised a payload the real server never sends. Recognising
 * mcode's fields INSIDE `data` is what makes shape 2 work.
 */
function normalise(p: Record<string, unknown>): { event: string; d: Record<string, unknown> } | null {
  const inner = p.data && typeof p.data === "object"
    ? (p.data as Record<string, unknown>)
    : null;
  // mcode's own field names, whether or not the server wrapped them.
  const flat = inner && (inner.subSessionId || inner.taskId) ? inner : null;
  const loose = !inner && (p.subSessionId || p.taskId) ? p : null;
  const mcode = flat ?? loose;
  if (mcode) {
    // mcode's status IS the lifecycle step: `started`/`running`/`queued` mean
    // live, anything else (completed, failed, cancelled) is an end.
    const status = String(mcode.status ?? "").toLowerCase();
    const event = status === "started" || status === "running" || status === "queued"
      ? "subagent.started"
      : "subagent.finished";
    return {
      event,
      d: {
        childSessionId: mcode.subSessionId ?? mcode.taskId,
        provider: "mcode",
        status: mcode.status,
        name: mcode.name,
      },
    };
  }
  if (inner) return { event: String(p.event ?? ""), d: inner };
  return null;
}

/**
 * The child's id, whichever name the event uses.
 *
 * TWO NAMES FOR ONE THING. DSH's lifecycle pair says `childSessionId`
 * (`subagent.started` is `{parentSessionId, childSessionId}`), while
 * `subagent/catalog` says `childId` — verified against the source, not guessed:
 * `SubagentCatalogEvent` is `{version, childId, childCreatedAt, mode, label?}`
 * (packages/subagent/subagent/src/catalog.ts:24-32). Reading only the first name
 * meant a catalog entry never matched a row, which is how the one human-readable
 * name DSH sends for a subagent was thrown away.
 */
function childIdOf(d: Record<string, unknown>): string {
  return String(d.childSessionId ?? d.childId ?? "");
}

/** One line of text out of the content-block shape a finish event carries. */
function lastText(blocks: unknown): string {
  if (!Array.isArray(blocks)) return "";
  const parts: string[] = [];
  for (const b of blocks) {
    if (b && typeof b === "object") {
      const t = (b as { text?: unknown }).text;
      if (typeof t === "string" && t) parts.push(t);
    }
  }
  // Concatenated, NOT joined with a space: a message arrives as a list of
  // arbitrary slices, so "found " + "three" is one sentence and joining on a
  // space inserts a double space at every block boundary.
  return parts.join("").slice(0, LAST_CLIP);
}

/**
 * Fold one `subagent` SSE payload into the list, returning a NEW array.
 *
 * Unknown event names return the list unchanged rather than appending a blank
 * row: a newer backend naming a lifecycle step this build does not know is not a
 * reason to render an empty entry.
 */
export function foldSubagent(rows: Subagent[], payload: unknown): Subagent[] {
  if (!payload || typeof payload !== "object") return rows;
  const n = normalise(payload as Record<string, unknown>);
  if (!n) return rows;
  const { event, d } = n;

  const id = childIdOf(d);
  if (!id) return rows;

  // The name travels on DSH's descriptor/catalog events, which are NOT lifecycle
  // steps — they can arrive before the start or after the finish. So a name is
  // applied to an existing row whenever one is present, on every shape, rather
  // than only at the two lifecycle transitions below.
  const label = d.name != null && String(d.name) ? String(d.name)
    : (d.label != null && String(d.label) ? String(d.label) : undefined);

  if (event === "subagent.started") {
    // A repeat start for an id already running is ignored rather than
    // duplicated — the same child reported twice is still one child.
    const at = rows.findIndex((r) => r.id === id);
    if (at >= 0) {
      return label ? rows.map((r, i) => (i === at ? { ...r, name: label } : r)) : rows;
    }
    return [...rows, { id, state: "running", name: label }];
  }

  if (event === "subagent.finished") {
    const last = lastText(d.lastAssistantMessage);
    const patch: Subagent = {
      id,
      state: "done",
      provider: d.provider != null ? String(d.provider) : undefined,
      stopReason: d.stopReason != null ? String(d.stopReason) : undefined,
      status: d.status != null ? String(d.status) : undefined,
      last: last || undefined,
      name: label,
    };
    // A finish with no start — a child spawned before this turn began, or a
    // dropped event — is ADDED rather than dropped. Showing a finished child is
    // strictly better than pretending it never ran.
    const at = rows.findIndex((r) => r.id === id);
    if (at < 0) return [...rows, patch];
    return rows.map((r, i) => (i === at ? { ...r, ...patch } : r));
  }

  // Not a lifecycle step, but it may still be a name for a child already known
  // (DSH's `subagent/descriptor` and `subagent/catalog`). Apply it and stop.
  if (label) {
    const at = rows.findIndex((r) => r.id === id);
    if (at >= 0) return rows.map((r, i) => (i === at ? { ...r, name: label } : r));
  }
  return rows;
}
