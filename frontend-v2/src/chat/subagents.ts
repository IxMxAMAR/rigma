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
}

/** The most a subagent's closing message may add to a row. */
const LAST_CLIP = 400;

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
  const p = payload as Record<string, unknown>;

  // TWO SHAPES AGAIN. DSH wraps its lifecycle in `{event, data}`; mcode reports a
  // task's ids flat, from the `details` of a `task`/`task_output` result, and its
  // status IS the lifecycle step. Rather than branch in the renderer, both are
  // reduced to (event, data) here.
  let event = String(p.event ?? "");
  let d: Record<string, unknown>;
  if (p.data && typeof p.data === "object") {
    d = p.data as Record<string, unknown>;
  } else if (p.taskId || p.subSessionId) {
    // mcode: `started`/`running` mean live, anything else is an end.
    const status = String(p.status ?? "").toLowerCase();
    event = status === "started" || status === "running" || status === "queued"
      ? "subagent.started"
      : "subagent.finished";
    d = {
      childSessionId: p.subSessionId ?? p.taskId,
      provider: "mcode",
      status: p.status,
    };
  } else {
    return rows;
  }

  const id = String(d.childSessionId ?? "");
  if (!id) return rows;

  if (event === "subagent.started") {
    // A repeat start for an id already running is ignored rather than
    // duplicated — the same child reported twice is still one child.
    if (rows.some((r) => r.id === id)) return rows;
    return [...rows, { id, state: "running" }];
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
    };
    // A finish with no start — a child spawned before this turn began, or a
    // dropped event — is ADDED rather than dropped. Showing a finished child is
    // strictly better than pretending it never ran.
    if (!rows.some((r) => r.id === id)) return [...rows, patch];
    return rows.map((r) => (r.id === id ? { ...r, ...patch } : r));
  }

  return rows;
}
