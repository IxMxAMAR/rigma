// DSH's compaction lifecycle, made visible.
//
// WHY THIS EXISTS. Rigma's NATIVE compaction already reports itself: the server
// emits `masked`, `housekeeping` and `compacted`, and the transcript draws all
// three ("compacted 12 messages into the summary"). A DSH turn had none of that.
// DSH reports its compaction as four session events and Rigma recognised none of
// them, so a long DSH turn that was busy summarising its own context looked
// exactly like a turn that had hung.
//
// The four events are a LIFECYCLE, not four facts, which is why this folds them
// rather than appending them: `compaction/start` opens a bracket and
// `compaction/end` closes it, paired by `compactionId`. Showing them as separate
// lines would report "compaction started" forever if the end never arrived.
//
// Shapes quoted from DSH's own declarations
// (packages/compaction/compaction/src/types.ts:24-89):
//
//   compaction/start    {compactionId, turn, sourceCommandId?}
//   compaction/summary  {compactionId, summary: ContentBlock[], shadowedRange,
//                        shadowedSeqs, shadowedTokenCount, provider, model, ...}
//   compaction/prune    {shadowedRange, shadowedSeqs, shadowedTokenCount}
//   compaction/end      {compactionId, turn, sourceCommandId?, error?}
//
// Note `prune` carries NO compactionId — it is a model-free replacement priced by
// the metering event right before it, so it is attributed to the compaction that
// is currently open. That is what `openId` is for.
//
// All four are `log-only` in DSH's words: durable and replayable, never part of
// the model transcript. Nothing here can be acted on, and nothing here pretends
// to be — it is a progress and audit surface.

export interface Compaction {
  id: string;
  /** Set by `start`; cleared by `end`. */
  open: boolean;
  /** Tokens the summariser folded away — the whole point of showing this. */
  shadowedTokens: number;
  /** Messages whose surface nodes were shadowed. */
  shadowedCount: number;
  /** The summary text, once one has arrived. */
  summary: string;
  /** Which route wrote the summary, e.g. `deepseek-official`. */
  provider: string;
  model: string;
  /** DSH's own failure text. A failed compaction is the one thing a reader must
   *  not miss: the context did NOT shrink and the turn may be about to fail. */
  error: string;
}

export interface Compactions {
  items: Compaction[];
  /** The id of the still-open compaction, or "".
   *
   *  Tracked explicitly because `compaction/prune` has no id of its own and
   *  because "is a compaction running right now" is the question the UI actually
   *  asks. `items` alone cannot answer it without a scan. */
  openId: string;
}

export const EMPTY_COMPACTIONS: Compactions = { items: [], openId: "" };

/** Flatten DSH's `ContentBlock[]` summary into text.
 *
 *  Blocks are not strings: a summary may be a mix of text and other block kinds.
 *  Joined with "" rather than " " — a space between blocks inserts a gap inside a
 *  word wherever the provider split a block mid-token, which is the same bug
 *  already fixed once in the DSH subagent rendering. Non-text blocks are dropped
 *  rather than stringified: `[object Object]` in a status line is worse than
 *  nothing. */
export function summaryText(blocks: unknown): string {
  if (!Array.isArray(blocks)) return "";
  const parts: string[] = [];
  for (const b of blocks) {
    if (typeof b === "string") {
      parts.push(b);
      continue;
    }
    if (b && typeof b === "object") {
      const t = (b as Record<string, unknown>).text;
      if (typeof t === "string") parts.push(t);
    }
  }
  return parts.join("").trim();
}

function num(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 0;
}

function str(v: unknown): string {
  return typeof v === "string" ? v : "";
}

/** Fold one `compaction/*` payload in. Returns `gov` unchanged for anything that
 *  is not a compaction event, so the caller's reducer arm can be unconditional. */
export function foldCompaction(prev: Compactions, payload: unknown): Compactions {
  if (!payload || typeof payload !== "object") return prev;
  const p = payload as Record<string, unknown>;
  const raw = str(p.event);
  // The prefix is REQUIRED, not merely stripped: stripping a prefix that is not
  // there is a no-op, so `{event: "goal"}` would otherwise be recorded as a
  // compaction kind called "goal". Same bug already fixed in governance.ts.
  if (!raw.startsWith("compaction/")) return prev;
  const kind = raw.slice("compaction/".length);
  const d = (p.data && typeof p.data === "object" ? p.data : {}) as Record<string, unknown>;
  const id = str(d.compactionId);

  const items = prev.items.slice();
  const find = (want: string) => items.findIndex((c) => c.id === want);

  switch (kind) {
    case "start": {
      // A repeated start for the same id (a replay) must not duplicate the row.
      const at = id ? find(id) : -1;
      if (at >= 0) {
        items[at] = { ...items[at], open: true };
      } else {
        items.push({
          id, open: true, shadowedTokens: 0, shadowedCount: 0,
          summary: "", provider: "", model: "", error: "",
        });
      }
      return { items, openId: id };
    }
    case "summary": {
      const at = id ? find(id) : -1;
      if (at < 0) {
        // A summary whose start we never saw. Kept, not dropped: the fold knows
        // real work happened even when the bracket is missing.
        items.push({
          id, open: false,
          shadowedTokens: num(d.shadowedTokenCount),
          shadowedCount: Array.isArray(d.shadowedSeqs) ? d.shadowedSeqs.length : 0,
          summary: summaryText(d.summary),
          provider: str(d.provider), model: str(d.model), error: "",
        });
        return { items, openId: prev.openId };
      }
      items[at] = {
        ...items[at],
        shadowedTokens: num(d.shadowedTokenCount),
        shadowedCount: Array.isArray(d.shadowedSeqs) ? d.shadowedSeqs.length : 0,
        summary: summaryText(d.summary) || items[at].summary,
        provider: str(d.provider) || items[at].provider,
        model: str(d.model) || items[at].model,
      };
      return { items, openId: prev.openId };
    }
    case "prune": {
      // No id of its own — attributed to the open compaction, or to the most
      // recent one if the bracket has already closed.
      const at = prev.openId ? find(prev.openId)
        : (items.length > 0 ? items.length - 1 : -1);
      if (at < 0) return prev;
      items[at] = {
        ...items[at],
        shadowedTokens: items[at].shadowedTokens + num(d.shadowedTokenCount),
        shadowedCount: items[at].shadowedCount
          + (Array.isArray(d.shadowedSeqs) ? d.shadowedSeqs.length : 0),
      };
      return { items, openId: prev.openId };
    }
    case "end": {
      const at = id ? find(id) : -1;
      if (at < 0) return prev;
      items[at] = { ...items[at], open: false, error: str(d.error) };
      return { items, openId: prev.openId === id ? "" : prev.openId };
    }
    default:
      // An event name from a newer DSH than this Rigma knows. Ignored rather
      // than guessed at, the same rule the runner applies to unknown kinds.
      return prev;
  }
}

/** The still-open compaction, or null. What the UI asks to decide whether to
 *  show "compacting…" rather than a frozen turn. */
export function running(c: Compactions): Compaction | null {
  if (!c.openId) return null;
  return c.items.find((x) => x.id === c.openId) ?? null;
}

/** One line describing what a finished compaction did, or null if it has nothing
 *  worth saying. Built from the numbers so it cannot claim more than it knows. */
export function compactionLine(c: Compaction): string | null {
  if (c.error) return `compaction failed — ${c.error}`;
  const bits: string[] = [];
  if (c.shadowedCount > 0) {
    bits.push(`${String(c.shadowedCount)} message${c.shadowedCount === 1 ? "" : "s"}`);
  }
  if (c.shadowedTokens > 0) {
    bits.push(`${c.shadowedTokens.toLocaleString()} tokens`);
  }
  if (bits.length === 0) return null;
  return `compacted ${bits.join(" · ")} into a summary`;
}
