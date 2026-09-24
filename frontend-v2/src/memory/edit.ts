// Editing one learned rule (IMP-6).
//
// PATCH /api/memory/{id} accepts text, kind, status and outcome_score
// (MemoryStore.EDITABLE). The UI edits the rule text and its status; the pure
// part is deciding WHICH fields to send — a write of unchanged text makes the
// store recompute the embedding and re-run the raw-trace guard for nothing.

export const MEMORY_STATUSES = ["draft", "verified", "retired"] as const;

export interface MemoryPatch {
  text?: string;
  status?: string;
}

/** Only the fields that actually changed. A blank text box is treated as "not
 *  an edit" rather than as "erase the rule" — a rule with no text is not
 *  something this UI can create. */
export function memoryPatch(
  row: { text: string; status: string },
  text: string,
  status: string,
): MemoryPatch {
  const out: MemoryPatch = {};
  const t = text.trim();
  if (t && t !== row.text) out.text = t;
  if (status && status !== row.status) out.status = status;
  return out;
}
