// Chat keyboard rules, as pure predicates (IMP-9).
//
// They live here rather than inline in the composer because the two that matter
// are about NOT stealing a key: Esc closes the command palette, and Esc cancels
// an edit in a single-line input. Only the composer (a textarea) and the page
// itself should stop a turn.

/** Enter sends; Shift+Enter is a newline. */
export function isSendKey(e: { key: string; shiftKey: boolean }): boolean {
  return e.key === "Enter" && !e.shiftKey;
}

/** Esc stops a streaming turn — unless the palette is open (Esc belongs to it)
 *  or focus is in an `<input>`/`<select>`, where Esc means "cancel this edit"
 *  (the workspace path, the session filter, the palette's own box).
 *  `target` is typed loosely so a real KeyboardEvent (whose target is an
 *  EventTarget) can be passed straight in. */
export function isStopKey(
  e: { key: string; target?: unknown },
  ctx: { streaming: boolean; paletteOpen: boolean },
): boolean {
  if (e.key !== "Escape" || !ctx.streaming || ctx.paletteOpen) return false;
  const tag = (e.target as { tagName?: string } | null | undefined)
    ?.tagName?.toUpperCase();
  return tag !== "INPUT" && tag !== "SELECT";
}

/** Does this event come from somewhere the user is typing, so a bare-key
 *  shortcut (like `/` for the session filter) must be left alone? */
export function isTypingTarget(
  target: { tagName?: string; isContentEditable?: boolean } | null | undefined,
): boolean {
  if (!target) return false;
  if (target.isContentEditable) return true;
  const tag = target.tagName?.toUpperCase();
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}
