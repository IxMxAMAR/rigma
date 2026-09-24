// Focus the chat surface's session filter.
//
// IMP-9 asks for ⌘/Ctrl+K to focus the session filter, but Ctrl+K is already
// the command palette's binding and UI-REWORK-PLAN makes the palette a
// first-class citizen — taking the key would break it. So the rail is focused
// by `/` (when not typing) and by a "Filter chats" palette command, which is
// reachable through Ctrl+K anyway.
export const CHAT_FILTER_EVENT = "rigma:focus-chat-filter";

export function focusChatFilter(): void {
  // Deferred: the caller may be switching to the chat surface, whose rail only
  // registers the listener on the next render.
  const defer = typeof requestAnimationFrame === "function"
    ? requestAnimationFrame
    : (fn: () => void) => window.setTimeout(fn, 0);
  defer(() => window.dispatchEvent(new Event(CHAT_FILTER_EVENT)));
}
