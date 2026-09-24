import { describe, expect, it } from "vitest";

import { isSendKey, isStopKey, isTypingTarget } from "./keyboard";

// IMP-9: the two rules that matter are about not stealing a key. Esc belongs to
// the command palette when it is open, and to a single-line input that is
// cancelling an edit; Enter sends only without Shift.
describe("chat keyboard rules", () => {
  it("sends on Enter and not on Shift+Enter", () => {
    expect(isSendKey({ key: "Enter", shiftKey: false })).toBe(true);
    expect(isSendKey({ key: "Enter", shiftKey: true })).toBe(false);
    expect(isSendKey({ key: "a", shiftKey: false })).toBe(false);
  });

  it("stops on Esc only while a turn is streaming", () => {
    expect(isStopKey({ key: "Escape", target: { tagName: "BODY" } },
                     { streaming: true, paletteOpen: false })).toBe(true);
    expect(isStopKey({ key: "Escape", target: { tagName: "BODY" } },
                     { streaming: false, paletteOpen: false })).toBe(false);
  });

  it("never stops the turn when the palette is open", () => {
    // The palette's own Esc handler closes it; stopping the reply as a side
    // effect of dismissing the palette would be a surprise.
    expect(isStopKey({ key: "Escape", target: { tagName: "BODY" } },
                     { streaming: true, paletteOpen: true })).toBe(false);
  });

  it("leaves Esc to a single-line input that is cancelling an edit", () => {
    expect(isStopKey({ key: "Escape", target: { tagName: "INPUT" } },
                     { streaming: true, paletteOpen: false })).toBe(false);
    expect(isStopKey({ key: "Escape", target: { tagName: "SELECT" } },
                     { streaming: true, paletteOpen: false })).toBe(false);
    // the composer is a textarea: Esc there does stop
    expect(isStopKey({ key: "Escape", target: { tagName: "TEXTAREA" } },
                     { streaming: true, paletteOpen: false })).toBe(true);
  });

  it("treats a bare-key shortcut as off-limits while typing", () => {
    expect(isTypingTarget({ tagName: "INPUT" })).toBe(true);
    expect(isTypingTarget({ tagName: "textarea" })).toBe(true);
    expect(isTypingTarget({ tagName: "DIV", isContentEditable: true })).toBe(true);
    expect(isTypingTarget({ tagName: "DIV" })).toBe(false);
    expect(isTypingTarget(null)).toBe(false);
  });
});
