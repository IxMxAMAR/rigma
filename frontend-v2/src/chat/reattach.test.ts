import { describe, expect, it } from "vitest";

import type { ChatMessage } from "../lib/api";
import { liveTail } from "./reattach";

// D3b. The server checkpoints a turn every 20 s into a `partial` message whose
// own notice reads "this reply was interrupted". On a RELOAD the reader did
// nothing to interrupt anything — the turn is still running — so that sentence is
// false. The rule that decides is `streaming` (D3a) and `partial` (AUDIT F7), and
// it lives here so a test can reach it without a renderer.
//
// The notice string is copied from `serve.py`'s `_partial_message`.

const NOTICE =
  "_(this reply was interrupted — the text above is what had been generated. "
  + "Say **continue** to resume it.)_";

const partial = (over: Partial<ChatMessage> = {}): ChatMessage => ({
  role: "assistant",
  content: "half a reply",
  partial: true,
  ckpt_id: "ck1",
  notice: NOTICE,
  ...over,
});

const final = (over: Partial<ChatMessage> = {}): ChatMessage => ({
  role: "assistant",
  content: "a whole reply",
  ...over,
});

describe("liveTail", () => {
  it("suppresses the false interruption while the server is still generating", () => {
    const tail = liveTail([partial()], true);
    expect(tail.text).toContain("still generating on the server");
    expect(tail.text).not.toContain("interrupted");
    expect(tail.running).toBe(true);
    // The server's own sentence is kept, so a caller can tell what was suppressed.
    expect(tail.notice).toBe(NOTICE);
  });

  it("keeps the server's own notice when the turn really did stop", () => {
    const tail = liveTail([partial()], false);
    expect(tail.text).toBe(NOTICE);
    expect(tail.text).toContain("interrupted");
    expect(tail.running).toBe(false);
  });

  it("says nothing about a finished message, and never calls it interrupted", () => {
    expect(liveTail([final()], false).text).toBe("");
    expect(liveTail([final()], true).text).toBe("");
  });

  it("still reports a live chat with no checkpoint stored yet", () => {
    // A turn can be generating before its first 20 s checkpoint.
    expect(liveTail([], true)).toEqual({ text: "", running: true, notice: "" });
    expect(liveTail([final()], true).running).toBe(true);
    expect(liveTail([final()], false).running).toBe(false);
  });

  it("reads only the TAIL, so an older interruption is not re-labelled", () => {
    // An earlier turn's partial that was genuinely interrupted, followed by a
    // finished message: the tail is finished, so there is nothing to say.
    const tail = liveTail([partial({ ckpt_id: "old" }), final()], true);
    expect(tail.text).toBe("");
    // And an older partial must not borrow the tail's liveness either: the
    // notice shown belongs to the message being rendered.
    expect(tail.notice).toBe("");
  });

  it("passes through a notice on a final message untouched", () => {
    const m = final({ notice: "compacted" });
    expect(liveTail([m], false).text).toBe("");
    expect(liveTail([m], false).notice).toBe("compacted");
  });
});
