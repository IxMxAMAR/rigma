import type { ChatMessage } from "../lib/api";

/** D3b: what the tail of a transcript should say, and whether it is still live.
 *
 *  WHY THIS IS PURE AND SEPARATE. The transcript reloads from the server, and a
 *  mid-turn chat has a `partial` message carrying the server's own notice —
 *  "this reply was interrupted". On a RELOAD that sentence is read by someone who
 *  did nothing to interrupt anything: the turn is still running on the server, and
 *  the reloaded text is simply the last checkpoint. The false notice is the defect
 *  D3a/D3b exist for. The rule that decides is one boolean (`streaming`) and one
 *  flag (`partial`), so it lives here where a test can reach it rather than in the
 *  render.
 *
 *  NOTHING IS INVENTED. With `streaming` false the server's own sentence is passed
 *  through unchanged — this module decides when a notice is WRONG, never what it
 *  should have said.
 */
export interface LiveTail {
  /** The sentence to draw beside the last message, or "" when there is none. */
  text: string;
  /** Whether the SERVER is still generating this chat — drives the rail dot. */
  running: boolean;
  /** The server's own notice on the last message, unchanged — "" when there is
   *  none. Kept so a caller can tell a SUPPRESSED notice from an absent one. */
  notice: string;
}

export function liveTail(messages: ChatMessage[], streaming: boolean): LiveTail {
  const last = messages.length > 0 ? messages[messages.length - 1] : null;
  const notice = last?.notice ?? "";
  // A finished message has nothing to say about liveness. The rail dot is still
  // reported: a chat can be generating with no partial stored yet.
  if (!last || last.partial !== true) {
    return { text: "", running: streaming, notice };
  }
  if (streaming) {
    // The checkpoint is not an interruption. Say what is actually true, and say
    // where the text came from — the reader is looking at the last 20 s snapshot,
    // not the whole reply.
    return {
      text: "still generating on the server — showing the last saved checkpoint",
      running: true,
      notice,
    };
  }
  return { text: notice, running: false, notice };
}
