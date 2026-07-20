// SSE over fetch + ReadableStream (not EventSource): POST bodies and
// AbortController-based instant cancel, per the stack decision.
export interface SseEvent {
  event: string;
  data: unknown;
}

/** Incremental SSE parser. Feed chunks, get complete events. Pure — the
 *  unit-testable half of streaming. */
export function makeSseParser() {
  let buf = "";
  return (chunk: string): SseEvent[] => {
    buf += chunk;
    const out: SseEvent[] = [];
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) !== -1) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7).trim();
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data || data === "[DONE]") continue;
      try {
        out.push({ event, data: JSON.parse(data) });
      } catch {
        /* partial or malformed frame — drop, never crash the stream */
      }
    }
    return out;
  };
}

/** POST a chat turn and deliver parsed events until the stream closes. */
export async function streamChat(
  sessionId: string,
  body: Record<string, unknown>,
  onEvent: (ev: SseEvent) => void,
  signal: AbortSignal,
): Promise<void> {
  const r = await fetch(`/api/sessions/${sessionId}/chat`, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok || !r.body) {
    const err = await r.json().catch(() => ({}) as { error?: string });
    throw new Error(
      (err as { error?: string }).error ?? `server replied ${r.status}`,
    );
  }
  const parse = makeSseParser();
  const reader = r.body.getReader();
  const dec = new TextDecoder();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const ev of parse(dec.decode(value, { stream: true }))) onEvent(ev);
  }
}
