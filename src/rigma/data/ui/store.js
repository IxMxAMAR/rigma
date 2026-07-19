// store.js — API client + SSE stream machinery. Server-authoritative:
// every mutation goes through these helpers; UI state is a thin cache.
"use strict";

async function api(method, path, body) {
  const r = await fetch(path, {
    method,
    headers: body !== undefined ? {"content-type": "application/json"} : undefined,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.error || ("server replied " + r.status));
  }
  return r.json();
}

// Plain-language explainer for a gguf quant name (e.g. "IQ4_XS", "UD-Q3_K_XL").
// Pure string -> string; used as tooltips + the Models/Bazaar quant glossary.
function quantHelp(q) {
  const s = String(q || "").toUpperCase();
  if (s === "F16" || s === "BF16")
    return "16-bit — full quality, largest file. Overkill for chat; " +
           "use only if you have VRAM to spare.";
  if (s === "F32") return "32-bit reference precision — enormous, never " +
           "needed for inference.";
  const m = s.match(/^(UD-)?(I?)Q(\d)/);
  if (!m) return "GGUF weights.";
  const ud = !!m[1], iq = m[2] === "I", bits = +m[3];
  const quality = bits >= 6 ? "near-lossless"
    : bits === 5 ? "excellent quality"
    : bits === 4 ? "very good — the size/quality sweet spot"
    : bits === 3 ? "usable, with some quality loss"
    : "small but noticeably degraded";
  const family = iq
    ? "i-quant: uses an importance matrix to spend bits where they matter, " +
      "so it's smaller for the same quality (tiny extra GPU compute)"
    : "K-quant: classic block quantization — well-tested and maximally " +
      "compatible";
  const variant = (s.match(/_(XXS|XS|L|M|S|P)$/) || [, ""])[1];
  const vnote = {XXS: " (XXS = smallest of this level)",
    XS: " (XS = extra-small)", S: " (S = small)",
    M: " (M = medium, balanced)", L: " (L = largest, best quality)",
    P: ""}[variant] || "";
  let note = "~" + bits + "-bit, " + quality + ". " + family + vnote + ".";
  if (ud) note = "Unsloth Dynamic — smarter per-tensor bit allocation for " +
    "better quality at this size. " + note;
  return note;
}

// The quant a given machine should usually pick: the largest (best quality)
// one that still fits. `quants` items look like {quant, fit:{ok}} (bytes DESC).
function recommendedQuant(quants) {
  const fits = (quants || []).filter((q) => q.fit && q.fit.ok);
  return fits.length ? fits[0].quant : null;   // list is largest-first
}

// Pure SSE frame parser: complete events out, unconsumed tail back.
// No DOM, no fetch — node-testable (tests/test_store_js.py).
function sseParse(buffer) {
  buffer = buffer.replace(/\r\n/g, "\n");
  const events = [];
  let rest = buffer;
  for (;;) {
    const cut = rest.indexOf("\n\n");
    if (cut === -1) break;
    const frame = rest.slice(0, cut);
    rest = rest.slice(cut + 2);
    let event = "", data = "";
    for (const ln of frame.split("\n")) {
      if (ln.startsWith("event: ")) event = ln.slice(7).trim();
      else if (ln.startsWith("data: ")) data += (data ? "\n" : "") + ln.slice(6);
    }
    if (data) events.push({event, data});
  }
  return {events, rest};
}

// One chat turn. handlers: {delta, error, citations, meta, done(aborted)}.
// Returns {abort()} — the caller owns persisting any partial text it kept.
function streamTurn(sessionId, payload, handlers) {
  const ctl = new AbortController();
  (async () => {
    let buf = "";
    try {
      const r = await fetch("/api/sessions/" + sessionId + "/chat", {
        method: "POST", headers: {"content-type": "application/json"},
        body: JSON.stringify(payload), signal: ctl.signal,
      });
      if (!r.ok) {
        const e = await r.json().catch(() => ({}));
        throw new Error(e.error || ("server replied " + r.status));
      }
      const reader = r.body.getReader(), dec = new TextDecoder();
      for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        buf += dec.decode(value, {stream: true});
        const parsed = sseParse(buf);
        buf = parsed.rest;
        for (const ev of parsed.events) {
          if (ev.data === "[DONE]") continue;
          let d;
          try { d = JSON.parse(ev.data); } catch { continue; }
          if (ev.event === "error") handlers.error && handlers.error(d);
          else if (ev.event === "citations")
            handlers.citations && handlers.citations(d.citations || []);
          else if (ev.event === "meta") handlers.meta && handlers.meta(d);
          else if (ev.event === "think")
            handlers.think && handlers.think(d.delta || "");
          else if (ev.event === "tool")
            handlers.tool && handlers.tool(d);
          else if (ev.event === "tool_result")
            handlers.toolResult && handlers.toolResult(d);
          else if (ev.event === "compacted")
            handlers.compacted && handlers.compacted(d);
          else if (d.delta) handlers.delta && handlers.delta(d.delta);
        }
      }
      handlers.done && handlers.done(false);
    } catch (err) {
      if (err.name === "AbortError") {
        handlers.done && handlers.done(true);
      } else {
        handlers.error &&
          handlers.error({message: "Couldn't reach the model: " + err.message});
        handlers.done && handlers.done(false);
      }
    }
  })();
  return {abort: () => ctl.abort()};
}
