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
