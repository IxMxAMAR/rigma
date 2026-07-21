// Methods: the catalog, macro preview, and the macro run stream. Types live
// here rather than beside one consumer because the Sidecar, the MacroStrip
// and the store all need them.
import { makeSseParser, type SseEvent } from "./sse";

export interface MacroDef {
  id: string;
  label: string;
  hint?: string;
}

export interface RuleDef {
  id: string;
  kind: "standing" | "trigger";
  text?: string;
}

export interface Method {
  id: string;
  name: string;
  tagline: string;
  guide: string[];
  builtin?: boolean;
  rules?: RuleDef[];
  macros?: MacroDef[];
  workflows?: MacroDef[];
}

export interface MacroPreview {
  macro_id: string;
  label: string;
  effectful: boolean;
  trusted: boolean;
  needs_confirm: boolean;
  preview: string;
  asks: string[];
}

const JSON_HEADERS = { "content-type": "application/json" };

export async function listMethods(): Promise<Method[]> {
  const r = await fetch("/api/methods");
  if (!r.ok) throw new Error("could not load methods");
  return ((await r.json()) as { methods: Method[] }).methods ?? [];
}

export async function previewMacro(
  sid: string,
  macroId: string,
  answers?: Record<string, string>,
): Promise<MacroPreview> {
  const r = await fetch(`/api/sessions/${sid}/macro/preview`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ macro_id: macroId, answers }),
  });
  const d = (await r.json()) as MacroPreview & { error?: string };
  if (!r.ok) throw new Error(d.error ?? "preview failed");
  return d;
}

export async function createDraft(): Promise<
  { session_id: string; draft_id: string }
> {
  const r = await fetch("/api/methods/draft", {
    method: "POST",
    headers: JSON_HEADERS,
    body: "{}",
  });
  if (!r.ok) throw new Error("could not start a method");
  return (await r.json()) as { session_id: string; draft_id: string };
}

export async function getDraft(did: string): Promise<Method | null> {
  const r = await fetch(`/api/methods/draft/${did}`);
  if (!r.ok) return null;
  return (await r.json()) as Method;
}

export async function promoteDraft(did: string): Promise<Method> {
  const r = await fetch(`/api/methods/draft/${did}/promote`, {
    method: "POST",
  });
  const d = (await r.json()) as Method & { errors?: string[] };
  if (!r.ok) throw new Error((d.errors ?? ["could not save"]).join("; "));
  return d;
}

/** Streams the macro run, handing each SSE event to `onEvent`. Mirrors
 *  streamChat so the store can fold both through the same reducer. */
export async function runMacroStream(
  sid: string,
  body: {
    macro_id: string;
    confirm?: "run" | "always";
    answers?: Record<string, string>;
    selection?: string;
  },
  onEvent: (ev: SseEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const res = await fetch(`/api/sessions/${sid}/macro`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    let msg = `macro failed (${res.status})`;
    try {
      msg = ((await res.json()) as { error?: string }).error ?? msg;
    } catch { /* keep the status-code message */ }
    throw new Error(msg);
  }
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  const parse = makeSseParser();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    for (const ev of parse(decoder.decode(value, { stream: true }))) {
      onEvent(ev);
    }
  }
}
