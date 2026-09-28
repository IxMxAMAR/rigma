// The context firewall, made visible.
//
// `delegate` hands a research question to a fresh-context helper. The helper's
// tool calls deliberately do NOT re-enter the main context — that is the entire
// point, and it is why they are recorded on the assistant message as a UI-only
// field (`delegate_trace`) rather than in `tool_trace`, which feeds context and
// run scoring. The consequence was that the whole exploration was invisible: the
// transcript showed one `delegate` chip and then an answer, with no way to see
// that twelve file reads had happened behind it.
//
// This is pure logic so it can be tested; `@testing-library/react` is not a
// dependency of this project and there are no `*.test.tsx` files.

export interface DelegateCall {
  name: string;
  args?: unknown;
  ok?: boolean;
  blocked?: boolean;
  ts?: number;
}

export interface DelegateSummary {
  /** How many calls the helper actually made, excluding blocked ones. */
  ran: number;
  /** How many it was refused, because the helper's roster is read-only. */
  blocked: number;
  /** How many of the ones that ran reported an error. */
  failed: number;
  /** The distinct tools it used, in first-use order. */
  tools: string[];
}

/** Count a delegate trace. Pure; never mutates its input. */
export function summariseDelegate(rows: DelegateCall[]): DelegateSummary {
  const tools: string[] = [];
  let ran = 0;
  let blocked = 0;
  let failed = 0;
  for (const r of rows) {
    const name = String(r?.name ?? "");
    if (r?.blocked) {
      blocked += 1;
      continue;
    }
    ran += 1;
    if (r?.ok === false) failed += 1;
    if (name && !tools.includes(name)) tools.push(name);
  }
  return { ran, blocked, failed, tools };
}

/** The one-line summary. Says what ran, and says plainly when something did
 *  not — a helper quietly refused a tool is a fact about the answer. */
export function delegateSentence(s: DelegateSummary): string {
  if (s.ran === 0 && s.blocked === 0) return "a research helper ran, and used no tools";
  const parts: string[] = [];
  parts.push(
    `a research helper ran ${s.ran} tool call${s.ran === 1 ? "" : "s"}`,
  );
  if (s.tools.length) {
    const shown = s.tools.slice(0, 6).join(", ");
    parts.push(`using ${shown}${s.tools.length > 6 ? `, +${s.tools.length - 6} more` : ""}`);
  }
  if (s.failed) parts.push(`${s.failed} failed`);
  if (s.blocked) parts.push(`${s.blocked} refused`);
  return parts.join(" — ");
}

/** One argument value, short enough to sit on a row. */
export function argHint(args: unknown): string {
  if (args == null || typeof args !== "object") return "";
  const entries = Object.entries(args as Record<string, unknown>);
  if (!entries.length) return "";
  return entries
    .slice(0, 3)
    .map(([k, v]) => {
      const s = typeof v === "string" ? v : JSON.stringify(v);
      const t = (s ?? "").length > 60 ? `${(s ?? "").slice(0, 60)}…` : (s ?? "");
      return `${k}=${t}`;
    })
    .join(" ");
}
