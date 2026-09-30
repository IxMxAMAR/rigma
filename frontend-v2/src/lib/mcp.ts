// D4c: `GET /api/mcp` — the configured/running MCP servers and the tools they
// contribute.
//
// The route is NOT polled and NOT fetched on mount, on purpose: its handler
// calls `mcp_client.manager().status()`, and `status()` calls `_ensure()`, which
// STARTS every configured server (mcp_client.py:433-444). Opening the Settings
// page must not spawn processes, so the card fetches only when the user asks.
import { bodyError } from "./listFetch";

export interface McpStatus {
  configured: string[];
  running: string[];
  failed: Record<string, string>;
  dead?: Record<string, string>;
  wedged?: Record<string, number>;
  oversize_frames?: Record<string, number>;
  tools: string[];
  /** Present when no config exists at all; names the file to create. */
  hint?: string;
}

export type McpResult =
  | { ok: true; status: McpStatus }
  | { ok: false; error: string };

const strList = (v: unknown): string[] =>
  Array.isArray(v) ? v.filter((x): x is string => typeof x === "string") : [];

/** The route's body, or null when it is not the shape the route promises. */
export function mcpBody(d: unknown): McpStatus | null {
  if (!d || typeof d !== "object" || Array.isArray(d)) return null;
  const o = d as Record<string, unknown>;
  // The unconfigured arm answers a hint and empty lists; both are real.
  if (o.configured !== undefined && !Array.isArray(o.configured)) return null;
  if (o.running !== undefined && !Array.isArray(o.running)) return null;
  if (o.tools !== undefined && !Array.isArray(o.tools)) return null;
  if (o.failed !== undefined
      && (o.failed === null || typeof o.failed !== "object"
          || Array.isArray(o.failed))) {
    return null;
  }
  return {
    configured: strList(o.configured),
    running: strList(o.running),
    failed: (o.failed ?? {}) as Record<string, string>,
    dead: (o.dead ?? {}) as Record<string, string>,
    wedged: (o.wedged ?? {}) as Record<string, number>,
    oversize_frames: (o.oversize_frames ?? {}) as Record<string, number>,
    tools: strList(o.tools),
    hint: typeof o.hint === "string" ? o.hint : undefined,
  };
}

/** The one-line verdict, pure so it is testable without a fetch. */
export function mcpLine(s: McpStatus): string {
  const bits = [
    `${s.configured.length} configured`,
    `${s.running.length} running`,
    `${s.tools.length} ${s.tools.length === 1 ? "tool" : "tools"}`,
  ];
  const broken = Object.keys(s.failed).length;
  if (broken) bits.push(`${broken} failed`);
  const dead = Object.keys(s.dead ?? {}).length;
  if (dead) bits.push(`${dead} dead`);
  const wedged = Object.keys(s.wedged ?? {}).length;
  if (wedged) bits.push(`${wedged} wedged`);
  return bits.join(" · ");
}

/** `GET /api/mcp`. Never throws: the 500's `{error}` is the server's own. */
export async function loadMcp(): Promise<McpResult> {
  let r: Response;
  try {
    r = await fetch("/api/mcp");
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  const body: unknown = await r.json().catch(() => null);
  if (!r.ok) return { ok: false, error: bodyError(body, r) };
  const status = mcpBody(body);
  return status
    ? { ok: true, status }
    : { ok: false, error: "the MCP API returned an unexpected body" };
}
