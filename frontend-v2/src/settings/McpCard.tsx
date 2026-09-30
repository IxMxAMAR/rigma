// D4c: `GET /api/mcp` — a status report for the MCP servers in the user's own
// config, and the tools they contribute to a turn.
//
// It is deliberately MANUAL. The route's handler calls
// `mcp_client.manager().status()`, which calls `_ensure()` and therefore STARTS
// every configured server; a card that fetched on mount would spawn processes
// every time the Settings page opened. The button says so.
import { useState } from "react";

import { loadMcp, mcpLine, type McpStatus } from "../lib/mcp";

export default function McpCard() {
  const [status, setStatus] = useState<McpStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const check = async () => {
    setBusy(true);
    setError(null);
    const r = await loadMcp();
    setBusy(false);
    if (r.ok) setStatus(r.status);
    else {
      setStatus(null);
      setError(r.error);
    }
  };

  const failed = Object.entries(status?.failed ?? {});
  const dead = Object.entries(status?.dead ?? {});
  const wedged = Object.entries(status?.wedged ?? {});

  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        mcp servers
      </h3>
      <p className="text-[13px] text-secondary mb-3">
        Model Context Protocol servers configured for this machine, and the
        tools they add to a turn. Checking starts them, which is why nothing
        happens until you ask.
      </p>
      <button
        type="button"
        onClick={() => void check()}
        disabled={busy}
        className="rounded-md bg-surface hover:bg-float text-secondary px-3 py-1 text-[13px] disabled:opacity-40"
      >
        {busy ? "checking…" : status ? "check again" : "check mcp servers"}
      </button>
      {error !== null && (
        <div role="alert" className="text-red text-[12.5px] mt-2 break-words">
          {error}
        </div>
      )}
      {status && (
        <div className="mt-2 flex flex-col gap-1.5">
          <p className="font-mono text-[12px] text-secondary">
            {mcpLine(status)}
          </p>
          {status.hint && (
            <p className="text-[12.5px] text-amber">{status.hint}</p>
          )}
          {status.configured.length > 0 && (
            <p className="text-[12.5px] text-secondary">
              <span className="text-muted">configured: </span>
              {status.configured.join(", ")}
            </p>
          )}
          {status.tools.length > 0 && (
            <p className="text-[12.5px] text-secondary break-words">
              <span className="text-muted">tools: </span>
              {status.tools.join(", ")}
            </p>
          )}
          {failed.map(([name, why]) => (
            <p key={name} className="text-[12.5px] text-red break-words">
              {name} failed — {why}
            </p>
          ))}
          {dead.map(([name, why]) => (
            <p key={name} className="text-[12.5px] text-red break-words">
              {name} died — {why}
            </p>
          ))}
          {wedged.map(([name, n]) => (
            <p key={name} className="text-[12.5px] text-amber">
              {name} wedged — {n} timeouts
            </p>
          ))}
        </div>
      )}
    </section>
  );
}
