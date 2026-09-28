import { useState } from "react";

import { api } from "../lib/api";
import {
  CONTROL_OPS,
  controlAvailability,
  controlOpError,
  controlParams,
  controlResultText,
  type ControlOp,
} from "./controlPlane";

/** R6-ACP-CONTROL: mcode's control plane, driven from the chat window.
 *
 *  WHY THIS EXISTS. Every operation here was reachable over ACP and none of it was
 *  reachable from the product: the goal, the queue and the steering all arrived as
 *  notifications, so the panel could DRAW them and a user could not touch them. A
 *  control plane you can only watch is not a control plane.
 *
 *  WHAT IT DELIBERATELY DOES NOT DO. It does not re-spell the server's operation names
 *  or re-implement its validation — `controlOpError` mirrors the one rule that keeps a
 *  button disabled, and the server's allowlist is what actually decides. It also does not
 *  try to be a full console: the per-row actions (delete a queued message, steer a
 *  specific one, stop a specific delegate) belong on the rows that already carry those
 *  ids, not in a form.
 *
 *  EVERY OPERATION IS A MUTATION, so the result is always shown. A button that appears
 *  to do nothing is indistinguishable from one that failed.
 */
export default function ControlPanel({ sessionId, harness, transport, hasSession }: {
  sessionId: string;
  harness: string;
  transport: string;
  hasSession: boolean;
}) {
  const [op, setOp] = useState<ControlOp>(CONTROL_OPS[0]);
  const [arg, setArg] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  const blocked = controlAvailability(harness, transport, hasSession);
  const argError = controlOpError(op, arg);

  const send = async () => {
    if (argError || busy) return;
    setBusy(true);
    setResult("");
    setError("");
    try {
      const r = await api.control(sessionId, op.op, controlParams(op, arg));
      setResult(controlResultText(op.op, r.result));
      // Cleared only on success: a rejected goal should not also lose what was typed.
      if (op.arg) setArg("");
    } catch (e) {
      setError(e instanceof Error ? e.message : "the operation failed");
    } finally {
      setBusy(false);
    }
  };

  return (
    <details className="rounded-md bg-surface px-2.5 py-1.5 text-[11.5px]">
      <summary className="cursor-pointer text-secondary">
        mcode control plane
      </summary>
      {blocked ? (
        /* The reason, not disabled buttons. `exec` is the DEFAULT transport, so this is
           the state most users are in — and a panel that simply did not render would
           leave them with no idea the control plane exists at all. */
        <p className="mt-1 text-muted leading-snug">{blocked}</p>
      ) : (
        <div className="mt-1 flex flex-col gap-1.5">
          <label className="flex items-center gap-2">
            <span className="w-20 shrink-0 text-secondary">operation</span>
            <select
              value={op.op}
              aria-label="mcode control operation"
              onChange={(e) => {
                const next = CONTROL_OPS.find((o) => o.op === e.target.value);
                if (next) {
                  setOp(next);
                  setArg("");
                  setResult("");
                  setError("");
                }
              }}
              className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                         outline-none"
            >
              {CONTROL_OPS.map((o) => (
                <option key={o.op} value={o.op}>{o.label}</option>
              ))}
            </select>
          </label>
          {op.arg && (
            <label className="flex items-center gap-2">
              <span className="w-20 shrink-0 text-secondary">{op.arg}</span>
              <input
                value={arg}
                aria-label={`mcode control ${op.arg}`}
                placeholder={op.arg === "objective" ? "what should it work toward?"
                                                     : "what should it do?"}
                onChange={(e) => setArg(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") void send(); }}
                className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                           outline-none"
              />
            </label>
          )}
          <p className="text-muted leading-snug">{op.hint}</p>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void send()}
              disabled={busy || argError !== ""}
              title={argError}
              className="rounded-md bg-surface px-2 py-1 text-[11.5px] text-secondary
                         disabled:opacity-50"
            >
              {busy ? "working…" : op.label}
            </button>
            {result && <span className="text-moss">{result}</span>}
            {error && <span className="text-amber">{error}</span>}
          </div>
        </div>
      )}
    </details>
  );
}
