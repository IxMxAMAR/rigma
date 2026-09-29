import { useState } from "react";

import { api } from "../lib/api";
import { selectStreaming, useChat, type AcpConfigOption } from "./chatStore";
import {
  CONTROL_OPS,
  controlAvailability,
  controlOpError,
  controlParams,
  controlResultText,
  type ControlOp,
} from "./controlPlane";

/** One value a server-sent option list offers, as a string. */
function optionValue(o: unknown): string {
  if (o && typeof o === "object") {
    const v = (o as { value?: unknown }).value;
    return v === undefined || v === null ? "" : String(v);
  }
  return o === undefined || o === null ? "" : String(o);
}

/** The display name for one value, falling back to the value itself. */
function optionLabel(o: unknown): string {
  if (o && typeof o === "object") {
    const n = (o as { name?: unknown }).name;
    if (typeof n === "string" && n !== "") return n;
  }
  return optionValue(o);
}

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

  /* THE SERVER'S OWN OPTION LISTS. These are the values mcode said it will ACCEPT, which
     is why the settings below are selects and not text fields: the set is closed, and a
     typed value would be refused. Both were already arriving and being discarded. */
  const config = useChat((s) => selectStreaming(s)?.acpConfig ?? []);
  const settingsBusy = busy;

  const blocked = controlAvailability(harness, transport, hasSession);
  const argError = controlOpError(op, arg);

  /** Change one server-sent session option, and say what happened. */
  const setOption = async (optionId: string, value: string) => {
    if (busy) return;
    setBusy(true);
    setResult("");
    setError("");
    try {
      const r = await api.control(sessionId, "config_set", { optionId, value });
      setResult(controlResultText("config_set", r.result));
    } catch (e) {
      setError(e instanceof Error ? e.message : "the change failed");
    } finally {
      setBusy(false);
    }
  };

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

          {/* R6-ACP-SETTINGS. `mode_set` and `config_set` were reachable over HTTP with no
              control at all. They are drawn as selects from the lists the server sent,
              because the values are a closed set — the whole point of `configOption` is
              that the server enumerates what it accepts.

              Rendered per OPTION rather than as one "change a setting" form: each option
              is a thing a user can see and change on its own, and making them pick the
              option id from a dropdown first would be a worse control than the value
              list already implies. */}
          {config.length > 0 && (
            <div className="flex flex-col gap-1 border-t border-muted/20 pt-1.5 mt-0.5">
              <span className="text-[10.5px] text-muted uppercase tracking-[0.08em]">
                session settings
              </span>
              {config.map((c: AcpConfigOption) => {
                const values = Array.isArray(c.options) ? c.options : [];
                const current = optionValue(c.currentValue);
                return (
                  <label key={String(c.id)} className="flex items-center gap-2">
                    <span className="w-20 shrink-0 text-secondary">
                      {String(c.name || c.id)}
                    </span>
                    <select
                      value={current}
                      disabled={settingsBusy || values.length === 0}
                      aria-label={`mcode ${String(c.id)}`}
                      onChange={(e) => void setOption(String(c.id), e.target.value)}
                      className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                                 outline-none disabled:opacity-50"
                    >
                      {/* The current value is always an option, even when the server did
                          not list it — otherwise the select would silently show the first
                          entry as if it were the live one. */}
                      {!values.some((o) => optionValue(o) === current) && (
                        <option value={current}>{current}</option>
                      )}
                      {values.map((o, i) => (
                        <option key={`${optionValue(o)}-${i}`} value={optionValue(o)}>
                          {optionLabel(o)}
                        </option>
                      ))}
                    </select>
                  </label>
                );
              })}
            </div>
          )}
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
