import { useState } from "react";

import { api } from "../lib/api";
import { selectStreaming, useChat, type AcpConfigOption } from "./chatStore";
import {
  CONTROL_OPS,
  acpModeList,
  controlAvailability,
  controlOpError,
  controlParams,
  controlReady,
  controlResultText,
  type ControlOp,
} from "./controlPlane";

/** The empty option list, as ONE shared array. It is NOT frozen — nothing mutates it,
 *  and `Object.freeze` would only turn a future mistake into a throw.
 *
 *  NOT `?? []` INSIDE THE SELECTOR. zustand v5 compares a selector's result with
 *  `Object.is`, so returning a fresh array literal whenever no turn is streaming makes the
 *  store believe state changed on every render — the documented "getSnapshot should be
 *  cached" loop, and it would fire for every session with no live turn, which is most of
 *  them. A stable module-level value is what makes the selector safe.
 */
const NO_CONFIG: AcpConfigOption[] = [];

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
 *  specific one) belong on the rows that already carry those ids, not in a form. There is
 *  no per-delegate action at all — `delegation_stop` is session-wide, which is why it is
 *  in the operation list above rather than on a member row.
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
  /** The chosen value for an operation whose parameter is a closed set. */
  const [choice, setChoice] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState("");
  const [error, setError] = useState("");

  /* THE SERVER'S OWN OPTION LISTS. These are the values mcode said it will ACCEPT, which
     is why the settings below are selects and not text fields: the set is closed, and a
     typed value would be refused. Both were already arriving and being discarded. */
  // `?? NO_CONFIG`, never `?? []`: see the comment on NO_CONFIG.
  const config = useChat((s) => selectStreaming(s)?.acpConfig ?? NO_CONFIG);
  const recordConfigOption = useChat((s) => s.setAcpConfigOption);
  /* B6d: the modes the SESSION advertised. `mode_set` had no control because the
     valid ids never reached a consumer; they now ride the same `acp_config` event
     as the options above. `acpModeList` never fabricates a list, so an absent or
     empty one renders as "unknown" rather than a guessed `plan`/`default`. */
  const acpModes = useChat((s) => selectStreaming(s)?.acpModes ?? null);
  const recordMode = useChat((s) => s.setAcpMode);
  const modes = acpModeList(acpModes);
  const settingsBusy = busy;

  const blocked = controlAvailability(harness, transport, hasSession);
  const argError = controlOpError(op, arg);
  // One gate for the button: text operations use `argError`, choice operations use the
  // presence of a value. `controlReady` is the single place that decides, so the button
  // and the send path cannot disagree about whether an operation is attemptable.
  const ready = controlReady(op, arg, choice);

  /** Change one server-sent session option, and say what happened. */
  const setOption = async (optionId: string, value: string) => {
    if (busy) return;
    setBusy(true);
    setResult("");
    setError("");
    try {
      const r = await api.control(sessionId, "config_set", { optionId, value });
      setResult(controlResultText("config_set", r.result));
      // RECORD IT, because nothing else will. The select reads the live turn's
      // `acpConfig`, which is written only by a `config_option_update` notification —
      // and a control operation runs on its OWN short-lived mcode process whose
      // notifications nobody reads. So without this the change SUCCEEDS, the panel says
      // "done", and the select snaps back to the old value and stays stale.
      //
      // After the await, so this records an accepted fact rather than a guess.
      recordConfigOption(optionId, value);
    } catch (e) {
      setError(e instanceof Error ? e.message : "the change failed");
    } finally {
      setBusy(false);
    }
  };

  /** B6d: change the session's mode, from the list the session itself advertised.
   *
   *  The id comes from that list, never from a constant here. The confirmation
   *  names the mode (`controlResultText("mode_set", …, modeId)`) because ACP does
   *  not promise to echo it back, and `recordMode` mirrors the accepted change so
   *  the select does not snap back — the same two problems `setOption` solves. */
  const setMode = async (modeId: string) => {
    if (busy || !modeId) return;
    setBusy(true);
    setResult("");
    setError("");
    try {
      const r = await api.control(sessionId, "mode_set", { modeId });
      setResult(controlResultText("mode_set", r.result, modeId));
      recordMode(modeId);
    } catch (e) {
      setError(e instanceof Error ? e.message : "the mode change failed");
    } finally {
      setBusy(false);
    }
  };

  const send = async () => {    if (!ready || busy) return;
    setBusy(true);
    setResult("");
    setError("");
    try {
      const r = await api.control(sessionId, op.op, controlParams(op, arg, choice));
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
                  setChoice("");
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
          {/* A CLOSED SET is a select, not a text field: the server defines the valid
              values, and typing one would be a way to be refused. */}
          {op.choices && (
            <label className="flex items-center gap-2">
              <span className="w-20 shrink-0 text-secondary">{op.choices.param}</span>
              <select
                value={choice || op.choices.values[0] || ""}
                aria-label={`mcode control ${op.choices.param}`}
                onChange={(e) => setChoice(e.target.value)}
                className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                           outline-none"
              >
                {op.choices.values.map((v, i) => (
                  <option key={v} value={v}>
                    {op.choices?.labels?.[i] ?? v}
                  </option>
                ))}
              </select>
            </label>
          )}
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
              {config.map((c: AcpConfigOption, i) => {
                const values = Array.isArray(c.options) ? c.options : [];
                const current = optionValue(c.currentValue);
                // A configOption with neither an id nor a name is a malformed payload.
                // `String(undefined)` would render the literal word "undefined", give
                // every such entry the SAME React key, and label the control
                // "mcode undefined" for a screen reader — so the position is used as a
                // last resort, which is unique and honest about being a fallback.
                const key = String(c.id ?? c.name ?? `#${i}`);
                const shown = String(c.name ?? c.id ?? `option ${i + 1}`);
                return (
                  <label key={key} className="flex items-center gap-2">
                    <span className="w-20 shrink-0 text-secondary">{shown}</span>
                    <select
                      value={current}
                      disabled={settingsBusy || values.length === 0}
                      aria-label={`mcode ${key}`}
                      onChange={(e) => void setOption(String(c.id), e.target.value)}
                      className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                                 outline-none disabled:opacity-50"
                    >
                      {/* The current value is always an option, even when the server did
                          not list it — otherwise the select would silently show the first
                          entry as if it were the live one.
                          NOT WHEN IT IS EMPTY, though: `value=""` with no label renders a
                          selected option that displays NOTHING, so "unset" becomes
                          indistinguishable from "unknown". An unset option says so. */}
                      {current === "" ? (
                        <option value="">(not set)</option>
                      ) : !values.some((o) => optionValue(o) === current) && (
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
          {/* B6d. `mode_set` is drawn from the SESSION's own list, and from
              nothing else. An absent or empty list is rendered as "unknown": a
              fabricated `plan`/`default` pair is a control whose values stop
              matching the server on the next mcode version, which is exactly why
              this operation had no UI until the real list reached a consumer. */}
          <div className="flex flex-col gap-1 border-t border-muted/20 pt-1.5 mt-0.5">
            <span className="text-[10.5px] text-muted uppercase tracking-[0.08em]">
              session mode
            </span>
            {modes.unknown ? (
              <p className="text-muted leading-snug">
                unknown — this session has not advertised a mode list, so there is
                nothing to choose. Rigma will not offer a guessed one.
              </p>
            ) : (
              <label className="flex items-center gap-2">
                <span className="w-20 shrink-0 text-secondary">mode</span>
                <select
                  value={modes.current || modes.options[0].id}
                  disabled={settingsBusy}
                  aria-label="mcode session mode"
                  onChange={(e) => void setMode(e.target.value)}
                  className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12px]
                             outline-none disabled:opacity-50"
                >
                  {/* The session's current mode is always an option, even when it
                      is not in the advertised list — otherwise the select would
                      show the first entry as if it were the live one. */}
                  {modes.current !== "" &&
                    !modes.options.some((m) => m.id === modes.current) && (
                    <option value={modes.current}>{modes.current}</option>
                  )}
                  {modes.options.map((m) => (
                    <option key={m.id} value={m.id}>{m.label}</option>
                  ))}
                </select>
              </label>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => void send()}
              disabled={busy || !ready}
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
