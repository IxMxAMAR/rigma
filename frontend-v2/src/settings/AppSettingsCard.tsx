// D4a: the `/api/settings` card.
//
// The route had no UI at all, so the idle auto-unload timeout — the one knob
// that decides whether the loaded model gives its VRAM back — could only be
// changed from the API or by hand-editing `~/.rigma/settings.json`.
//
// `IdleUnloadForm` is split out and exported as a pure props component so the
// markup a user reads (the effective value, the env-override refusal, the
// server's own 400) is assertable with `renderToStaticMarkup`, without a store
// or an event. The stateful card only loads and saves.
import { useCallback, useEffect, useState } from "react";

import LoadError from "../LoadError";
import {
  MAX_IDLE_MINUTES,
  loadSettings,
  minutesFromInput,
  saveSettings,
  type AppSettingsBody,
} from "../lib/appSettings";

export function IdleUnloadForm({
  draft,
  effective,
  envOverride,
  loading,
  loadError,
  busy,
  canSave,
  saved,
  saveError,
  onDraft,
  onSave,
  onRetry,
}: {
  draft: string;
  /** The value actually in force. Differs from `draft` while the env overrides
   *  it, which is exactly when the difference has to be on screen. */
  effective: number;
  envOverride: boolean;
  loading: boolean;
  loadError: string | null;
  busy: boolean;
  canSave: boolean;
  saved: boolean;
  saveError: string | null;
  onDraft: (v: string) => void;
  onSave: () => void;
  onRetry: () => void;
}) {
  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        engine
      </h3>
      {loadError !== null ? (
        <LoadError message={`could not load settings: ${loadError}`}
                   onRetry={onRetry} />
      ) : loading ? (
        <p className="font-mono text-[12px] text-muted">loading…</p>
      ) : (
        <>
          <p className="text-[13px] text-secondary mb-3">
            Unload the model from VRAM after this many minutes with no turns.
            0 keeps it loaded until you unload it yourself.
          </p>
          <div className="flex items-end gap-3 flex-wrap">
            <label className="flex flex-col gap-1">
              <span className="font-mono text-[11px] text-muted">
                idle unload (minutes)
              </span>
              <input
                type="number"
                min={0}
                max={MAX_IDLE_MINUTES}
                step={1}
                value={draft}
                disabled={envOverride || busy}
                onChange={(e) => onDraft(e.target.value)}
                aria-label="Idle unload minutes"
                className="w-28 rounded-md bg-surface px-3 py-1.5 font-mono text-[13px] outline-none disabled:opacity-50"
              />
            </label>
            <button
              type="button"
              onClick={onSave}
              disabled={!canSave}
              className="rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold disabled:opacity-40"
            >
              {busy ? "saving…" : saved ? "saved" : "save"}
            </button>
            <span className="font-mono text-[11.5px] text-muted pb-1.5">
              in effect: {effective} min
            </span>
          </div>
          {envOverride && (
            // A control that changes nothing reads as broken, so the reason is
            // stated and the input is disabled rather than silently ignored.
            <p className="text-[12.5px] text-amber mt-2">
              RIGMA_KEEP_ALIVE_MIN is set, so this value has no effect while it
              is — unset it to control the timeout from here.
            </p>
          )}
          {saveError !== null && (
            <div role="alert" className="text-red text-[12.5px] mt-2">
              {saveError}
            </div>
          )}
        </>
      )}
    </section>
  );
}

export default function AppSettingsCard() {
  const [body, setBody] = useState<AppSettingsBody | null>(null);
  const [draft, setDraft] = useState("");
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [saved, setSaved] = useState(false);

  const apply = useCallback((b: AppSettingsBody) => {
    setBody(b);
    setDraft(String(b.settings.idle_unload_minutes));
    setSaved(false);
  }, []);

  const refresh = useCallback(async () => {
    setLoading(true);
    setLoadError(null);
    const res = await loadSettings();
    setLoading(false);
    if (res.ok) apply(res.body);
    else setLoadError(res.error);
  }, [apply]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const minutes = minutesFromInput(draft);
  const canSave = !busy && body !== null && minutes !== null &&
    minutes !== body.settings.idle_unload_minutes;

  const save = async () => {
    if (minutes === null) return;
    setBusy(true);
    setSaveError(null);
    const res = await saveSettings(minutes);
    setBusy(false);
    if (res.ok) {
      apply(res.body);
      setSaved(true);
    } else {
      setSaveError(res.error);
    }
  };

  return (
    <IdleUnloadForm
      draft={draft}
      effective={body?.idle_unload_minutes ?? 0}
      envOverride={body?.env_override === true}
      loading={loading}
      loadError={loadError}
      busy={busy}
      canSave={canSave}
      saved={saved}
      saveError={saveError}
      onDraft={(v) => {
        setDraft(v);
        setSaveError(null);
        setSaved(false);
      }}
      onSave={() => void save()}
      onRetry={() => void refresh()}
    />
  );
}
