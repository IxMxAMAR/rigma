// `/api/settings` — the app's own settings, from the UI's side.
//
// D4a: the route had NO UI at all. Neither frontend called it, so the idle
// auto-unload timeout was reachable only from the API or by hand-editing
// `~/.rigma/settings.json`. The transport and the shape guards live here, apart
// from the card, for the reason the rest of this project states: the decisions
// (is this body readable, what does a refusal say, is this input a number the
// server will accept) are what a test can reach.

import { responseError } from "./listFetch";

/** Mirrors `app_settings.MAX_IDLE_MINUTES` (serve.py's route validates against
 *  it). Duplicated deliberately and narrowly: the input must refuse an
 *  impossible number BEFORE a round trip, and the server's own 400 remains the
 *  authority for anything that gets through. */
export const MAX_IDLE_MINUTES = 24 * 60;

export interface AppSettingsBody {
  settings: { idle_unload_minutes: number };
  /** The value actually in force — may come from `RIGMA_KEEP_ALIVE_MIN`. */
  idle_unload_minutes: number;
  /** `RIGMA_KEEP_ALIVE_MIN` is set, so a write here has no effect. */
  env_override: boolean;
}

export type SettingsResult =
  | { ok: true; body: AppSettingsBody }
  | { ok: false; error: string };

/** The route's body, or null when it is not the shape the route promises.
 *
 *  Null rather than a cast: a FastAPI 500's `{detail}` or an HTML error page
 *  would otherwise reach the form as `undefined` minutes and render as a blank
 *  input that silently saves 0. */
export function settingsBody(d: unknown): AppSettingsBody | null {
  if (!d || typeof d !== "object") return null;
  const o = d as Record<string, unknown>;
  const s = o.settings;
  if (!s || typeof s !== "object") return null;
  const stored = (s as Record<string, unknown>).idle_unload_minutes;
  if (typeof stored !== "number" || !Number.isFinite(stored)) return null;
  if (typeof o.idle_unload_minutes !== "number"
      || !Number.isFinite(o.idle_unload_minutes)) return null;
  return {
    settings: { idle_unload_minutes: stored },
    idle_unload_minutes: o.idle_unload_minutes,
    // Strictly `true`: an absent or malformed flag must not disable the input
    // and claim an override that is not there.
    env_override: o.env_override === true,
  };
}

/** What the minutes field holds, or null when it is not a value the server
 *  will accept. `0` is a real answer (never unload), so it is not falsy here. */
export function minutesFromInput(raw: string): number | null {
  const t = raw.trim();
  if (t === "") return null;
  const n = Number(t);
  if (!Number.isFinite(n) || n < 0 || n > MAX_IDLE_MINUTES) return null;
  return n;
}

async function read(r: Response): Promise<SettingsResult> {
  if (!r.ok) return { ok: false, error: await responseError(r) };
  const body = settingsBody(await r.json().catch(() => null));
  return body
    ? { ok: true, body }
    : { ok: false, error: "the settings API returned an unexpected body" };
}

/** `GET /api/settings`. Never throws: a failure is a value the card renders. */
export async function loadSettings(): Promise<SettingsResult> {
  let r: Response;
  try {
    r = await fetch("/api/settings");
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  return read(r);
}

/** `POST /api/settings`. The route validates before writing and answers 400
 *  with its own sentence, which is returned verbatim — it names the bound. */
export async function saveSettings(minutes: number): Promise<SettingsResult> {
  let r: Response;
  try {
    r = await fetch("/api/settings", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ idle_unload_minutes: minutes }),
    });
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  return read(r);
}
