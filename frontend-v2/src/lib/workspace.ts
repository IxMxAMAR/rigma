// Saving a chat's workspace folder, without lying about it.
//
// The header fired the POST, swallowed every outcome (`.catch(() => {})`), and
// then wrote the new path into the UI store unconditionally. A refused save —
// a path that does not exist, a server that was down — therefore left the
// header showing a working directory the session was not using, and the next
// turn ran somewhere else with nothing on screen saying so (A10).

import { responseError } from "./listFetch";

export type SaveWorkspaceResult =
  | { ok: true }
  | { ok: false; error: string };

/** Persist `v` as `sid`'s workspace. Never throws: a refusal is a value the
 *  caller can render, because resolving anyway is exactly how the UI and the
 *  server came to disagree. */
export async function saveWorkspace(
  sid: string,
  v: string,
): Promise<SaveWorkspaceResult> {
  let r: Response;
  try {
    r = await fetch(`/api/sessions/${sid}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ workspace: v }),
    });
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  if (!r.ok) return { ok: false, error: await responseError(r) };
  return { ok: true };
}

export type PickWorkspaceResult =
  | { ok: true; path: string }
  | { ok: false; error: string };

/** Ask the server to show the OS folder picker and return the chosen path.
 *
 *  Never throws, like `saveWorkspace` above. A CANCEL comes back as
 *  `{ ok: true, path: "" }` — closing the dialog is a normal outcome, not an
 *  error to render — while `{ ok: false }` means no dialog could be shown
 *  (a server with no desktop, a picker that timed out), which the caller should
 *  say out loud rather than swallow. This does NOT save: the caller feeds the
 *  path through `saveWorkspace`, so the session's workspace keeps exactly one
 *  write path. */
export async function pickWorkspace(
  sid: string,
  initial = "",
): Promise<PickWorkspaceResult> {
  let r: Response;
  try {
    r = await fetch(`/api/sessions/${sid}/workspace/pick`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ initial }),
    });
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  if (r.status === 204) return { ok: true, path: "" };
  if (!r.ok) return { ok: false, error: await responseError(r) };
  const d: unknown = await r.json().catch(() => null);
  const path =
    d && typeof d === "object" && typeof (d as { path?: unknown }).path === "string"
      ? (d as { path: string }).path
      : "";
  return { ok: true, path };
}
