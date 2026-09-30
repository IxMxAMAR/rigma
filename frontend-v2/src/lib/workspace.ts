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
