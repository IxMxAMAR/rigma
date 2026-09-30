// D4c: `/api/backup` (GET) and `/api/restore` (POST), from the UI's side.
//
// `/api/restore` REPLACES memory, settings and methods (OD-15 option 1): the
// memory store is replaced outright, the settings are written with
// `app_settings.replace` (defaults overlaid with the document's keys, so a key
// the document omits goes back to its default), and every user method the
// document does not name is deleted (`serve.py::_apply_restore`). Sessions,
// RAG sources/index, method drafts, macro trust, models and calibration are
// NOT part of the file and are untouched — so "the whole store" is a lie the
// copy must not tell.
// The decision this module owns is not "how do we render it" but "what,
// exactly, is about to change". `restoreSummary` answers that from the file
// itself, before anything is sent, and the card refuses to post until the user
// has confirmed that sentence. The deleted-method set is only knowable AFTER
// the restore, from the route's additive `deleted` list.
//
// The route is all-or-nothing (A11): it validates the WHOLE document and then
// applies it, so the only errors the UI can show are the server's own — a bad
// version, a bad method, an unwritable disk. They are passed through verbatim.
import { bodyError } from "./listFetch";

export const BACKUP_URL = "/api/backup";

export interface BackupDoc {
  rigma_backup: number;
  app_version?: string;
  created_at?: string;
  settings?: Record<string, unknown>;
  methods?: unknown[];
  memory?: unknown[];
}

export type ParseBackup =
  | { ok: true; doc: BackupDoc }
  | { ok: false; error: string };

/** Read a chosen file. Everything checked here is checked again by the server;
 *  doing it first is what lets the confirmation name real counts instead of
 *  asking the user to confirm a document nobody has looked at. */
export function parseBackup(text: string): ParseBackup {
  let d: unknown;
  try {
    d = JSON.parse(text);
  } catch {
    return { ok: false, error: "that file is not JSON" };
  }
  if (!d || typeof d !== "object" || Array.isArray(d)) {
    return { ok: false,
             error: "that file is not a Rigma backup (expected a JSON object)" };
  }
  const o = d as Record<string, unknown>;
  if (typeof o.rigma_backup !== "number") {
    return { ok: false,
             error: "that file is not a Rigma backup — it carries no "
                    + "`rigma_backup` version" };
  }
  if (o.settings !== undefined
      && (o.settings === null || typeof o.settings !== "object"
          || Array.isArray(o.settings))) {
    return { ok: false, error: "settings: must be an object" };
  }
  if (o.methods !== undefined && !Array.isArray(o.methods)) {
    return { ok: false, error: "methods: must be a list" };
  }
  if (o.memory !== undefined && !Array.isArray(o.memory)) {
    return { ok: false, error: "memory: must be a list" };
  }
  return { ok: true, doc: o as unknown as BackupDoc };
}

/** What the file carries — the sentence the confirmation shows. It names the
 *  three sections the route applies, with the counts the document actually
 *  carries, so "restore" is never a blind click. */
export function restoreSummary(doc: BackupDoc): string {
  const keys = doc.settings ? Object.keys(doc.settings).length : 0;
  const methods = doc.methods?.length ?? 0;
  const memory = doc.memory?.length ?? 0;
  return [
    `${keys} settings ${keys === 1 ? "key" : "keys"}`,
    `${methods} ${methods === 1 ? "method" : "methods"}`,
    `${memory} memory ${memory === 1 ? "row" : "rows"}`,
  ].join(", ");
}

export type RestoreResult =
  | { ok: true; methods: number; memory: { before: number; after: number };
      /** The method ids the restore deleted, or `null` when the response
       *  carried no `deleted` field at all — an older server. `null` is NOT
       *  the same as `[]`: "the server did not say" must never render as
       *  "nothing was deleted". */
      deleted: string[] | null }
  | { ok: false; error: string };

/** The line the card shows for the methods a restore deleted. Pure so the
 *  empty-list and absent-field cases are assertable without a mount. */
export function deletedLine(deleted: string[] | null): string {
  if (deleted === null) {
    return "the server did not report which methods were deleted";
  }
  if (deleted.length === 0) {
    return "nothing was deleted";
  }
  const n = deleted.length;
  return `deleted ${n} ${n === 1 ? "method" : "methods"}: ${deleted.join(", ")}`;
}

/** `POST /api/restore`. Never throws: a refusal is a value the card renders.
 *  The body's `{error}` is the server's own sentence, kept verbatim. */
export async function postRestore(doc: unknown): Promise<RestoreResult> {
  let r: Response;
  try {
    r = await fetch("/api/restore", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(doc),
    });
  } catch {
    return { ok: false, error: "could not reach the server" };
  }
  const body: unknown = await r.json().catch(() => null);
  if (!r.ok) return { ok: false, error: bodyError(body, r) };
  const o = (body ?? {}) as Record<string, unknown>;
  const mem = (o.memory ?? {}) as Record<string, unknown>;
  return {
    ok: true,
    methods: typeof o.methods === "number" ? o.methods : 0,
    memory: {
      before: typeof mem.before === "number" ? mem.before : 0,
      after: typeof mem.after === "number" ? mem.after : 0,
    },
    // Additive field (ODR-7). Absent or malformed is reported as "not said",
    // never silently as zero.
    deleted: Array.isArray(o.deleted)
      ? o.deleted.map((x) => String(x))
      : null,
  };
}
