// D4c: backup and restore.
//
// `GET /api/backup` is a download — the route answers one versioned JSON
// document with settings, the user's methods and memory — so it is an `<a>`,
// not a fetch.
//
// `POST /api/restore` REPLACES THE WHOLE STORE — memory, settings and methods
// (OD-15 option 1). The route (`serve.py::_apply_restore`) replaces the memory
// store outright, writes the settings with `app_settings.replace` (defaults
// overlaid with the document's keys, so a key the document omits goes back to
// its default), and deletes every user method the document does not name. That
// is the reason this card is two steps and not one: choosing a file only parses
// it and shows what it contains, and the replace button stays disabled until the
// user has ticked a confirmation that says the whole store is replaced. A single
// click that rewrites the whole store is not a control this UI may draw.
//
// The route is all-or-nothing (A11): it validates the whole document, then
// applies it under one lock and rolls back on failure. So the only errors here
// are the server's own, and they are rendered verbatim.
import { useRef, useState } from "react";

import {
  BACKUP_URL,
  parseBackup,
  postRestore,
  restoreSummary,
  type BackupDoc,
} from "../lib/backup";

/** The confirmation step, pure props so the sentence a user must read before
 *  rewriting the whole store is assertable without a store. No success line
 *  here: on success `replace()` clears `pending` (which unmounts this panel)
 *  before it sets `done`, so the card renders the result. */
export function RestoreConfirm({
  summary, confirmed, busy, error, onConfirm, onReplace, onCancel,
}: {
  summary: string;
  confirmed: boolean;
  busy: boolean;
  error: string | null;
  onConfirm: (v: boolean) => void;
  onReplace: () => void;
  onCancel: () => void;
}) {
  return (
    <div className="rounded-md bg-canvas/60 px-3 py-2.5 mt-2 flex flex-col gap-2">
      <p className="text-[12.5px] text-amber font-semibold">
        This replaces the whole store on this machine.
      </p>
      <p className="text-[12.5px] text-secondary">
        Restoring replaces the whole store: your memory becomes the file's
        memory rows, settings become the file's settings (a key the file omits
        goes back to its default), and methods the file does not name are
        deleted. Anything not in the file is gone. The file carries:{" "}
        <span className="font-mono">{summary}</span>.
      </p>
      <label className="flex items-center gap-2 text-[12.5px] text-secondary">
        <input type="checkbox" className="accent-amber" checked={confirmed}
               disabled={busy}
               aria-label="Confirm replacing the whole store"
               onChange={(e) => onConfirm(e.target.checked)} />
        I understand — replace the whole store
      </label>
      {error !== null && (
        <div role="alert" className="text-red font-mono text-[11.5px] break-words">
          {error}
        </div>
      )}
      <div className="flex items-center gap-2">
        <button
          type="button"
          disabled={!confirmed || busy}
          onClick={onReplace}
          className="rounded-md bg-red/15 text-red px-3 py-1 text-[13px] font-semibold disabled:opacity-40"
        >
          {busy ? "restoring…" : "replace the whole store"}
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={onCancel}
          className="rounded-md bg-surface hover:bg-float px-3 py-1 text-[13px]"
        >
          cancel
        </button>
      </div>
    </div>
  );
}

export default function BackupCard() {
  const [pending, setPending] =
    useState<{ doc: BackupDoc; summary: string } | null>(null);
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [parseError, setParseError] = useState<string | null>(null);
  const [done, setDone] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const choose = async (f: File) => {
    setParseError(null);
    setError(null);
    setDone(null);
    setConfirmed(false);
    setPending(null);
    const res = parseBackup(await f.text());
    if (!res.ok) {
      setParseError(res.error);
      return;
    }
    setPending({ doc: res.doc, summary: restoreSummary(res.doc) });
  };

  const replace = async () => {
    if (!pending) return;
    setBusy(true);
    setError(null);
    const r = await postRestore(pending.doc);
    setBusy(false);
    if (!r.ok) {
      setError(r.error);
      return;
    }
    setPending(null);
    setConfirmed(false);
    setDone(`restored ${r.methods} method(s); memory rows ${r.memory.before} `
      + `→ ${r.memory.after}`);
  };

  return (
    <section className="rounded-lg bg-panel p-4">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
        backup
      </h3>
      <p className="text-[13px] text-secondary mb-3">
        One versioned JSON file with your settings, your methods and memory —
        for moving to another machine. Restoring REPLACES the whole store:
        memory, settings and methods all come from the file, and anything not in
        the file is gone.
      </p>
      <div className="flex items-center gap-2 flex-wrap">
        <a
          href={BACKUP_URL}
          download="rigma-backup.json"
          className="rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold"
        >
          download a backup
        </a>
        <button
          type="button"
          onClick={() => fileRef.current?.click()}
          className="rounded-md bg-surface hover:bg-float text-secondary px-3 py-1 text-[13px]"
        >
          restore from a file…
        </button>
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          hidden
          aria-label="Backup file to restore"
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void choose(f);
          }}
        />
      </div>
      {parseError !== null && (
        <div role="alert" className="text-red text-[12.5px] mt-2">
          {parseError}
        </div>
      )}
      {pending && (
        <RestoreConfirm
          summary={pending.summary}
          confirmed={confirmed}
          busy={busy}
          error={error}
          onConfirm={setConfirmed}
          onReplace={() => void replace()}
          onCancel={() => {
            setPending(null);
            setConfirmed(false);
            setError(null);
          }}
        />
      )}
      {!pending && done !== null && (
        <p className="text-moss font-mono text-[11.5px] mt-2">{done}</p>
      )}
    </section>
  );
}
