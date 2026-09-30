// Settings: preset manager. App-level knobs stay minimal — most state is
// per-chat (sidecar) or per-model (registry), by design.
import { useCallback, useEffect, useRef, useState } from "react";

import EmptyState from "../EmptyState";
import LoadError from "../LoadError";
import InlineError from "../InlineError";
import AppSettingsCard from "./AppSettingsCard";
import BackupCard from "./BackupCard";
import McpCard from "./McpCard";
import { engineApi } from "../lib/engineApi";
import { readList, responseError } from "../lib/listFetch";
import { openaiBase } from "../lib/openaiBase";

interface Preset {
  id: string;
  name: string;
  system_prompt?: string;
  params?: Record<string, number>;
  builtin?: boolean;
}

const isBuiltin = (p: Preset) => p.builtin || p.id.startsWith("usecase:");

export default function SettingsSurface() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [draft, setDraft] = useState<{ name: string; system_prompt: string }>({
    name: "", system_prompt: "",
  });
  const [editing, setEditing] = useState<string | null>(null); // preset id
  const [err, setErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [rowErr, setRowErr] = useState<string | null>(null);
  // IMP-10: the empty state's action focuses the create form below.
  const nameRef = useRef<HTMLInputElement>(null);
  // AUDIT F11-6: this card used to hardcode `http://127.0.0.1:11499/v1` and
  // call it Rigma's OpenAI API. 11499 is llama-server's upstream port, so a
  // backend pointed there bypasses the session, the tool-call repair and the
  // idle-unload bookkeeping — and the URL is dead on any non-default launch.
  // /api/server publishes the real base (`openai_base`); read it, and fall back
  // to the origin that served this page rather than to a literal port.
  const [apiBase, setApiBase] = useState(() =>
    openaiBase(null, window.location.origin));

  useEffect(() => {
    engineApi.server()
      .then((i) => setApiBase(openaiBase(i, window.location.origin)))
      .catch(() => { /* keep the origin fallback */ });
  }, []);

  const refresh = useCallback(async () => {
    // AUDIT F11-3: a non-array /api/presets body used to reach .map and blank
    // the app; a failed load must not read as "no presets yet".
    try {
      const res = await readList<Preset>(await fetch("/api/presets"));
      if (!res.ok) {
        setLoadErr(res.error);
        setPresets([]);
        return;
      }
      setLoadErr(null);
      setPresets(res.rows);
    } catch (e) {
      setLoadErr((e as Error).message);
      setPresets([]);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[640px] mx-auto flex flex-col gap-4">
        {/* D4a: `/api/settings` had no UI anywhere — this is the app's OWN
            settings, so they belong on this page rather than in a chat's ⚙
            panel or a model's card. */}
        <AppSettingsCard />
        {/* D4c: `/api/backup` + `/api/restore` and the MCP status report.
            Restore replaces the whole store — memory, settings and methods
            (OD-15 option 1) — so it lives behind its own confirmation; the MCP
            report starts servers, so it is manual. */}
        <BackupCard />
        <McpCard />
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-3">
            presets
          </h3>
          {loadErr && (
            <LoadError message={`could not load presets: ${loadErr}`}
                       onRetry={() => void refresh()} />
          )}
          {!loadErr && presets.length === 0 && (
            <EmptyState
              title="no presets yet"
              body="A preset bundles a system prompt and sampling settings and can be applied to any chat."
              actionLabel="create a preset"
              onAction={() => nameRef.current?.focus()} />
          )}
          <p className="text-muted text-[12px] mb-2">
            Click a preset to edit it. Built-ins can be viewed, not changed —
            apply presets to a chat from the chat's ⚙ panel.
          </p>
          {rowErr && <InlineError message={`could not delete: ${rowErr}`} />}
          <ul className="flex flex-col gap-1 mb-3">
            {presets.map((p) => (
              <li key={p.id}
                  className={`group flex items-center gap-2 rounded-md hover:bg-surface px-3 py-1.5 cursor-pointer ${editing === p.id ? "bg-surface" : ""}`}
                  onClick={() => {
                    setEditing(p.id);
                    setDraft({ name: p.name,
                               system_prompt: p.system_prompt ?? "" });
                    setErr(null);
                  }}>
                <span className="flex-1 text-[13.5px]">
                  {p.name}
                  {isBuiltin(p) && (
                    <span className="font-mono text-[10px] text-muted ml-1.5">built-in</span>
                  )}
                </span>
                <span className="font-mono text-[11px] text-muted truncate max-w-[220px]">
                  {p.system_prompt?.slice(0, 48) ?? ""}
                </span>
                {!isBuiltin(p) && (
                  <button
                    className="text-muted hover:text-red px-1"
                    aria-label={`delete preset ${p.name}`}
                    onClick={async (e) => {
                      e.stopPropagation();
                      setRowErr(null);
                      // AUDIT F11-4: this ignored r.ok and refreshed, so a
                      // refused delete looked exactly like one that worked.
                      const r = await fetch(
                        `/api/presets/${encodeURIComponent(p.id)}`,
                        { method: "DELETE" });
                      if (!r.ok) {
                        setRowErr(await responseError(r));
                        return;
                      }
                      if (editing === p.id) setEditing(null);
                      void refresh();
                    }}
                  >
                    ×
                  </button>
                )}
              </li>
            ))}
          </ul>
          <form
            className="flex flex-col gap-2"
            onSubmit={async (e) => {
              e.preventDefault();
              if (!draft.name.trim()) return;
              setErr(null);
              const editingBuiltin = editing != null &&
                presets.some((p) => p.id === editing && isBuiltin(p));
              const path = editing && !editingBuiltin
                ? `/api/presets/${editing}` : "/api/presets";
              const r = await fetch(path, {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(draft),
              });
              if (!r.ok) {
                const b = (await r.json().catch(() => ({}))) as { error?: string };
                setErr(b.error ?? `server replied ${r.status}`);
                return;
              }
              setDraft({ name: "", system_prompt: "" });
              setEditing(null);
              void refresh();
            }}
          >
            <input
              ref={nameRef}
              value={draft.name}
              onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
              placeholder="preset name"
              aria-label="Preset name"
              className="rounded-md bg-surface px-3 py-1.5 text-[13px] outline-none placeholder:text-muted"
            />
            <textarea
              value={draft.system_prompt}
              onChange={(e) =>
                setDraft((d) => ({ ...d, system_prompt: e.target.value }))}
              placeholder="system prompt"
              aria-label="Preset system prompt"
              rows={3}
              className="rounded-md bg-surface px-3 py-1.5 text-[13px] outline-none resize-y placeholder:text-muted"
            />
            {err && <div className="text-red text-[12.5px]">{err}</div>}
            <div className="flex gap-2">
              <button className="rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold">
                {editing
                  ? presets.some((p) => p.id === editing && isBuiltin(p))
                    ? "save as copy"
                    : "save"
                  : "create"}
              </button>
              {editing && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(null);
                    setDraft({ name: "", system_prompt: "" });
                    setErr(null);
                  }}
                  className="rounded-md bg-surface hover:bg-float px-3 py-1 text-[13px]"
                >
                  new instead
                </button>
              )}
            </div>
          </form>
        </section>
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
            interfaces
          </h3>
          <p className="text-[13px] text-secondary">
            Legacy UI: <a href="/rizz" className="text-amber hover:underline">/rizz</a>
            {" · "}OpenAI-compatible API:{" "}
            <code className="font-mono text-[12px] bg-surface rounded px-1.5 py-0.5"
                  title={"Rigma's own /v1 passthrough — sessions, tool-call "
                         + "repair and idle-unload bookkeeping all apply. The "
                         + "engine's own port is different and bypasses them."}>
              {apiBase}
            </code>
          </p>
        </section>
      </div>
    </main>
  );
}
