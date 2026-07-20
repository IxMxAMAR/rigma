// Settings: preset manager. App-level knobs stay minimal — most state is
// per-chat (sidecar) or per-model (registry), by design.
import { useCallback, useEffect, useState } from "react";

interface Preset {
  id: string;
  name: string;
  system_prompt?: string;
  params?: Record<string, number>;
}

export default function SettingsSurface() {
  const [presets, setPresets] = useState<Preset[]>([]);
  const [draft, setDraft] = useState<{ name: string; system_prompt: string }>({
    name: "", system_prompt: "",
  });
  const [err, setErr] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/presets");
      setPresets((await r.json()) as Preset[]);
    } catch {
      setPresets([]);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[640px] mx-auto flex flex-col gap-4">
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-3">
            presets
          </h3>
          {presets.length === 0 && (
            <p className="text-muted text-[13px] mb-2">
              No presets yet — a preset bundles a system prompt + sampling and
              can be applied to any chat.
            </p>
          )}
          <ul className="flex flex-col gap-1 mb-3">
            {presets.map((p) => (
              <li key={p.id} className="group flex items-center gap-2 rounded-md hover:bg-surface px-3 py-1.5">
                <span className="flex-1 text-[13.5px]">{p.name}</span>
                <span className="font-mono text-[11px] text-muted truncate max-w-[220px]">
                  {p.system_prompt?.slice(0, 48) ?? ""}
                </span>
                <button
                  className="opacity-0 group-hover:opacity-100 text-muted hover:text-red px-1"
                  aria-label={`delete preset ${p.name}`}
                  onClick={async () => {
                    await fetch(`/api/presets/${p.id}`, { method: "DELETE" });
                    void refresh();
                  }}
                >
                  ×
                </button>
              </li>
            ))}
          </ul>
          <form
            className="flex flex-col gap-2"
            onSubmit={async (e) => {
              e.preventDefault();
              if (!draft.name.trim()) return;
              setErr(null);
              const r = await fetch("/api/presets", {
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
              void refresh();
            }}
          >
            <input
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
            <button className="self-start rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold">
              create
            </button>
          </form>
        </section>
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-2">
            interfaces
          </h3>
          <p className="text-[13px] text-secondary">
            Legacy UI: <a href="/rizz" className="text-amber hover:underline">/rizz</a>
            {" · "}OpenAI-compatible API:{" "}
            <code className="font-mono text-[12px] bg-surface rounded px-1.5 py-0.5">
              http://127.0.0.1:11499/v1
            </code>
          </p>
        </section>
      </div>
    </main>
  );
}
