// Right sidecar for the chat surface: grounding, sampling, system prompt.
// Collapsible; state persists to the session via the existing PATCH API.
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { useChat } from "./chatStore";

interface RagStatus {
  running: boolean;
  sources: string[];
  indexing: boolean;
  error: string;
}

function GroundingCard() {
  const currentId = useChat((s) => s.currentId);
  const [status, setStatus] = useState<RagStatus | null>(null);
  const [grounded, setGrounded] = useState(false);
  const [path, setPath] = useState("");

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/rag/status");
      setStatus((await r.json()) as RagStatus);
    } catch {
      setStatus(null);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    if (!currentId) return;
    api.getSession(currentId).then((s) => setGrounded(!!s.use_rag)).catch(() => {});
  }, [currentId]);

  const toggle = async () => {
    if (!currentId) return;
    const next = !grounded;
    setGrounded(next);
    try {
      await api.updateSession(currentId, { use_rag: next });
    } catch {
      setGrounded(!next);
    }
  };

  return (
    <section className="rounded-lg bg-panel p-3">
      <label className="flex items-center gap-2 cursor-pointer">
        <span
          className={`w-2 h-2 rounded-full ${status?.running ? "bg-moss" : "bg-float"}`}
        />
        <span className="flex-1 text-[13px]">Grounded chat</span>
        <input type="checkbox" checked={grounded} onChange={() => void toggle()}
               className="accent-[#8fb573]" aria-label="Ground this chat" />
      </label>
      <p className="text-[11.5px] text-muted mt-1">
        model reads your documents as it talks
      </p>
      <ul className="mt-2 flex flex-col gap-1">
        {(status?.sources ?? []).map((src) => (
          <li key={src} className="group flex items-center gap-1.5 font-mono text-[11.5px] text-secondary">
            <span className="flex-1 truncate" dir="rtl" title={src}>{src}</span>
            <button
              className="opacity-0 group-hover:opacity-100 text-muted hover:text-red px-1"
              aria-label={`stop indexing ${src}`}
              onClick={async () => {
                await fetch("/api/rag/sources", {
                  method: "DELETE",
                  headers: { "content-type": "application/json" },
                  body: JSON.stringify({ path: src }),
                });
                void refresh();
              }}
            >
              ×
            </button>
          </li>
        ))}
      </ul>
      <form
        className="mt-2 flex gap-1.5"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!path.trim()) return;
          await fetch("/api/rag/sources", {
            method: "POST",
            headers: { "content-type": "application/json" },
            body: JSON.stringify({ path: path.trim() }),
          }).catch(() => {});
          setPath("");
          void refresh();
        }}
      >
        <input
          value={path}
          onChange={(e) => setPath(e.target.value)}
          placeholder="add folder…"
          aria-label="Folder to index"
          className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 font-mono text-[12px] outline-none placeholder:text-muted"
        />
        <button className="rounded-md bg-surface hover:bg-float px-2 text-[13px]">+</button>
      </form>
      <div className={`font-mono text-[11px] mt-1.5 ${status?.error ? "text-red" : status?.running ? "text-moss" : "text-muted"}`}>
        {status?.indexing ? "● indexing…"
          : status?.error ? "▲ " + status.error
          : status?.running ? `● ready · ${status.sources.length} folder${status.sources.length === 1 ? "" : "s"}`
          : status?.sources.length ? "○ starts with your first grounded message"
          : "no folders indexed"}
      </div>
    </section>
  );
}

function SamplingCard() {
  const currentId = useChat((s) => s.currentId);
  const [params, setParams] = useState<Record<string, number>>({});
  const [prompt, setPrompt] = useState("");
  const [dirty, setDirty] = useState(false);

  useEffect(() => {
    if (!currentId) return;
    api.getSession(currentId).then((s) => {
      const raw = (s as unknown as { params?: Record<string, number>;
                                     system_prompt?: string });
      setParams(raw.params ?? {});
      setPrompt(raw.system_prompt ?? "");
      setDirty(false);
    }).catch(() => {});
  }, [currentId]);

  const save = async () => {
    if (!currentId) return;
    await api.updateSession(currentId, { params, system_prompt: prompt })
      .catch(() => {});
    setDirty(false);
  };

  const num = (key: string, label: string, step: number, max: number) => (
    <label className="flex items-center gap-2 text-[12.5px]">
      <span className="w-24 text-secondary">{label}</span>
      <input
        type="number"
        step={step}
        min={0}
        max={max}
        value={params[key] ?? ""}
        placeholder="default"
        onChange={(e) => {
          const v = e.target.value;
          setParams((p) => {
            const n = { ...p };
            if (v === "") delete n[key];
            else n[key] = Number(v);
            return n;
          });
          setDirty(true);
        }}
        className="flex-1 min-w-0 rounded-md bg-surface px-2 py-0.5 font-mono text-[12px] outline-none"
      />
    </label>
  );

  return (
    <section className="rounded-lg bg-panel p-3 flex flex-col gap-1.5">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em]">
        this chat
      </h3>
      <textarea
        value={prompt}
        onChange={(e) => { setPrompt(e.target.value); setDirty(true); }}
        placeholder="system prompt (empty = default)"
        aria-label="System prompt"
        rows={3}
        className="rounded-md bg-surface px-2 py-1.5 text-[12.5px] outline-none resize-y placeholder:text-muted"
      />
      {num("temperature", "temperature", 0.05, 2)}
      {num("dry_multiplier", "DRY", 0.05, 2)}
      {num("repeat_penalty", "repeat pen.", 0.01, 2)}
      {num("max_tokens", "max tokens", 256, 32768)}
      {dirty && (
        <button
          onClick={() => void save()}
          className="self-end rounded-md bg-amber/15 text-amber px-3 py-1 text-[12.5px] font-semibold"
        >
          save
        </button>
      )}
    </section>
  );
}

export default function Sidecar({ open }: { open: boolean }) {
  if (!open) return null;
  return (
    <aside className="w-[260px] shrink-0 border-l border-white/5 overflow-y-auto p-3 flex flex-col gap-3"
           aria-label="Chat settings">
      <GroundingCard />
      <SamplingCard />
    </aside>
  );
}
