// Right sidecar for the chat surface: grounding, sampling, system prompt.
// Collapsible; state persists to the session via the existing PATCH API.
import { useCallback, useEffect, useState } from "react";
import { api } from "../lib/api";
import { type MacroDef, type Method } from "../lib/methods";
import { useApp } from "../store";
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

interface PresetRow {
  id: string;
  name: string;
}

function SamplingCard() {
  const currentId = useChat((s) => s.currentId);
  // the cap follows the ENGINE's context — a hardcoded 32768 silently
  // clamped a 131072 the user had set (owner report 2026-07-21)
  const maxTok = useApp((s) => s.server?.ctx) || 262144;
  const [params, setParams] = useState<Record<string, number>>({});
  const [prompt, setPrompt] = useState("");
  const [dirty, setDirty] = useState(false);
  const [presets, setPresets] = useState<PresetRow[]>([]);
  const [presetId, setPresetId] = useState("");

  useEffect(() => {
    fetch("/api/presets")
      .then((r) => r.json())
      .then((d: unknown) => Array.isArray(d) && setPresets(d as PresetRow[]))
      .catch(() => {});
  }, []);

  useEffect(() => {
    if (!currentId) return;
    api.getSession(currentId).then((s) => {
      const raw = (s as unknown as { params?: Record<string, number>;
                                     system_prompt?: string;
                                     preset_id?: string });
      const loaded = { ...(raw.params ?? {}) };
      // retro-clamp: values saved before limits existed (the 2,500,000 max
      // tokens era) must not survive a reload
      if (loaded.max_tokens != null && loaded.max_tokens > maxTok)
        loaded.max_tokens = maxTok;
      setParams(loaded);
      setPrompt(raw.system_prompt ?? "");
      setPresetId(raw.preset_id ?? "");
      setDirty(false);
    }).catch(() => {});
  }, [currentId]);

  const LIMITS: Record<string, number> = {
    temperature: 2, dry_multiplier: 2, repeat_penalty: 2, max_tokens: maxTok,
  };
  const save = async () => {
    if (!currentId) return;
    const clamped: Record<string, number> = {};
    for (const [k, v] of Object.entries(params))
      clamped[k] = Math.min(LIMITS[k] ?? v, Math.max(0, v));
    setParams(clamped);
    await api.updateSession(currentId,
      { params: clamped, system_prompt: prompt }).catch(() => {});
    setDirty(false);
  };

  const num = (key: string, label: string, step: number, max: number,
               showMax = false) => (
    <label className="flex items-center gap-2 text-[12.5px]">
      <span className="w-24 text-secondary">
        {label}
        {showMax && (
          <span className="block font-mono text-[10px] text-muted">
            reply cap · max {max.toLocaleString()}
          </span>
        )}
      </span>
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
        onBlur={() => {
          // the max attribute only gates the SPINNER; typing walks straight
          // past it (owner demonstrated with 2,500,000 max tokens). Clamp
          // for real once focus leaves.
          setParams((p) => {
            const v = p[key];
            if (v == null) return p;
            const clamped = Math.min(max, Math.max(0, v));
            return clamped === v ? p : { ...p, [key]: clamped };
          });
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
      <label className="flex items-center gap-2 text-[12.5px]">
        <span className="w-24 text-secondary">preset</span>
        <select
          value={presetId}
          onChange={async (e) => {
            const v = e.target.value;
            setPresetId(v);
            if (currentId)
              await api.updateSession(currentId, { preset_id: v })
                .catch(() => {});
          }}
          aria-label="Preset"
          className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none"
        >
          <option value="">none</option>
          {presets.map((pr) => (
            <option key={pr.id} value={pr.id}>{pr.name}</option>
          ))}
        </select>
      </label>
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
      {num("max_tokens", "max tokens", 1024, maxTok, true)}
      <p className="text-[10.5px] text-muted leading-snug">
        max tokens caps one reply. The context window (the “of{" "}
        {Math.round(maxTok / 1024)}K” bar) is set on the{" "}
        <button
          className="text-amber hover:underline"
          onClick={() => useApp.getState().setSurface("engine")}
        >
          Engine page
        </button>{" "}
        and needs a model relaunch.
      </p>
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


// One-click workflow setups: prompt + sampler profile + effort + tool
// posture + a Notes template + the how-to guide, per activity. The method
// knowledge lives in the product, not in the user's memory.
function MethodCard({ onApplied }: { onApplied?: () => void }) {
  const currentId = useChat((s) => s.currentId);
  const open = useChat((s) => s.open);
  const loadSessions = useChat((s) => s.loadSessions);
  const [methods, setMethods] = useState<Method[]>([]);
  const [active, setActive] = useState("");
  const [expanded, setExpanded] = useState("");
  const [applied, setApplied] = useState("");
  const [macroBusy, setMacroBusy] = useState("");

  // Plan 2 replaces this with the MacroStrip above the composer and an inline
  // ConfirmBar. Until then the buttons live here so the book method keeps the
  // one-click move it had as a "ritual".
  const runMacro = async (macro: MacroDef) => {
    if (!currentId || macroBusy) return;
    setMacroBusy(macro.id);
    try {
      const pre = await fetch(`/api/sessions/${currentId}/macro/preview`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ macro_id: macro.id }),
      });
      const p = (await pre.json()) as
        { needs_confirm?: boolean; preview?: string };
      if (p.needs_confirm && !window.confirm(`${p.preview}\n\nRun it?`)) {
        setMacroBusy("");
        return;
      }
      const r = await fetch(`/api/sessions/${currentId}/macro`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ macro_id: macro.id, confirm: "run" }),
      });
      const text = await r.text();
      const m = /"new_session_id":\s*"([^"]+)"/.exec(text);
      await loadSessions();
      if (m) await open(m[1]);
      else if (currentId) await open(currentId);
    } catch { /* chat stays where it is */ }
    setMacroBusy("");
  };

  useEffect(() => {
    fetch("/api/methods")
      .then((r) => (r.ok ? r.json() : null))
      .then((d: { methods: Method[] } | null) => {
        if (d) setMethods(d.methods);
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    setApplied("");
    if (!currentId) return;
    api.getSession(currentId)
      .then((s) => setActive(String((s as { method?: string }).method ?? "")))
      .catch(() => {});
  }, [currentId]);

  const apply = async (id: string) => {
    if (!currentId) return;
    try {
      const r = await fetch(`/api/sessions/${currentId}/method`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ id }),
      });
      if (r.ok) {
        setActive(id);
        setApplied(id);
        onApplied?.();
      }
    } catch { /* stays unapplied */ }
  };

  return (
    <section className="rounded-lg bg-panel p-3 flex flex-col gap-1.5">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em]">
        method
      </h3>
      <p className="text-[11.5px] text-muted -mt-0.5">
        set this chat up for what you're doing — prompt, sampling, thinking
        and a notes template in one click
      </p>
      <ul className="flex flex-col gap-1 mt-1">
        {methods.map((m) => {
          const open = expanded === m.id;
          return (
            <li key={m.id} className="rounded-md bg-surface/60">
              <button
                className="w-full flex items-center gap-2 px-2.5 py-1.5 text-left"
                onClick={() => setExpanded(open ? "" : m.id)}
                aria-expanded={open}
              >
                <span className={`font-mono text-[11px] ${
                  active === m.id ? "text-moss" : "text-muted"}`}>
                  {active === m.id ? "●" : "○"}
                </span>
                <span className="text-[12.5px] text-primary">{m.name}</span>
                <span className="flex-1 truncate text-[11px] text-muted">
                  {m.tagline}
                </span>
                <span className="font-mono text-[11px] text-muted">
                  {open ? "▾" : "▸"}
                </span>
              </button>
              {open && (
                <div className="px-2.5 pb-2 flex flex-col gap-1.5">
                  <ul className="flex flex-col gap-1">
                    {m.guide.map((g, i) => (
                      <li key={i} className="text-[11.5px] text-secondary flex gap-1.5">
                        <span className="text-muted">·</span>
                        <span>{g}</span>
                      </li>
                    ))}
                  </ul>
                  <div className="flex items-center gap-2 flex-wrap">
                    <button
                      onClick={() => void apply(m.id)}
                      className="rounded-md bg-amber/15 text-amber px-2.5 py-1 text-[12px] font-semibold"
                    >
                      {applied === m.id ? "applied ✓"
                        : active === m.id ? "re-apply" : "use this method"}
                    </button>
                    {active === m.id && (m.macros ?? []).map((mac) => (
                      <button
                        key={mac.id}
                        onClick={() => void runMacro(mac)}
                        disabled={!!macroBusy}
                        title={mac.hint}
                        className="rounded-md bg-moss/15 text-moss px-2.5 py-1 text-[12px] font-semibold disabled:opacity-50"
                      >
                        {macroBusy === mac.id ? "working…" : mac.label}
                      </button>
                    ))}
                  </div>
                  {applied === m.id && (
                    <p className="text-[11px] text-muted">
                      prompt, sampling, effort and tools set — notes got the
                      template only if they were empty
                    </p>
                  )}
                </div>
              )}
            </li>
          );
        })}
      </ul>
    </section>
  );
}

// rendered inside the draggable FloatWindow — no layout chrome of its own
export default function Sidecar() {
  // applying a method rewrites prompt/params server-side; remount the cards
  // below so they re-fetch — without this the panel looked unchanged and
  // the apply button read as broken (owner report 2026-07-21)
  const [rev, setRev] = useState(0);
  return (
    <>
      <MethodCard onApplied={() => setRev((r) => r + 1)} />
      <GroundingCard key={`g${rev}`} />
      <SamplingCard key={`s${rev}`} />
    </>
  );
}
