// Right sidecar for the chat surface: grounding, sampling, system prompt.
// Collapsible; state persists to the session via the existing PATCH API.
import { useCallback, useEffect, useRef, useState } from "react";
import { api } from "../lib/api";
import {
  createDraft,
  exportUrl,
  importMethod,
  listMethods,
  type Method,
} from "../lib/methods";
import { engineApi, type ServerInfo } from "../lib/engineApi";
import { useApp } from "../store";
import MethodBuilder from "./MethodBuilder";
import { useChat } from "./chatStore";

const CTX_STEPS = [8192, 16384, 32768, 65536, 131072, 262144];
const KV_TYPES = ["f16", "q8_0", "q5_1", "q4_0"];

/** Engine settings, in the panel where they are actually needed.
 *
 * These were only on the Engine page, which is a different screen from the
 * chat whose context window they set — and the sampling card sat next to a
 * note telling you to go there. They live here now, but under their own
 * heading and their own warning, because they are NOT per-chat: one engine
 * serves every session, and each of these stops it and starts it again.
 */
function EngineCard() {
  const [srv, setSrv] = useState<ServerInfo | null>(null);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const streaming = useChat((s) => s.streaming);
  // Staged, not applied. Every control used to relaunch the engine the instant
  // it changed, so reaching a config that differs in context AND cache cost two
  // full reloads of a 13GB model to get to one setup (owner, 2026-08-19). The
  // endpoint has always accepted all three together; only the UI insisted on
  // one at a time.
  const [want, setWant] = useState<{ctx?: number; kv?: string; vision?: boolean}>({});

  const load = useCallback(() => {
    engineApi.server().then(setSrv).catch(() => setSrv(null));
  }, []);
  useEffect(() => { load(); }, [load]);

  if (!srv || srv.unloaded) return null;

  const apply = async (what: string, o: {
    ctx?: number; kv?: string; vision?: boolean;
  }) => {
    if (streaming) {
      setErr("a reply is still generating — stop it first, or wait: "
             + "relaunching now would cut it off mid-sentence.");
      return;
    }
    const ctx = o.ctx ?? srv.ctx ?? 0;
    if (!window.confirm(
      `${what}\n\nThis RESTARTS the engine:\n` +
      "  • the model unloads and reloads — tens of seconds on a large quant\n" +
      "  • every chat is served by this one engine, not just this one\n" +
      "  • your conversations are on disk and are NOT lost\n" +
      "  • a setting that doesn't fit will fail the launch, and Rigma says " +
      "so rather than starting something you didn't ask for\n\nGo ahead?")) {
      return;
    }
    setBusy(what);
    setErr(null);
    try {
      await engineApi.relaunchWith({ ctx, kv: o.kv, vision: o.vision });
      setWant({});          // applied — stop showing it as pending
    } catch (e) {
      setErr((e as Error).message);
    }
    setBusy(null);
    load();
  };

  const row = "flex items-center gap-2 text-[12.5px]";
  const label = "w-24 text-secondary";
  const input = "flex-1 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none disabled:opacity-40";
  const native = srv.native_ctx || 262144;

  return (
    <section className="rounded-lg bg-panel p-3 flex flex-col gap-1.5">
      <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em]">
        engine — affects every chat
      </h3>

      <label className={row} title="The context window: how much of the
conversation the model can see. Bigger windows cost VRAM, which competes with
the model's own weights.">
        <span className={label}>context</span>
        <select className={input} value={String(want.ctx ?? srv.ctx ?? "")}
                disabled={!!busy}
                aria-label="Context window"
                onChange={(e) =>
                  setWant((w) => ({ ...w, ctx: Number(e.target.value) }))}>
          {CTX_STEPS.filter((c) => c <= native).concat(
            srv.ctx && !CTX_STEPS.includes(srv.ctx) ? [srv.ctx] : [])
            .sort((a, b) => a - b)
            .map((c) => (
              <option key={c} value={c}>{Math.round(c / 1024)}K</option>
            ))}
        </select>
      </label>

      <label className={row} title="KV cache precision. Halving it roughly
doubles the context that fits, but cache error is written per token and every
later token attends over it, so it compounds over a long conversation.">
        <span className={label}>kv cache</span>
        <select className={input} value={want.kv ?? (srv.kv_cache || "")}
                disabled={!!busy}
                aria-label="KV cache precision"
                onChange={(e) => setWant((w) => ({ ...w, kv: e.target.value }))}>
          {!srv.kv_cache && <option value="">auto</option>}
          {KV_TYPES.map((k) => <option key={k} value={k}>{k}</option>)}
        </select>
      </label>

      {(want.ctx !== undefined || want.kv !== undefined
        || want.vision !== undefined) && (
        <button
          disabled={!!busy}
          onClick={() => {
            const bits = [
              want.ctx !== undefined ? `context → ${Math.round(want.ctx / 1024)}K` : "",
              want.kv !== undefined ? `kv → ${want.kv || "auto"}` : "",
              want.vision !== undefined ? (want.vision ? "vision on" : "vision off") : "",
            ].filter(Boolean);
            void apply(bits.join(", "), {
              ctx: want.ctx, kv: want.kv, vision: want.vision,
            });
          }}
          className="mt-1 rounded-md bg-amber/15 text-amber px-2.5 py-1 text-[12px] font-semibold disabled:opacity-40"
          title={"Applies everything you changed in ONE relaunch. Changing "
                 + "these one at a time used to reload the model once per "
                 + "setting."}
        >
          {busy ? "relaunching…" : "apply — one relaunch"}
        </button>
      )}
      {srv.has_mmproj && (
        <label className={row} title="The vision projector is loaded with the
weights and cannot be offloaded, so it costs VRAM for the whole session whether
or not you ever send an image. Turning it off frees that VRAM for context —
often the single biggest context lever a vision model has.">
          <span className={label}>vision</span>
          <span className="flex-1 flex items-center gap-2">
            <input type="checkbox" className="accent-amber"
                   disabled={!!busy}
                   checked={want.vision ?? !srv.no_vision}
                   onChange={(e) =>
                     setWant((w) => ({ ...w, vision: e.target.checked }))} />
            <span className="text-muted font-mono text-[11px]">
              {srv.no_vision ? "text-only — projector not loaded"
                : "images enabled"}
            </span>
          </span>
        </label>
      )}

      {busy && (
        <p className="font-mono text-[11.5px] text-amber">
          {busy} — relaunching the engine…
        </p>
      )}
      {err && (
        <div className="rounded-md bg-red/10 text-red px-2.5 py-1.5 font-mono text-[11.5px]">
          {err}
        </div>
      )}
      <p className="text-muted text-[11.5px] leading-snug">
        Each of these restarts the engine. Chats are on disk and survive it; a
        reply in progress does not. See every quant's context and VRAM budget
        on the <span className="text-secondary">Models</span> page.
      </p>
    </section>
  );
}

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
  // How hard the model thinks. Qwen3.8 publishes four reasoning levels and its
  // chat template reads `reasoning_effort`; Rigma could only send a binary
  // enable_thinking, and the UI could not send even that — the field was not
  // rendered anywhere. Applies per chat and needs NO engine restart.
  const [effort, setEffort] = useState("");

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
      setEffort(String((s as unknown as { effort?: string }).effort ?? ""));
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
      <label className="flex items-center gap-2 text-[12.5px]"
             title={"How hard the model thinks before answering. "
                    + "Qwen3.8 publishes four levels; its template turns the "
                    + "level into its own steering text. medium injects no "
                    + "instruction at all, which is why it costs nothing. "
                    + "Applies to THIS chat and takes effect on the next "
                    + "message — no engine restart."}>
        <span className="w-24 text-secondary">thinking</span>
        <select
          value={effort}
          aria-label="Thinking effort"
          onChange={async (e) => {
            const v = e.target.value;
            setEffort(v);
            if (currentId)
              await api.updateSession(currentId, { effort: v }).catch(() => {});
          }}
          className="flex-1 min-w-0 rounded-md bg-surface px-2 py-1 text-[12.5px] outline-none"
        >
          <option value="">model default</option>
          <option value="off">off — no thinking</option>
          <option value="low">low</option>
          <option value="medium">medium</option>
          <option value="high">high</option>
          <option value="xhigh">xhigh — slowest, most careful</option>
        </select>
      </label>
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
        max tokens caps ONE reply. The context window (the “of{" "}
        {Math.round(maxTok / 1024)}K” bar) is a different thing — how much of
        the conversation the model can see — and it is in{" "}
        <span className="text-secondary">engine</span> just below, because
        changing it restarts the engine.
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
  const [making, setMaking] = useState(false);
  const [importError, setImportError] = useState("");
  const importRef = useRef<HTMLInputElement>(null);
  const [methods, setMethods] = useState<Method[]>([]);
  const [active, setActive] = useState("");
  const [expanded, setExpanded] = useState("");
  const [applied, setApplied] = useState("");

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

  const importFile = async (f: File) => {
    try {
      const doc = JSON.parse(await f.text()) as unknown;
      const saved = await importMethod(doc);
      setMethods(await listMethods());
      setExpanded(saved.id);
    } catch (e) {
      setImportError((e as Error).message || "that file is not a method");
    }
  };

  // opens a chat bound to a draft: the model there is offered the builder
  // tools and nothing else, so it can only build a method
  const startDraft = async () => {
    if (making) return;
    setMaking(true);
    try {
      const d = await createDraft();
      await loadSessions();
      await open(d.session_id);
    } catch { /* the panel stays as it is */ }
    setMaking(false);
  };

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
      <div className="flex items-center gap-1.5 flex-wrap">
        <button
          onClick={() => void startDraft()}
          disabled={making}
          className="rounded-md bg-surface hover:bg-float text-secondary px-2.5 py-1 text-[12px] disabled:opacity-40"
        >
          {making ? "opening…" : "+ Create method"}
        </button>
        <button
          onClick={() => importRef.current?.click()}
          title="load a method someone shared with you"
          className="rounded-md bg-surface hover:bg-float text-secondary px-2.5 py-1 text-[12px]"
        >
          Import
        </button>
        <input
          ref={importRef}
          type="file"
          accept="application/json,.json"
          hidden
          onChange={(e) => {
            const f = e.target.files?.[0];
            e.target.value = "";
            if (f) void importFile(f);
          }}
        />
      </div>
      {importError && (
        <p className="text-[11px] text-red">{importError}</p>
      )}
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
                    <a
                      href={exportUrl(m.id)}
                      download={`${m.id}.json`}
                      title="save this method as a file you can share"
                      className="rounded-md bg-surface hover:bg-float text-muted px-2.5 py-1 text-[12px]"
                    >
                      Export
                    </a>
                  </div>
                  {/* what the method actually CONTAINS — running its macros
                      belongs to the strip above the composer, not here */}
                  {(m.rules?.length || m.macros?.length ||
                    m.workflows?.length) ? (
                    <div className="flex flex-col gap-1 pt-1">
                      {m.rules && m.rules.length > 0 && (
                        <p className="text-[11px] text-muted">
                          <span className="font-mono">rules</span>{" "}
                          {m.rules.length}
                        </p>
                      )}
                      {m.macros && m.macros.length > 0 && (
                        <p className="text-[11px] text-muted">
                          <span className="font-mono">macros</span>{" "}
                          {m.macros.map((x) => x.label).join(", ")}
                        </p>
                      )}
                      {m.workflows && m.workflows.length > 0 && (
                        <p className="text-[11px] text-muted">
                          <span className="font-mono">workflows</span>{" "}
                          {m.workflows.map((x) => x.label).join(", ")}
                        </p>
                      )}
                      {active === m.id && m.macros && m.macros.length > 0 && (
                        <p className="text-[11px] text-muted">
                          the buttons are above the message box
                        </p>
                      )}
                    </div>
                  ) : null}
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
  const currentId = useChat((s) => s.currentId);
  const [draftId, setDraftId] = useState("");

  // a creation chat carries method_draft_id; the builder panel replaces the
  // usual cards there, because none of them apply while building
  useEffect(() => {
    setDraftId("");
    if (!currentId) return;
    api.getSession(currentId)
      .then((s) =>
        setDraftId(String((s as { method_draft_id?: string })
          .method_draft_id ?? "")))
      .catch(() => {});
  }, [currentId]);

  if (draftId) return <MethodBuilder draftId={draftId} />;

  // Order is deliberate and owner-chosen (2026-07-30): what you touch every
  // turn first, what you set up occasionally next, what you configure once
  // last. Methods used to lead — six cards of setup above the two controls
  // actually reached for mid-conversation.
  return (
    <>
      <SamplingCard key={`s${rev}`} />
      <EngineCard key={`e${rev}`} />
      <MethodCard onApplied={() => setRev((r) => r + 1)} />
      <GroundingCard key={`g${rev}`} />
    </>
  );
}
