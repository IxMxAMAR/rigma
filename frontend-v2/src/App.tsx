// The inverted-L shell (docs/design/UI-REWORK-PLAN.md): left sidebar =
// navigation, top header = context + telemetry, main canvas = the surface.
// Phase 1 ships the chrome; each later phase fills one canvas.
import { useEffect, useState } from "react";
import AutonomousSurface from "./autonomous/AutonomousSurface";
import { useChat } from "./chat/chatStore";
import { Copy as CopyIcon, External } from "./Icon";
import ChatSurface from "./chat/ChatSurface";
import EngineSurface from "./engine/EngineSurface";
import MemorySurface from "./memory/MemorySurface";
import ModelsSurface from "./models/ModelsSurface";
import SettingsSurface from "./settings/SettingsSurface";
import Palette from "./Palette";
import WorkspacePanel from "./WorkspacePanel";
import { SURFACES, useApp } from "./store";

function Sidebar() {
  const surface = useApp((s) => s.surface);
  const setSurface = useApp((s) => s.setSurface);
  const width = useApp((s) => s.sidebarWidth);
  const setWidth = useApp((s) => s.setSidebarWidth);
  return (
    <nav
      className="relative shrink-0 bg-panel flex flex-col py-3"
      style={{ width }}
      aria-label="Primary"
    >
      <div
        role="separator"
        aria-orientation="vertical"
        aria-label="Resize sidebar"
        title="drag to resize"
        className="absolute right-0 top-0 h-full w-1.5 cursor-col-resize hover:bg-white/10"
        onPointerDown={(e) => {
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (e.buttons === 1) setWidth(e.clientX);
        }}
      />
      <div className="px-4 pb-3 font-mono font-semibold text-[13px] tracking-[0.08em] text-secondary">
        RIGMA
      </div>
      <ul className="flex flex-col gap-0.5 px-2">
        {SURFACES.map((s) => (
          <li key={s.id}>
            <button
              onClick={() => setSurface(s.id)}
              aria-current={surface === s.id ? "page" : undefined}
              className={`w-full text-left px-3 py-1.5 rounded-md text-[13.5px] ${
                surface === s.id
                  ? "bg-surface text-primary"
                  : "text-secondary hover:bg-white/5 hover:text-primary"
              }`}
            >
              {s.label}
            </button>
          </li>
        ))}
      </ul>
      <WorkspacePanel />
      <div className="mt-auto px-4 pt-3 text-muted font-mono text-[11px]">
        <button
          className="hover:text-secondary"
          onClick={() => useApp.getState().setPalette(true)}
        >
          ctrl+k commands
        </button>
        <a href="/rizz" className="block pt-1 hover:text-secondary">
          legacy ui →
        </a>
      </div>
    </nav>
  );
}

function Header() {
  const surface = useApp((s) => s.surface);
  const server = useApp((s) => s.server);
  const workspacePath = useApp((s) => s.workspacePath);
  const label = SURFACES.find((s) => s.id === surface)?.label ?? "";
  const currentId = useChat((s) => s.currentId);
  const [copied, setCopied] = useState(false);
  const [editingWs, setEditingWs] = useState(false);
  const setWorkspacePath = useApp((s) => s.setWorkspacePath);
  const saveWs = async (raw: string) => {
    setEditingWs(false);
    const v = raw.trim();
    if (v === workspacePath || !currentId) return;
    await fetch(`/api/sessions/${currentId}`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ workspace: v }),
    }).catch(() => {});
    setWorkspacePath(v);
  };
  return (
    <header className="h-12 shrink-0 flex items-center gap-4 px-5 bg-panel">
      <h1 className="text-[14px] font-semibold shrink-0">{label}</h1>
      {surface === "chat" && currentId && (
        <span className="flex items-center gap-1.5 min-w-0 max-w-[44%]">
          {editingWs ? (
            <input
              autoFocus
              defaultValue={workspacePath}
              placeholder="D:\Writing"
              aria-label="Workspace folder path"
              onKeyDown={(e) => {
                if (e.key === "Escape") setEditingWs(false);
                if (e.key === "Enter")
                  void saveWs((e.target as HTMLInputElement).value);
              }}
              onBlur={(e) => void saveWs(e.target.value)}
              className="min-w-[220px] rounded-md bg-surface px-2 py-0.5 font-mono text-[11.5px] outline-none placeholder:text-muted"
            />
          ) : (
            <button
              onClick={() => setEditingWs(true)}
              title={(workspacePath || "no workspace") + "  (click to edit)"}
              aria-label="Edit workspace path"
              dir="rtl"
              className="font-mono text-[11.5px] text-muted hover:text-secondary truncate min-w-0"
            >
              {workspacePath ? "‎" + workspacePath : "set workspace…"}
            </button>
          )}
          {workspacePath && !editingWs && (
            <>
              <button
                onClick={() => {
                  void navigator.clipboard.writeText(workspacePath).then(() => {
                    setCopied(true);
                    window.setTimeout(() => setCopied(false), 1200);
                  });
                }}
                aria-label="Copy workspace path"
                title={copied ? "copied!" : "copy path"}
                className="shrink-0 text-muted hover:text-secondary"
              >
                {copied ? <span className="font-mono text-[11px] text-moss">✓</span> : <CopyIcon />}
              </button>
              <button
                onClick={() => {
                  void fetch(`/api/sessions/${currentId}/workspace/open`,
                             { method: "POST" });
                }}
                aria-label="Open workspace in file manager"
                title="open in Explorer"
                className="shrink-0 text-muted hover:text-amber"
              >
                <External />
              </button>
            </>
          )}
        </span>
      )}
      <div className="ml-auto flex items-center gap-4 font-mono text-[12px]">
        {server ? (
          <>
            <span className="text-secondary">{server.model}</span>
            {server.tps != null && (
              <span className="text-amber">{server.tps.toFixed(1)} tok/s</span>
            )}
            <span
              className={`inline-block w-2 h-2 rounded-full ${
                server.healthy ? "bg-moss" : "bg-red"
              }`}
              aria-label={server.healthy ? "engine healthy" : "engine down"}
            />
          </>
        ) : (
          <span className="text-muted">connecting…</span>
        )}
      </div>
    </header>
  );
}

function Canvas() {
  const surface = useApp((s) => s.surface);
  const meta = SURFACES.find((s) => s.id === surface);
  if (surface === "chat") return <ChatSurface />;
  if (surface === "engine") return <EngineSurface />;
  if (surface === "models") return <ModelsSurface />;
  if (surface === "autonomous") return <AutonomousSurface />;
  if (surface === "memory") return <MemorySurface />;
  if (surface === "settings") return <SettingsSurface />;
  // Remaining surfaces: a deliberate, designed empty state each —
  // CONSTITUTION §7: "an empty screen is an invitation to act", never blank.
  return (
    <main className="flex-1 overflow-y-auto bg-canvas flex items-center justify-center">
      <div className="text-center max-w-[380px] px-6">
        <div className="font-mono text-[12px] text-muted mb-2 uppercase tracking-[0.1em]">
          {meta?.label}
        </div>
        <p className="text-secondary text-[13.5px]">
          {meta?.hint}. This surface arrives in a later phase — the legacy UI
          at{" "}
          <a href="/rizz" className="text-amber hover:underline">
            /rizz
          </a>{" "}
          keeps full functionality until parity.
        </p>
      </div>
    </main>
  );
}

export default function App() {
  const setServer = useApp((s) => s.setServer);
  useEffect(() => {
    let alive = true;
    const poll = async () => {
      try {
        const r = await fetch("/api/server");
        const d = await r.json();
        if (alive)
          setServer({
            model: d.model ?? "",
            quant: d.quant ?? "",
            tps: d.last_tg ?? d.tps ?? null,
            healthy: !d.unloaded && !!d.model,
            ctx: d.ctx ?? 0,
          });
      } catch {
        if (alive) setServer(null);
      }
    };
    poll();
    const t = setInterval(poll, 5000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [setServer]);

  return (
    <div className="h-full flex overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 min-h-0">
        <Header />
        <Canvas />
      </div>
      <Palette />
    </div>
  );
}
