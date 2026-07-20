// Sidebar workspace browser (owner request 2026-07-21): the active chat's
// workspace folder, always visible — what the model's file tools can touch.
// Refreshes when the chat changes and when a streaming turn finishes (that
// is when files appear).
import { useCallback, useEffect, useRef, useState } from "react";
import { useChat } from "./chat/chatStore";

interface Entry {
  name: string;
  dir: boolean;
  size: number;
}

const kb = (n: number) =>
  n < 1024 ? `${n} B`
  : n < 1024 ** 2 ? `${(n / 1024).toFixed(0)} KB`
  : `${(n / 1024 ** 2).toFixed(1)} MB`;

export default function WorkspacePanel() {
  const currentId = useChat((s) => s.currentId);
  const streaming = useChat((s) => s.streaming);
  const [data, setData] = useState<{ path: string; entries: Entry[];
                                     missing?: boolean } | null>(null);
  const wasStreaming = useRef(false);

  const refresh = useCallback(async () => {
    if (!currentId) { setData(null); return; }
    try {
      const r = await fetch(`/api/sessions/${currentId}/workspace`);
      const d: unknown = await r.json();
      if (r.ok && d && typeof d === "object" && "entries" in (d as object))
        setData(d as { path: string; entries: Entry[] });
      else setData(null);
    } catch { setData(null); }
  }, [currentId]);

  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    // falling edge of streaming = a turn just finished; files may have changed
    if (wasStreaming.current && !streaming) void refresh();
    wasStreaming.current = !!streaming;
  }, [streaming, refresh]);

  if (!data || !data.path) return null;
  const base = data.path.replace(/[\/]+$/, "").split(/[\/]/).pop();
  return (
    <div className="px-2 pt-3 border-t border-white/5 mx-2 min-h-0 flex flex-col">
      <div className="flex items-center gap-1.5 px-2 pb-1">
        <span className="font-mono text-[10.5px] text-muted uppercase tracking-[0.08em] flex-1 truncate"
              title={data.path}>
          workspace · {base}
        </span>
        <button onClick={() => void refresh()} aria-label="Refresh workspace"
                className="text-muted hover:text-secondary text-[11px]">↻</button>
      </div>
      {data.missing ? (
        <div className="px-2 font-mono text-[11px] text-red">folder missing</div>
      ) : data.entries.length === 0 ? (
        <div className="px-2 font-mono text-[11px] text-muted">empty</div>
      ) : (
        <ul className="overflow-y-auto max-h-[30vh] pb-1">
          {data.entries.map((e) => (
            <li key={e.name}
                className="flex items-baseline gap-2 px-2 py-0.5 rounded hover:bg-white/5"
                title={e.name}>
              <span className={`font-mono text-[11.5px] truncate flex-1 ${e.dir ? "text-secondary" : "text-primary/80"}`}>
                {e.dir ? e.name + "/" : e.name}
              </span>
              {!e.dir && (
                <span className="font-mono text-[10px] text-muted shrink-0">{kb(e.size)}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
