// Command palette — first-class from Phase 1 (UI-REWORK-PLAN: "every action
// reachable without the mouse"). Ctrl+K opens; type to filter; arrows + Enter
// to run; Esc closes. Actions are a flat, data-driven list so later phases
// register surface-specific commands without touching this component.
import { useEffect, useMemo, useRef, useState } from "react";
import { useChat } from "./chat/chatStore";
import { SURFACES, useApp } from "./store";

export interface Command {
  id: string;
  label: string;
  hint?: string;
  run: () => void;
}

export function usePaletteCommands(): Command[] {
  const setSurface = useApp((s) => s.setSurface);
  const newChat = useChat((s) => s.newChat);
  return useMemo(
    () => [
      {
        id: "new-chat",
        label: "New chat",
        hint: "fresh conversation with the loaded model",
        run: () => {
          setSurface("chat");
          void newChat();
        },
      },
      ...SURFACES.map((s) => ({
        id: `go-${s.id}`,
        label: `Go to ${s.label}`,
        hint: s.hint,
        run: () => setSurface(s.id),
      })),
      {
        id: "legacy",
        label: "Open legacy UI",
        hint: "/rizz — full functionality until v2 parity",
        run: () => {
          window.location.href = "/rizz";
        },
      },
    ],
    [setSurface, newChat],
  );
}

export default function Palette() {
  const open = useApp((s) => s.paletteOpen);
  const setPalette = useApp((s) => s.setPalette);
  const commands = usePaletteCommands();
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);

  const hits = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return commands;
    return commands.filter(
      (c) =>
        c.label.toLowerCase().includes(q) ||
        (c.hint ?? "").toLowerCase().includes(q),
    );
  }, [commands, query]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") {
        e.preventDefault();
        setPalette(!useApp.getState().paletteOpen);
      } else if (e.key === "Escape") {
        setPalette(false);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [setPalette]);

  useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(0);
      // focus after the element exists
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  if (!open) return null;

  const onInputKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setCursor((c) => Math.min(c + 1, hits.length - 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setCursor((c) => Math.max(c - 1, 0));
    } else if (e.key === "Enter" && hits[cursor]) {
      hits[cursor].run();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/40 flex items-start justify-center pt-[18vh]"
      onMouseDown={(e) => e.target === e.currentTarget && setPalette(false)}
      role="dialog"
      aria-modal="true"
      aria-label="Command palette"
    >
      <div className="w-[560px] max-w-[92vw] rounded-lg bg-float shadow-2xl overflow-hidden">
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setCursor(0);
          }}
          onKeyDown={onInputKey}
          placeholder="Type a command…"
          aria-label="Search commands"
          className="w-full bg-transparent px-4 py-3 text-[15px] text-primary placeholder:text-muted outline-none border-b border-white/5"
        />
        <ul className="max-h-[320px] overflow-y-auto py-1" role="listbox">
          {hits.length === 0 && (
            <li className="px-4 py-3 text-muted text-[13px]">
              Nothing matches “{query}”
            </li>
          )}
          {hits.map((c, i) => (
            <li key={c.id} role="option" aria-selected={i === cursor}>
              <button
                className={`w-full text-left px-4 py-2 flex items-baseline gap-3 ${
                  i === cursor ? "bg-white/5" : "hover:bg-white/5"
                }`}
                onMouseEnter={() => setCursor(i)}
                onClick={c.run}
              >
                <span className="text-primary text-[13.5px]">{c.label}</span>
                {c.hint && (
                  <span className="text-muted text-[12px]">{c.hint}</span>
                )}
              </button>
            </li>
          ))}
        </ul>
        <div className="px-4 py-2 border-t border-white/5 font-mono text-[11px] text-muted flex gap-4">
          <span>↑↓ navigate</span>
          <span>↵ run</span>
          <span>esc close</span>
        </div>
      </div>
    </div>
  );
}
