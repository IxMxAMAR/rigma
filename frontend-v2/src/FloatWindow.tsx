// A small draggable window (owner request 2026-07-21): chat settings live
// here instead of a fixed right pane that stole transcript width. Position
// persists per window id; dragging is pointer-capture on the title bar.
import { useEffect, useRef, useState, type ReactNode } from "react";

export default function FloatWindow({
  id, title, open, onClose, children,
}: {
  id: string;
  title: string;
  open: boolean;
  onClose: () => void;
  children: ReactNode;
}) {
  const key = `rigma.float.${id}`;
  const [pos, setPos] = useState<{ x: number; y: number }>(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(key) ?? "");
      if (typeof saved.x === "number" && typeof saved.y === "number")
        return saved;
    } catch { /* first open */ }
    return { x: window.innerWidth - 340, y: 64 };
  });
  const drag = useRef<{ dx: number; dy: number } | null>(null);

  // keep it reachable if the viewport shrank since last session
  useEffect(() => {
    setPos((p) => ({
      x: Math.min(Math.max(0, p.x), window.innerWidth - 80),
      y: Math.min(Math.max(0, p.y), window.innerHeight - 48),
    }));
  }, [open]);

  if (!open) return null;

  return (
    <div
      className="fixed z-40 w-[300px] max-h-[80vh] flex flex-col rounded-lg bg-float shadow-2xl border border-white/8"
      style={{ left: pos.x, top: pos.y }}
      role="dialog"
      aria-label={title}
    >
      <div
        className="flex items-center gap-2 px-3 py-2 cursor-grab active:cursor-grabbing select-none border-b border-white/5"
        onPointerDown={(e) => {
          drag.current = { dx: e.clientX - pos.x, dy: e.clientY - pos.y };
          (e.target as HTMLElement).setPointerCapture(e.pointerId);
        }}
        onPointerMove={(e) => {
          if (!drag.current) return;
          const next = {
            x: Math.min(Math.max(0, e.clientX - drag.current.dx),
                        window.innerWidth - 80),
            y: Math.min(Math.max(0, e.clientY - drag.current.dy),
                        window.innerHeight - 48),
          };
          setPos(next);
        }}
        onPointerUp={() => {
          drag.current = null;
          localStorage.setItem(key, JSON.stringify(pos));
        }}
      >
        <span className="font-mono text-[11px] text-secondary uppercase tracking-[0.08em] flex-1">
          {title}
        </span>
        <button
          onClick={onClose}
          aria-label="Close"
          className="text-muted hover:text-primary text-[14px] leading-none px-1"
        >
          ×
        </button>
      </div>
      <div className="overflow-y-auto p-2.5 flex flex-col gap-2.5">
        {children}
      </div>
    </div>
  );
}
