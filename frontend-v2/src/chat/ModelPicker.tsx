// No model on the engine and the user opened a chat: offer the fix in place
// (owner request 2026-07-21). Lists on-disk models, one click to load, live
// status while the engine comes up. Dismissible — the composer stays usable
// for typing; sending is what needs a model.
import { useEffect, useRef, useState } from "react";
import { engineApi, type ModelCard } from "../lib/engineApi";
import { useApp } from "../store";

export default function ModelPicker() {
  const server = useApp((s) => s.server);
  const surface = useApp((s) => s.surface);
  const [cards, setCards] = useState<ModelCard[] | null>(null);
  const [state, setState] = useState<"idle" | "loading" | "error">("idle");
  const [err, setErr] = useState("");
  const [dismissed, setDismissed] = useState(false);
  const pickedRef = useRef<string | null>(null);

  const engineDown = server === null || !server.healthy;
  const show = surface === "chat" && engineDown && !dismissed;

  useEffect(() => {
    if (!show || cards !== null) return;
    engineApi.models()
      .then((d) => setCards(d.models.filter(
        (m) => m.quants.some((q) => q.on_disk))))
      .catch(() => setCards([]));
  }, [show, cards]);

  // engine came up (poll in App noticed): close ourselves
  useEffect(() => {
    if (!engineDown) {
      setDismissed(false);
      setState("idle");
      pickedRef.current = null;
    }
  }, [engineDown]);

  if (!show) return null;

  const pick = async (slug: string) => {
    setState("loading");
    setErr("");
    pickedRef.current = slug;
    try {
      await engineApi.switchTo(slug);
      // App's 5s poll flips server.healthy and this modal closes itself
    } catch (e) {
      setState("error");
      setErr((e as Error).message);
    }
  };

  return (
    <div className="fixed inset-0 z-40 bg-black/40 flex items-center justify-center"
         role="dialog" aria-modal="true" aria-label="Choose a model">
      <div className="w-[440px] max-w-[92vw] rounded-lg bg-float shadow-2xl overflow-hidden">
        <div className="px-4 pt-4 pb-2">
          <h2 className="text-[14px] font-semibold">No model is loaded</h2>
          <p className="text-[12.5px] text-secondary mt-0.5">
            Pick one to start chatting — these are already on disk.
          </p>
        </div>
        {state === "loading" ? (
          <div className="px-4 py-8 text-center">
            <div className="font-mono text-[12.5px] text-amber animate-pulse">
              loading {pickedRef.current}…
            </div>
            <p className="text-[11.5px] text-muted mt-1">
              first load can take a couple of minutes
            </p>
          </div>
        ) : (
          <ul className="max-h-[300px] overflow-y-auto py-1">
            {cards === null && (
              <li className="px-4 py-3 font-mono text-[12px] text-muted">loading…</li>
            )}
            {cards?.length === 0 && (
              <li className="px-4 py-3 text-[12.5px] text-secondary">
                No models on disk yet — install one from the Models page.
              </li>
            )}
            {cards?.map((m) => (
              <li key={m.slug}>
                <button
                  onClick={() => void pick(m.slug)}
                  className="w-full text-left px-4 py-2 hover:bg-white/5 flex items-baseline gap-2"
                >
                  <span className="text-[13.5px] text-primary truncate">{m.slug}</span>
                  <span className="font-mono text-[11px] text-muted shrink-0">
                    {m.quants.filter((q) => q.on_disk).map((q) => q.quant).join(" ")}
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
        {err && (
          <div className="px-4 py-2 text-[12px] text-red">{err} — try the Engine page.</div>
        )}
        <div className="flex items-center justify-between px-4 py-2 border-t border-white/5">
          <button
            onClick={() => useApp.getState().setSurface("models")}
            className="font-mono text-[11.5px] text-muted hover:text-secondary"
          >
            open Models →
          </button>
          <button
            onClick={() => setDismissed(true)}
            className="font-mono text-[11.5px] text-muted hover:text-secondary"
          >
            not now
          </button>
        </div>
      </div>
    </div>
  );
}
