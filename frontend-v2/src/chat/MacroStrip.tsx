// The method's macros, one click from the composer. A macro that changes
// things asks first — inline, because a window.confirm cannot show WHICH
// file it is about to touch and reads as a browser error rather than a
// choice. The preview sentence comes from the server, generated from the
// macro's own steps after placeholder substitution, so it names real paths.
import { useEffect, useState } from "react";
import {
  listMethods,
  previewMacro,
  type MacroDef,
  type MacroPreview,
} from "../lib/methods";
import { selectStreaming, useChat } from "./chatStore";

export default function MacroStrip() {
  const currentId = useChat((s) => s.currentId);
  const streaming = useChat(selectStreaming);
  const runMacro = useChat((s) => s.runMacro);
  const [macros, setMacros] = useState<MacroDef[]>([]);
  const [pending, setPending] = useState<MacroPreview | null>(null);
  const [answers, setAnswers] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    let alive = true;
    setPending(null);
    setErr(null);
    if (!currentId) {
      setMacros([]);
      return;
    }
    void (async () => {
      try {
        const [methods, session] = await Promise.all([
          listMethods(),
          fetch(`/api/sessions/${currentId}`).then((r) => r.json()),
        ]);
        const mid = (session as { method?: string }).method ?? "";
        const m = methods.find((x) => x.id === mid);
        if (alive) setMacros(m?.macros ?? []);
      } catch {
        if (alive) setMacros([]);
      }
    })();
    return () => {
      alive = false;
    };
  }, [currentId]);

  if (!currentId || macros.length === 0) return null;

  const click = async (macro: MacroDef) => {
    if (streaming) return;
    setErr(null);
    try {
      const p = await previewMacro(currentId, macro.id);
      if (p.needs_confirm || p.asks.length > 0) {
        setAnswers(Object.fromEntries(p.asks.map((a) => [a, ""])));
        setPending(p);
        return;
      }
      await runMacro(macro.id);
    } catch (e) {
      // AUDIT F11-4: a preview the server refuses used to leave the strip
      // exactly as it was — indistinguishable from a dead button.
      setErr((e as Error).message);
    }
  };

  const go = async (confirm: "run" | "always") => {
    const p = pending;
    if (!p) return;
    setPending(null);
    const a = answers;
    setAnswers({});
    await runMacro(p.macro_id, confirm, a);
  };

  if (pending) {
    return (
      <div className="flex flex-col gap-2 rounded-lg bg-surface px-3 py-2 mb-2">
        <p className="text-[12px] text-secondary">{pending.preview}</p>
        {pending.asks.map((label) => (
          <input
            key={label}
            value={answers[label] ?? ""}
            onChange={(e) =>
              setAnswers((a) => ({ ...a, [label]: e.target.value }))
            }
            placeholder={label}
            aria-label={label}
            className="rounded-md bg-float px-2 py-1 text-[12.5px] text-primary placeholder:text-muted"
          />
        ))}
        <div className="flex items-center gap-2">
          <button
            onClick={() => void go("run")}
            className="rounded-md bg-amber/15 text-amber px-2.5 py-1 text-[12px] font-semibold"
          >
            Run
          </button>
          {pending.needs_confirm && (
            <button
              onClick={() => void go("always")}
              title="don't ask again for this macro"
              className="rounded-md bg-surface hover:bg-float text-secondary px-2.5 py-1 text-[12px]"
            >
              Always allow
            </button>
          )}
          <button
            onClick={() => {
              setPending(null);
              setAnswers({});
            }}
            className="rounded-md bg-surface hover:bg-float text-muted px-2.5 py-1 text-[12px]"
          >
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-1 pb-2">
      {err && (
        <p role="alert" className="text-red text-[11.5px]">{err}</p>
      )}
      <div className="flex items-center gap-1.5 flex-wrap">
        {macros.map((m) => (
          <button
            key={m.id}
            onClick={() => void click(m)}
            disabled={!!streaming}
            title={m.hint}
            className="rounded-md bg-surface hover:bg-float text-secondary px-2.5 py-1 text-[12px] disabled:opacity-40"
          >
            {m.label}
          </button>
        ))}
      </div>
    </div>
  );
}
