import { useCallback, useEffect, useRef, useState } from "react";

import EmptyState from "../EmptyState";
import LoadError from "../LoadError";
import InlineError from "../InlineError";
import { readList, responseError } from "../lib/listFetch";

interface Skill {
  id: string;
  name: string;
  filename: string;
  content: string;
}

export default function SkillsSurface() {
  const [skills, setSkills] = useState<Skill[]>([]);
  const [draft, setDraft] = useState<{ name: string; content: string }>({
    name: "", content: "",
  });
  const [editing, setEditing] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [rowErr, setRowErr] = useState<string | null>(null);
  // IMP-10: the empty state's action focuses the create form below.
  const nameRef = useRef<HTMLInputElement>(null);

  const refresh = useCallback(async () => {
    // AUDIT F11-3: a 500's {detail} body used to be cast to Skill[] and reach
    // .map during render, blanking the whole app. Check the status AND the
    // shape; a failed load must never render as "no skills yet".
    try {
      const res = await readList<Skill>(await fetch("/api/skills"));
      if (!res.ok) {
        setLoadErr(res.error);
        setSkills([]);
        return;
      }
      setLoadErr(null);
      setSkills(res.rows);
    } catch (e) {
      setLoadErr((e as Error).message);
      setSkills([]);
    }
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);

  return (
    <main className="flex-1 overflow-y-auto p-6">
      <div className="max-w-[720px] mx-auto flex flex-col gap-4">
        <section className="rounded-lg bg-panel p-4">
          <h3 className="font-mono text-[11px] text-muted uppercase tracking-[0.08em] mb-3">
            global skills (`SKILL.md`)
          </h3>
          <p className="text-muted text-[13px] mb-2">
            Skills bundle instructions and domain knowledge. Invoke any skill in chat anytime using <code className="bg-surface px-1.5 py-0.5 rounded font-mono">/skillName</code> or <code className="bg-surface px-1.5 py-0.5 rounded font-mono">/skill:name</code>.
          </p>
          {loadErr && (
            <LoadError message={`could not load skills: ${loadErr}`}
                       onRetry={() => void refresh()} />
          )}
          {!loadErr && skills.length === 0 && (
            <EmptyState
              title="no global skills yet"
              body="A skill is a SKILL.md the model can pull into any chat with /skillName. Write one below."
              actionLabel="create a skill"
              onAction={() => nameRef.current?.focus()} />
          )}
          {rowErr && <InlineError message={`could not delete: ${rowErr}`} />}
          <ul className="flex flex-col gap-1 mb-3">
            {skills.map((s) => (
              <li key={s.id}
                  className={`group flex items-center gap-2 rounded-md hover:bg-surface px-3 py-1.5 cursor-pointer ${editing === s.id ? "bg-surface" : ""}`}
                  onClick={() => {
                    setEditing(s.id);
                    setDraft({ name: s.name, content: s.content });
                    setErr(null);
                  }}>
                <span className="flex-1 text-[13.5px] font-mono">
                  /{s.name}
                </span>
                <span className="font-mono text-[11px] text-muted truncate max-w-[280px]">
                  {s.content.slice(0, 50)}...
                </span>
                <button
                  className="text-muted hover:text-red px-1"
                  aria-label={`delete skill ${s.name}`}
                  onClick={async (e) => {
                    e.stopPropagation();
                    setRowErr(null);
                    // AUDIT F11-4: this ignored r.ok and refreshed, so a
                    // refused delete looked exactly like one that worked.
                    const r = await fetch(
                      `/api/skills/${encodeURIComponent(s.name)}`,
                      { method: "DELETE" });
                    if (!r.ok) {
                      setRowErr(await responseError(r));
                      return;
                    }
                    if (editing === s.id) setEditing(null);
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
              if (!draft.name.trim() || !draft.content.trim()) return;
              setErr(null);
              const r = await fetch("/api/skills", {
                method: "POST",
                headers: { "content-type": "application/json" },
                body: JSON.stringify(draft),
              });
              if (!r.ok) {
                const b = (await r.json().catch(() => ({}))) as { error?: string };
                setErr(b.error ?? `server replied ${r.status}`);
                return;
              }
              setDraft({ name: "", content: "" });
              setEditing(null);
              void refresh();
            }}
          >
            <input
              ref={nameRef}
              value={draft.name}
              onChange={(e) => setDraft((d) => ({ ...d, name: e.target.value }))}
              placeholder="skill name (e.g. wildcard)"
              aria-label="Skill name"
              className="rounded-md bg-surface px-3 py-1.5 text-[13px] outline-none placeholder:text-muted font-mono"
            />
            <textarea
              value={draft.content}
              onChange={(e) =>
                setDraft((d) => ({ ...d, content: e.target.value }))}
              placeholder="# SKILL: Wildcard Generator&#10;&#10;Detailed instructions and guidelines..."
              aria-label="Skill content"
              rows={8}
              className="rounded-md bg-surface px-3 py-1.5 text-[13px] font-mono outline-none resize-y placeholder:text-muted"
            />
            {err && <div className="text-red text-[12.5px]">{err}</div>}
            <div className="flex gap-2">
              <button className="rounded-md bg-amber/15 text-amber px-3 py-1 text-[13px] font-semibold">
                {editing ? "save skill" : "create skill"}
              </button>
              {editing && (
                <button
                  type="button"
                  onClick={() => {
                    setEditing(null);
                    setDraft({ name: "", content: "" });
                    setErr(null);
                  }}
                  className="rounded-md bg-surface hover:bg-float px-3 py-1 text-[13px]"
                >
                  new instead
                </button>
              )}
            </div>
          </form>
        </section>
      </div>
    </main>
  );
}
