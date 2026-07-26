import { useCallback, useEffect, useState } from "react";

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

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/skills");
      setSkills((await r.json()) as Skill[]);
    } catch {
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
          {skills.length === 0 && (
            <p className="text-muted text-[13px] mb-2">
              No global skills yet. Create one below!
            </p>
          )}
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
                  className="opacity-0 group-hover:opacity-100 text-muted hover:text-red px-1"
                  aria-label={`delete skill ${s.name}`}
                  onClick={async (e) => {
                    e.stopPropagation();
                    await fetch(`/api/skills/${s.name}`, { method: "DELETE" });
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
