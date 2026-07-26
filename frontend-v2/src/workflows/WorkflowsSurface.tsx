import { useCallback, useEffect, useState } from "react";

interface WorkflowMethod {
  id: string;
  name: string;
  tagline?: string;
  builtin?: boolean;
  apply?: { system_prompt?: string };
}

export default function WorkflowsSurface() {
  const [methods, setMethods] = useState<WorkflowMethod[]>([]);
  const [err] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/methods");
      const data = (await r.json()) as { methods?: WorkflowMethod[] };
      setMethods(data.methods ?? []);
    } catch {
      setMethods([]);
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
            workflows & methods
          </h3>
          <p className="text-muted text-[13px] mb-4">
            Workflow methods bundle instructions, macros, and rules into reusable structured missions.
          </p>
          {err && <div className="text-red text-[12.5px] mb-3">{err}</div>}
          {methods.length === 0 && (
            <p className="text-muted text-[13px] mb-2">
              No workflows found.
            </p>
          )}
          <ul className="flex flex-col gap-2">
            {methods.map((m) => (
              <li key={m.id} className="rounded-md bg-surface p-3 flex flex-col gap-1">
                <div className="flex items-center gap-2">
                  <span className="text-[14px] font-semibold text-primary">{m.name}</span>
                  {m.builtin && (
                    <span className="font-mono text-[10px] text-muted bg-panel px-1.5 py-0.5 rounded">built-in</span>
                  )}
                  <span className="font-mono text-[11px] text-muted ml-auto">{m.id}</span>
                </div>
                {m.tagline && (
                  <p className="text-[13px] text-secondary">{m.tagline}</p>
                )}
                <div className="font-mono text-[11.5px] text-muted truncate mt-1">
                  {m.apply?.system_prompt?.slice(0, 80) ?? "(no system prompt)"}
                </div>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </main>
  );
}
