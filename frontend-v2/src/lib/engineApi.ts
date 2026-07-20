// Engine + Hangar API surface, typed against serve.py's actual shapes.
async function j<T>(method: string, path: string, body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method,
    headers: body !== undefined ? { "content-type": "application/json" } : {},
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });
  if (!r.ok) {
    const e = (await r.json().catch(() => ({}))) as { error?: string };
    throw new Error(e.error ?? `server replied ${r.status}`);
  }
  return r.json() as Promise<T>;
}

export interface ServerInfo {
  model?: string;
  quant?: string;
  backend?: string;
  ctx?: number;
  unloaded?: boolean;
  calibrating?: unknown;
  engine_version?: string;
  last_tg?: number | null;
  expected_tg?: number | null;
  verdict?: string;
  openai_base?: string;
  [k: string]: unknown;
}

export interface SwitchOption {
  model: string;
  quant: string;
  ctx: number;
  backend: string;
  reason: string;
}

export interface Pull {
  status?: string;
  done?: number;
  bps?: number | null;
  eta?: number | null;
  error?: string;
}

export interface QuantRow {
  file: string;
  quant: string;
  bytes: number;
  on_disk: boolean;
  pullable: boolean;
  pull?: Pull | null;
}

export interface ModelCard {
  slug: string;
  family: string;
  kind: string;
  custom: boolean;
  capabilities: string[];
  native_ctx: number;
  quants: QuantRow[];
  mmproj?: (QuantRow & { quant?: string }) | null;
  running: boolean;
}

export interface HfHit {
  id: string;
  [k: string]: unknown;
}

export const engineApi = {
  server: () => j<ServerInfo>("GET", "/api/server"),
  switchOptions: () => j<SwitchOption[]>("GET", "/api/server/switch-options"),
  switchTo: (model: string) => j<unknown>("POST", "/api/server/switch", { model }),
  setCtx: (ctx: number) => j<unknown>("POST", "/api/server/ctx", { ctx }),
  load: () => j<unknown>("POST", "/api/server/load", {}),
  unload: () => j<unknown>("POST", "/api/server/unload", {}),
  recalibrate: () => j<unknown>("POST", "/api/server/recalibrate", {}),
  log: async (lines = 120): Promise<string> => {
    const r = await fetch(`/api/server/log?lines=${lines}`);
    return r.ok ? r.text() : "";
  },

  models: () => j<{ models: ModelCard[]; [k: string]: unknown }>("GET", "/api/models"),
  pull: (slug: string, file: string) =>
    j<unknown>("POST", `/api/models/${slug}/pull`, { file }),
  deleteFile: (slug: string, file: string) =>
    j<unknown>("DELETE", `/api/models/${slug}/files/${encodeURIComponent(file)}`),
  hfSearch: (q: string) =>
    j<HfHit[]>("GET", `/api/hf/search?q=${encodeURIComponent(q)}`),
  hfAdd: (repo: string) => j<unknown>("POST", "/api/hf/add", { repo }),
};

export const gb = (n: number) => (n / 2 ** 30).toFixed(1) + " GB";
export const eta = (s: number | null | undefined) => {
  if (s == null) return "";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};
