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
  kv_cache?: string;
  native_ctx?: number;
  /** this launch deliberately left the vision projector off */
  no_vision?: boolean;
  /** the model HAS a projector, so the vision toggle means something */
  has_mmproj?: boolean;
  calibrating?: unknown;
  engine_version?: string;
  last_tg?: number | null;
  expected_tg?: number | null;
  verdict?: string;
  openai_base?: string;
  /** Compute backends THIS GPU can run. `ready` means the engine build is
   *  already unpacked; choosing one that isn't costs a ~1.2GB download, so the
   *  UI has to say so before it relaunches. */
  backends?: { name: string; ready: boolean; buildable: boolean }[];
  /** Where the card's memory has gone. `desktop_mb` is what OTHER processes
   *  hold; on Windows that is the difference between a resident model and one
   *  the driver silently pages over PCIe, and it is the only number here the
   *  user can act on. null when it could not be measured. */
  vram?: {
    total_mb: number; desktop_mb: number | null; usable_mb: number;
    assumed_mb: number; pressured: boolean;
  } | null;
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

/** Whether this quant runs on THIS machine, from resolve.quant_verdicts.
 *  `speed` is how much of it sits on the GPU: weights spilled to RAM run on
 *  the CPU every token, so a bigger quant that only fits via heavy offload is
 *  slower, not better. Empty object when the hardware probe failed. */
export interface Fit {
  ok?: boolean;
  ctx?: number;
  n_cpu_moe?: number;
  /** GPU layers the resolver planned (dense); 99 = all */
  ngl?: number;
  /** KV cache type it settled on, e.g. "f16" | "q8_0" */
  kv?: string;
  /** % of weights left in system RAM — 0 means fully GPU-resident */
  offload_pct?: number;
  speed?: "gpu" | "light" | "offload" | "no";
}

/** Reference quality for the quant FORMAT — see src/rigma/quant_quality.py.
 *  `ppl_pct` is the typical % perplexity increase over BF16 for this format,
 *  NOT a measurement of this model. null when the repo names its files its own
 *  way (I-Compact etc.) and no published figure exists. */
export interface Quality {
  bpw: number;
  ppl_pct: number;
  tier: string;
  note: string;
  basis: string;
}

/** Everything given up vs the true reference — BF16 weights AND an f16 cache.
 *  The two sources compose as ratios, so this is not simply weights + kv. */
export interface TotalLoss {
  weights_pct: number;
  kv_pct: number;
  total_pct: number;
  tier: string;
  basis: string;
}

/** Where the VRAM goes, in MB — the arithmetic behind a single "8K". */
export interface Budget {
  file_mb: number;
  mmproj_mb: number;
  kv_mb: number;
  budget_mb: number;
  /** positive = this much OVER the budget, so weights spill to RAM */
  over_mb: number;
  ctx: number;
  kv_type: string;
}

/** The explorer's knobs. These only change what the fit math ASSUMES — nothing
 *  is launched or saved, so the page can be driven freely. */
export interface FitConfig {
  /** "" = the model's own policy ladder; "q4_0"; or "q8_0,q4_0" for split K/V */
  kv: string;
  /** false drops the vision projector, which is otherwise always resident */
  vision: boolean;
  /** "speed" never trades a GPU layer for context; "context" spends up to 15% */
  grow: "speed" | "context";
}

export const DEFAULT_FIT: FitConfig = { kv: "", vision: true, grow: "speed" };

export function fitQuery(c?: FitConfig): string {
  if (!c) return "";
  const p = new URLSearchParams();
  if (c.kv) p.set("kv", c.kv);
  if (!c.vision) p.set("vision", "0");
  if (c.grow !== "speed") p.set("grow", c.grow);
  const s = p.toString();
  return s ? `?${s}` : "";
}

export interface QuantRow {
  file: string;
  quant: string;
  bytes: number;
  on_disk: boolean;
  pullable: boolean;
  fit?: Fit & { budget?: Budget };
  quality?: Quality | null;
  total?: TotalLoss | null;
  pull?: Pull | null;
  /** Does THIS file carry the multi-token-prediction draft head? Per FILE:
   *  whether MTP survives is the quantiser's decision per artefact, so the
   *  model-level capability cannot answer it. null = not read yet. */
  mtp?: boolean | null;
  /** Bits per weight this file actually spends — bytes x 8 / parameters, both
   *  counted. The honest cell for a repo whose own naming (I-Compact) has no
   *  published quality figure. NOT a quality percentage. */
  bpw?: number | null;
  /** Set when the label's nominal bits/weight is not what the file spends —
   *  a custom mix that kept its base format's name. */
  label_drift?: { measured_bpw: number; label_bpw: number; drift_pct: number } | null;
  /** this exact quant is the one the engine has loaded right now */
  running?: boolean;
  /** The quant format alone — "Q3_K_M" — with the repo's variant suffixes
   *  split off into `variants`. Big repos ship a GRID, not a list: 0bserverx's
   *  Qwen3.8-27B is 103 files that are 26 quants x multilingual x mtp x vision.
   *  The card groups rows on this and filters them on `variants`. */
  base?: string;
  /** e.g. ["multilingual", "mtp"]. Empty for the plain build, and empty for
   *  every row in the great majority of repos, which ship no variants. */
  variants?: string[];
}

/** Facts probed from the gguf itself, shared by the library card and the
 *  pre-download repo view so the two cannot disagree. */
export interface ProbedFacts {
  params?: number;
  n_layers?: number;
  full_attn_layers?: number;
  mtp_layers?: number;
  /** false = the gguf shipped no tokenizer.chat_template, so an empty
   *  capability list is missing evidence rather than a finding. */
  has_template?: boolean;
  /** a repaired chat template is installed at ~/.rigma/templates/<slug>.jinja
   *  and passed to llama-server, so the model is NOT on a fallback format */
  template_override?: boolean;
}

export interface ModelCard extends ProbedFacts {
  slug: string;
  family: string;
  kind: string;
  custom: boolean;
  capabilities: string[];
  native_ctx: number;
  quants: QuantRow[];
  mmproj?: (QuantRow & { quant?: string }) | null;
  /** best quality that still runs at GPU speed here — resolve.recommended_quant */
  recommended?: string | null;
  /** the HF repo this came from. The slug is the gguf's own general.name and
   *  is often nothing like the repo, so the card shows both. */
  source?: string;
  running: boolean;
}

// Mirrors hf_browse.search() EXACTLY — see tests/test_hf_browse.py, which
// pins this shape. The repo id arrives as `repo`, not `id`: the endpoint
// renames Hugging Face's own `id` field on the way out. No index signature
// here on purpose — with one, `h.id` type-checked fine and rendered as
// undefined, which is how HF adding shipped broken.
export interface HfHit {
  repo: string;
  downloads: number;
  likes: number;
  updated: string;
}

/** hf_browse.inspect_repo — the pre-download view. Quant rows are shaped like
 *  QuantRow on purpose, so one component renders both this and /api/models. */
export interface HfRepoDetail extends ProbedFacts {
  repo: string;
  name: string;
  family: string;
  kind: string;
  native_ctx: number;
  capabilities: string[];
  /** already in the library — the add button becomes a no-op */
  already: boolean;
  mmproj?: { file: string; bytes: number } | null;
  /** multi-part .gguf files skipped; they aren't supported yet */
  split_skipped: number;
  ggufs: QuantRow[];
  recommended?: string | null;
}

export const engineApi = {
  server: () => j<ServerInfo>("GET", "/api/server"),
  switchOptions: () => j<SwitchOption[]>("GET", "/api/server/switch-options"),
  switchTo: (model: string, quant?: string) =>
    j<unknown>("POST", "/api/server/switch", quant ? { model, quant } : { model }),
  /** Relaunch the RUNNING model with different engine settings. Every one of
   *  these stops the engine and starts it again — see EngineCard's warning. */
  relaunchWith: (o: { ctx: number; kv?: string; vision?: boolean;
                      backend?: string }) =>
    j<unknown>("POST", "/api/server/ctx", {
      ctx: o.ctx,
      ...(o.kv ? { kv: o.kv } : {}),
      ...(o.vision === undefined ? {} : { vision: o.vision }),
      ...(o.backend ? { backend: o.backend } : {}),
    }),
  relaunch: (ctx: number, kv?: string) =>
    j<unknown>("POST", "/api/server/ctx", kv ? { ctx, kv } : { ctx }),
  load: () => j<unknown>("POST", "/api/server/load", {}),
  unload: () => j<unknown>("POST", "/api/server/unload", {}),
  recalibrate: () => j<unknown>("POST", "/api/server/recalibrate", {}),
  log: async (lines = 120): Promise<string> => {
    const r = await fetch(`/api/server/log?lines=${lines}`);
    return r.ok ? r.text() : "";
  },

  models: (cfg?: FitConfig) =>
    j<{ models: ModelCard[]; [k: string]: unknown }>(
      "GET", `/api/models${fitQuery(cfg)}`),
  pull: (slug: string, file: string) =>
    j<unknown>("POST", `/api/models/${slug}/pull`, { file }),
  deleteFile: (slug: string, file: string) =>
    j<unknown>("DELETE", `/api/models/${slug}/files/${encodeURIComponent(file)}`),
  hfSearch: (q: string) =>
    j<HfHit[]>("GET", `/api/hf/search?q=${encodeURIComponent(q)}`),
  /** The full quant table for a repo BEFORE downloading anything — the header
   *  is read over a ranged request, so it costs megabytes not gigabytes. */
  hfRepo: (id: string, cfg?: FitConfig) => {
    const q = fitQuery(cfg);
    return j<HfRepoDetail>(
      "GET", `/api/hf/repo?id=${encodeURIComponent(id)}${q.replace("?", "&")}`);
  },
  hfAdd: (repo: string) => j<unknown>("POST", "/api/hf/add", { repo }),
  deleteModel: (slug: string) => j<unknown>("DELETE", `/api/models/${slug}`),
};

export const gb = (n: number) => (n / 2 ** 30).toFixed(1) + " GB";
export const eta = (s: number | null | undefined) => {
  if (s == null) return "";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};
