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
   *  UI has to say so before it relaunches.
   *
   *  NOTE this is llama.cpp's COMPUTE backend (vulkan/rocm/cuda/cpu). The ENGINE
   *  itself is `engine`/`engineRuntimes` below — two different axes, and
   *  conflating them is what would put "vllm" in this picker. */
  backends?: { name: string; ready: boolean; buildable: boolean }[];
  /** Which ENGINE RUNTIME is actually serving — "llamacpp" or "vllm".
   *
   *  Undefined/null when the record predates the field or the launch did not name
   *  one, and the UI must show NOTHING in that case rather than assume llama.cpp:
   *  a wrong label here is worse than no label. (R3-VLLM-4) */
  engine?: string | null;
  /** WHICH BINARY is serving — `{kind, name, path, source}`, or null.
   *
   *  A third axis again: `backends` is llama.cpp's compute backend, `engine` is the
   *  runtime, and this is the actual executable. It matters because a *registered*
   *  third-party build can now be selected over Rigma's pin when it is the only one
   *  that can load a model's tensor types — so "which engine is running" is no longer
   *  answerable from the version string. null for a record written before a launch
   *  named its binary; show nothing rather than assume the pin. (R3-ENG-9) */
  engine_binary?: {
    kind: "registered" | "pinned";
    name: string;
    path: string;
    source: string;
  } | null;
  /** Which engine runtimes this machine could run, and exactly why not for the
   *  ones it cannot. Separate from `backends` on purpose. Named with the
   *  server's own snake_case, because this layer does NOT rename keys. */
  engine_runtimes?: {
    engine: string; available: boolean; state: string; reason: string;
  }[];
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

/** R3-VLLM-4: what the Sidecar should say about the running engine, or null to
 *  say nothing.
 *
 *  Pure and separate from the component so it can be tested, which matters here
 *  because the two ways to get this wrong are both silent:
 *
 *   1. Assuming llama.cpp when the record does not say. Records written before
 *      the field existed have no engine, and the whole class of bug this fixes is
 *      a UI that names an engine that is not running.
 *   2. Not flagging the contradiction when the running engine is one this machine
 *      reports as UNAVAILABLE. That is a real inconsistency — a vLLM serving on a
 *      host whose verdict says it cannot — and quietly rendering "vllm" as if it
 *      were expected hides the thing worth looking at.
 *
 *  Returns the label plus a `warn` flag rather than a formatted string, so the
 *  caller owns the styling and this stays testable.
 */
export function engineLabel(srv: {
  engine?: string | null;
  engine_runtimes?: { engine: string; available: boolean }[];
}): { name: string; warn: boolean } | null {
  const name = (srv.engine ?? "").trim();
  if (!name) return null;                       // (1) never assume
  const row = (srv.engine_runtimes ?? []).find((r) => r.engine === name);
  return { name, warn: row ? !row.available : false };   // (2) flag it
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
  /** A16: the fit ARITHMETIC failed, so this row carries no verdict at all.
   *
   *  Only a broken fit sets this. The three shapes the Models page can receive
   *  are otherwise indistinguishable on `ok`/`speed` alone:
   *    - a quant the resolver computed and refused: `{ok:false, speed:"no"}`
   *    - a fit that threw while being computed:      `{ok:false, speed:"no", error}`
   *    - no hardware profile at all:                 `{}`
   *  `hangar.list_models` attaches `error: "fit failed"` to every row when
   *  `quant_verdicts` raises (hangar.py:949-950). */
  error?: string;
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
  /** A2b: the recurrent-state term the fit charges for a hybrid's SSM/linear
   *  attention buffers, times `LAUNCH_PARALLEL`. It was missing from this row,
   *  so a row could read "fits" while the fit it described offloaded a layer.
   *  Optional because a server older than A2b does not send it. */
  rs_mb?: number;
  /** A2d-budget: `true` when the model gives evidence of recurrent layers but
   *  no geometry to size them with, so `rs_mb` is an ESTIMATE (or a 0 charged
   *  for lack of evidence, not because there is nothing to allocate). A boolean
   *  BESIDE `rs_mb` on purpose — the term is still charged, and every numeric
   *  consumer of the row is untouched.
   *
   *  Optional, and ABSENT is its own state rather than a synonym for `false`:
   *  a backend older than A2d-budget sends no marker, and a consumer must not
   *  read that silence as "measured" — `budgetHint` labels a charged term whose
   *  provenance is unstated instead of printing a bare figure. */
  rs_unknown?: boolean;
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

/** How a model comes up, as stored on its spec (`LaunchDefaults.model_dump()`).
 *
 *  Every field has a FALSY SENTINEL rather than null — `""` or `0` — because
 *  "unset" means NO OPINION, and `vision` is the one tri-state (`null` = keep
 *  whatever the last launch used). The POST route treats an explicit `null` as
 *  "clear this field", which is why a caller that means to remove a default must
 *  send null and NOT the empty string the sentinel would suggest. */
export interface LaunchDefaults {
  quant?: string | null;
  ctx?: number | null;
  kv?: string | null;
  vision?: boolean | null;
  spec_type?: string | null;
  spec_n_max?: number | null;
  backend?: string | null;
  /** C10: `-b`/`-ub`/`-ngl`. 0 on either batch field and -1 on `ngl` are the
   *  "no opinion" sentinels, so a caller that means to CLEAR one sends null and
   *  not the sentinel. `-ngl 0` is a real request (every layer on the CPU). */
  batch?: number | null;
  ubatch?: number | null;
  ngl?: number | null;
}

/** `GET /api/models/{slug}/defaults` — the D2 route. `first_load` is the
 *  server's own honest signal: nothing is pinned AND no turn has ever finished
 *  for this model. `custom` is the other half — a registry model's launch
 *  settings are hand-authored and the POST refuses to overwrite them, so the
 *  dialog must be able to explain that refusal rather than swallow it. */
export interface ModelDefaults {
  slug: string;
  launch: LaunchDefaults | null;
  custom: boolean;
  last_used: number | null;
  first_load: boolean;
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
  /** D2: the stored launch defaults, straight from `hangar.list_models`. null
   *  when the spec pins nothing, which is the same fact `first_load` reports. */
  launch?: LaunchDefaults | null;
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

/** A decision the engine announced once at load and never mentioned again.
 *
 *  The point of these is that the flag being SET and the flag being IN EFFECT
 *  are different facts: `--cache-reuse 256` is passed on every launch and the
 *  engine refuses it on a hybrid model, silently, forever. Nothing else in the
 *  app can see that. */
export interface EngineFinding {
  id: string;
  severity: "info" | "warn";
  message: string;
  count: number;
  /** Observed in this machine's own logs, as opposed to matched best-effort
   *  from upstream reports. */
  confirmed_here: boolean;
  example: string;
}

/** The lifetime odometer, read from ~/.rigma/stats.json by serve.server_stats.
 *  `by_model` maps a model id to the tokens IT generated. The IMP-7 enrichment
 *  adds a pre-sorted `models[]` with a per-model turn count and a last-used
 *  stamp; all three are optional because an older server does not send them. */
export interface UsageStats {
  total_tokens: number;
  total_turns: number;
  by_model: Record<string, number>;
  by_model_turns?: Record<string, number>;
  last_used?: Record<string, number>;
  models?: { model?: string; tokens?: number; turns?: number;
             last_used?: number | null }[];
}

/** The body `POST /api/server/switch` and `/api/server/ctx` answer with.
 *
 *  It is the new server state plus, when the switch had something to say, a
 *  `notice` — today the KV-cache restore being refused (server_ops.py:856-874).
 *  Both routes returned it all along; `j<unknown>` erased it and every caller
 *  discarded the value, so the sentence never reached the screen (A8c). */
export interface SwitchResult {
  notice?: string;
  [k: string]: unknown;
}

/** The server's `notice` from a switch/relaunch body, trimmed; `""` when there
 *  is none or the shape is wrong. Pure, so the decision is testable without a
 *  fetch. A step-down notice is NOT a failure, so callers render it neutrally
 *  rather than in the error slot. */
export function switchNotice(r: unknown): string {
  if (!r || typeof r !== "object") return "";
  const n = (r as { notice?: unknown }).notice;
  return typeof n === "string" ? n.trim() : "";
}

export const engineApi = {
  server: () => j<ServerInfo>("GET", "/api/server"),
  switchOptions: () => j<SwitchOption[]>("GET", "/api/server/switch-options"),
  switchTo: (model: string, quant?: string) =>
    j<SwitchResult>("POST", "/api/server/switch", quant ? { model, quant } : { model }),
  /** Relaunch the RUNNING model with different engine settings. Every one of
   *  these stops the engine and starts it again — see EngineCard's warning. */
  relaunchWith: (o: { ctx: number; kv?: string; vision?: boolean;
                      backend?: string }) =>
    j<SwitchResult>("POST", "/api/server/ctx", {
      ctx: o.ctx,
      ...(o.kv ? { kv: o.kv } : {}),
      ...(o.vision === undefined ? {} : { vision: o.vision }),
      ...(o.backend ? { backend: o.backend } : {}),
    }),
  relaunch: (ctx: number, kv?: string) =>
    j<SwitchResult>("POST", "/api/server/ctx", kv ? { ctx, kv } : { ctx }),
  load: () => j<unknown>("POST", "/api/server/load", {}),
  unload: () => j<unknown>("POST", "/api/server/unload", {}),
  recalibrate: () => j<unknown>("POST", "/api/server/recalibrate", {}),
  /** Read from the WHOLE log, not the tail: these land in the first few hundred
   *  lines of a launch and are long gone by the time anyone looks. */
  findings: () => j<{ findings: EngineFinding[] }>("GET", "/api/server/findings"),
  /** Lifetime tokens/turns and the per-model split (IMP-7). */
  stats: () => j<UsageStats>("GET", "/api/server/stats"),
  log: async (lines = 120): Promise<string> => {
    // R3-UI-2: this resolved to "" on a non-ok response, and "" is exactly what
    // an EMPTY log looks like. The caller then rendered "(empty)" with no
    // explanation, contradicting its own comment ("a read failure leaves the
    // previous text rather than blanking the panel"). A read that failed is not
    // a log that is empty — and this is the panel where a failed model load
    // explains itself, so it is the worst place to answer "nothing to report"
    // to a question that was never asked.
    //
    // NOT through `j`: the route answers `Response(text, media_type="text/plain")`
    // (serve.py:3519), so `r.json()` would throw on every successful read and
    // break the panel this is meant to fix. Reject explicitly instead.
    const r = await fetch(`/api/server/log?lines=${lines}`);
    if (!r.ok) throw new Error(`server replied ${r.status}`);
    return r.text();
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
  /** Pin how a model comes up. Send only the fields you mean to set; null
   *  clears one. A plain load then lands here instead of on whatever the
   *  resolver picked — which on this hardware is the difference between
   *  38 and 53 tok/s, not a matter of taste. */
  setDefaults: (slug: string, d: LaunchDefaults) =>
    j<{ slug: string; launch: LaunchDefaults | null }>(
      "POST", `/api/models/${slug}/defaults`, d),
  /** D2: what is pinned for this model, whether it is custom, and whether this
   *  is a first load. The GET did not exist (the POST answered 405), so a
   *  dialog could WRITE a default but never show the one already stored. */
  modelDefaults: (slug: string) =>
    j<ModelDefaults>("GET", `/api/models/${slug}/defaults`),
  /** D4b: re-read the gguf and correct the stored geometry/capabilities. Only
   *  custom models; a registry spec is hand-authored and the route answers 409
   *  with its own sentence. Costs a ranged header read over the network, which
   *  is why it is explicit rather than automatic. */
  reprobeModel: (slug: string) =>
    j<ProbedFacts & { slug: string }>(
      "POST", `/api/models/${slug}/reprobe`),
  /** D4b: rename a custom model, carrying its template and calibration rows.
   *  409 with the server's sentence when it is not custom, the name is empty,
   *  the name is taken, or the model is running. */
  renameModel: (slug: string, newSlug: string) =>
    j<{ slug: string }>(
      "POST", `/api/models/${slug}/rename`, { slug: newSlug }),
};

export const gb = (n: number) => (n / 2 ** 30).toFixed(1) + " GB";
export const eta = (s: number | null | undefined) => {
  if (s == null) return "";
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${s % 60}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
};
