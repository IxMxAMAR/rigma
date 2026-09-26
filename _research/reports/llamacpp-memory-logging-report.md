# llama.cpp memory logging + auto-fit: verified findings

Sources: raw source at pinned tags (`raw.githubusercontent.com`), GitHub REST API, official `tools/server/README.md`, real startup logs in issue bodies, DuckDuckGo. `web_search` was unavailable (no DEEPSEEK_API_KEY); DDG was used via `html.duckduckgo.com/html/?q=` (`lite.` and POST variants serve a CAPTCHA from this host). Latest release: **b11195 (2026-09-26)**.

## 1. Exact log lines

**KV summary - CURRENT** (`src/llama-kv-cache.cpp`, inside `llama_kv_cache::llama_kv_cache(...)`, so `__func__` = `llama_kv_cache`):
```
LLAMA_LOG_INFO("%s: size = %7.2f MiB (%6u cells, %3d layers, %2u/%u seqs), K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", __func__, ...)
```
Real rendered (issue #27872):
`llama_kv_cache: size =  354.75 MiB (  8448 cells,  43 layers,  1/1 seqs), K (f16):  354.75 MiB, V (f16):    0.00 MiB`

**KV per-buffer - CURRENT, unchanged since b3600 (2024-08-17):**
`LLAMA_LOG_INFO("%s: %10s KV buffer size = %8.2f MiB\n", ...)` -> real (issue #28115): `llama_kv_cache:        CPU KV buffer size = 17408.00 MiB`

**Compute buffer - CURRENT** (`src/llama-context.cpp`, inside `llama_context::sched_reserve()`):
`LLAMA_LOG_INFO("%s: %10s compute buffer size = %8.2f MiB\n", ...)` -> real (issue #28378):
```
sched_reserve:      CUDA0 compute buffer size =   126.77 MiB
sched_reserve:  CUDA_Host compute buffer size =    78.77 MiB
```
**Model buffer:** `%s: %12s model buffer size = %8.2f MiB` -> `load_tensors:        CUDA0 model buffer size =  6378.40 MiB`
**Output buffer:** `%s: %10s  output buffer size = %8.2f MiB` (DOUBLE space) -> `llama_context:  CUDA_Host  output buffer size =     0.49 MiB`

**Historical KV lines.** 2024 era, prefix `llama_new_context_with_model:` - real log (HF TheBloke/phi-2-GGUF discussions/13): `llama_new_context_with_model: KV self size = 160.00 MiB, K (f16): 80.00 MiB, V (f16): 80.00 MiB`, alongside `CPU input buffer size` / `CPU compute buffer size`. Last present in b5400 (2025-05-15): `LLAMA_LOG_INFO("%s: KV self size  = %7.2f MiB, K (%s): %7.2f MiB, V (%s): %7.2f MiB\n", ...)`, in `llama_kv_cache_unified` and `llama_kv_cache_recurrent`. (The two spaces come from the source format string; web pages collapse them.)

## 2. Stability: FRAGILE, already broken

| Tag (date) | Change |
|---|---|
| 2024 era | prefix `llama_new_context_with_model:` |
| <= b5400 (2025-05-15) | `KV self size  = ...`, prefix `llama_kv_cache_unified:` |
| b5600 (2025-06-06) | renamed to `size = ... (%6u cells, %3d layers, %2u seqs)` |
| b5950 (2025-07-21) | gained total: `%2u/%2u seqs` |
| b6100 (~2025-08) | `%2u/%2u` -> `%2u/%u` (unchanged to master) |
| b7600 (2026-01-01) -> b8200 (2026-03-09) | compute-buffer prefix `llama_context:` -> `sched_reserve:` |

Four prefix/format regimes for the KV line and one for the compute line in under two years.

Second hazard, verified in #27872: one startup prints **multiple** `llama_kv_cache: size = ...` lines (ISWA/SWA, DSV4 raw/CSA/HCA/lightning-indexer), including a decoy `llama_kv_cache: size =    0.00 MiB (524288 cells,   0 layers,  1/1 seqs), K (f16):    0.00 MiB, V (f16):    0.00 MiB`. "Find the KV line" is ambiguous; sum the per-buffer lines instead.

## 3. Better alternative: `--log-jsonl`

Official `tools/server/README.md`: `--log-jsonl, --no-log-jsonl` - "Log as JSONL (one JSON object per line) to stdout, this also disables colored logging (default: disabled)" (env `LLAMA_ARG_LOG_JSONL`). Under it, `common_memory_breakdown_print()` (`common/fit.cpp`, called from `tools/server/server.cpp`) emits:
```
{"type":"fit_memory_breakdown","data":{"unit":"MiB","rows":[...]}}
```
Row kinds: `device` (name, description, total, free, self, model, context, unaccounted); `host` and `buffer_type` (name, self, model, context, compute). All integers in MiB. **This is the recommendation.** Caveat: emitted only after a context is created successfully, so it is absent on the OOM path. `--fit-print`/`-fitp` prints a non-JSON table instead.

HTTP endpoints (source + official README agree):
- `/props` GET: **no memory fields**. Verified keys: default_generation_settings, total_slots, model_alias, model_ftype, model_path, modalities, media_marker, endpoint_slots/props/metrics, ui, ui_settings, chat_template(+_caps, _tool_use), bos_token, eos_token, build_info, is_sleeping, cors_proxy_enabled. Read-only unless started with `--props`. Use `build_info` to detect the version.
- `/metrics`: "only accessible if `--metrics` is set", else errors "This server does not support metrics endpoint. Start it with `--metrics`". Documented metrics are throughput/latency only: prompt/token counters, `*_seconds` rates, `requests_processing`, `requests_deferred`, `n_tokens_max`, `n_decode_total`, `n_busy_slots_per_decode`, and four `spec_decode_*` series. **No memory metrics at all.**
- `/health` -> `{"status":"ok"}`. `/slots` on by default; no memory.

## 4. Native auto-fit: YES, default ON

Official README: `-fit, --fit [on|off]` (default **on**, env `LLAMA_ARG_FIT`), `-fitt, --fit-target MiB0,MiB1,...` (default 1024 per device), `-fitc, --fit-ctx N` (default 4096), `-fitp, --fit-print [on|off]`. Introduced between b7450 (2025-12-14) and b7550 (2025-12-27); already default-on at b7600. Behavior (`common/fit.cpp`): reduces n_ctx toward `--fit-ctx`, moves MoE tensors to CPU, assigns per-device layer counts using layer-fraction overflow (UP/GATE/ATTN) - silent partial offload, not refusal.

## 5. Not-fitting behavior: WARN then hard OOM - except on Windows

Verified verbatim from #27872:
```
common_params_fit_impl: projected to use 24840 MiB of device memory vs. 22341 MiB of free device memory
common_params_fit_impl: cannot meet free memory target of 1024 MiB, need to reduce device memory by 3523 MiB
common_fit_params: failed to fit params to free device memory: model_params::tensor_buft_overrides already set by user, abort
ggml_backend_cuda_buffer_type_alloc_buffer: allocating 14656.12 MiB on device 0: cudaMalloc failed: out of memory
ggml_gallocr_reserve_n_impl: failed to allocate CUDA0 buffer of size 15368060928
graph_reserve: failed to allocate compute buffers
llama_init_from_model: failed to initialize the context: failed to allocate compute pp buffers
```
`common_fit_params` catches `common_params_fit_exception` -> `LOG_WRN`, returns FAILURE, and the server keeps loading unfitted, then dies at compute-buffer allocation. `--fit off` is the workaround.

**Windows/WDDM - the important exception (verified).** On Windows the NVIDIA driver's Sysmem Fallback Policy lets a CUDA allocation exceeding dedicated VRAM succeed out of system RAM, shown as "Shared GPU memory" in Task Manager. The allocation does not fail, so llama.cpp prints no error and the model just runs very slowly over PCIe. Introduced in NVIDIA driver branch **536.40**; the switch is NVIDIA Control Panel > Manage 3D Settings > **CUDA - Sysmem Fallback Policy** ("Prefer No Sysmem Fallback" disables it). An independent vendor doc (Cognex) says driver versions above **572.83** require disabling it explicitly. Real llama.cpp reports: issue #9964 and discussion #9960 (both 2024-10-20) - "any model i'm trying to load gets pushed to shared GPU memory (per Task Manager), instead of dedicated one", with magnitude-slower inference on Windows only.

llama.cpp has **no detection, warning, or env var** for this: grepping `ggml-cuda.cu` for CUDA env vars yields only `GGML_CUDA_DEVICES`, `_P2P`, `_GRAPH_OPT`, `_DISABLE_FUSION`, `_CUBLAS_COMPUTE_TYPE`, `_ALLREDUCE`, `_NO_PINNED`, `_REGISTER_HOST`, `_ENABLE_UNIFIED_MEMORY` - nothing for sysmem/shared memory. Free VRAM comes from `cudaMemGetInfo` (ggml-cuda.cu:4914, 5057), which reports dedicated VRAM. Consequence for Rigma: **on Windows, absence of an OOM error is not evidence the model fits**, and Rigma's own prediction is more valuable there, not less.

## What I could NOT verify

1. Exact commits/PRs for the `--fit` introduction (bracketed b7450..b7550), the `sched_reserve` move (bracketed b7600..b8200; b7800/b8000 raw fetches failed), and the three KV-line changes (tag-bracketed only). `pulls/23232` was unfetchable.
2. **Whether `--fit` accounts for sysmem fallback on Windows.** Since `--fit` sizes against `cudaMemGetInfo` (dedicated VRAM) while the driver may silently satisfy an over-budget allocation from system RAM, `--fit`'s own probe allocation could succeed when it should not. This is inference from the two verified facts above, NOT something I read in the source or observed.
3. I ran no `llama-server`. Rendered lines are derived from the exact format string plus argument types, or copied from an issue body, as labelled.
4. Which of these flags/endpoints exist in the build Rigma actually ships.
5. Whether `common_memory_breakdown_print`'s table needs `-v`; the JSON emission is gated only on `--log-jsonl`, but I did not confirm verbosity conditions at the `server.cpp` call site.
6. The NVIDIA driver-version claims come from a vendor doc and a blog, not NVIDIA's own documentation, which I did not locate.
