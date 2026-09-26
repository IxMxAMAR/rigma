# Rigma VRAM planner: closed-loop correction vs. llama.cpp native fitting

Research date: 2026-09-26. Primary sources only; no source below is an AI-generated SEO page.
Note: `web_search` was unavailable (no API key); searches were run via DuckDuckGo HTML/Lite and
all substantive claims come from fetched primary sources.

## Verdict

**1. Is closed-loop measured correction the right approach?**
No — the loop already exists upstream and should be consumed, not rebuilt. llama.cpp has a native
automatic VRAM-fitting feature (`--fit`, **on by default**) that solves the same problem *without a
formula*: it reads real free device memory and computes exact per-device model/context/compute
accounting from a no-alloc dummy load, then shrinks context and offload to fit a safety margin. The
same mechanism already prints a full measured memory breakdown to the log Rigma is discarding.

**2. Correct pattern for "predict, then verify"?**
Every authoritative system found uses *exact accounting where possible, plus a fixed additive safety
margin where it is not* — not a learned multiplicative correction. llama.cpp enforces a hard
`--fit-margin` (default 1024 MiB, raised to 2160 MiB in one real multimodal case); the most rigorous
community effort (symbolic regression over 19,517 real measurements) still needed an additive
"+577 MB buffer" to reach 95% confidence. Neither applies a self-correcting delta.

**3. Failure modes of closed-loop correction.**
Not directly documented for this specific case (see "could not verify"). However, the upstream
design deliberately avoids the loop: the fit is recomputed from scratch on every load from current
free memory. A real bug report shows the fit *does* change between cold start and sleep-resume on
the same machine (`ngl` 64 → 59), which is evidence that a persisted learned correction would be
applied against a moving baseline.

**4. Is post-launch VRAM measurement reliable?**
Only device-level free/total is trustworthy; per-process and vendor "used" counters are not.
On Windows/WDDM, NVIDIA's per-process memory APIs return `-1 (N/A)`; on ROCm, `amd-smi` `USED_VRAM`
was measured to move by **1 MB for a 2 GB allocation** while `hipMemGetInfo()` correctly reported
2058 MB. A value read once is a sample, never a peak — the fit's own `compute` term is a *reservation*,
not an observed peak.

## Best solution

**Do not build the closed-loop corrector. Consume llama.cpp's fit output instead.** Concretely:

1. **Ensure the pinned llama.cpp build has `--fit` and leave it enabled.** Do not pass `--fit off`.
   Stop hand-computing `--ctx-size` / `-ngl` for the purpose of fitting; let the engine choose. Set
   `--fit-target` to Rigma's desired per-device headroom (MiB) instead of Rigma's `compute_buffer`
   constant. If Rigma must pin a context ceiling, `--fit-ctx` sets the minimum context the fit may
   reduce to.
2. **Replace the `resolve.py` fit arithmetic with the measured breakdown, and delete the
   "errs toward overcommitting" term.** Parse the already-captured engine log for these exact lines:
   - `common_memory_breakdown_print: | memory breakdown [MiB] | total free self model context compute unaccounted |`
   - `common_params_fit_impl: projected to use <N> MiB of device memory vs. <M> MiB of free device memory`
   - `common_params_fit_impl: will leave <X> >= <Y> MiB of free device memory, no changes needed`
   - `common_params_fit_impl: cannot meet free memory target of <Y> MiB, need to reduce device memory by <Z> MiB`
   - `llama_kv_cache: size = <S> MiB (<cells> cells, <layers> layers, <n>/<m> seqs), K (f16): ... V (f16): ...`
   These give Rigma the *actual* KV size, compute buffer, and per-device totals — free, measured, and
   already per-architecture-correct.
3. **Store the measured breakdown as a record, not as a correction factor.** Persist
   `{model_hash, build, device, driver, n_ctx, ngl, cache_type, predicted, measured, unaccounted}` for
   the bench/throughput system that already exists. Use it for *reporting and anomaly detection*
   (e.g. warn when `unaccounted` is large), and to seed a conservative prior — never to silently
   scale future predictions.
4. **Where Rigma still needs a pre-launch estimate** (before any engine runs), use exact GGUF tensor
   accounting for weights, and for KV use the engine's own printed formula per layer rather than a
   single global `full_attn_layers * kv_heads * head_dim` product — the real log shows *two* caches
   with different sizes in one model.

## Evidence

- [`tools/server/README.md`](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md) —
  `-fit, --fit [on|off]  whether to adjust unset arguments to fit in device memory (default: 'on')`;
  also `--fit-ctx`, `--fit-target`, `--fit-margin`. `/props` exposes model/slot/generation settings
  and `build_info` but **no memory fields**. `/metrics` is Prometheus-compatible and gated behind
  `--metrics`, but the documented metric list contains only throughput/token counters — **no VRAM
  metric**.
- [`common/fit.cpp`](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/fit.cpp) —
  `mparams_copy.no_alloc = true; mparams_copy.load_mode = LLAMA_LOAD_MODE_NONE;` then
  `llama_get_memory_breakdown(ctx)` aggregated per `ggml_backend_dev_memory()`. This is exact
  accounting, not a formula. It also defines `llama-fit-params` (a CLI that prints
  `device model context compute` MiB columns), usable as a dry-run oracle.
- [PR #16653](https://github.com/ggml-org/llama.cpp/pull/16653) (merged; author JohannesGaessler) —
  introduced `llama_params_fit`. Real log: `llama_params_fit: cannot fulfill margin of 1024 MiB on all
  devices, need to use 97337 MiB less in total`, `context size reduced from 65536 to 4096 -> need
  13440 MiB less memory`. Confirms margin-first, context-second, offload-third policy.
- [Issue #26401](https://github.com/ggml-org/llama.cpp/issues/26401) — full breakdown log format and
  the sleep-resume instability: cold start `will leave 2522 >= 2160 MiB ... no changes needed` with
  `n_layer=64`; after resume `cannot meet free memory target of 3296 MiB ...` ending at 59 layers.
- [Issue #18996](https://github.com/ggml-org/llama.cpp/issues/18996) — a gpt-oss model with
  `n_swa=128` produces two caches: non-SWA `KV buffer size = 384.00 MiB (16384 cells, 12 layers)` and
  SWA `KV buffer size = 24.00 MiB (1024 cells, 12 layers)`. Direct evidence that one global
  per-token KV product is structurally wrong for hybrid/SWA models.
- [PR #13194](https://github.com/ggml-org/llama.cpp/pull/13194) (ggerganov) — SWA cache sized as
  `PAD(n_swa*n_seq_max + n_batch)`, i.e. independent of `ctx`. Rigma's formula does not model this.
- [oobabooga: A formula that predicts GGUF VRAM usage](https://oobabooga.github.io/blog/posts/gguf-vram-formula/) —
  19,517 measurements over 60 quants, `gpu_layers` 0→max, `ctx` 512→131072, cache fp16/q8_0/q4_0.
  Explicitly reports the naive hypothesis (`VRAM ≈ gpu_layers/n_layers * size_on_disk + ctx term`)
  "was not good"; final symbolic-regression formula has **median absolute error 365 MiB** and still
  needs **+577 MB** for 95% confidence. Single-author but from the text-generation-webui maintainer,
  with a reproducible methodology and dataset — the strongest empirical source found.
- [ROCm/amdsmi #175](https://github.com/ROCm/amdsmi/issues/175) (AMD-assigned, closed) — measured:
  2 GB allocation moved `amd-smi` `USED_VRAM` by **Δ=1 MB**; `hipMemGetInfo()` showed **Δ=2058 MB**.
- [NVIDIA Developer Forums, NVML/WDDM](https://forums.developer.nvidia.com/t/nvml-problems-for-windows-not-available-in-wddm-driver-model/77557) —
  "all process memory get -1 (N/A) because of 'Used GPU Memory : Not available in WDDM driver
  model'"; per-process memory is not exposed on Windows.
- [PyTorch HIP notes](https://docs.pytorch.org/docs/main/notes/hip.html) — "the unused memory managed
  by the allocator will still show as if used in amd-smi". Vendor "used" counters include caching
  allocator slack, so they cannot be subtracted to infer a model's true footprint.

## Risks / what would break

- **Persisted learned deltas are applied against a moving baseline.** `--fit` recomputes from live
  free memory each load; the resume case above changed the outcome with the *same* model and machine.
  A stored delta would be added to a number that already moved.
- **Cold start is unfixable by learning.** First launch on a new model has no data, so the loop
  provides nothing exactly when the user is most likely to hit an OOM.
- **Cross-machine transfer is invalid.** The margin depends on desktop VRAM footprint (Rigma already
  knows this — its 700 MB drift guard), driver, and `compute` buffer, which varies by backend and
  batch size. A delta learned on one GPU/OS is not evidence for another.
- **Windows per-process measurement is unavailable**, so any Rigma design that attributes VRAM to the
  engine process cannot work on the primary consumer platform; only device free/total deltas work,
  and those are contaminated by other applications.
- **A single post-launch read is a sample, not a peak.** Peak compute-buffer use happens during
  prompt processing / graph reservation, typically before or at warmup; sampling after the server is
  idle can under-read. The `compute` figure in the fit log is a reservation, not an observation.
- **Rigma's existing arithmetic cannot be validated by one delta anyway.** With a median error of
  ~365 MiB in the best published fit, a single measurement cannot distinguish a systematic formula
  error from run-to-run noise.

## What I could NOT verify

- **The exact release/build number that introduced `--fit`.** PR #16653 is the origin; I did not
  resolve its merge commit to a `bXXXX` tag, so whether *Rigma's currently pinned build* has it is
  unverified — check `llama-server --help | findstr fit` on the pinned binary.
- **Whether llama.cpp exposes a live "current VRAM usage" API.** I verified `/props` has no memory
  fields and `/metrics` is throughput-only from the README. I did **not** exhaustively audit the
  server source for an undocumented route.
- **Any authoritative prior-art source** (Kubernetes, query planners, CUDA pool sizing, Ollama/LM
  Studio/vLLM) prescribing additive vs. multiplicative vs. learned correction. Searches returned only
  generic OOMKilled troubleshooting content and unrelated control-theory papers. The "additive
  margin" conclusion is inferred from llama.cpp's and oobabooga's independent choices, not from a
  stated engineering principle.
- **Documented oscillation / overfitting reports** for closed-loop memory planners. Nothing specific
  was found; do not treat the upstream design as proof that oscillation was observed.
- **ROCm-on-Windows behaviour** specifically. The amdsmi issue is Linux (ROCm 7.0.0, MI355X); the
  NVIDIA WDDM finding is Windows. I could not verify AMD Windows VRAM reporting.
- **Whether `--fit` can be trusted to prevent OOM in all cases.** One open issue (#29386) shows a
  server still being SIGKILLed under sustained load with `--fit off` explicitly set, so that case
  does not test `--fit`. I found no clean measurement of `--fit` OOM-prediction accuracy.
