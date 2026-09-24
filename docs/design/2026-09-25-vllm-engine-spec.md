# vLLM as a second engine runtime — specification

**Status: the registry, the availability verdict, the argv builder and the CLI
report are IMPLEMENTED and tested. The launch path, the fit arithmetic and the
KV-cache story are SPECIFIED ONLY. Nothing in this document has been executed
against a real vLLM on any machine.**

| | |
|---|---|
| Written | 2026-09-25 |
| vLLM revision read | `vllm-project/vllm` `main`, docs banner "latest developer preview"; the nightly wheel URL in the CUDA install page is `vllm-0.23.1rc1.dev901+g2f3f441f8`, so `main` is at ~0.23.x. Stable docs live at `docs.vllm.ai/en/stable/`. |
| Implemented in | `src/rigma/engines.py`, `tests/test_engines.py`, `rigma engine-runtimes`, `server_ops.available_engine_runtimes()` |
| NOT implemented | everything in §7 phase 1 and later |
| Executed against real vLLM | **no — not once, on any platform** |

---

## 0. The verdict for THIS machine, first

> **vLLM cannot run on this machine today.** The owner's box is Windows 11 with
> an AMD Radeon RX 9070 XT and an Intel iGPU, Python 3.12 in a venv, no vLLM,
> no torch, and no verified WSL2-with-ROCm. vLLM's stated requirement is
> `OS: Linux`, and its own documentation says plainly that it "does not support
> Windows natively". The GPU is *not* the blocker — the RX 9070 XT is gfx1201,
> which is explicitly on vLLM's supported ROCm list, and the venv's Python 3.12
> is exactly the version the ROCm prebuilt wheels are published for.

It becomes runnable under any one of:

1. **WSL2** with a Linux distribution that can see the GPU, a WSL ROCm driver,
   ROCm 7.0 or 7.2.1, glibc >= 2.35, and a Python **3.12** environment — then
   `uv pip install vllm --extra-index-url https://wheels.vllm.ai/rocm/`.
   *(UNVERIFIED: whether a WSL2 ROCm stack can actually expose gfx1201 to
   vLLM is not something this document measured, and AMD's WSL support for
   RDNA4 is the specific thing to test first.)*
2. A **Linux partition** with the same ROCm stack.
3. An **NVIDIA GPU** on Linux (compute capability 7.5+), which needs none of
   the ROCm caveats.

None of those was attempted. Installing vLLM here would mean a multi-GB CUDA
wheel that fails at runtime with `libcudart.so: cannot open shared object
file` — the exact mistake this document exists to prevent.

`rigma engine-runtimes` says this on the machine, in one sentence, and is the
executable form of the paragraph above.

---

## 1. What vLLM is, and why it is worth having as a second engine

vLLM is a high-throughput LLM inference and serving library. From its README
([source](https://raw.githubusercontent.com/vllm-project/vllm/main/README.md)):

- **PagedAttention** — KV memory managed in fixed-size blocks with a page
  table, so a sequence's cache does not have to be one contiguous allocation.
  This is the paper the project is built on (SOSP 2023, arXiv 2309.06180).
- **Continuous batching** of incoming requests, plus chunked prefill.
- **Automatic prefix caching** — shared prefixes are reused across requests
  without any flag; it is **on by default** (`CacheConfig.enable_prefix_caching
  = True`).
- **Tensor, pipeline, data, expert and context parallelism** — one model over
  several GPUs, which llama.cpp does not do for a single process.
- **HuggingFace/safetensors weights**, "seamless integration with popular Hugging
  Face models" — the artefact is a repo of safetensors plus a `config.json`, not
  a single gguf.
- An **OpenAI-compatible API server** (`vllm serve`), which is what makes it
  drop-in behind Rigma's existing proxy at all.

The practical difference for a *local, single-user* Rigma install is narrower
than the marketing suggests, and the honest framing is this: llama.cpp is
already excellent at one user, one conversation, one GPU, and Rigma has spent
real effort (fit math, calibration, KV fingerprints, slot snapshots) making
that path good. vLLM's advantages — concurrency, batching, multi-GPU — mostly
do not apply to that workload. What vLLM genuinely offers Rigma is:

- a **second opinion on throughput** for the same weights, which is a
  measurement Rigma's `bench`/`sweep` machinery could act on;
- **quantisation formats llama.cpp cannot read** (FP8, AWQ, GPTQ, compressed
  tensors, NVFP4) as first-class artefacts;
- **tensor parallelism**, the only route to serving a model larger than one
  card without expert-offload tricks;
- and a hedge: if llama.cpp's Windows/ROCm path regresses, there is a
  supported alternative.

### What it costs

| Cost | Detail |
|---|---|
| **Linux only** | `OS: Linux`. Windows is unsupported natively; WSL2 or a community fork are the only routes. Rigma supports Windows first-class. |
| **Multi-GB torch dependency** | The wheel "bundles PyTorch and all required dependencies". Compare with Rigma's current engine: a ~200 MB llama.cpp zip, pinned by SHA-256, extracted into `~/.rigma/engines/`. |
| **A different model format** | safetensors/HF repo layout. Rigma's entire model library, `GgufFile`, the downloader, the on-disk inventory and the probe are gguf-shaped. |
| **GGUF support is not a bridge** | vLLM *does* document GGUF, but calls it "**highly experimental and under-optimized**", has moved it out of tree to `vllm-gguf-plugin`, and still wants a HuggingFace config/tokenizer beside the file. See §5. |
| **A build-per-platform story Rigma does not control** | Rigma pins one llama.cpp build id. vLLM's wheels are per (CUDA/ROCm variant, Python version), and the ROCm ones lag: the ROCm 7.0 wheel table lists supported versions `0.14.0` to `0.18.0` while `main` is ~0.23.x. |
| **A silent failure mode** | On an AMD host with any Python other than 3.12, the installer falls back to the CUDA wheel **without saying so**. See §2. |
| **No on-disk KV snapshot** | llama.cpp's `--slot-save-path` + `/slots/{n}?action=save` has no equivalent in the flags Rigma emits. See §6. |

---

## 2. The hard capability matrix

Every row below is quoted from the page named beside it. Where a claim comes
from vLLM's *source* rather than its docs, that is stated, because the docs
generate their argument reference from those classes and the generated page is
not a stable URL.

### 2.1 The tuple

| Dimension | vLLM requires | Source |
|---|---|---|
| **OS** | `Linux`. "vLLM does not support Windows natively. To run vLLM on Windows, you can use the Windows Subsystem for Linux (WSL) with a compatible Linux distribution, or use some community-maintained forks, e.g. https://github.com/SystemPanic/vllm-windows" | [`docs/getting_started/installation/gpu.md`](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/) |
| **Python** | `3.10 -- 3.13` | same |
| **ROCm (AMD)** | ROCm **6.3 or above**; prebuilt wheels for ROCm **7.0** (`rocm700`) and **7.2.1** (`rocm721`) | [`gpu.rocm.inc.md`](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.rocm.inc.md) |
| **ROCm wheel Python** | **3.12 only.** "ROCm pre-built wheels are only available for **Python 3.12**. If you are using a different Python version (e.g. 3.11 or 3.13), the installer **will silently fall back** to the CUDA wheel from PyPI, which will fail on AMD GPUs with errors like `libcudart.so: cannot open shared object file`." | same |
| **glibc** | `>= 2.35` for both ROCm wheel variants | same |
| **AMD GPU arch** | MI200s (gfx90a), MI300 (gfx942), MI350 (gfx950), Radeon RX 7900 (gfx1100/1101), **Radeon RX 9000 (gfx1200/1201)**, Ryzen AI MAX / AI 300 (gfx1151/1150) | same |
| **CUDA (NVIDIA)** | GPU **compute capability 7.5 or higher**; binaries compiled against **CUDA 12.9**, with 12.8 and 13.0 variants published | [`gpu.cuda.inc.md`](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.cuda.inc.md) |
| **ROCm wheel version window** | `rocm700`: vLLM `0.14.0`–`0.18.0`. `rocm721`: nightly only, after commit `171775f306a333a9cf105bfd533bf3e113d401d9` | `gpu.rocm.inc.md` |

### 2.2 The verdicts that matter

- **Windows: not supported.** Not "unsupported, try anyway" — the requirement is
  stated as an OS line, and the Windows note offers WSL or a third-party fork.
  Rigma must therefore never offer vLLM on Windows as an option it will launch;
  it reports *why* instead. **Implemented.**
- **gfx1201 (RX 9070 XT): supported.** It is named in the ROCm GPU list. The
  owner's card is not the problem. **Implemented** (a test asserts gfx1201 +
  Python 3.12 + Linux is `runnable`).
- **Python-3.12-only ROCm wheels: the trap.** Because the fallback is silent,
  a Python-3.11 user gets a *successful-looking* install and a runtime failure
  much later. Rigma's availability verdict must say this before anything is
  installed, and it does. **Implemented.**
- **glibc >= 2.35.** Ubuntu 22.04+ / Debian 12+. **Implemented** (blocking when
  measured below the floor; reported as *unverified* when `os.confstr` cannot
  answer, rather than assumed to pass).
- **ROCm 6.3 floor.** **Implemented** (read from `/opt/rocm/.info/version`).
- **CUDA compute capability 7.5.** **NOT verified by Rigma** — Rigma does not
  read compute capability, so the CUDA-path verdict says so in the sentence
  instead of implying a check that did not happen.
- **An unidentifiable GPU is `unverified-gpu`, not `available`.** If `rocminfo`
  yields no gfx target and there is no `nvidia-smi`, Rigma cannot rule out the
  AMD-Python trap, so it refuses and explains.

---

## 3. The design

### 3.1 The naming problem, solved in one comment

Three different things in this codebase are spelled "backend":

1. the llama.cpp **compute** backend — `vulkan`/`cuda`/`rocm`/`cpu`
   (`RunPlan.backend`, `server_ops.available_backends`, the
   `"windows/vulkan"` keys in `data/engines.json`);
2. the **harness** backend — `native`/`dsh`/`mcode` (`harness.py`);
3. the **engine runtime** — `llamacpp`/`vllm` (this feature).

`engines.py` opens with exactly that list and never uses the bare word
"backend" for (3). A future reader will otherwise conflate (1) and (3), because
both are per-launch choices surfaced in the CLI.

### 3.2 Every function added or changed

**`src/rigma/engines.py` (new, no imports of `vllm` or `torch`, no work at
import time):**

| Name | Kind | What it is |
|---|---|---|
| `LLAMACPP`, `VLLM`, `ENGINE_RUNTIMES`, `DEFAULT_ENGINE_RUNTIME` | data | The axis, named once. |
| `SUPPORTED_ROCM_ARCHES`, `ROCM_WHEEL_PYTHON`, `ROCM_WHEEL_GLIBC`, `ROCM_MIN_VERSION`, `VLLM_PYTHON_MIN/MAX`, `CUDA_MIN_COMPUTE` | data | The matrix from §2, as data, so a test can assert against it. |
| `LLAMACPP_ONLY_FLAGS`, `VLLM_RETIRED_FLAGS` | data | Flags a `vllm serve` line must never carry. |
| `_os_name`, `_python_version`, `_vllm_executable`, `_vllm_module_present`, `_gpu_arch`, `_rocm_version`, `_cuda_present`, `_glibc_version` | probes | One fact each, so a test replaces exactly the fact it is about. `_gpu_arch` runs `rocminfo` on Linux only and returns `None` elsewhere — Rigma's own GPU table names the arch `rdna4` (a family), not a gfx target, so guessing is refused. |
| `EngineAvailability` | dataclass | `engine`, `available`, `state`, `reason`, `evidence`, plus `as_dict()`. `state` is deliberately finer than the boolean. |
| `vllm_availability()` | public | The verdict. Check order: OS → installed → accelerator → Python → arch → ROCm → glibc. |
| `llamacpp_availability(registry=None)` | public | The same shape for llama.cpp, delegating to `server_ops.available_backends` rather than re-deriving it. |
| `engine_runtimes(registry=None)` | public | Both verdicts, in a fixed order. |
| `vllm_argv(...)` | public | The `vllm serve` command line, with real validation. |
| `EngineRuntimeDecision` | dataclass | `runtime`, `requested`, `reason`, `availability`. |
| `detect_engine_runtime(requested=None)` | public | Chooses; default is always `llamacpp`. |
| `_vllm_failure_hint(tail)` | private, pure | Turns the two failures a Rigma user will hit into a diagnosis. |
| `launch_vllm_server(...)` | public | Popen + poll `GET /health`; returns the same `runtime.ServerProcess` handle the llama.cpp path returns. **Never executed against a real vLLM.** |

**Changed (additive only):**

| File | Change |
|---|---|
| `src/rigma/server_ops.py` | **New** `available_engine_runtimes(registry=None)`. `available_backends()` is untouched: it answers a different question and two `serve.py` call sites render its rows. |
| `src/rigma/cli.py` | **New** `@app.command("engine-runtimes")`. No existing command, option or output changed. |
| `.gitignore` | Re-includes `docs/design/` — a spec is a deliverable, not a working note. `docs/*` still hides everything else under `docs/`. |

**Deliberately NOT changed:** `runtime.py` (not one line), `models.py`,
`kvcache.py`, `prefixcache.py`, `resolve.py`, `serve.py`, `frontend-v2/`, and
anything named `harness*`. llama.cpp's behaviour is unchanged, and
`tests/test_audit_serve.py`, `tests/test_cli.py`, `tests/test_audit_ops.py`,
`tests/test_backend_choice.py` and `tests/test_kvcache.py` all still pass
(112 passed).

### 3.3 Why the launch path is *not* wired in

`detect_engine_runtime()` is the seam a real launch would call. It is not
called from `server_ops.perform_switch` or `cli.up` yet, on purpose:

- `perform_switch` resolves a `RunPlan` whose `gguf` is a `GgufFile` and whose
  flags are `ComboFlags` (`-ngl`, `-c`, `--n-cache-type-*`). A vLLM launch
  needs a *different model artefact* (an HF repo id or a safetensors directory)
  and a *different flag set*. Threading that through the existing launch path
  is a model-artefact change, not an engine-runtime change, and doing it
  half-way is how you get a launch that reads a gguf with vLLM and dies in the
  GGUF plugin.
- Auto-calibration (`bench.auto_calibrate`) sweeps llama.cpp flags. It must not
  be run for a vLLM launch, and nothing today would stop it.

So: the availability report and the argv builder are real and tested; the
launch is specified in §7 phase 2.

---

## 4. Fit arithmetic does not transfer — and must not be silently reused

This is the part of the design where a wrong number is not a bad benchmark, it
is an **OOM or a takeover of the card**.

### llama.cpp, today

- Model shape comes from **GGUF metadata** (`gguf_meta.py`: `block_count`,
  `kv_heads`, `head_dim`, `full_attention_interval`, `swa_*`, MoE expert
  counts).
- `resolve.kv_bytes_per_token` sizes the cache from `models.CACHE_BYTES` (a
  block-size table: q4_0 18 B/32, q8_0 34 B/32, f16 2 B/element, …) times
  `ctx`.
- **`--n-cpu-moe`** pushes expert tensors to CPU RAM, which is how a MoE model
  fits a 16 GB card at all.
- `-ngl` decides how many layers go to the GPU; `-c` is the context;
  `-fa` and `--cache-type-k/v` choose the attention path and the cache
  precision; `--parallel 2 --kv-unified` gives two slots over one KV pool.
- The budget is **bytes**: `_budgets(profile)` returns usable VRAM in MB, and
  every decision downstream is arithmetic on megabytes.

### vLLM

- Model shape comes from the **HuggingFace config** of the checkpoint, not from
  gguf metadata.
- The KV cache is sized by vLLM **itself**, after a GPU-memory profiling run,
  from `--gpu-memory-utilization` — "The fraction of GPU memory to be used for
  the model executor, which can range from 0 to 1", **default 0.92**
  (`CacheConfig.gpu_memory_utilization`, `Field(default=0.92, gt=0, le=1)`).
  Rigma does not compute a cache size; vLLM computes it and there is no
  `--kv-cache-bytes` in the flags Rigma emits.
- `--max-model-len` caps prompt+output length, and accepts vLLM's own `-1`
  / `auto` sentinel meaning "largest that fits".
- `--tensor-parallel-size` shards the model across GPUs.
- `--quantization` names the *scheme already in the checkpoint* (AWQ, GPTQ,
  FP8, compressed-tensors, …); it is not a request to quantise.
- `--kv-cache-dtype` is the cache dtype (`cache_dtype`; `auto`, `fp8`, …).
- `--max-num-seqs` caps concurrency.
- **There is no `--n-cpu-moe`.** Expert-offload in vLLM is a different
  mechanism and was **not** researched for this document — do not assume one
  exists under another name.

### The rule

> **Rigma's GGUF-based fit math does NOT transfer to vLLM, and no code may
> reuse it.** The two share a *word* (`vram_mb`) and nothing else.

Concretely, in the code that exists today:

- `vllm_argv` **validates `gpu_memory_utilization` as a fraction in (0, 1]** and
  raises on anything else. Handing it `15000` (Rigma's units) would otherwise be
  clamped by vLLM to `1.0` — the whole card — with no error anywhere. There is a
  test for exactly that, and its docstring says why.
- `vllm_argv` **refuses a `.gguf` model path** unless `gguf_plugin=True` is
  passed explicitly, because the tempting move — give vLLM the path llama.cpp
  gets — is the one that must fail loudly. See §5.
- Phase 2 must derive `gpu_memory_utilization` as *(measured free device memory
  − Rigma's `VRAM_RESERVE_MB` + compute buffer) / total device memory*, and must
  use `probe.gpu_used_mb()`-style **measured** numbers, not the llama.cpp
  budget. **Not implemented.**

---

## 5. GGUF: what vLLM actually offers

vLLM does document GGUF, and the README lists it among supported quantisation
formats — so "vLLM cannot read gguf" would be false. The accurate statement is
in vLLM's own GGUF page
([source](https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/gguf.md)):

> "Please note that GGUF support in vLLM is **highly experimental and
> under-optimized** at the moment, it might be incompatible with other
> features."
>
> "GGUF support has migrated to OOT
> [vllm-gguf-plugin](https://github.com/vllm-project/vllm-gguf-plugin). Make
> sure you have GGUF plugin installed before serving a GGUF model."
>
> "We recommend using the tokenizer from base model instead of GGUF model.
> Because the tokenizer conversion from GGUF is time-consuming and unstable…"
>
> "GGUF assumes that HuggingFace can convert the metadata to a config file. In
> case HuggingFace doesn't support your model you can manually create a config
> and pass it as hf-config-path."

So a Rigma gguf is **not** a drop-in: it needs an extra package, an
HF-compatible config, and usually a separate tokenizer, and the result is
explicitly not feature-complete. It is not a bridge between the two engines and
must not be presented as one.

**Implemented:** `vllm_argv` raises on a `.gguf` path unless the caller passes
`gguf_plugin=True`, with the reasoning in the exception text.

---

## 6. KV cache and prefix cache: the fingerprint must carry the engine runtime

`kvcache.py` exists because a saved llama.cpp slot cache is only meaningful
under the exact configuration that produced it, and restoring one under a
different configuration "generates fluent, subtly wrong text from a cache that
does not describe its own history". The safety property is that the file is
**named** for its configuration: `kvcache.fingerprint()` hashes every field in
`FINGERPRINT_FIELDS`, and `restore()` can only ever ask for the file whose name
matches the configuration being launched now.

### What is llama.cpp-specific

| Rigma today | vLLM |
|---|---|
| `kvcache.launch_fingerprint(rp, exe)` → `kv-<fp>.bin` under `~/.rigma/sessions` | no equivalent in the flags Rigma emits |
| `kvcache.save/restore` → `POST /slots/{slot}?action=save\|restore` | **vLLM does not serve `/slots/*`.** The call would 404. |
| `--slot-save-path` passed by `runtime.server_argv` on every launch | no peer; deliberately not emitted |
| `prefixcache.py` snapshot/restore, same `/slots` endpoint | no peer |
| `--cache-reuse 256`, `--checkpoint-min-step 4096` | vLLM's scheduler does its own thing; neither flag exists |

### What vLLM offers instead

**Automatic prefix caching**, enabled by default
(`CacheConfig.enable_prefix_caching = True`). It works *within a running
server*, across requests, with no configuration — which covers the in-process
reuse llama.cpp's `--cache-reuse` covers. It does **not** cover what
`kvcache.py` was written for: surviving a **process restart**. Unloading a model
to free the card and loading it again still re-prefills from zero under vLLM.
Rigma's four-minutes-of-prefill problem has **no vLLM solution implemented
here**, and that is a real regression a vLLM user would feel. Phase 3 must
either find vLLM's own KV-offload/persistence route (there is a KV-offloading
feature in vLLM; it was **not** researched for this document) or accept and
document the loss.

### The fingerprint hazard, precisely

`kvcache.FINGERPRINT_FIELDS` is:

```
model, quant, gguf, backend, engine, ctx, cache_type_k, cache_type_v,
ngl, n_cpu_moe, spec_type, spec_n_max, flash_attn
```

`engine` is `str(exe)` — the **llama.cpp binary path**. It varies with the
engine *build*, and therefore already separates two llama.cpp builds. It does
**not** separate engine *runtimes*: nothing in that tuple says "this cache came
from llama.cpp". A future vLLM launch that recorded a fingerprint and a later
llama.cpp launch that derived the same one would look like a match.

Two facts blunt the impact today, and both are luck rather than design:

1. `kvcache.restore` is called on the llama.cpp launch path with
   `runtime.rigma_home() / "sessions"`, and every `/slots` call is wrapped in
   `except Exception: pass`, so a 404 against vLLM is a slow start, not a
   corruption. (The call site in `server_ops.perform_switch` is
   `try: kvcache.restore(...) except Exception: pass`.)
2. `config_of()` derives `engine` from an `exe` argument; a vLLM launch has no
   llama.cpp exe to pass.

**The requirement, therefore:** the engine runtime (`llamacpp` | `vllm`) must
become part of the engine-runtime identity that `FINGERPRINT_FIELDS` hashes, so
a snapshot can never be restored across runtimes *by construction* — the same
"the name it would need does not exist" guarantee the module was built on.

**Specified only, deliberately not implemented.** Adding a field to
`FINGERPRINT_FIELDS` changes the hash of every configuration, which invalidates
**every saved KV cache on every machine** — and `kvcache.py` says so in its own
comment ("changing it invalidates every saved cache"). That is a behaviour
change to llama.cpp's path, and this task's rule is that llama.cpp's behaviour
must not change. The correct vehicle is a phase-3 commit that (a) adds
`engine_runtime` to `FINGERPRINT_FIELDS`, (b) records it in `config_of`, and
(c) accepts the one-time cache invalidation knowingly, with a note in the
release. Anyone implementing vLLM launch support **before** that lands is
introducing the cross-runtime hazard.

---

## 7. Phased plan

| Phase | What | State |
|---|---|---|
| **0** | **Engine-runtime registry, availability verdict, `vllm serve` argv builder, `rigma engine-runtimes`, 38 tests, mutation-checked.** | **DONE** |
| 1 | Run vLLM **once** on real supported hardware (Linux + ROCm 7.x + Python 3.12, or Linux + NVIDIA). Verify `GET /health` answers 200 and that `ServerProcess.is_healthy()` works unchanged. Verify every flag in `vllm_argv` against `vllm serve --help=all` on the installed version, and re-check `--tensor-parallel-size` / `--max-num-seqs` (confirmed here from docs and from the `--{field}` registration rule in `cli_args.py`, not from a live `--help`). Measure real cold-start time to replace the 600 s default. | **NOT STARTED** |
| 2 | Launch plumbing: a vLLM model artefact (HF repo id / safetensors directory) beside `GgufFile`; a fit step that derives `gpu_memory_utilization` as a **fraction** of measured device memory; wire `detect_engine_runtime()` into `server_ops.perform_switch` and `cli.up` behind an explicit opt-in; make sure `bench.auto_calibrate` is skipped for vLLM. | **NOT STARTED** |
| 3 | KV/prefix cache: add the engine runtime to `kvcache.FINGERPRINT_FIELDS` (accepting the one-time invalidation), and decide whether vLLM gets an on-disk cache story or documents its absence. | **NOT STARTED** |
| 4 | UI: an engine-runtime selector. Requires `frontend-v2/src` changes **and** a rebuilt `src/rigma/data/ui_v2/` bundle in the same commit — explicitly out of scope here. `server_ops.available_engine_runtimes()` is the seam it would read. | **NOT STARTED** |
| 5 | WSL2 as a first-class state: detect a WSL2 environment with ROCm/CUDA visibility and report "supported, but through WSL" rather than a flat `unsupported-os`. | **NOT STARTED** |

**Do not mistake this document for a working feature.** Only phase 0 exists.

---

## 8. How these tests were checked

38 tests in `tests/test_engines.py`, none of which import `vllm` or `torch` or
touch a real GPU — every probe is replaced by `monkeypatch`. A test that cannot
fail is not a test, so each of these mutations was applied to `engines.py` in
turn and the suite was re-run; every one produced a failure in the test named:

| Mutation | Test that failed |
|---|---|
| `--gpu-memory-utilization` renamed to `--gpu-mem-util` | `test_gpu_memory_utilization_is_a_fraction_and_megabytes_are_refused` |
| `--n-cpu-moe 4` injected into the argv | `test_argv_contains_no_llamacpp_only_flag` |
| `detect_engine_runtime` returns `vllm` unconditionally | `test_an_unknown_engine_runtime_is_an_error_not_a_fallback` (and the default test) |
| the OS check moved after the install check | `test_vllm_is_not_chosen_when_asked_but_unavailable` |
| the Windows reason reworded to contain "not installed" | `test_windows_is_unavailable_and_the_reason_is_wsl` |
| `ROCM_WHEEL_PYTHON` changed to `(3, 11)` | the ROCm-trap test |
| the log tail dropped from the launch failure | `test_a_crashing_launch_reports_the_log_tail_and_the_real_cause` |
| the `.gguf` guard removed | `test_a_gguf_model_is_refused_unless_the_plugin_is_declared` |

Commands, verbatim, from the worktree root:

```
$env:PYTHONPATH="<worktree>\src;C:\ComfyUI\RD\rigma-review\.scratch\00-recon\plug"
python -m ruff check src tests tools          -> All checks passed!
python -m pytest tests/test_engines.py -q -p no:cacheprovider -p dsh_tmpfix \
    --basetemp=<...>\bt-vllm                  -> 38 passed, exit 0
python -m pytest tests/test_audit_serve.py tests/test_cli.py \
    tests/test_audit_ops.py tests/test_backend_choice.py tests/test_kvcache.py \
    -q ...                                    -> 112 passed, exit 0
```

---

## 9. What is UNVERIFIED, stated plainly

- **vLLM has never been run by this work**, on this machine or any other. Not
  one weight has been loaded, not one request served.
- **`GET /health` was not exercised.** The assumption that `vllm serve` answers
  it comes from vLLM's own serve-arg help, which names `"/health,/metrics,/ping"`
  as the endpoints worth excluding from access logs. If vLLM's health path ever
  differs, `runtime.ServerProcess.is_healthy()` — which polls `GET /health` —
  would spin until timeout. Phase 1 must confirm this first.
- **`--tensor-parallel-size` and `--max-num-seqs`** were confirmed from the
  documented GGUF example (`--tensor-parallel-size 2`) and from
  `vllm serve --help=max-num-seqs` in the CLI guide, plus the registration rule
  read in `vllm/entrypoints/launchers/cli_args.py`
  (`f"--{key.replace('_','-')}"` over the config dataclasses). They were **not**
  read off a live `vllm serve --help` of an installed version.
- **The wheel indices were not fetched.** The version window
  (`0.14.0`–`0.18.0` for `rocm700`) is quoted from the docs table; nothing was
  downloaded from `wheels.vllm.ai`.
- **Whether WSL2 + ROCm can expose gfx1201 to vLLM was not tested.** That is
  the single most important unknown for this machine, and it is a hardware
  question, not a documentation one.
- **vLLM's KV offloading / persistence options were not researched**, so the
  phase-3 statement "there is no on-disk cache peer in the flags Rigma emits"
  is a statement about *those flags*, not about vLLM as a whole.
- **`launch_vllm_server`'s polling loop** is tested against a fake process only.

## 10. Sources

All read 2026-09-25.

- Requirements, OS/Python, the Windows note — <https://docs.vllm.ai/en/latest/getting_started/installation/gpu/>
  (raw: `docs/getting_started/installation/gpu.md`)
- ROCm: version floor, wheel table, Python-3.12-only warning, GPU arch list —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.rocm.inc.md>
- CUDA: compute capability 7.5, CUDA 12.9 binaries —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/getting_started/installation/gpu.cuda.inc.md>
- `vllm serve`, config-file keys (`model`, `host`, `port`) —
  <https://docs.vllm.ai/en/latest/configuration/serve_args/>
- CLI guide: `vllm serve <model> --port`, `--help=all`, `--help=max-num-seqs`,
  human-readable integer suffixes — <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/cli/README.md>
- Serve argument registration (`FrontendArgs`: `host`, `port`, `api_key`, `uds`;
  positional `model_tag`; the `/health,/metrics,/ping` example) —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/entrypoints/launchers/cli_args.py>
- `ModelConfig`: `served_model_name`, `max_model_len`, `quantization`, `dtype`,
  `revision`, `tokenizer`, `trust_remote_code`, `enforce_eager` —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/model.py>
- `CacheConfig`: `gpu_memory_utilization` default 0.92, `enable_prefix_caching`
  default `True`, `cache_dtype` —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/vllm/config/cache.py>
- GGUF: experimental, plugin, tokenizer caveat —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/features/quantization/gguf.md>
- OpenAI-compatible API surface and `--api-key` limitations —
  <https://raw.githubusercontent.com/vllm-project/vllm/main/docs/serving/online_serving/openai_compatible_server.md>
- Key features (PagedAttention, continuous batching, prefix caching, tensor
  parallelism, HF/safetensors) — <https://raw.githubusercontent.com/vllm-project/vllm/main/README.md>
