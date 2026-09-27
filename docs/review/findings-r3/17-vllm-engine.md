# R3 findings — area: engine runtimes (vLLM as a second engine)

This area's deliverable is a *feature* (a second engine runtime), so the
findings below are what building it exposed about the code that already
existed, not defects in new code. The spec is
`docs/design/2026-09-25-vllm-engine-spec.md` (committed); the module is
`src/rigma/engines.py`; the tests are `tests/test_engines.py`.

**VLLM-4 (added by the orchestrator, `f0efd61` + `1e42ea7`): the feature was
UNREACHABLE, and that is itself the finding.** Everything above was built and
tested — `detect_engine_runtime`, `vllm_argv`, `launch_vllm_server`,
`rigma engine-runtimes` — and nothing called any of it: no `--engine` flag, no
state field recording which engine was running, no API or UI surface to read. So
the area delivered a correct report about a road with no on-ramp, and "vLLM as a
selectable backend" was one wiring pass short of true. See RUN3.md for the design
point it turns on (refuse rather than fall back on a command line, while keeping
the fallback for a stored preference) and for the three bugs the wiring's own tests
caught.

---

### VLLM-1 [HIGH] The KV-cache fingerprint does not identify the engine RUNTIME, so "a cache that does not describe its own history cannot be restored" holds only between llama.cpp builds

- **Where:** `src/rigma/kvcache.py:43-58` (`FINGERPRINT_FIELDS`),
  `src/rigma/kvcache.py:114-126` (`config_of`),
  `src/rigma/server_ops.py:607-609` (the restore call site)
- **Trigger:** any launch that is not llama.cpp recording a `kv_fp`. Nothing
  does that today — this is the latent half of adding a second engine runtime,
  and it is being added now.
- **Consequence:** `kvcache.py`'s stated safety property is "restore only ever
  asks for the file whose name matches the configuration being launched right
  now … A mismatch cannot be restored because the name it would need does not
  exist." That is true across llama.cpp *builds* and false across engine
  *runtimes*: the `engine` field is `str(exe)`, a llama.cpp BINARY path. If a
  vLLM launch ever records a fingerprint (which is exactly what reusing
  `server_ops.perform_switch` would do) and a later llama.cpp launch derives the
  same one, the restore asks llama.cpp's `/slots/0?action=restore` for a file
  describing a cache vLLM produced. Today that is harmless **by luck, not by
  design**: vLLM does not serve `/slots/*`, and the call site swallows every
  exception (`except Exception: pass`, `server_ops.py:609`), so a cross-runtime
  restore is a silent 404 rather than corruption — but the silence is the same
  silence that would hide a real mismatch.
- **Cause:** `FINGERPRINT_FIELDS` was written when there was exactly one engine,
  and `engine` was added to separate engine *builds* (its comment is about the
  binary). Nothing in the tuple says "llama.cpp".
- **Fix:** add the engine runtime (`llamacpp` | `vllm`) to `FINGERPRINT_FIELDS`
  and to `config_of()`, so the cross-runtime case is impossible by construction.
  **NOT implemented, deliberately.** Changing that tuple changes the hash of
  every configuration and therefore invalidates every saved KV cache on every
  machine — `kvcache.py` says so itself ("changing it invalidates every saved
  cache") — which is a behaviour change to llama.cpp's path, and this task's
  rule is that llama.cpp's behaviour must not change. Written up as phase 3 in
  the spec (§6). Anyone wiring vLLM into the launch path before that lands is
  creating this hazard.
- **Verified:** read `kvcache.py:1-199` and `server_ops.py:600-615`; the field
  list and the `"engine": engine` mapping are quoted above at the exact lines.

---

### VLLM-2 [MEDIUM] Nothing in Rigma can answer "which GPU arch is this" off Linux, so every vendor/arch-gated decision is undecidable on the platform Rigma supports first-class

- **Where:** `src/rigma/probe.py:28-44` (`classify_gpu`, `arch=row["arch"]` at
  `probe.py:36`), `src/rigma/models.py:50-56` (`GpuInfo.arch`)
- **Trigger:** asking a vendor/arch question about an AMD card on Windows. vLLM's
  supported-ROCm list is written in **gfx targets** (`gfx1200/1201` for the RX
  9000 series); Rigma's registry row stores `arch: "rdna4"`, a marketing family.
- **Consequence:** `HardwareProfile` cannot decide "is this card on the
  ROCm-supported list". `engines._gpu_arch()` has to shell out to `rocminfo`
  (Linux only) and returns `None` everywhere else, which forces the honest but
  weak `unverified-gpu` verdict instead of a real answer. Any future arch-gated
  feature — which ROCm wheel to install, per-arch kernels, whether a card is
  supported at all — is unanswerable on the owner's own machine.
- **Cause:** `arch` in the gpu table exists for display and for the resolver's
  prose; nothing needed a gfx target until a second engine did.
- **Fix:** carry a gfx target per registry row (or map family → target) and let
  `_gpu_arch()` fall back to `probe_hardware().primary_gpu` before giving up.
  **Not implemented** — out of scope here, and on Windows the OS check fires
  before the arch is consulted, so the None costs nothing *yet*.
- **Verified:** read `probe.py:1-301` and `models.py:50-56`; on this box
  `engines._gpu_arch()` returns `None` because `_os_name() != "linux"`.

---

### VLLM-3 [MEDIUM] Rigma's fit arithmetic is in megabytes and vLLM's memory knob is a fraction; nothing converts between them, and the MoE offload that makes Rigma's plans fit has no peer

- **Where:** `src/rigma/resolve.py` (`_budgets`, `kv_bytes_per_token`),
  `src/rigma/models.py:20-28` (`CACHE_BYTES`),
  `src/rigma/models.py:304-323` (`ComboFlags.n_cpu_moe`),
  `src/rigma/server_ops.py:561-585`
- **Trigger:** reusing an existing budget/plan for a vLLM launch.
- **Consequence:** vLLM's `gpu_memory_utilization` is
  `Field(default=0.92, gt=0, le=1)` — a FRACTION of device memory — and it sizes
  the KV cache itself after a profiling run. Rigma's number is bytes, derived
  from gguf metadata, `CACHE_BYTES` and `ctx`. Handing vLLM a megabytes value
  either errors or is clamped to 1.0 (the whole card) with no other signal, and
  `--n-cpu-moe` — the flag that makes Rigma's large-MoE plans fit a 16 GB card
  at all — has no vLLM equivalent among the flags Rigma emits. A plan that
  "fits" per Rigma's arithmetic is therefore an OOM or a card takeover under
  vLLM.
- **Cause:** one engine, one unit, and the fit math is written against gguf
  metadata that vLLM does not read.
- **Fix:** implemented only as a **guard** —
  `engines.vllm_argv` validates `gpu_memory_utilization` as `(0, 1]` and raises
  with the reason in the message, covered by
  `test_gpu_memory_utilization_is_a_fraction_and_megabytes_are_refused`. The
  actual conversion (measured free device memory − `VRAM_RESERVE_MB`, over total
  device memory) is **specified only** (spec §4, phase 2).
- **Verified:** `pytest tests/test_engines.py -q …` → **38 passed**, exit 0;
  `ruff check src tests tools` → All checks passed. The failing-before/passing-
  after direction was established by mutation: renaming
  `--gpu-memory-utilization` makes that test fail.

---

### VLLM-4 [LOW] `platform.system()` is a dict key in four places, so any OS outside {Windows, Linux, Darwin} is a `KeyError`, not a message

- **Where:** `src/rigma/server_ops.py:369`, `src/rigma/server_ops.py:561`,
  `src/rigma/cli.py:1203`, `src/rigma/cli.py:1594`
- **Trigger:** running Rigma on FreeBSD / OpenBSD / any other POSIX.
- **Consequence:** `KeyError: 'FreeBSD'` out of `rigma doctor` or `rigma up`
  instead of the honest `no pinned engine build for freebsd/vulkan`.
- **Cause:** the same expression copied four times; each one indexes a dict with
  a key that only three platforms provide.
- **Fix:** `{...}.get(platform.system(), platform.system().lower())`, which is
  what the new `engines._os_name()` does. **Not implemented** — it is outside
  this task's additive rule for `cli.py`/`server_ops.py`, and it is not a vLLM
  finding; recorded so it is not lost.
- **Verified:** grep of the four sites (`Select-String` on the two files), lines
  quoted above.

---

## Not findings, but decided here

- `server_ops.available_backends()` was **not** widened to carry engine
  runtimes. It answers the llama.cpp COMPUTE-backend question and two `serve.py`
  call sites render its rows; the engine-runtime axis got its own function
  (`available_engine_runtimes()`) instead. A `"vllm"` entry in a list the
  backend picker iterates would be a contract change, not an addition.
- `--enable-prefix-caching` is **not** emitted by `engines.vllm_argv` because
  vLLM's automatic prefix caching is already on by default
  (`CacheConfig.enable_prefix_caching = True`). Emitting it would imply Rigma
  arranged a default.
- GGUF is not a bridge between the engines. vLLM documents GGUF as "highly
  experimental and under-optimized", moved it to the out-of-tree
  `vllm-gguf-plugin`, and still wants an HF config/tokenizer beside the file.
  `vllm_argv` therefore **refuses** a `.gguf` path unless `gguf_plugin=True`.

<!-- coverage: src/rigma/engines.py (read, written), tests/test_engines.py (read, written), src/rigma/runtime.py L1-365 (read), src/rigma/models.py L1-437 (read), src/rigma/server_ops.py L320-672 (read), src/rigma/kvcache.py L1-199 (read), src/rigma/probe.py L1-301 (read), src/rigma/cli.py L1-30,428-700,1150-1210,1580-1600 (read), src/rigma/data/engines.json (read), .gitignore (read, written), pyproject.toml L1-80 (read), .githooks/pre-commit (read), docs/design/2026-09-25-vllm-engine-spec.md (written) -->
