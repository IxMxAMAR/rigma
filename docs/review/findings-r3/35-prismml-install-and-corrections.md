# R3-ENG-7..11 — the PrismML install, and three corrections to my own earlier work

**Status:** two source bugs fixed and tested; the install verified live; two of my own
earlier conclusions corrected.
**Branch:** `review/deep-audit-2026-09-22` in the fork `C:\ComfyUI\RD\rigma-review`.

---

## What was asked

Install `%USERPROFILE%\Downloads\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf` and
test it, on PrismML's fork of llama.cpp, using Vulkan.

That file is **not** the same model as the already-installed `Ternary-Bonsai-2-27B-PQ2_0`
— same architecture and same tensor-type histogram, different weights (first divergence
at byte 3,441,389,284).

## What it is

Both Bonsai files are `arch=qwen35`: 64 layers, 16 of them full attention
(`full_attention_interval=4`), 48 SSM, `kv_heads=4`, `head_dim=256`, `n_embd=5120`,
`native_ctx=262144`. Both report `{0: 353, 30: 96, 142: 402}` — **402 tensors of
`GGML_TYPE_PQ2_0 = 142`**, a type private to `PrismML-Eng/llama.cpp`. Mainline's enum
stops at `Q2_0 = 42`.

Installed through Rigma's own path (`hangar.install_model`), which slugged it
`ternary-bonsai-2-27b-uncensored-heretic-pq2-0` — no collision, because `general.name` is
`Hf` (two characters, below `model_slug`'s minimum) so it fell back to the filename.

PrismML's release `prism-b10743-adfffbe` ships both `bin-win-hip-radeon-x64` and
`bin-win-vulkan-x64`. Both are installed and registered:

```
prism-b10743-hip      backend=rocm    36 types, max 142
prism-b10743-vulkan   backend=vulkan  36 types, max 142
```

---

## R3-ENG-7 — `check_types` rejected a fork type the engine actually has

`check_types` is documented as taking `engine_type_count` = "the authoritative bound for
THAT build". It then ignored that bound for fork types:

```python
forks = {t: FORK_TYPES[t][0] for t in ids if t in FORK_TYPES}   # no bound consulted
...
if forks:
    return Compatibility(ok=False, ...)                          # unconditional
```

So PrismML's own build — table size 144, which therefore *contains* `PQ2_0 = 142` — was
told a PrismML model could not load, and advised to "install a build of
PrismML-Eng/llama.cpp" while running one. Measured before the fix:

```
check_types({0:353, 30:96, 142:402}, engine_type_count=144) -> ok=False
```

### The fix, and the mistake I made fixing it

`forks` should mean *fork types this build lacks*. My first attempt filtered with
`t < bound`, which is exactly inverted — `142 < 144` is true, so it kept precisely the
types the engine supports. The correct predicate is `>=`: the engine refuses any id at or
above its table size.

`above` needed the same clause. It flagged anything past `MAINLINE_MAX_TYPE` regardless of
the engine, so even with `forks` corrected, a bound-144 engine still reported the model
"was made for a different engine".

After the fix, the full matrix:

| `engine_type_count` | means | verdict on the Bonsai file |
|---|---|---|
| `None` | no engine context (mainline default) | not ok — needs a fork |
| `42` | pinned b9867 `[0,42)` | not ok — needs a fork, names PrismML |
| `43` | mainline master `[0,43)` | not ok — needs a fork, names PrismML |
| `144` | PrismML fork `[0,144)` | **ok** |

### A test was pinning the bug

`test_a_fork_type_is_never_excused_by_a_high_count` asserted `ok is False` for
`engine_type_count=144`, reasoning that "a fork build reporting COUNT = 144 still does not
make it mainline". That is true about the *name* and false about the *capability* — two
questions had been collapsed into one. The test now splits them: a fork type **is**
loadable by the build that defines it, and is still **named as a fork** for a build that
lacks it.

Worth recording: `docs/review/findings-r3/20-engine-compat-r3.md:207` already stated the
correct behaviour ("PrismML fork (COUNT 144) | the same model | **silent** — it loads").
The documentation and the test disagreed, and the test was wrong.

---

## R3-ENG-8 — "PQ2_0 has no Vulkan kernel" was false

`engine_compat.check_engine` appended, for `backend == "vulkan"`:

> note that PQ2_0 has no Vulkan kernel even on a PrismML build — use the group-64 Q2_0
> variant on Vulkan

Every clause is wrong, and the advice pointed at a format that will not load:

* PrismML's fork README lists **"Vulkan and SYCL backend support"** and carries a Vulkan
  row in its build table.
* Its release ships `bin-win-vulkan-x64`.
* **Measured**: that build loaded all 402 PQ2_0 tensors and decoded at ~52 t/s (below).
* `Q2_0` on older repos is the **deprecated legacy format** — the README says it "does not
  load on these builds". Recommending it was the one thing guaranteed not to work.

The claim came from reading the model card's list of *preferred* backends
(Metal/CUDA/HIP/CPU) as a list of *supported* ones, and from treating the existence of a
third-party Vulkan fork as evidence the kernel was missing.

---

## R3-ENG-9 — `engine_registry.select` had no callers

The module docstring explains the whole situation — the owner installed the fork by hand,
`ensure_engine` trusted a `.ready` sentinel, Rigma "could not name it, could not verify
it". Registration was built to fix that. But nothing in the launch path ever called
`select`:

```
$ grep engine_registry src/rigma/*.py
cli.py:715  for e in sorted(engine_registry.load().values(), ...)   # listing only
cli.py:840  engine_registry.register(...)
cli.py:859  engine_registry.forget(...)
```

So a registered engine was **describable but never selectable**, and `rigma up` still chose
the pinned b9867 build for a PQ2_0 model and died with `invalid ggml type 142. should be in
[0, 42)` — the exact failure registration was added to prevent.

**Fix:** `server_ops._registered_engine_for(gguf, backend)` reads the model's real tensor
types from its header and asks `engine_registry.select`. `perform_switch` prefers a
registered engine that covers every type in the file, and falls back to the pin otherwise.
Verified against the real install:

```
backend=vulkan  -> prism-b10743-vulkan
backend=rocm    -> prism-b10743-hip
```

### The first fix was incomplete, and only a real launch showed it

Wiring `perform_switch` was not enough. Rigma launches an engine from **three** places, and
each called `runtime.ensure_engine` directly:

| path | used by |
|---|---|
| `cli.up` (with its own fallback ladder) | `rigma up --model …`, `rigma up --detach` |
| `cli.sweep` | `rigma sweep` |
| `server_ops.perform_switch` | the UI's model switcher |

So after fixing only `perform_switch`, `rigma up --model <this model>` **still** launched the
pinned b9867 build, the pin refused type 142, and the fallback ladder silently served
SmolLM2-135M. Observed verbatim in `~/.rigma/logs/detached-11500.log`:

```
plan: ternary-bonsai-2-27b-uncensored-heretic-pq2-0 Q2_0 on vulkan (calculator+ctx-override)
argv: llama-server -m <model> --port 11499 … -ngl 99 -c 65536 … --cache-type-k q5_1 --cache-type-v q5_1
starting llama-server: ternary-bonsai-2-27b-uncensored-heretic-pq2-0 Q2_0 (first load can take minutes)...
llama-server failed to become healthy on :11499
falling back -> smollm2-135m-instruct Q2_K (fallback:floor)
```

Note the argv: **`q5_1`**, not the pinned `q8_0` — see R3-ENG-12. The unit tests all passed
while this was broken, because they tested the helper rather than the launch path. Only
running the command found it.

The fix is one exported seam, `server_ops.engine_binary_for(gguf, backend, os_name)`,
returning `(exe, record)`, used by all three. A test now asserts that `cli.py` contains no
direct `runtime.ensure_engine` call at all, since this failure was purely one of wiring.

**Verified end to end after the fix** — the launch that had failed:

```
C:\Users\dev\.rigma\engines\prism-b10743\vulkan\llama-server.exe
  -m …\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf
  --port 11499 -ngl 99 -c 65536 --parallel 2 --kv-unified -fa on
  --cache-type-k q8_0 --cache-type-v q8_0 …

load_model: loading model '…Heretic-PQ2_0.gguf'
model loaded
```

and through Rigma's own OpenAI API on `:11500`:

```
"What is 17 * 23?"                                  -> "391"        correct
"A farmer has 17 sheep. All but 9 run away…"         -> "9 sheep are left."  correct
```

with `reasoning_content` streaming separately, as `docs/AGENTS.md` describes.

Guards, each with a test: a registered engine that does *not* cover the file is not chosen;
one with unknown capabilities is not chosen (unknown is not a licence to pick); one for
another backend is not chosen; a truncated header read yields no opinion; a registry entry
whose binary has vanished yields no opinion; no registry at all yields no opinion; and the
pin is still returned when nothing registered fits.

The choice is recorded as `state["engine_binary"]` — a **separate** field, because
`state["engine"]` is the engine *runtime* (`llamacpp`/`vllm`), a string that `bench`/`hwid`
compare for equality when attributing calibration. Putting a description there would have
corrupted calibration provenance.

---

## R3-ENG-10 — Vulkan is faster, and I was wrong to say otherwise

The steer was "Use Vulkan, it's faster and better". I measured HIP first, got a good
number, and reported that HIP was better. **That was wrong, and the way it was wrong is
the useful part.**

Sequential measurements, same build, same flags, minutes apart:

| run | vulkan | hip |
|---|---|---|
| `llama-bench -r 5` | 55.91 ± 2.06 | 35.24 ± 21.56 |
| `llama-bench -r 12` | 46.65 ± 12.82 | 53.13 ± 2.80 |

The *same* Vulkan build measured 55.91 and then 46.65 — a 9 t/s swing. HIP measured 35.24
and then 53.13. Run-to-run drift dominated, so each sequential pass measured the drift as
much as the backend, and I over-read one of them.

**Interleaved A/B**, 6 rounds of alternating 4-rep blocks so drift lands on both builds
equally:

```
vulkan  mean=52.40  sd=2.86
hip     mean=49.20  sd=4.51
paired difference (vulkan - hip): mean=+3.20  sd=3.12  se=1.27  t=+2.51 on 5 df
per-round: [+1.08, +2.17, +6.85, +1.10, +0.51, +7.47]
```

**Vulkan won all six rounds.** Paired mean +3.20 t/s (~6.5%), `p < 0.05` (t = 2.51 vs the
2.015 critical value at 5 df) but not `p < 0.01`. The consistent direction across every
round is the strong evidence; the magnitude is modest and should not be quoted to more
than one significant figure.

Vulkan also wins on prompt processing in the interleaved HTTP test (~880 vs ~800 t/s for a
5k prompt), though a single `llama-bench -p 512` pass favoured HIP — another instance of
the same noise, and the reason the interleaved method is the one to trust.

`--cache-type-k q8_0 --cache-type-v q8_0`, `-ngl 99`, ctx 65536, all 64 layers on
`ROCm0`/Vulkan device.

### The GPU is not stable under sustained load

A `bench3` run collapsed to **0.14 t/s** decode and a 110-second warmup, on the same build
that had just done 55 t/s. Two causes were identified and one is a real trap:

* **Vulkan compiles shaders on the first call.** An unwarmed measurement reports
  compilation, not throughput. Every measurement here is warmed.
* The remaining variance is environmental (thermal or background) and is *not* a property
  of either backend — it affects both, which is exactly why the interleaved design was
  necessary.

---

## R3-ENG-11 — I was wrong: `_engine_type_count` works

**This section originally claimed `_engine_type_count` "always returns `None`" because
`llama-fit-params` "exits 0 and prints no error for a file the engine cannot load". Both
halves are false.** It is left here, corrected, because the wrong version was reported as
fact and the method error is the interesting part.

What I originally ran:

```
prism-vulkan        fit_params_bin -> ...\vulkan\llama-fit-params.exe
                    load_error=''   parse_type_count -> None
prism-hip           load_error=''   parse_type_count -> None
pinned-b9867-rocm   load_error=''   parse_type_count -> None
```

Every one of those three builds **loads this file**, so `None` was the correct answer and I
read it as a malfunction. `pinned-b9867-rocm` was not the mainline build I assumed it was —
the ROCm tree also holds a `rocm-mainline-b9867` variant — and I never checked which binary I
had actually pointed at.

Reading the source shows why the claim was wrong on its own terms:

* `tools/fit-params/fit-params.cpp` calls `exit(1)` when `common_fit_params` does not return
  `COMMON_PARAMS_FIT_STATUS_SUCCESS`. A refusal is a non-zero exit, not a silent zero.
* `_engine_type_count` never parses the fit oracle's *output*. It reads `FitResult.load_error`,
  which `memtruth` fills from `_LOAD_FAIL_RE` — a regex that already matches
  `invalid ggml type`. My probe inspected the wrong thing and I generalised from it.

Measured against the real mainline binaries and the real PQ2_0 file:

```
llama-fit-params, b9867/vulkan (mainline)   exit=1
  E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 42)
  E gguf_init_from_reader: failed to read tensor info
llama-fit-params, b9867/cpu    (mainline)   exit=1   (same)
llama-fit-params, prism/vulkan (fork)       exit=0   stdout: -c 512 -ngl 1

_engine_type_count(b9867/vulkan, model) -> 42
_engine_type_count(b9867/cpu,    model) -> 42
_engine_type_count(prism/vulkan, model) -> None
```

The fingerprint works exactly as its docstring claims: **42** for the pinned build, and
`None` for the fork because the fork *loaded* the file — which is the authoritative answer
"this engine can load it", and correctly suppresses the warning. The verdict that follows is
right too:

```
b9867/vulkan -> ok=False  this engine's type table ends at 42 …
prism/vulkan -> (no verdict: the engine loaded it)
```

There is no gap here. R3-ENG-2's "explain why this engine refuses your model" path fires
exactly when an engine refuses.

### What is genuinely unavailable, and should stay that way

An engine's type count is discoverable *only* by making it refuse a file. When an engine loads
a model successfully Rigma learns nothing about that engine's table — it cannot, and it does
not need to: a successful load is the strongest available evidence, and recording a bound for
a working engine would store a guess in a field that reads like a fact.

### The method error

I had a probe that called the wrong function against the wrong binaries, and I reported its
`None` as a property of the code rather than of the probe. The tell was there: the finding
said three different builds all returned `None`, including one whose whole purpose is to load
this model. A result that uniform should have prompted a second look before it was written
down as a latent gap.

---

## R3-ENG-12 — the cache policy, the VRAM reservation, and a measurement trap

### The budget is 14954 MB, not 9190 MB, and I first read it wrong

An early resolution reported `vram=9190MB` and I built a conclusion on it — that Rigma
reserves ~5.7 GB on a 16304 MiB card and therefore steps the cache down. Re-running after
stopping a benchmark server gave `vram=14954MB`. The 9190 MB reading was taken **while a
llama-server I had launched was still holding VRAM**; Rigma read the real free memory and
budgeted accordingly. The instrument was contaminated, not the arithmetic.

With the correct budget, q8_0 at ctx 65536 fits comfortably — `kv=2176MB` against
`vram=14954MB` — and the cache policy works as intended. The launch that previously used
q5_1 now uses q8_0 (argv verified, R3-ENG-9). **Nothing was wrong here.**

### Why the detached launch showed q5_1

The failed detached launch printed `--cache-type-k q5_1`. That was a *consequence* of the
engine bug, not a separate one: the resolver ran against the contaminated 9190 MB reading
and legitimately stepped the ladder down. With the pin set and the budget correct, the
resolved plan and the real argv both say q8_0. Recorded because the two symptoms looked
independent and were not.

### A measurement trap worth remembering

Rigma's VRAM figure is live free memory. Any leftover engine — mine, from a benchmark —
changes the plan it produces. Every measurement here that disagreed with another was
eventually traced to this or to GPU drift, never to a bug in the arithmetic.

### `_GROW_LAYER_BUDGET` buys context at a very steep price

The resolver grows context and is allowed to spend `_GROW_LAYER_BUDGET = 0.15` of a dense
model's layers doing it. For this 64-layer model that permits spilling up to 9 layers to
reach a larger window, and it chose ctx 131072 with 58/64 layers on the GPU over ctx 65536
with all 64. Measured, interleaved, 3 rounds each:

```
ctx 65536, -ngl 99    mean = 48.5 t/s  sd = 1.0   [49.4, 47.5, 48.6]
ctx 131072, -ngl 58   mean =  9.0 t/s  sd = 0.9   [ 8.0, 9.9, 9.0]
```

**Spilling 6 of 64 layers — 644 MB — costs 82% of decode throughput.** The module's own
comment already warns that "offloading half a gigabyte cost 60% of throughput"; this
measures 82% on the same kind of trade, and the two agree in direction and rough size. The
0.15 budget is therefore far more expensive for a dense hybrid model than its framing
suggests.

> **CORRECTED 2026-09-29 (R3-ENG-14).** The comparison above moved `-c` and `-ngl` together
> (65536/ngl 99 vs 131072/ngl 58), so it could not separate context from spill — the same
> confound this document flags a few lines down. The controlled runs in
> `36-partial-offload-cliff.md` fix the context and move only `-ngl`: 54.5 t/s at 0 CPU
> layers, 12.6 t/s at 7, 5.9 t/s at 17 — a 4.3x loss for 7 layers, not 82% for 6. The
> mechanism was also wrong: the cost is the CPU matmul throughput of the PQ2_0 weights
> (~9.2 ms per CPU layer at ~11.4 GiB/s), not a context-proportional KV cache read, because
> attention only ever runs over the occupied padded cells (`n_kv` = 512 for a ~260-token
> run). And **262144 fits fully on this card at q5_1/q5_1** (13016 MB against a 14954 MB
> budget); the spill at 262144 was caused by a pinned q8_0 cache policy, now removed.
> See `37-262k-context.md`.

This is **not changed here** — it is a global policy, and other models may genuinely prefer
the window. The model's spec is pinned to `launch.ctx = 65536` with `--ctx` available as an
explicit override, which re-fits `ngl` (verified: `rigma up --ctx 65536 --dry-run` produces
`-ngl 99 -c 65536 … --cache-type-k q8_0`). Worth a follow-up decision: whether 15% is the
right allowance for dense models at all.

### The hybrid KV math is correct

For the record, since it was a suspected bug and is not one:

```
f16   65,536 bytes/token  ->  4,096 MB at 65536
q8_0  34,816 bytes/token  ->  2,176 MB
q5_1  24,576 bytes/token  ->  1,536 MB
q4_0  18,432 bytes/token  ->  1,152 MB
```

`full_attn_layers = 16` is used, not `n_layers = 64`, which is the correct treatment of a
4:1 hybrid and was my first suspicion. For the same reason a comment in `resolve.py` that
says a hybrid's KV "costs 4x what a hybrid-attention model's actually does" is stale — the
code already handles it.

---

## Verification

* Full suite: `pytest tests` — exit 0, no failures.
* `tests/test_r3_engine_compat.py` — 36 pass, including the two rewritten ones.
* `tests/test_r3_engine_binary_choice.py` — 17 new tests pass, covering the helper, the
  shared seam, and the wiring assertion that `cli.py` no longer calls `ensure_engine`.
* `ruff check src tests` — clean.
* Live, through Rigma's own CLI and API: the launch that previously fell back to SmolLM2 now
  serves the right model on PrismML's Vulkan build with `-ngl 99 -c 65536` and q8_0 KV,
  answering `391` and `9` correctly with `reasoning_content` streaming.
* **Not verified:** whether the HIP 35 t/s figure is reproducible (it is inside the noise);
  whether `_engine_type_count` can be re-pointed at a build that actually refuses; whether
  the `--ctx` override re-fits correctly at contexts other than 65536.

### A note on the intermittent suite hang

A full-suite run appeared to hang for 6.5 minutes with no output and was killed. It was not
hung: PowerShell's `Select-Object` buffers the pipeline, so nothing is emitted until pytest
exits, and the suite takes ~8 minutes. `pytest-timeout` is still not installed (finding 20,
item 6), and `-o faulthandler_timeout=180` is the available substitute. The lesson is the
standing order's: wait on the artifact, not the pipe.
