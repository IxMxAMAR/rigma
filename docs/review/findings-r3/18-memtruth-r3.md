# R3-MEM-1 — Rigma's VRAM plan was never checked against the engine

**Severity: HIGH (silent, and it is the failure the tool exists to prevent)**
**Status: fixed** — `src/rigma/memtruth.py`, `tests/test_r3_memtruth.py`,
`rigma plan --verify`, `rigma up --verify`.

## The defect

`resolve.py` predicts whether a model fits in VRAM with a closed-form formula
over GGUF metadata:

    kv_bytes_per_token = full_attn_layers * kv_heads * head_dim * (bytes_k + bytes_v)
    kv_mb = ctx * kv_bytes_per_token / 2**20 + swa_kv_bytes(...)

Nothing has ever compared that number to reality. The formula is the *entire*
basis for the central decision the tool makes, and it is unvalidated. Its own
source admits one term "errs toward overcommitting the card rather than refusing
a plan" — so the known direction of the error is the dangerous one.

**Cause:** the arithmetic was written when there was nothing to check it against.
The pinned llama.cpp build ships `llama-fit-params`, which does a no-alloc dummy
load (`no_alloc = true`, `load_mode = LLAMA_LOAD_MODE_NONE`), asks the backend for
exact per-device accounting, and compares it against real free device memory. It
was on disk the whole time and nothing called it.

**Why the margin is real and not theoretical:** the research found models where a
single global product *cannot* be right — one gpt-oss model produces TWO KV caches
in one load (384 MiB non-SWA + 24 MiB SWA; llama.cpp#18996), and the SWA cache is
sized by the window, independent of ctx. Rigma's formula has one term for that.

## What was implemented

`memtruth.py` runs the oracle for the plan Rigma is about to launch and reads:

```
common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + ( 899 =    82 +     720 +      97) +        -735 |
common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed
```

`model + context + compute` is what Rigma has been estimating by hand; `free` is
what it has been assuming. `rigma plan --verify` and `rigma up --verify` print
both and name the gap when there is one.

**It is a verification, not a planner, and that is deliberate.** llama.cpp only
auto-adjusts arguments the caller left UNSET, and Rigma sets `-c` and `-ngl`
explicitly (verified: the oracle echoes `-c 32768 -ngl 99` back unchanged). So the
oracle reports on Rigma's decision rather than making it — which is the useful
direction, because Rigma decides and then asks whether the decision fits.

### The Windows trap, and why the check is arithmetic

WDDM — and NVIDIA's Sysmem Fallback Policy (driver branch 536.40+) — lets an
allocation LARGER than dedicated VRAM **succeed** out of system RAM. The model
then runs over PCIe at a fraction of the speed and llama.cpp prints **no error**.
llama.cpp has no detection, warning, or env var for it, and its free-VRAM reading
comes from `cudaMemGetInfo`, which reports dedicated VRAM only.

So on Windows *"the server started" is not evidence that the model fits.* Rigma
therefore does not trust the fit verdict: if what the engine says it will hold
exceeds what it says is free, that is a refusal regardless. That check cannot be
fooled by the driver handing out memory the card does not have, and it is pinned
by `test_a_claimed_fit_that_exceeds_free_memory_is_refused`, whose fixture has the
oracle reporting success while holding 9000 MiB of 4000 MiB free.

### And then the ground under that check moved

Chasing a note from the owner that ComfyUI had been running during the earlier
measurements produced a worse finding, and it is the important one.

With a real `llama-server` holding ~13 GB of a 16 GB card, the Windows GPU
performance counter read **13,149 MiB dedicated in use**, and `llama-fit-params`
reported at the same moment:

```
common_memory_breakdown_print: | - ROCm0 (RX 9070 XT) | 16304 = 16140 + (899 = 82 + 720 + 97) + -735 |
common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed
```

**16,140 MiB free on a card with 3,155 MiB free — and its own fit verdict said
"fits".** Whatever `hipMemGetInfo` returns on this driver, it is not available
VRAM. Reproduced deliberately under load:

```
verify:  engine measures 359 MiB (model 82 + context 180 + compute 97) against 3917 MiB free of 16304 MiB
         engine's own fit verdict: UNRELIABLE (see below)
         the engine reports 15,416 MiB free but the OS says 3,918 MiB is free
         (12,386 MiB of 16,304 MiB is in use by other processes) — the engine's fit
         verdict was made against the wrong number, so only its accounting is usable here
```

This is the AMD/Windows form of the same trap. The consequence is that the
engine's fit **verdict** is not evidence on this platform and only its per-device
**accounting** is usable. Rigma cross-checks `free` against the OS counter,
believes the OS, rewrites `free`, re-runs the overcommit check against it, and
marks the verdict UNRELIABLE rather than printing "fits" over a warning. When the
two agree the engine's figure is left untouched — the cross-check corrects a wrong
reading, it does not substitute its own.

The arithmetic check above is therefore still the right *shape* — do not trust a
verdict, compare numbers — but the numbers have to come from the OS, not only from
the engine.

### A correction to my own earlier reading

I first reported the earlier `--verify` runs as "contaminated by ComfyUI". That was
half wrong. `gpu_used_mb()` reads **residency**, not allocation: with ComfyUI alive
but idle it reads **22 MiB**, and during an active generation it read
**12,586 MiB**. The earlier runs were taken while ComfyUI held nothing, so their
`free` figures were right at that instant. The 16,140-vs-3,155 measurement is the
valid one and it is the one that matters.

## The oracle also caught a plan Rigma's own engine cannot serve

Measured on the owner's machine: the resolver planned `ternary-bonsai-2-27b`
(PQ2_0, ctx 262144) and the pinned engine answered

```
gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 42)
```

Build b9867 accepts ggml types `[0, 42)`; PQ2_0 is type 142. **Rigma planned a
model its own pinned engine cannot load.** That is a separate defect
(engine-pin/quant incompatibility) and is reported, not fixed here — but it was
invisible until the oracle ran, which is the argument for the oracle.

## A correction I have to make

The parallel/KV research reported that Rigma passes `--parallel 2` without
`--kv-unified`, so unified stays off and the KV pool doubles — contradicting
Rigma's own comment. **That finding was wrong, and my first test of it was
wrong in the same way.**

I measured `kv_unified = 'false'` by launching `llama-server` with `--parallel 2`
and *no* `--kv-unified` flag — i.e. I tested a command line Rigma does not use.
Rigma adds the flag in `_launch_extra`, not in `ModelSpec.server_args`, so
grepping the latter missed it. With Rigma's real argv:

| flags | `n_ctx_slot` | total KV |
|---|---|---|
| `--parallel 2 --kv-unified` | 32768 | 32768 (one pool) |
| `--parallel 2 --no-kv-unified` | 16384 | 32768 (two pools of 16384) |

So the flag IS applied, `-c` is the TOTAL across slots, the total KV does **not**
multiply by slot count, and the comment's claim is correct as written. The
research agent's caveat — "whether Rigma's bundled build matches current master" —
was the right thing to flag, and it is where the error was.

## Deliberately NOT built: the closed loop

The obvious alternative — measure VRAM after launch, store the delta, correct
future predictions — was researched and rejected on evidence:

* Windows/WDDM exposes no per-process GPU memory at all; a device-level sample is
  a sample, never a peak.
* AMD's own tracker has a measured case where `amd-smi` `USED_VRAM` moved **1 MB
  for a 2 GB allocation** while `hipMemGetInfo()` correctly showed 2058 MB.
* `--fit` returns different answers on the same machine and model before and after
  a sleep/resume (llama.cpp#26401), so a persisted delta would be applied to a
  baseline that had already moved.

The oracle is per-launch and reads the live device, so none of that applies.

## Verification

* 32 tests in `tests/test_r3_memtruth.py`, all fixtures taken VERBATIM from the
  pinned build's real output on this machine.
* Mutation-checked: dropping the overcommit guard fails 2 tests; making
  `planned_mb` return a silent `0.0` fails the guard that exists because an
  earlier version did exactly that.
* A live test runs the real binary against a real model, so a llama.cpp upgrade
  that renames these strings is caught rather than silently un-parsed. It passes.
* Live on the owner's install: `plan --model smollm2-135m-instruct --verify` →
  `engine measures 359 MiB (model 82 + context 180 + compute 97) against 15416 MiB
  free of 16304 MiB` — and Rigma's formula independently predicts 82 + 180 = 262,
  so the compute buffer is real (~97 MiB) and unbudgeted.
* `plan --model ternary-bonsai-2-27b --verify` → names the ggml type 142 failure.

## Open, and named as open

1. **`--log-jsonl` is not in the pinned build** (verified: it answers
   `error: invalid argument`), so the structured path in `parse_fit_jsonl` is
   written from the documented schema and exercised only by tests against that
   schema. It has never run against an emitting build. The stderr table parser is
   the one verified live; the JSONL path becomes primary if a future pin gains
   the flag.
2. **The compute buffer is still not modelled by the resolver.** The oracle now
   shows it (~97 MiB here, larger at bigger ctx/batch), but `resolve.py` does not
   budget it. Folding the measured value back into the plan is the natural next
   step and is not done.
3. **The formula was not replaced.** On `smollm2-135m` it is exactly right
   (180 MiB predicted, 180 measured at ctx 8192). The evidence says it is wrong
   for dual-cache/SWA models, but I did not find such a model locally to measure,
   so that remains the research's finding rather than my measurement.
4. **Engine-pin vs quant type** (ggml 142) is reported, not fixed.
