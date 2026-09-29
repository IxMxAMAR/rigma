# R3-ENG-18 — 262K context on the 16 GB card: it fits, it was pinned out, and what it will run at

Answers the owner's two challenges from the earlier session, both of which were right:

> "the model is only about 6.8GBs and on 16GB VRAM Card there should be Plenty of space?
> for the Context and the Whole model itself"
>
> "the 9tok/s at 131k sounds very off, even at 262K it shouldn't be that low"

Everything below is either **MEASURED** (already in `35-`/`36-`, or in
`~/.rigma/calibration.json`) or **PREDICTION** (arithmetic from a measured constant), and each
row says which. **No GPU work was run for this document** — the card was busy, so the
measurement half is `37a-262k-verification-plan.md`.

---

## 1. The answer: 262144 fits fully, and the spill was a pinned cache policy

The resolver's own fit, on the owner's card (`usable_vram = 14954 MB`; 16304 MiB total, desktop
on the iGPU, 150 MB compute reserve), model file 6872.3 MiB:

**With the spec as it was — `cache_type_policy.pinned: true`, q8_0/q8_0** (REPRODUCED from
`fit_gguf`; the mechanism is `_cache_candidates` returning only q8_0 when pinned):

```
     ctx   ngl   CPU layers   kv MB   model+kv   vs budget
  131072    99            0    4352      11224       -3730
  163840    99            0    5440      12312       -2642
  196608    99            0    6528      13400       -1554
  229376    99            0    7616      14488        -466
  245760    63            2    8160      15032         +78   <- spills
  262144    58            7    8704      15576        +622   <- spills, ~4x slower
```

**Unpinned — the ladder q8_0 -> q5_1 -> q4_0 is tried before any layer moves:**

```
     ctx   ngl   CPU layers        kv      kv MB   model+kv   vs budget
  245760    99            0   q5_1/q5_1    5760      12632       -2322
  262144    99            0   q5_1/q5_1    6144      13016       -1938
```

**q5_1/q5_1 at 262144 leaves 1938 MB of headroom with all 64 layers on the GPU.** The context
was never the card's limit; the pin was. `resolve()` on the unpinned spec now returns
`ctx=262144 ngl=99 kv=q5_1/q5_1` (REPRODUCED offline).

The pin's stated reason was a VRAM reading taken while a benchmark server held VRAM, and was
retracted in `35-`. It also violated a documented invariant — `CachePolicy.pinned` is the
Models-page explorer's knob and "must never be on a stored spec". The owner's spec is now
unpinned (`cache_type_policy.pinned: false`, timestamped `.bak` beside it), and the launch path
treats a requested cache type as a **ceiling** rather than a hard pin (commit `e520b06`).

There was a second, worse half. `server_ops.perform_switch` and `cli.up` applied a stored
`launch.kv` **after** the fit, so even an unpinned policy could be defeated: the resolver fitted
the fully-resident q5_1, then the launch path overwrote it with q8_0 — an argv the fit had just
rejected. WDDM accepts that allocation, pages ~622 MB to system RAM, and prints no error. That
is the "9 t/s" the owner challenged, and it is now impossible: the requested type goes *into*
the fit (`resolve.fit_for_launch`), and a step-down is reported in the plan explain, in the
returned state as `notice`, and on stderr.

---

## 2. The measurement stands; the earlier explanation did not

Repeated in one server, six consecutive generations, ctx 131072, ngl 58 (MEASURED, doc 36):

```
  13.03  13.33  13.12  10.59  12.76  12.73 t/s
  mean 12.59   sd 1.01   min 10.59   max 13.33
```

So ~12.6 t/s is stable, not drift. All-GPU at the same context: **54.52 t/s** (MEASURED). That
is 4.3x for 7 of 64 layers. The number was right; the reason given for it was wrong, and
`36-partial-offload-cliff.md` now carries a correction banner. The verified mechanism:

* **Attention runs over occupied cells, not the allocated window.** `get_n_kv`
  (`llama-kv-cache.cpp`) returns `min(cells.size(), max(n_pad, 256, GGML_PAD(used_max_p1,
  max(n_pad, 256))))`; `get_k`/`get_v` use `ne[2] = n_kv`. For a ~260-token run `n_kv` is 512,
  and `llama-kv-cache.h` calls it *"a heuristic, to avoid attending the full cache if it is not
  yet utilized"*. The 131072-token allocation is charged to VRAM but never attended over at
  short depth. The CPU attention term is <=0.3 ms.
* **The cost is the CPU matmul.** Three points, one line (MEASURED):
  `ms/token ~= 0.280 x GPU layers + 9.18 x CPU layers`, R^2 0.99992, RMS residual 0.54 ms.
  ~107 MB of PQ2_0 weights per layer at **~11.4 GiB/s** — compute-bound (desktop DRAM is
  40-80 GB/s), not bandwidth-bound. The corrected comment is at `resolve._GROW_LAYER_BUDGET`.
* **`ngl` counts the output layer.** `i_gpu_start = max(n_layer_all + 1 - n_gpu_layers, 0)`
  (`llama-model.cpp`), so `-ngl 58` leaves **7** layers on the CPU and `-ngl 48` leaves **17** —
  doc 36 said 6 and 16. `_spilled` reports the *weight* fraction (6/64); the two are different
  numbers. The off-by-one in `_spilled`'s display is a known open item.

---

## 3. What 262K will actually run at (PREDICTION)

Constant: **BW_eff = 374.66 GB/s**, from the one fully-anchored measurement — 6.872 GB /
18.3419 ms/token at 54.52 t/s (MEASURED, doc 36) = 58.5% of the 640 GB/s nominal. Per-token KV
traffic = 16384 x (CB[k] + CB[v]) bytes with 16384 = 16 full-attention layers x 4 kv_heads x
256 head_dim; occupancy = **depth** x per-token bytes, not ctx x per-token.

```
t/s at depth        0        32768      131072     245760     (fully filled)
q8_0/q8_0        54.52 M    46.76 P    32.76 P    24.28 P *   M = MEASURED anchor
q5_1/q5_1        54.52      48.80      37.12      29.02       P = PREDICTION
q4_0/q4_0        54.52      50.12      40.34      32.86
q8_0/q4_0        54.52      48.38      36.16      27.93       best asymmetric
* does not fit at 245760: 15032 MB against 14954, doc 36 spills 1 layer.
```

**The single most important unmeasured cell is q8_0/q8_0 at a fully filled 131072: predicted
32.76 t/s, 60% of the anchor (-40%), and nobody has ever measured it.** It fits (11224/14954),
it is the resolver's default, and it is the first thing to run when the card is free.

Error bars are **+/-20-30%** on every PREDICTION. The slope is optimistic: llama.cpp issue
#24483 on this same card measured depth costing *more* than KV bytes alone (observed -35.7% at
50k against -21.4% predicted from KV, with 39% run-to-run variance), so treat the table as an
upper bound at depth. External cross-checks (all URLs in `.scratch/r3-37/a4-depth-speed.md`):
an RX 7900 XTX on the same qwen35 27B measured tg128 37.09 -> 36.33 t/s at depth 4096 (-2.05%,
vs -0.8..-1.5% predicted); a PrismML PQ2_0 run on an RTX 5090 hit 135.41 t/s = 973 GB/s (54% of
nominal), consistent with the 58.5% effective-bandwidth fraction here.

Calibration note: `~/.rigma/calibration.json` has `ternary-bonsai-2-27b:PQ2_0:rocm` (the base
model, not the Heretic variant) at **35.64 t/s** with prompt 2048 / gen 128 on engine `b9867`
rocm, 12495 MB VRAM, 2026-09-21 (MEASURED). That is a different engine and a different file, so
it is context, not a control.

---

## 4. Which cache at 262K (PREDICTION, from published measurements)

Rigma runs K and V at the same type. llama.cpp PR 7412's **symmetric** K=V perplexity cost over
f16 (the figures now in `quant_quality._KV_PPL`, corrected in `6e916e5`):

```
q8_0  +0.035%    q5_1  +0.430%    q5_0  +0.749%    q4_1  +1.770%    q4_0  +3.309%
```

K is more sensitive than V (q4_0: K +3.03% vs V +0.31%), which is why the K-only column looked
different — an earlier version of the table mixed the two and understated q4_1/q4_0 by ~2x.
Asymmetric pairs are still skipped (B3): on Vulkan at 87268f77 a mixed pair *is* fused, but on
the CUDA/HIP path it is not, Rigma runs both, and the Vulkan path dequantizes both sides to an
F16 scratch anyway when they are dense — so the quality margin of acting is zero-to-negative.

**The pick at 262144 is q5_1/q5_1**: 6144 MB, 1938 MB of slack, fused flash-attention on both
backends, +0.43% (and lower with rotation, below). q8_0 is preferable whenever it fits — at
<=229376 it does, fully resident — and the ceiling logic now makes that automatic.

**Rotation changes the numbers in our favour.** The fork at 87268f77 enables KV rotation (the
Walsh-Hadamard transform of PR #21551) **by default** for a quantized cache with
`head_dim % 64 == 0` (`llama-kv-cache.cpp:315-339`); there is no CLI flag, only the env
`LLAMA_ATTN_ROT_DISABLE=1`, which Rigma never sets. Measured with rotation: q4_0/q4_0 +0.19% on
llama-2-7B (vs +2.42% without) and +1.22% on gemma3-4b; q8_0 +0.02%. PR 7412 predates rotation,
so the table above is an **upper bound** on the loss for a quantized cache on this build. Long
-context needle/RULER evidence for these types is **UNVERIFIED**.

---

## 5. Flash attention and mixed K/V (E3, VERIFIED in source)

The pinned commit `87268f77` and the shipped release `prism-b10743-adfffbe` have SHA256-identical
`ggml-vulkan.cpp`, `llama-context.cpp`, `fattn.cu` and `ggml/CMakeLists.txt`, so the source read
is the shipped source.

* **Vulkan accepts K and V independently.** Two separate `fa_kv_ok()` calls
  (`ggml-vulkan.cpp:18321-18339`); only a BF16/non-BF16 mix is rejected. The shaders are
  genuinely asymmetric (`FaTypeK`/`FaTypeV` are separate spec constants, and
  `flash_attn_dequant.glsl` says "Asymmetric K/V flash attention"). So the old comment "fused FA
  needs ctk == ctv" is **refuted for Vulkan**.
* **It is still true for CUDA/HIP.** `fattn.cu:442-446` selects no kernel for `K->type !=
  V->type` without `GGML_CUDA_FA_ALL_QUANTS`, which the shipped PrismML HIP build does not define
  (verified in the binary); HIP also rejects q4_1/q5_0/q5_1 even symmetric. Rigma runs a ROCm
  backend, so `_symmetric_kv` keeps a real purpose there.
* **`-fa on` with an unsupported pair falls back to CPU attention silently.** `-fa auto` runs
  the fork's probe and auto-disables with a warning; `-fa on` skips the probe and the scheduler
  assigns the unsupported node to the CPU backend. A quantized V cache rewrites `-fa auto` to
  ENABLED, so it behaves like `-fa on`.

Rigma does not need to change its symmetric rule; the comments in `models.py`, `serve.py` and
`resolve.py` now give the per-backend reason instead of the false universal one (`6e916e5`).

---

## 6. What changed, and where

| commit | what |
|---|---|
| `e520b06` | R3-ENG-14: `fit_for_launch`; the requested cache is a ceiling; pinned-spill explain; `_GROW_LAYER_BUDGET` corrected; 11 new tests |
| `294616c` | R3-ENG-15: the growth policy's explain line and the Models-page tooltip price a CPU layer at the measured cost |
| `dafc068` | R3-ENG-16: doc 36's mechanism, layer counts and bytes/value corrected; doc 35's confounded 82% annotated; doc 36a partly superseded |
| `6e916e5` | R3-ENG-17: `_KV_PPL` re-sourced to PR 7412's symmetric column; the flash-attention rationale corrected per backend |
| (outside the repo) | the owner's spec unpinned, `.bak-20260929-235923` beside it |

Not changed, deliberately: `_GROW_LAYER_BUDGET` stays 0.15 (a global default should not move on
one machine's measurement); `launch.ctx = 65536` and `launch.kv = q8_0` stay as the owner's idle
preference, now safe as a ceiling; asymmetric K/V stays skipped.

---

## 7. What is not verified

* Everything in section 3 beyond the depth-0 q8_0 anchor is PREDICTION.
* The 262144 readings in doc 36 (11.44 / 12.18 t/s) remain unreliable — they were taken with the
  over-budget q8_0 cache, so they measured WDDM paging. The repeat belongs in `37a`.
* Whether the depth slope is as shallow as the table predicts on this card (issue #24483 says no).
* HIP: `-fa on` and mixed K/V behaviour is source-verified but not measured here; whether RADV
  exposes subgroup shuffle/vote on gfx1201 is a PREDICTION.
* Rotation's effect on *this* model and *this* context length is unmeasured.
* The PQ2_0 CPU kernel's ~11.4 GiB/s has no published counterpart (UNVERIFIED against outside
  numbers; it is derived from the three in-house points).

## Provenance

Source: `PrismML-Eng/llama.cpp` at `87268f77` via `raw.githubusercontent.com` (`llama-kv-cache.cpp`,
`llama-kv-cache.h`, `llama-model.cpp`, `llama-context.cpp`, `ggml-vulkan.cpp`, `fattn.cu`,
`ggml-cpu.c`, `arch/x86/quants.c`, `tools/llama-bench/llama-bench.cpp`). Research: DuckDuckGo HTML
plus direct fetches (no search API in this session). Measurements: `35-`, `36-`, and
`~/.rigma/calibration.json` (metadata only). Full research notes:
`.scratch/r3-37/a1-e2.md`, `a2-e3.md`, `a3-cache-quality.md`, `a4-depth-speed.md`.
