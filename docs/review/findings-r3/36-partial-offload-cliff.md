# R3-ENG-13 — the partial-offload slowdown: measured, and the mechanism (corrected)

Follow-up to `35-prismml-install-and-corrections.md`, answering two challenges:

1. **The user's:** "the model is only about 6.8GBs and on 16GB VRAM Card there should be
   Plenty of space? for the Context and the Whole model itself"
2. **The user's:** "the 9tok/s at 131k sounds very off, even at 262K it shouldn't be that low"

> **CORRECTED 2026-09-29 (R3-ENG-14).** The measurements below stand; the *mechanism* this
> document first gave for them did not. It claimed the cost was a CPU-resident attention
> layer's context-proportional KV cache, read in full every token. Source verification
> (`llama-kv-cache.h/.cpp`, `llama-model.cpp`) shows llama.cpp attends only over the
> **occupied, padded** cells — `n_kv` = 512 for a ~260-token run, not 131072 — so the CPU
> attention term is ≤0.3 ms of the 9.2 ms each CPU layer costs. The cost is the **CPU matmul
> throughput of the PQ2_0 weights** (~107 MB/layer at ~11.4 GiB/s). The corrected derivation
> is in `37-262k-context.md`; the corrected comment is at `resolve._GROW_LAYER_BUDGET`. The
> first section is also corrected: **262144 fits fully on this card at q5_1/q5_1** — the
> spill was caused by a pinned q8_0 policy, now fixed.

Both challenges were productive. (1) is correct and exposes a mistake of mine; (2) is
correct that my *explanation* was wrong, though the measurement itself holds up under
repetition.

---

## The answer to (1): it fits, and the spill was a pinned cache policy

Rigma's own fit function, asked with the model's **pinned** `q8_0/q8_0` policy, keeps all 64
layers on the GPU up to ctx 229376 and spills from 245760 on:

```
usable_vram = 14954 MB     model = 6872 MB
=> KV headroom for all layers = 8082 MB

     ctx  pinned q8_0         kv   model+kv   vs budget
  131072  ngl 99, 0 CPU     4352M     11224M      -3730M
  163840  ngl 99, 0 CPU     5440M     12312M      -2642M
  196608  ngl 99, 0 CPU     6528M     13400M      -1554M
  229376  ngl 99, 0 CPU     7616M     14488M       -466M
  245760  ngl 63, 2 CPU     8160M     15032M        +78M
  262144  ngl 58, 7 CPU     8704M     15576M       +622M
```

The spill at 262144 is **not** what the card requires — it is what the *pin* requires. The
resolver has a ladder (q8_0 → q5_1 → q4_0) that it tries before spilling any layer; a pinned
policy disables the ladder, so only q8_0 is ever considered. Remove the pin and the same
context keeps every layer on the GPU:

```
     ctx  unpinned ladder     kv   model+kv   vs budget
  245760  ngl 99, q5_1/q5_1  5760M     12632M      -2322M
  262144  ngl 99, q5_1/q5_1  6144M     13016M      -1938M
```

`q5_1/q5_1` at 262144 is 13016 MB against the 14954 MB budget — **1938 MB of headroom, all
64 layers on the GPU**. The pin's stated reason ("measured on this card: ... ~52 tok/s, so
q8_0 is known to fit") came from a VRAM reading taken while a benchmark server held VRAM and
was later retracted. R3-ENG-14 removed it (`cache_type_policy.pinned: false`, with a
timestamped `.bak` beside the spec) and made `fit_for_launch` treat a requested cache type as
a **ceiling** rather than a hard pin, so a stored `launch.kv: q8_0` can no longer re-impose a
cache the fit just rejected. The strict pass still picks q8_0 whenever it fits, so the pin
only ever removed options.

`resolve()` with no override and the pin removed returns **ctx=262144, ngl=99, q5_1/q5_1** —
fully on the GPU. With the pin it stopped at ctx 131072, q8_0 (also fully resident, 54.5 t/s).

---

## The answer to (2): the measurement is real, my explanation was not

Repeated in one server, six consecutive generations, ctx 131072, ngl 58:

```
  13.03  13.33  13.12  10.59  12.76  12.73 t/s
  mean 12.59   sd 1.01   min 10.59   max 13.33
```

So ~12.6 t/s is stable, not drift. The same server at ngl 99, same context:

```
  54.52 t/s (260 tokens)   — and 209.74 t/s on a short run
```

**4.3x from spilling 7 of 64 transformer layers.** The number stands. My *explanation* did
not:

| my claim | verdict |
|---|---|
| The spilled weights cross PCIe every token | **false** — weights stay resident; one ~10 KB hidden state crosses per token |
| ~0.63 GB of weights per token is re-read | **false** — that is a one-time load |
| `PQ2_0` may have no CPU kernel, falling back to a slow path | **false** — it has a vectorized AVX2/AVX-VNNI one |
| Thread oversaturation (llama.cpp's documented trap) | **ruled out** — 10 physical cores, 10 threads, `n_threads = 10` |
| The KV cache follows its layer's device | **TRUE — verified in source**, but not the cost |
| One CPU attention layer re-reads its **whole context-proportional** cache every token | **FALSE (R3-ENG-14)** — attention runs over occupied, padded cells only |
| CPU attention over that cache is the dominant term | **FALSE (R3-ENG-14)** — ≤0.3 ms of 9.2 ms per layer |
| The cost is a "cliff" in fraction of layers | **false** — it is a straight line in CPU layers |
| `PQ2_0` has no scalar fallback | **imprecise** — a generic scalar `#else` exists, but is not taken on x86-64 |

### The corrected mechanism, verified in the fork's own source

Three facts, each from `PrismML-Eng/llama.cpp` at `87268f77`:

**(a) Attention sees occupied cells, not the allocated window.** `get_n_kv`
(`src/llama-kv-cache.cpp`) returns
`min(cells.size(), max(n_pad, 256, GGML_PAD(used_max_p1, max(n_pad, 256))))`, where
`used_max_p1` is the highest occupied cell + 1. `get_k`/`get_v` build their views with
`ne[2] = n_kv`; `get_size()` is only the stride. The header says it outright:

```cpp
// a heuristic, to avoid attending the full cache if it is not yet utilized
// as the cache gets filled, the benefit from this heuristic disappears
int32_t n_kv;
```

For a ~260-token run `n_kv` is 512 (padded to 256). The 131072-token allocation is charged
against VRAM by the fit, but it is never attended over.

**(b) Offload is from the END, and `ngl` counts the output layer.**
`src/llama-model.cpp`: `i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0)`; layers
`il < i_gpu_start` go to the CPU; the output layer is placed by the same rule and is
explicitly decremented out of `n_repeating`. So with 64 layers:

```
-ngl 99  -> i_gpu_start 0   ->  0 CPU layers
-ngl 58  -> i_gpu_start 7   ->  7 CPU layers   (doc 36 first said 6)
-ngl 48  -> i_gpu_start 17  -> 17 CPU layers   (doc 36 first said 16)
```

**(c) The cost is a straight line.** With those counts, the three measured points fit

```
ms/token ~= 0.280 x GPU layers + 9.18 x CPU layers     (R^2 0.99992, RMS residual 0.54 ms)
```

i.e. ~0.28 ms per GPU layer and **~9.18 ms per CPU layer**, intercept ≈0. Each CPU-resident
layer streams ~107 MB of PQ2_0 weights (2.125 bpw, 128 values/block) through the CPU 2-bit
unpack + `dpbusd` kernels at **~11.4 GiB/s**. That is far below desktop DRAM (40–80 GB/s),
so the kernel is **compute-bound**, not bandwidth-bound — and the CPU attention term at
`n_kv=512` is ~1.1 MB/layer/token (≤0.09 ms at that rate) plus ~12.6 MFLOP of QK+PV
(≤0.25 ms), i.e. **≤4% of the 9.18 ms**. The earlier doc's KV arithmetic also used `34/64`
bytes per q8_0 value; the correct block is `34/32` — it was 2x understated.

So the slowdown is not context-sized and not PCIe: it is what a 2-bit CPU matmul costs on
this machine, and it is linear in the number of CPU-resident layers.

---

## Where the offload boundary falls

llama.cpp offloads from the **end**; the CPU keeps the FIRST layers. With 64 layers and
`-ngl 58`, layers 0-6 (seven) stay on the CPU and layers 7-63 plus the output go to the GPU.
I had assumed the opposite. For a hybrid with `full_attention_interval = 4`, the first 7
layers contain attention layers 3 and 7, so the CPU holds at least one attention layer — and
its cache — but per (a) that cache is not what the 9 ms is spent on.

---

## A measurement I do not trust

`ctx 262144 ngl 99` reported **11.44 t/s**, and `ctx 262144 ngl 58` reported **12.18 t/s**.
`fit_gguf` says ngl 99 does not fit at 262144 with a q8_0 cache (15576 MB against a
14954 MB budget) — that is the pinned-cache configuration of section (1). The likely reading
is that llama.cpp accepted the allocation anyway — the documented Windows trap, where WDDM
pages VRAM to system RAM and prints no error — so that run measured paging, not offload.
**Recorded as unreliable rather than explained.** With q5_1/q5_1 the same context fits with
1938 MB to spare, so the 262144 measurement should be repeated on that configuration.

The earlier "48.5 t/s at 65536 vs 9.0 at 131072" was not a controlled comparison: `-c` and
`-ngl` moved together, so it could not separate context from spill. The controlled runs now
say:

```
ctx 131072, ngl 99 (0 CPU layers)   54.5 t/s
ctx 131072, ngl 58 (7 CPU layers)   12.6 t/s
ctx 131072, ngl 48 (17 CPU layers)   5.9 t/s
```

which is a monotone curve in **CPU-resident layers at fixed context** — linear, with no
context confound.

---

## What changed in the code

R3-ENG-13 added a comment only. **R3-ENG-14** (`37-262k-context.md`) adds the actual fix:

* `resolve.fit_for_launch` — a requested cache type (explicit `kv` or a stored launch
  default) is a **ceiling** fitted in before placement, never re-applied after it. A stored
  `pinned` flag is read as "no opinion" on the launch path.
* `resolve.fit_gguf` / `quant_verdicts` — a pinned policy that spills now says which rung
  would have fit and how many layers the pin costs.
* `server_ops.perform_switch` and `cli.up` use it, so ctx 262144 + a stored `kv: q8_0` yields
  the all-GPU q5_1 plan with a notice, not a 622 MB page to system RAM.
* The owner's spec is unpinned (`.bak` beside it).

---

## Verification status

* Verified: the fit boundary tables (Rigma's own `fit_gguf`, both policies); 12.59 t/s mean
  over six runs with sd 1.01; 54.52 t/s reproduced; `i_gpu_start` and the output-layer count
  in `llama-model.cpp`; `get_n_kv` padding and the `n_kv` heuristic in
  `llama-kv-cache.cpp/.h`; KV-on-layer-device in the same file; the PQ2_0 CPU `vec_dot` in
  `ggml-cpu.c`/`arch/x86/quants.c`; 10 physical cores and `n_threads = 10`.
* Corrected here: the mechanism (CPU attention over the allocated cache), the layer counts
  (7 and 17, not 6 and 16), the q8_0 bytes/value (34/32, not 34/64), and the claim that the
  curve is a cliff.
* Not verified: the exact effective rate of the PQ2_0 CPU kernel against a published figure
  (none exists — UNVERIFIED); the 262144 readings above, recorded as unreliable; whether the
  same curve appears on HIP (only Vulkan was measured).
* Method note: `web_search` is unconfigured in this session, so research went through
  DuckDuckGo HTML and direct `raw.githubusercontent.com` fetches of the fork at `87268f77`.
