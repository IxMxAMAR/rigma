# llama.cpp 5x partial-offload penalty: findings

> **PARTLY SUPERSEDED 2026-09-29 (R3-ENG-14).** The placement facts below are confirmed
> against the fork at `87268f77` and stand: offload is from the end, `ngl` counts the output
> layer, weights stay resident, and the PQ2_0 CPU kernel is vectorized. **Q1's "cliff" and
> Q3's arithmetic are refuted.** Attention does not read the whole context-proportional
> cache: `get_n_kv` returns the occupied, padded cell count (`n_kv` = 512 for a ~260-token
> run), so the KV term is ≤0.3 ms per CPU layer, not 45-70 ms/token. The cost is the CPU
> matmul of the PQ2_0 weights: a straight line, ~9.18 ms per CPU-resident layer at
> ~11.4 GiB/s (R² 0.99992), which is compute-bound rather than DRAM-bound. The source quotes
> below are from `master`; the pinned-commit equivalents are in
> `37-262k-context.md`. See also the correction banner in `36-partial-offload-cliff.md`.

Method: `web_search` was unavailable (no API key), so I used DuckDuckGo HTML + direct fetches. I verified llama.cpp internals by cloning **ggml-org/llama.cpp** (master) and the actual fork **PrismML-Eng/llama.cpp** (commit `87268f77`, 2026-09-28) and reading the source locally. Code refs below are from those clones.

## Premise corrections (source-verified)

**1. The offload direction is the opposite of the premise.** [llama-model.cpp:1585](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-model.cpp#L1585) computes `i_gpu_start = std::max(n_layer_all + 1 - n_gpu_layers, 0)`, and `get_layer_buft_list()` (`:1587-1597`) sends layer `il` to CPU when `il < i_gpu_start`. With 64 layers and `-ngl 58`: `i_gpu_start = 65-58 = 7`, so **layers 0-6 go to CPU; 7-63 plus the output layer go to GPU.** llama.cpp offloads from the END; CPU keeps the FIRST layers. The input embedding is always on CPU (`:1599-1601`).

**2. Weights do not cross PCIe per token.** They stay resident. With 7 leading CPU layers the boundary is crossed once per token carrying one hidden state: 5120 floats ~= 10 KB at f16. That is microseconds. The 0.63 GB is a one-time load cost. **PCIe transfer volume cannot explain 11.7 t/s.**

## Q1 - shape of the -ngl curve
I could **not** verify a published llama-bench `-ngl` sweep dataset - UNVERIFIED. But the source shows why the curve is a cliff rather than linear in layer fraction: **the KV cache follows the layer's device.** [llama-kv-cache.cpp:215-219](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-kv-cache.cpp#L215):

```cpp
ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
if (offload) { auto * dev = model.dev_layer(il); buft = ggml_backend_dev_buffer_type(dev); }
```

`offload` is `cparams.offload_kqv`, default true ([llama-context.cpp:3828](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-context.cpp#L3828)). So each CPU layer drags its **context-proportional** KV cache into host RAM - per-layer cost grows with `-c`. That is the cliff.

## Q2 - hybrids
I found **no** llama.cpp issue/PR specifically reporting hybrid/SSM models as pathologically slow under partial offload - UNVERIFIED (several phrasings returned nothing on point). Verified from source: hybrids get no special placement. [llama-memory-hybrid.cpp:34-65](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-hybrid.cpp#L34) passes one shared `offload` flag to both caches; [llama-memory-recurrent.cpp:83-90](https://github.com/ggml-org/llama.cpp/blob/master/src/llama-memory-recurrent.cpp#L83) places SSM state by the same `model.dev_layer(i)` rule. **The mechanism is shared with dense models, not hybrid-specific.** Hybrid nuance: only 16/64 layers own a KV cache, so context cost moved to CPU is quantized in 1/16 steps - one CPU-resident attention layer brings ~1/16 of the whole KV cache to host RAM.

## Q3 - the arithmetic (n_embd 5120, 131072 ctx, 4 kv_heads, head_dim 256)
- Activation across the single boundary: ~10 KB/token - negligible.
- CPU layer weights re-read per token: 6.7 GB x 7/64 = **0.73 GB/token**.
- Per CPU-resident attention layer, K+V per token: 131072 x 4 x 256 x 2 x 2 B = **1.07 GB** (f16).

Total **1.8-2.9 GB/token from system RAM**. At a realistic ~40 GB/s dual-channel figure that is 45-70 ms/token against 85 ms measured - the budget closes. Note this is **host RAM, not PCIe**: attention runs on the CPU where the cache lives, so the KV never crosses the bus. RX 9070 XT (Navi 48) is PCIe 5.0 per [Wikipedia, citing AMD](https://en.wikipedia.org/wiki/Radeon_RX_9000_series), but that is irrelevant here.

**Critical confound:** `-c` changed together with `-ngl` (65536 -> 131072). The KV term scales linearly with `-c`, so the two variables are entangled.

## Q4 - PQ2_0 CPU kernels: hypothesis FALSIFIED
The fork has a real vectorized CPU kernel. `ggml/src/ggml-cpu/ggml-cpu.c:240-245`:

```cpp
[GGML_TYPE_PQ2_0] = { .from_float = quantize_row_pq2_0, .vec_dot = ggml_vec_dot_pq2_0_q8_K,
                      .vec_dot_type = GGML_TYPE_Q8_K, .nrows = 1 },
```

Implementation at `ggml/src/ggml-cpu/arch/x86/quants.c:4365`, using AVX2/AVX-VNNI `dpbusd`. A comment at `ggml-cpu.c:1183-1184` states narrow rows "still [use] the AVX2 kernel rather than the generic scalar dot" - there is no dequantize-to-F32 fallback. Caveat: that kernel is annotated "tier 2, 2026-09-18", so an older build gets the slower Q8_0-activation path. I verified existence and vectorization, not relative speed.

## Q5 - threads (verified verbatim)
[token_generation_performance_tips.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/development/token_generation_performance_tips.md): "It's extremely important that this parameter is not too large. If your token generation is extremely slow, try setting this number to 1." Its table: `-ngl 2000000` alone -> **<0.1 t/s**; `-t 4 -ngl 2000000` -> **9.1 t/s**; `-t 7` -> 8.7. Guidance: `-t` = number of **physical** cores "even if you utilize a GPU". Effect can be ~100x. Cheap to rule out.

## Q6 - better strategies
`-ncmoe/--n-cpu-moe` is MoE-only - [common/arg.cpp:2756-2757](https://github.com/ggml-org/llama.cpp/blob/master/common/arg.cpp#L2756): "keep the Mixture of Experts (MoE) weights of the first N layers in the CPU". It does not help hybrids. `-ot/--override-tensor` (`:2743`) is a generic per-tensor buffer override and does apply. No documented hybrid-specific strategy found - UNVERIFIED.

## Best single-sentence explanation
The 5x is not PCIe weight transfer: `-ngl 58` pins the first 7 layers **and their context-proportional attention KV cache** to host RAM, so at 131072 context every token re-reads ~0.73 GB of 2-bit weights plus ~1-2 GB of KV from system RAM and runs that attention on the CPU - a cost that scales with context and is invisible in the "9% of layers" framing.

**Confidence:** high on the placement facts and on "PCIe weight transfer is not the mechanism"; medium on the exact 0.73/1.07 GB split (not measured).

**What would falsify it:** run `-c 65536 -ngl 58`. If it is also ~11 t/s, `-ngl` is the cause and context is irrelevant; if ~40 t/s, context length is the cause. Then `-c 131072 -ngl 58 -ctk q8_0 -ctv q8_0` should roughly halve the KV term. Also confirm from the startup log which layers went to CPU.
