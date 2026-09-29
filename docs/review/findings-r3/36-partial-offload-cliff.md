# R3-ENG-13 — the partial-offload cliff: measured, and the mechanism verified in source

Follow-up to `35-prismml-install-and-corrections.md`, answering two challenges:

1. **The user's:** "the model is only about 6.8GBs and on 16GB VRAM Card there should be
   Plenty of space? for the Context and the Whole model itself"
2. **The user's:** "the 9tok/s at 131k sounds very off, even at 262K it shouldn't be that low"

Both were productive. (1) is correct and exposes a mistake of mine; (2) is correct that my
*explanation* was wrong, though the measurement itself holds up under repetition.

---

## The answer to (1): it does fit, and the spill was mine

Rigma's own fit function, asked directly, keeps **all 64 layers on the GPU** at the default
context and well beyond it:

```
usable_vram = 14954 MB     model = 6872 MB
=> KV headroom for all layers = 8082 MB

     ctx  fit ngl  spill         kv   model+kv   vs budget
  131072       99      0      4352M     11224M      -3730M
  163840       99      0      5440M     12312M      -2642M
  196608       99      0      6528M     13400M      -1554M
  229376       99      0      7616M     14488M       -466M
  245760       63      1      8160M     15032M        +78M
  262144       58      6      8704M     15576M       +622M
```

`resolve()` with no override returns **ctx=131072, ngl=99, kv=q8_0/q8_0, origin=calculator** —
fully on the GPU, 11.2 GB of a 16.3 GB card.

The `-ngl 58` I had been measuring came from **my own `--ctx 65536` override**, which I passed
while chasing an unrelated question. The resolver's default never spilled. I then reported the
spilled configuration's throughput as if it were what Rigma chooses, which it is not.

**So the honest framing of the original finding was wrong.** `_GROW_LAYER_BUDGET` is not
"Rigma's policy silently costs 82%" — it is a policy the user selects on the Models page, and
it only engages past ctx 229376, where the card genuinely runs out.

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

**4.3x from spilling 6 of 64 layers.** The number stands. My *explanation* did not:

### What I claimed, and what is actually true

| my claim | verdict |
|---|---|
| The spilled weights cross PCIe every token | **false** — weights stay resident; one ~10 KB hidden state crosses per token |
| ~0.63 GB of weights per token is re-read | **false** — that is a one-time load |
| `PQ2_0` may have no CPU kernel, falling back to a slow path | **false** — it has a vectorized one |
| Thread oversaturation (llama.cpp's documented trap) | **ruled out** — 10 physical cores, 10 threads, `n_threads = 10` |
| The KV cache follows its layer's device | **TRUE — verified in source** |

### The mechanism, verified in the fork's own source

`llama-kv-cache.cpp`, in the constructor, per layer:

```cpp
ggml_backend_buffer_type_t buft = ggml_backend_cpu_buffer_type();
if (offload) {
    auto * dev = model.dev_layer(il);
    buft = ggml_backend_dev_buffer_type(dev);
}
```

The KV cache is allocated **on its layer's device**. So spilling one *attention* layer drags
that layer's entire context-proportional K/V into host RAM, where the attention then runs on
the CPU every token. This is why the cost is a cliff rather than a slope in "fraction of
layers": the unit that matters is not the layer's 0.6 GB of weights but the context-sized
cache behind it.

For this model at ctx 131072 with q8_0 (16 attention layers, 4 kv_heads, head_dim 256):

```
per attention layer per token = 2 * 4 * 256 * (34/64) = 1088 B
  -> ctx 131072 = 136 MB per layer   (2176 MB across all 16)
6 spilled layers, of which ~1-2 are attention
  -> ~136-272 MB read from host RAM per token
at ~40 GB/s dual-channel -> 3.4-6.8 ms/token, against ~79 ms observed
```

Host-RAM traffic is therefore **not** the dominant term; the CPU-side attention compute is
(the cache is read once and reused across the 24 query heads). Either way the mechanism is
the same and it is not PCIe: **spilling a layer moves its cache off the GPU.**

The `PQ2_0` CPU kernel exists and is vectorized — verified in the fork's `ggml-cpu.c`:

```c
[GGML_TYPE_PQ2_0] = {
    .from_float   = quantize_row_pq2_0,
    .vec_dot      = ggml_vec_dot_pq2_0_q8_K,
    .vec_dot_type = GGML_TYPE_Q8_K,
    .nrows        = 1,
},
```

implemented in `arch/x86/quants.c` with AVX2/AVX-VNNI. So the CPU layers are not running a
scalar fallback.

---

## Where the offload boundary actually falls (corrected)

llama.cpp offloads from the **end**: `i_gpu_start = n_layer_all + 1 - n_gpu_layers`, and
layers `il < i_gpu_start` go to the CPU. With 64 layers and `-ngl 58`, the **first 7 layers
stay on the CPU** and layers 7-63 plus the output go to the GPU. I had assumed the opposite.
This matters for hybrids: with `full_attention_interval = 4`, the first 7 layers contain
attention layers 3 and 7, so the CPU always holds at least one attention layer and its cache.

---

## A measurement I do not trust

`ctx 262144 ngl 99` reported **11.44 t/s**, and `ctx 262144 ngl 58` reported **12.18 t/s**.
The first is impossible to reconcile with `ctx 131072 ngl 99 = 54.5 t/s` on the same build
minutes earlier, and `fit_gguf` says ngl 99 does not fit at 262144 (15576 MB against a
14954 MB budget). The likely reading is that llama.cpp accepted the allocation anyway —
the documented Windows trap, where WDDM pages VRAM to system RAM and prints no error — so
that run measured paging, not offload. **Recorded as unreliable rather than explained.**

This also invalidates the one comparison I had hoped would be decisive. The earlier
"48.5 t/s at 65536 vs 9.0 at 131072" was not a controlled comparison: `-c` and `-ngl` moved
together, so it could not separate context from spill. The controlled runs now say:

```
ctx 131072, ngl 99 (0 spilled)   54.5 t/s
ctx 131072, ngl 58 (6 spilled)   12.6 t/s
ctx 131072, ngl 48 (16 spilled)   5.9 t/s
```

which is a monotone curve in **spilled layers at fixed context** — the cliff, with no
context confound.

---

## What changed in the code

Nothing functional. `_GROW_LAYER_BUDGET` stays at 0.15: it is user-selected, other models
genuinely prefer the window, and changing a global default on one machine's measurement
would be worse than documenting it. What was added is a comment at the constant recording
the measured 4.3x and the source-verified reason, because the name "15% of the layers"
badly understates the cost and the next reader deserves the number.

The honest open item is a **UI** one, not a resolver one: the Models page growth-policy
dropdown prices "context" as more window and never as "up to 4x slower". The measurement to
put a number on it now exists.

---

## Verification status

* Verified: the fit boundary table (Rigma's own `fit_gguf`); the default resolve
  (ctx 131072 / ngl 99); 12.59 t/s mean over six runs with sd 1.01; 54.52 t/s reproduced;
  KV-on-layer-device in `llama-kv-cache.cpp`; `PQ2_0` CPU kernel in `ggml-cpu.c`;
  10 physical cores and `n_threads = 10`.
* Not verified: the exact split of the 79 ms between host-RAM KV traffic and CPU attention
  compute; the 262144 readings above, which are recorded as unreliable; whether the same
  cliff appears on HIP (only Vulkan was measured).
* Method note: `web_search` is unconfigured in this session, so research went through
  DuckDuckGo HTML and direct `raw.githubusercontent.com` fetches of the fork at
  `87268f77`, rather than through a search API.
