# Rigma `--parallel 2 --kv-unified` — Evidence Review

**Verdict:** The flag is real and the memory direction is right, but the rationale's key
assumption is **factually wrong**: `--kv-unified` is **not** default-on when `--parallel` is
passed explicitly. Rigma is actually running **non-unified** KV. The prompt-cache claim is
also partly wrong — a prompt cache exists, is on by default, and preserves the evicted prompt.

All llama.cpp findings are from **current master** source (read directly, not from docs prose).

---

## 1. `--kv-unified`: real, but not default-on under explicit `--parallel`

**Exact help text** (`common/arg.cpp`):

```
{"-kvu", "--kv-unified"}, {"-no-kvu", "--no-kv-unified"},
"use single unified KV buffer shared across all sequences (default: enabled if number of slots is auto)"
```

**Source behavior** (`src/llama-context.cpp`):

```cpp
if (cparams.kv_unified) {
    cparams.n_ctx_seq = cparams.n_ctx;
} else {
    cparams.n_ctx_seq = cparams.n_ctx / cparams.n_seq_max;
    cparams.n_ctx_seq = GGML_PAD(cparams.n_ctx_seq, 256);
}
```

**Allocation** (`src/llama-kv-cache.cpp:233-234`):

```cpp
ggml_tensor * k = has_k ? ggml_new_tensor_3d(ctx, type_k, n_embd_k_gqa, kv_size, n_stream) : nullptr;
```

with `n_stream(unified ? 1 : n_seq_max)` (line 84) and `kv_size = cparams.n_ctx_seq`
(`src/llama-model.cpp:2377`).

So:

| mode | total KV cells | per-slot context |
|---|---|---|
| unified | `n_ctx_seq × 1` = `n_ctx` | `n_ctx` |
| non-unified | `n_ctx_seq × n_seq_max` = `n_ctx` | `n_ctx / n_seq_max` |

Rigma's "one shared pool of size ctx" is correct **for unified**. But:

**The bug.** `common/common.h` declares `bool kv_unified = false;`. The auto branch in
`tools/server/server.cpp` is:

```cpp
if (params.n_parallel < 0) {
    params.n_parallel = 4;
    params.kv_unified = true;
}
```

`params.n_parallel = -1` is set only for `LLAMA_EXAMPLE_SERVER` in `arg.cpp:1404`. Because Rigma
passes `--parallel 2`, `n_parallel = 2`, the `< 0` test fails, the branch is **skipped**, and
`kv_unified` stays at its struct default of **false**.

The help text "enabled if number of slots is auto" describes exactly this. **Rigma is getting
non-unified KV** — the opposite of what the comment claims.

**Not deprecated.** The word "unified" was dropped from internal naming in
[PR #15467](https://github.com/ggml-org/llama.cpp/pull/15467) (naming only; the flag is
unaffected). A genuine bug where unified was active despite not passing the flag is
[issue #17450](https://github.com/ggml-org/llama.cpp/issues/17450) (closed). `--no-kv-unified`
exists as the counterpart. Unified-cache work appears at least as early as
[PR #12381](https://github.com/ggml-org/llama.cpp/pull/12381) (Mar 2025), before b5912.

## 2. `--parallel 2` — defensible but suboptimal

**Exact help** (`arg.cpp:2541`, server-specific section):

```
{"-np", "--parallel"}, "N",
"number of server slots (default: %d, -1 = auto)"
```

`--parallel 0` throws `invalid_argument`. Auto resolves to **4** (with unified on). `n_parallel`
maps to `n_seq_max` (`common/common.cpp`).

**Batch interaction** (`server-context.cpp`): the batch is sized
`batch.init(std::max(n_batch, params_base.n_parallel), n_embd)` and the comment states
"the update_slots() logic will always submit a maximum of n_batch or n_parallel tokens".
With `n_batch = 2048` (upstream default) there is ample headroom for 2 slots, so the claim that
2 slots split batch capacity into a meaningful single-stream penalty is **not supported** by the
scheduling code. No measured benchmark for N=1 vs 2 vs 4 on single-stream throughput was found.

Verdict: **2 is reasonable and conservative**, not optimal. Upstream's own default is 4.

## 3. Prompt-cache eviction rationale — partly wrong

A prompt cache exists, is on by default, and is exactly the mechanism Rigma says is missing:

- `--cache-prompt` — "whether to enable prompt caching (default: enabled)".
- API param `cache_prompt`: *"Re-use KV cache from a previous request if possible. This way the
  common prefix does not have to be re-processed, only the suffix that differs... Default:
  `true`"* (server README).
- Server-side `get_available_slot()` selects a slot by **longest-common-prefix similarity**
  (default `slot_prompt_similarity = 0.1`), falling back to **LRU**. On LRU or when
  `f_keep < 0.5` it calls `prompt_save()` into a RAM prompt cache (`--cache-ram`, default
  8192 MiB) and then `prompt_load()`.
- `--cache-reuse N` — "min chunk size to attempt reusing from the cache via KV shifting"
  (default 0 = disabled). Relevant but not required.
- `--slot-save-path` persists slot state to disk.

So a second request does **not** blindly force a full re-prefill; the evicted prompt is saved and
restored from RAM.

**Important inversion.** Idle-slot clearing happens **only when unified is on**:

```cpp
bool try_clear_idle_slots() {
    if (!params_base.kv_unified) { return res; }
    ...
}
```

and `server-context.cpp:1428` comments: *"without a unified KV cache, clearing a slot frees no
reusable room, so we only publish a RAM-cache copy of idle slots (their KV stays in VRAM)"*.

Consequence: **non-unified (what Rigma actually runs) keeps the idle slot's KV resident in
VRAM.** The isolation Rigma wants is real — but it comes from the doubled KV pool, not from
`--kv-unified`.

## 4. What other tools default to

- **llama.cpp**: `-np 4` + unified, via the auto path.
- **Ollama**: `OLLAMA_NUM_PARALLEL` **default 1**. Official FAQ: *"Required RAM will scale by
  OLLAMA_NUM_PARALLEL * OLLAMA_CONTEXT_LENGTH."*
  ([docs.ollama.com/faq](https://docs.ollama.com/faq)). The multiply-by-N model is independently
  visible in [ollama#17408](https://github.com/ollama/ollama/issues/17408), which logs
  `-c 65536 -np 2` derived from `32768 × NUM_PARALLEL 2`.
- **LM Studio / vLLM / text-generation-webui**: **not verified** from primary sources.

## 5. Better-known solutions to aux-call contention

Not verified from maintainer statements. Rigma's separate-slot approach is legitimate. Other
standard options are a second small model, a second server instance, or queuing aux calls. No
authoritative benchmark comparing them was found.

---

## Recommended configuration

1. **Minimal fix:** add `--kv-unified` explicitly if the shared-pool memory behaviour is wanted:
   `--parallel 2 --kv-unified`.
2. **Or drop both flags** and inherit upstream's single-model default (`-np 4` + unified).
3. **Current behavior** (`--parallel 2`, no `--kv-unified`) is defensible for strict isolation but
   doubles KV. If `-c` is left at 0 (model default), total KV becomes `2 × n_ctx_train` — the
   "overflow" the comment feared. The comment is right about the cost, wrong that the flag
   prevents it.

---

## Could NOT verify

1. The exact llama.cpp version that first introduced `--kv-unified`.
2. Any measured benchmark for N=1 vs 2 vs 4 single-stream throughput, or for the stated
   "20-60K tokens / ~30-90s dead" re-prefill figure — **that number is unverified**.
3. LM Studio, vLLM, and text-generation-webui default parallel/slot configurations.
4. Whether Rigma's bundled llama.cpp matches current master; if it pins an older revision the
   auto/`kv_unified` interaction could differ.
5. No runtime measurement was performed — all memory math is from source reading, not an
   executed server.

## Evidence

- [common/arg.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/arg.cpp) — flag definitions, `n_parallel = -1` default
- [common/common.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/common.h) — `bool kv_unified = false;`
- [tools/server/server.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/server.cpp) — the `n_parallel < 0` auto branch
- [src/llama-context.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-context.cpp) — `n_ctx_seq` derivation
- [src/llama-kv-cache.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-kv-cache.cpp) — tensor allocation / `n_stream`
- [src/llama-model.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-model.cpp) — `kv_size = cparams.n_ctx_seq`
- [tools/server/server-context.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/server-context.cpp) — slot selection, prompt cache, idle-slot clearing
- [tools/server/README.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/tools/server/README.md) — flag/param default tables
- [PR #15467](https://github.com/ggml-org/llama.cpp/pull/15467), [issue #17450](https://github.com/ggml-org/llama.cpp/issues/17450)
- [Ollama FAQ](https://docs.ollama.com/faq), [ollama#17408](https://github.com/ollama/ollama/issues/17408)
