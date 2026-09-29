# R3-ENG-18a — the 262K verification plan (run when the GPU is free)

This is the measurement half of `37-262k-context.md`. **Nothing in it was run tonight** — the
card was busy, and the rule for the audit was no GPU work of any kind. It is written so that it
can be executed top-to-bottom without re-deriving anything, and every prediction it tests is
labelled with the value `37-` predicted for it.

Success is not "the numbers match". It is: **one row of the depth table is MEASURED instead of
predicted, or the prediction is shown wrong with a number attached.**

---

## 0. Preconditions and hygiene

1. **The card is genuinely free.** `35-` documents the trap: Rigma's VRAM figure is *live free
   memory*, and a leftover engine changes every plan it produces. Before each block:
   * `rigma plan --verify` must not report another engine on the port;
   * the desktop must be on the iGPU (the calibration notes say ~1200 MB is reserved when it
     is; `calibration.json`'s backfilled `vram_used_mb: 3955` entries were taken with the
     desktop on the dGPU and are marked superseded).
2. **No `taskkill`/`Stop-Process` by name.** Track the PID of anything launched and stop exactly
   that.
3. **One configuration per process.** `llama-bench` reuses the model across instances when the
   model params match, but a new `llama_context` is created per instance and the depth state is
   only reused when it is compatible, so mixing configs in one invocation risks a stale-state
   re-fill. Separate invocations keep the arithmetic clean.
4. **A/B/A bracketing.** Every block starts and ends with the same control
   (`-d 0 -p 512 -n 128`, q5_1). If the two controls differ by more than 5%, discard the block —
   issue #24483 measured 39% run-to-run variance on this card.
5. **`-t 10`.** Doc 36 measured 10 physical cores and `n_threads = 10`; `llama-bench`'s default
   is `common_cpu_get_num_math()`, which may be the logical count.

## 1. Binaries (verified present on this machine)

```
vulkan   C:\Users\amren\.rigma\engines\prism-b10743\vulkan\llama-bench.exe
hip      C:\Users\amren\.rigma\engines\prism-b10743\hip\llama-bench.exe
mainline C:\Users\amren\.rigma\engines\b9867\vulkan\llama-bench.exe
model    C:\Users\amren\.rigma\models\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf
```

`prism-b10743` is the release whose `ggml-vulkan.cpp`, `llama-context.cpp`, `fattn.cu` and
`ggml/CMakeLists.txt` are SHA256-identical to the pinned `87268f77` tree that `37-` cites
(VERIFIED). `--list-devices` first, and record the device description in the results — the
whole point is that this is the same card as the anchor.

## 2. Flags — each one verified in the fork's own parser

`tools/llama-bench/llama-bench.cpp` at `87268f77`:

| flag | default | verified at |
|---|---|---|
| `-m, --model <file>` | models/7B/… | `print_usage`, `parse_cmd_params` |
| `-p, --n-prompt <n>` | 512 | usage; `parse_int_range` accepts `0` |
| `-n, --n-gen <n>` | 128 | usage |
| `-d, --n-depth <n>` | 0 | usage; pre-fills the cache, state cached across reps (lines 2388-2414) |
| `-ctk, --cache-type-k <t>` | f16 | usage; `ggml_type_from_name` accepts f16, bf16, q8_0, q4_0, q4_1, q5_0, q5_1, iq4_nl |
| `-ctv, --cache-type-v <t>` | f16 | same |
| `-b, --batch-size <n>` | 2048 | usage |
| `-ub, --ubatch-size <n>` | 512 | usage |
| `-t, --threads <n>` | num_math | usage |
| `-ngl, --n-gpu-layers <n>` | -1 | usage |
| `-fa, --flash-attn <on\|off\|auto>` | auto | usage |
| `-r, --repetitions <n>` | 5 | usage |
| `-o, --output <csv\|json\|jsonl\|md\|sql>` | md | usage |
| `--no-warmup`, `--progress`, `--delay <s>`, `--list-devices`, `-dev`, `-sm`, `-lm` | — | usage |

Two behaviours that matter, both from source:

* **`n_ctx = n_prompt + n_gen + n_depth`** (`to_llama_cparams`, line 1285). So
  `-p 0 -n 128 -d 131072` runs a 128-token generation at depth 131072 in a context of 131200,
  and the test is printed as `tg128 @ d131072`.
* **`-fa auto` is not neutral with a quantized V cache** — it is rewritten to ENABLED
  (`llama-context.cpp:3881-3883`), so it behaves like `-fa on`. Pass `-fa on` explicitly so the
  intent is in the log.

Depth prefill uses **random tokens** (`test_prompt`), so these runs measure speed only, never
quality. That is the point; quality is `quant_quality`'s published table plus the rotation
caveat in `37-`.

## 3. The core table — depth sweep at the resolver's plan

Every row: Vulkan, prism-b10743, `-fa on -t 10 -ngl 99 -r 3 -o md`, `-p 0 -n 128`, varying
`-d`. The right-hand column is what `37-` predicts.

```
$B = "C:\Users\amren\.rigma\engines\prism-b10743\vulkan\llama-bench.exe"
$M = "C:\Users\amren\.rigma\models\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf"

# control, and the fully-filled native window (262144 = 261888 + 256)
& $B -m $M -ngl 99 -ctk q5_1 -ctv q5_1 -fa on -t 10 -p 512 -n 128 -d 0     -r 3
& $B -m $M -ngl 99 -ctk q5_1 -ctv q5_1 -fa on -t 10 -p 0   -n 256 -d 261888 -r 3
```

| config | depth | PREDICTED t/s | measured |
|---|---|---:|---|
| q5_1/q5_1 | 0 (control, `-p 512`) | 54.52 (MEASURED anchor, doc 36) | |
| q5_1/q5_1 | 32768 | 48.80 | |
| q5_1/q5_1 | 131072 | 37.12 | |
| q5_1/q5_1 | 245760 | 29.02 | |
| q5_1/q5_1 | 261888 (native fill) | ~28.5 | |
| q4_0/q4_0 | 131072 | 40.34 | |
| q4_0/q4_0 | 261888 | ~32.5 | |
| q8_0/q8_0 | 0 (control) | 54.52 (MEASURED) | |
| q8_0/q8_0 | 131072 | **32.76 — the top priority cell** | |
| q8_0/q8_0 | 229376 (last all-GPU) | ~26.5 | |
| q8_0/q4_0 | 131072 | 36.16 | |

The **q8_0/q8_0 @ 131072** row is the one that matters most: it is the resolver's default
configuration, it fits (11224/14954 MB), it has never been measured, and `37-` predicts a 40%
loss against the shallow anchor. If it comes back near 54 t/s instead, the depth model in `37-`
is wrong and the whole KV-bytes story needs re-deriving.

**Do not run q8_0/q8_0 at 245760 or 261888.** It does not fit (15032/15576 MB against 14954),
so the run would measure WDDM paging. Reproducing that failure *deliberately* is a separate,
optional block (section 6) — it is what produced doc 36's unreliable 11.44 / 12.18 t/s.

## 4. The spill curve, controlled (validates section 2 of `37-`)

Doc 36's curve moved `-ngl` at fixed ctx 131072. Reproduce it once, with the control bracket, to
confirm the corrected layer counts (7 and 17, not 6 and 16) and the linear model:

```
for ($ngl in 99,58,48) {
  & $B -m $M -ngl $ngl -ctk q5_1 -ctv q5_1 -fa on -t 10 -p 0 -n 128 -d 131072 -r 3
}
```

| ngl | CPU layers (VERIFIED: `n_layer_all+1-ngl`) | PREDICTED t/s |
|---|---:|---:|
| 99 | 0 | 37.12 (all-GPU q5_1 at depth) |
| 58 | 7 | ~12.6 (MEASURED at shallow depth, doc 36) |
| 48 | 17 | ~5.9 (MEASURED at shallow depth) |

The linear model is `ms/token = 0.280*L_gpu + 9.18*L_cpu` (R^2 0.99992). At depth the GPU term
grows (more KV traffic) while the CPU term is flat, so the *ratio* should compress with depth:
7 CPU layers at depth 131072 should be **less** than 4.3x slower than all-GPU. That is a
falsifiable prediction of the corrected mechanism, and it is the cleanest way to test it.

## 5. Rotation A/B (sizes the quality caveat in `37-` §4)

`37-` says the fork enables KV rotation by default for a quantized cache with
`head_dim % 64 == 0`, so PR 7412's perplexity figures are an upper bound. Speed effect, if any,
is unmeasured. One A/B, q4_0/q4_0 at depth 131072:

```
& $B -m $M -ngl 99 -ctk q4_0 -ctv q4_0 -fa on -t 10 -p 0 -n 128 -d 131072 -r 3
$env:LLAMA_ATTN_ROT_DISABLE="1"
& $B -m $M -ngl 99 -ctk q4_0 -ctv q4_0 -fa on -t 10 -p 0 -n 128 -d 131072 -r 3
Remove-Item Env:\LLAMA_ATTN_ROT_DISABLE
```

There is **no CLI flag** for rotation — the env var is the only switch, and Rigma never sets it.
If the two differ materially, Rigma's `total_loss` tooltip needs a rotation-aware figure.

## 6. The paging reproduction (optional, and only with a monitor)

The purpose is to replace doc 36's "unreliable" 262144 readings with a documented failure. Launch
`-ngl 99 -ctk q8_0 -ctv q8_0 -p 0 -n 128 -d 261888` and watch dedicated GPU memory while it
runs. If it completes at ~5-12 t/s with memory above the budget, that is the WDDM page, and the
result should be recorded as *the trap*, not as a benchmark. Do not use it to price anything.

## 7. HIP, mixed K/V, and `-fa on` (tests `37-` §5)

HIP rejects a mixed pair and falls back to CPU attention **silently** under `-fa on`
(`fattn.cu:442-446`; the shipped HIP binary does not define `GGML_CUDA_FA_ALL_QUANTS`). Verify
the failure mode is visible as a throughput cliff:

```
$H = "C:\Users\amren\.rigma\engines\prism-b10743\hip\llama-bench.exe"
& $H -m $M -ngl 99 -ctk q4_0 -ctv q4_0 -fa on -t 10 -p 0 -n 128 -d 32768 -r 3   # symmetric: fast
& $H -m $M -ngl 99 -ctk q8_0 -ctv q4_0 -fa on -t 10 -p 0 -n 128 -d 32768 -r 3   # mixed: expect a cliff
```

If the mixed row is not a cliff, `-fa on` is not reaching the FA node at all (e.g. it is being
disabled earlier), and the source reading needs revisiting. Vulkan should show the opposite:
`-ctk q8_0 -ctv q4_0` is accepted and fused.

## 8. The Rigma path (the number the owner actually sees)

`llama-bench` is the clean instrument; Rigma is the product. Do both, and compare.

1. **Plan.** `rigma up --model ternary-bonsai-2-27b-uncensored-heretic-pq2-0 --ctx 262144 --dry-run`
   Expect: `-ngl 99 -c 262144 --cache-type-k q5_1 --cache-type-v q5_1`, and (because the stored
   `launch.kv` is q8_0) a step-down notice. If the argv still says q8_0, the `e520b06` fix did
   not take — that is the regression test.
2. **Launch** for real, then POST a **synthetic** prompt (repeated filler tokens, never owner
   prose) of ~130000 tokens with `max_tokens: 128` to
   `http://127.0.0.1:11500/v1/chat/completions`, and read `timings.predicted_per_second` and
   `timings.prompt_per_second` from the response JSON. Those fields are engine metadata, not
   content.
3. **Cross-check the VRAM** against `14954 MB`. `rigma plan --verify` prints the engine's own
   numbers; a `-c 262144` q5_1 run should hold 13016 MB and leave ~1938 MB. If it holds ~15576,
   it is running the pinned q8_0 plan and paging.

Predicted: ~37 t/s at a filled 131072 and ~29 t/s at 245760, against doc 36's 9-12.6 t/s at
131072. The gap between those two is the entire finding.

## 9. Runtime budget (PREDICTION, from the pp/tg rates above)

| block | cost |
|---|---|
| control + native-fill q5_1 (prefill 261888 tokens at ~840 t/s) | ~6 min |
| q5_1 depth sweep, 4 depths, 3 reps | ~15 min |
| q8_0 @ 131072 (prefill 131200) | ~4 min |
| q4_0 / q8_0-q4_0 cells | ~10 min |
| spill curve (3 × prefill 131200) | ~12 min |
| rotation A/B | ~8 min |
| HIP A/B | ~8 min |
| Rigma-path server run | ~15 min |

Roughly **1.5 hours**, dominated by prefill. `--delay 5` between tests costs little and lets the
card cool.

## 10. What would falsify what

| observation | conclusion |
|---|---|
| q8_0 @ 131072 near 54 t/s, not 33 | the KV-bytes depth model is wrong; `37-` §3 must be re-derived |
| q8_0 @ 131072 near 33 t/s | `37-` §3 confirmed at the most important cell |
| spill curve ratio *grows* with depth | the CPU attention term dominates after all; `37-` §2 is wrong |
| spill curve ratio *shrinks* with depth | corrected mechanism confirmed (CPU matmul is depth-flat) |
| Rigma argv still says q8_0 at 262144 | `e520b06` regressed |
| HIP mixed K/V is not a cliff | the `-fa on` source reading is wrong |
| rotation A/B differs >5% in speed | the quality caveat needs a speed caveat too |

## Provenance

Flags and semantics: `PrismML-Eng/llama.cpp` at `87268f77`, `tools/llama-bench/llama-bench.cpp`
(fetched via `raw.githubusercontent.com`; full text cached at the DSH spill path for this
session). Predictions: `37-262k-context.md` §3. Measurements quoted for comparison: `35-`, `36-`,
`~/.rigma/calibration.json`. Binary paths: directory listing of `~/.rigma/engines` (read-only).
