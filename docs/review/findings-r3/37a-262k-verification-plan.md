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
vulkan   C:\Users\dev\.rigma\engines\prism-b10743\vulkan\llama-bench.exe
hip      C:\Users\dev\.rigma\engines\prism-b10743\hip\llama-bench.exe
mainline C:\Users\dev\.rigma\engines\b9867\vulkan\llama-bench.exe
model    C:\Users\dev\.rigma\models\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf
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
$B = "C:\Users\dev\.rigma\engines\prism-b10743\vulkan\llama-bench.exe"
$M = "C:\Users\dev\.rigma\models\Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf"

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

### The CPU term is NOT depth-flat — reworked prediction

`-ngl` keeps the FIRST layers on the CPU, and full attention is exactly `il % 4 == 3`
(`src/models/qwen35.cpp:21-27` at `87268f77`, VERIFIED), so a CPU-resident *attention* layer has
its KV slice in host RAM (the KV-follows-layer-device fact from doc 36) and reads it every token:

| ngl | CPU layers (`n_layer_all+1-ngl`) | CPU attention layers | which |
|---|---:|---:|---|
| 99 | 0 | 0 | — |
| 58 | 7 (0..6) | **1** | il=3 |
| 48 | 17 (0..16) | **4** | il 3, 7, 11, 15 |

So the CPU term has three parts, and only the first is depth-flat:

```
T_cpu(d) = 9.18 * c                                  # PQ2_0 CPU matmul, MEASURED (37- §2)
         + a_c * (1536 * d / BW_host)                # that layer's KV from system RAM
         + a_c * (20480 * d / F_cpu)                 # CPU attention compute: QK+PV,
                                                     # 20 q-heads x 256 head_dim x 4 flops/key
```
with `1536 B/token` = q5_1 per attention layer (24576/16), `BW_host` 40-80 GB/s and `F_cpu`
100-300 GFLOP/s — the last two are PREDICTION, and `F_cpu` is the widest error bar in this
document (there is no in-house measurement of it). The GPU side keeps the measured decomposition:
`0.280 ms` per GPU layer, plus `0.537 ms` per GPU attention layer at d=131072, scaling with d
(from the all-GPU column: 26.94 ms at 131072 against 18.34 ms at depth 0 = 8.60 ms for 16 layers).

At **depth 131072**, with the arithmetic shown (`T_gpu` includes the GPU-side KV read):

| ngl | T_cpu | T_gpu | total | PREDICTED t/s | at depth 0 (MEASURED) |
|---:|---:|---:|---:|---:|---:|
| 99 | 0 | 26.5 ms | 26.5 ms | 37.7 | 54.5 |
| 58 | 75.7-96.1 ms | 24.0 ms | 99.7-120.1 ms | **8.3-10.0** | 12.59 |
| 48 | 201.7-283.3 ms | 19.6 ms | 221.3-302.9 ms | **3.3-4.5** | 5.9 |

Worked example, ngl 58: `15.96 (57 GPU layers) + 64.26 (7 CPU layers) + 8.06 (15 GPU attention
layers) + [2.5..5.0] (192 MiB host KV) + [8.9..26.8] (2.68 GFLOP of CPU attention)` = 99.7-120.1 ms.
ngl 48: `13.16 + 156.06 (17 CPU layers) + 6.44 (12 GPU attention) + 4x[2.5..5.0] + 4x[8.9..26.8]`.

**The ratio ngl58/ngl99 is therefore roughly FLAT with depth, not compressing and not
exploding**: 4.4x at depth 0 (measured 4.3x), 3.8-4.5x at depth 131072 — the CPU attention term
grows with depth at about the same rate as the GPU's KV traffic, so the two effects very nearly
cancel. That is the falsifiable claim: if the measured ratio at depth is far outside ~3.8-4.5x,
the CPU-attention term is mis-sized. `37-` §2's linear model is a SHORT-DEPTH model and is tested
by the depth-0 row, not by the depth behaviour.

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
$H = "C:\Users\dev\.rigma\engines\prism-b10743\hip\llama-bench.exe"
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
| ngl58/ngl99 ratio at depth 131072 far ABOVE ~4.5x | the CPU attention term is more expensive than assumed — `F_cpu` below ~100 GFLOP/s, or the GPU KV term smaller |
| ngl58/ngl99 ratio at depth 131072 far BELOW ~3.8x | the CPU attention term is cheaper than assumed (`F_cpu` above ~300 GFLOP/s) or the host KV read is not happening |
| the ratio near 3.8-4.5x at depth 131072 | §4's reworked prediction holds: the CPU attention term grows with depth at the same rate as the GPU KV term |
| ngl 48 collapses far below 3.3 t/s | 4 CPU attention layers cost more than the model allows for; `F_cpu` is low |
| the depth-0 spill curve moves | `37-` §2's measured linear model (0.280 / 9.18 ms) is not reproducible |
| Rigma argv still says q8_0 at 262144 on Vulkan | `e520b06` regressed |
| Rigma argv says q5_1 on ROCm | `08b029b` regressed — q5_1 has no FA kernel there |
| HIP mixed K/V is not a cliff | the `-fa on` source reading is wrong |
| rotation A/B differs >5% in speed | the quality caveat needs a speed caveat too |

`37-` §2's linear model is a SHORT-DEPTH model. Testing it means testing the depth-0 rows; testing
the *mechanism* means testing whether the CPU term grows with depth as §4 predicts.

## 11. Optional later experiment — `--kv-mean-center` (do not wire into Rigma)

The fork ships a K-cache bias correction that is relevant to the ROCm/CUDA pick (q4_0) in `37-`
§4: `--kv-mean-center FNAME` (`common/arg.cpp:2451-2459`, env `LLAMA_ARG_KV_MEAN_CENTER`) subtracts
a precomputed per-(kv-head, channel) bias from K before Q4_0 quantization. It is softmax-invariant
(the `q·k̄` term is constant per row), claimed to cost nothing at decode time, and requires
`-ctk q4_0` (enforced at `llama-context.cpp:3913-3917`); it supports hybrid memory
(`llama-context.cpp:504-506` collects the hybrid attention cache). **The fork publishes no
perplexity and no speed number** — `docs/kv-mean-center.md` says so explicitly and defers to
`tools/kv-mean-center/README.md`, which reports logit-KLD only (rotation alone 0.00144, centering
alone 0.00149, matched rotated-basis bias 0.00111). So it is a candidate to make q4_0 on ROCm
closer to q8_0 quality, not a measured win.

Testing it needs calibration work (the `llama-kv-mean-center` tool over a text corpus) plus a
quality run — GPU/CPU work for another night. It is recorded here so the ROCm q4_0 fallback has a
documented path to better quality, and deliberately **not** wired into Rigma until a number exists.

## The consolidated NEEDS-GPU list (folded in 2026-09-30, from the program hand-off)

This is the single place the program's GPU-blocked questions live. **Do not start a second plan.**
Everything below needs a free card and is labelled with what it would settle.

1. **The hybrid at 2 slots (the highest-value one).** Launch
   `Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf` (arch `qwen35`, 64 layers, 48 recurrent,
   `full_attention_interval=4`) at **ctx 65536, ubatch 512, `--parallel 2 --kv-unified`**, and capture
   the `load_tensors` / `sched_reserve` / `llama_memory_recurrent` / `graph splits` lines. It answers
   four separate code questions at once:
   - (a) does `token_embd` still land in `CPU_Mapped` host memory at 2 slots — and is that a property
     of the fork or of the model? (`CPU_Mapped model buffer size = 322.07 MiB` in the existing log.)
   - (b) what is `compute buffer size` at 2 slots? `COMPUTE_BUFFER_MB = 150` in `resolve.py` is a
     **differenced** quantity (the draft head's buffers were subtracted), and C10 now scales the
     engine's **measured** `410.28 MiB` above ubatch 512 — so this turns a PREDICTION into a
     measurement. `E2`/`G4`: raising the constant blindly re-introduces a double count.
   - (c) **A17b's** code expects `graph splits = 2` on exactly this load, and `.scratch/prism-v.log`
     is the only evidence that it does.
   - (d) **DR2-3's** second ubatch point: run it once more at **ubatch 1024 or 2048** and the linear
     scaling `compute_buffer_mb = 410.28 * ubatch/512` becomes measured instead of predicted.
2. **The 262K-context verification itself** — the depth table in `37-262k-context.md` §3–§4, run
   top-to-bottom per this document.
3. **One live mcode session** — answering `ask_user` end-to-end and the DSH ACP client (OD-6). Also
   the only way to observe the **Windows** ACP stop (DR2) against a real `mcode.cmd` + node agent
   rather than the fake, and the only way to see whether the DR6 grep wall clock is ever hit by a
   legitimate scan.
4. **vLLM phases 1–5** need Linux/WSL + a GPU; on Windows `vllm_availability` refuses
   (`engines.py:288-291`). Only unit-level work with fakes is possible on this host.
5. **A multi-GPU machine** would settle **A17e** — the device-count split baseline is derived from
   the load's own device labels and cannot be executed here — and would give DR2-1's
   `weights_are_device_resident` a real two-device spill to judge.
6. **A quality run for the fork's K-cache bias correction** (`--kv-mean-center`, see the section
   above): it needs the `llama-kv-mean-center` calibration over a corpus plus a perplexity run. It is
   deliberately **not** wired into Rigma until a number exists.

7. **Is `plan.flags.ubatch` the engine's physical `n_ubatch`?** (`impl/ubatch` @ `d2557d8`, the
   DR21RN1-n1 fix.) The plan-side compute charge now uses the `-ub` the launch emitted, read back from
   `state.json`. Rigma refuses `ubatch > batch` at write time, but **llama.cpp can clamp `n_ubatch` to
   `n_batch`**, and no engine ran to confirm the two agree. A single real load with `rigma up --ubatch
   N` for an `N` above and below `batch`, reading the engine's own `n_ubatch`, settles it. If they
   diverge the charge is wrong in the same false-positive direction the fix closed.
8. **The compute charge at a second ubatch point** (item 1(d), restated with a reason). `resolve.py`
   scales `compute_buffer_mb` linearly from the one measured **410.28 MiB at ub 512**. The 150 MiB
   default charge plus the 512 MiB slack is what keeps a healthy load quiet (DR21RN1-n2), and that
   arithmetic is a PREDICTION. One load at ubatch 1024 or 2048 turns it into a measurement and says
   whether the slack is still enough at depth.
9. **W13B-3's ambient-env lever** (deferred, owner decision). The sweep's quality-lever gate cannot
   see a lever exported in the process env while `runtime.launch_server` merges `os.environ` into the
   child; the naive fix keeps **0 of 8** configs, so it was correctly deferred. Deciding whether the
   sweep should be ambient-aware needs a real sweep with `LLAMA_ATTN_ROT_DISABLE=1` (or another
   quality lever) exported, on a card, to see whether a crowned config is still representative. It is
   a **measurement caveat, not a write-path hole** — the override path and the plain launch run under
   the same exported env.

## Provenance

Flags and semantics: `PrismML-Eng/llama.cpp` at `87268f77`, `tools/llama-bench/llama-bench.cpp`
(fetched via `raw.githubusercontent.com`; full text cached at the DSH spill path for this
session), `src/models/qwen35.cpp` (attention indices), `common/arg.cpp` and
`docs/kv-mean-center.md` (section 11). Predictions: `37-262k-context.md` §3 and §4. Measurements
quoted for comparison: `35-`, `36-`, `~/.rigma/calibration.json`. Binary paths: directory listing
of `~/.rigma/engines` (read-only). Round-2 verification notes: `.scratch/r3-37/r2-verify.md`.
