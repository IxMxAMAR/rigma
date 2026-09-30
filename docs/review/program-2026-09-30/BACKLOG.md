# BACKLOG — Rigma improvement program, 2026-09-30

Head Agent: DeepSeek-V4.1-Flash on tokenjuice. Integration branch `review/deep-audit-2026-09-22`
(base `7ea1827`). Baseline at program start: full suite `-m "not hardware"` = **2935 tests, all
pass, exit 0**; `ruff check src tests` clean.

Built from six read-only recon passes, each with its own report in `.scratch/orchestrator/`:

| recon | report | scope |
|---|---|---|
| R1 | `recon-R1-record.md` | the written record: docs/review, findings-r3, HANDOFF/RESUME/backlog |
| R2 | `recon-R2-r3harness.md` | the unmerged branch `r3/harness` + checkpoints 23,24,27–34 |
| R3 | `recon-R3-code.md` | code sweep: markers, dead controls, orphans, swallowed errors, races |
| R4 | `recon-R4-levers.md` | engine levers: Rigma argv vs `common/arg.cpp` at both pins |
| R5 | `recon-R5-seam.md` | Rigma↔DSH and Rigma↔mcode seam |
| R6 | `recon-R6-seeds.md` | seeds S1–S8, confirmed/refuted against the tree |

Tags: **DO** = implement now · **NEEDS-GPU** = the change is ready but its verification needs the
card · **OWNER-DECISION** = a policy/irreversible choice · **SKIP** = deliberately not done, with
the reason written down (the owner's "leave it be" rule).

Ranking rule (brief §3): things that can make something else fail (data loss, wrong plan, silent
fallback, wrong chat) > the harness seam > missing levers > UI/UX polish.

---

## Tier A — can make something else fail

| id | statement | cause | proposed resolution | evidence | risk | tag |
|---|---|---|---|---|---|---|
| A1 | A chat turn whose final save loses all 3 `StaleWriteError` retries is reported as saved; the tail of the reply is never written and only the older `partial` checkpoint survives a reload | `serve.py:3740-3752`: `alive = True` before the retry loop, `break` on success, no `alive = False` on exhaustion; then `:3758` sets `_ckpt["saved"]=False, final=True` | track `saved` explicitly; on exhaustion keep `alive`/checkpoint unfinal and surface "could not save" | R3 report; code read | LOW (localised, testable) | **DO** (W1) |
| A2 | The fit charges the whole GGUF to VRAM and charges the hybrid's recurrent state (RS) **zero**; a 27B hybrid at `--parallel 2` therefore over-commits by the RS buffer | `resolve.py:520` `file_mb = gguf.bytes / 2**20`; no RS term anywhere in `resolve.py`; `COMPUTE_BUFFER_MB = 150` (`:28`) used only at `:312` | charge `RS = (n_seq_max) × per-seq RS` computed from GGUF `ssm.*` fields; charge the host-resident `token_embd` only when the engine keeps it host-side | `.scratch/prism-v.log:4486-4487` (CPU_Mapped 322.07, ROCm0 6539.67), `:4643` RS 149.62, `:4669` compute 410.28, `:4672` splits 2; formula 48 layers → 144.00+5.62 MiB matches the log | MED — see note below | **DO** (W1, RS half) |
| A3 | `_spilled` reports 6/64 CPU layers where the engine leaves 7 (the output layer `ngl` counts); the explorer's `offload_pct` and `speed` bucket inherit the off-by-one | `resolve.py:411` `(n - min(ngl, n)) / n` vs `_cpu_layers` `:381-382` `(n+1) - ngl` | make the dense branch use `_cpu_layers`/`n`, clamped to 1 | `resolve.py:369-382` docstring + the measured regression `ms/token = 0.28×gpu + 9.18×cpu` (R² 0.99992): 7 CPU layers → 80.5 ms predicted vs 79.4 measured; 6 → 71.3 | LOW | **DO** (W1, with A2) |
| A4 | `**/` in `find_files`/`grep` patterns is not collapsed → a catastrophic-backtracking regex, 2.19 s **per file**, on a tool that auto-runs | `tools.py:2010` `_glob_re` | collapse repeated `**/` groups before compiling; bound the compiled pattern | `findings-r3/04-sandbox-r3.md:386` | LOW | **DO** (W1) |
| A5 | `_engine_compat_note` swallows the engine's authoritative `check_engine` verdict and falls through to a stale model-side check → a loadable model can be reported "THIS ENGINE CANNOT LOAD THIS MODEL" | `cli.py:1048-1056` (contradicts its own comment at `:1040-1047`) | prefer the engine verdict; use the model-side check only when the engine cannot be asked, and say which was used | R3 report | LOW | **DO** (W2) |
| A6 | Any `update_engines_manifest()` failure (offline, HTTP error, corrupt file) is printed as "no newer pin published" — indistinguishable from "already current" | `cli.py:429-434` returns a bare `False` for every failure | return a reason; print the real failure; distinguish offline from up-to-date | R3 report | LOW | **DO** (W2) |
| A7 | An unknown run `profile` is silently coerced to `"all"` (full network + delete) instead of 400 | `serve.py:6324` | reject an unknown profile with 400 and name the allowed values | R3 report; `serve.py:2630,6262` default is already `"all"` | MED (behaviour change) | **DO** (W2) |
| A8 | KV restore failure is swallowed but `kv_fp` is still written → the engine silently re-prefills a 4-minute prompt on every restart | `server_ops.py:816-820` | on restore failure, do not write `kv_fp`; log the reason and say so in the plan | R3 report | LOW | **DO** (W2) |
| A9 | Idle auto-unload failure under `except Exception: pass` with no log — the card stays occupied and nothing says why | `serve.py:6919-6925` | log the failure with its reason; keep the idle state | R3 report | LOW | **DO** (W2) |
| A10 | Workspace save `.catch(()=>{})` then an optimistic `setWorkspacePath` → the UI and the server disagree about the working directory | `frontend-v2/src/App.tsx:95-99` | surface the failure, do not set the optimistic path | R3 report | LOW | **DO** (W2, frontend) |
| A11 | `/api/restore` applies settings and methods sequentially; a mid-loop `OSError` leaves a half-restored store and the client sees only a 500 | `findings-r3/01-http-r3.md:57` | stage the restore, then commit; report which stage failed | R1 report | MED | **DO** (W2) |
| A12 | The chat prompt queue is unbounded in RAM and silently popped on **any** unwind, including a turn exception | `findings-r3/01-http-r3.md:80` | bound the queue; only pop on delivery, not on unwind | R1 report | MED | **DO** (W3) |
| A13 | VLLM-1: the KV-cache fingerprint omits the engine runtime (`FINGERPRINT_FIELDS` carries only `engine` = binary path), so a cache written by a different runtime is accepted | `kvcache.py:43-58`; `findings-r3/17-vllm-engine.md:22` | add the runtime/version to the fingerprint | R1 report | MED | **DO** (W2) |
| A14 | `add_consolidated` decides the conflict verdict on a pre-`await` snapshot | `findings-r3/10-memory-r3.md:53` | re-read after the await, or take the lock across the decision | R1 report | MED | **DO** (W3) |
| A15 | Method drafts are unbounded and have no DELETE route | `findings-r3/10-memory-r3.md:60` | bound the draft store; add `DELETE /api/methods/drafts/{id}` | R1 report | LOW | **DO** (W3) |
| A16 | `list_models` swallows fit exceptions; the GGUF version check is floor-only; `params` is unbounded; `engine_log` attribution is wrong (06R3-7/8/9/10) | `findings-r3/06-fit-r3.md:111-157` | surface the fit error per model; bound `params`; fix the attribution | R1 report | LOW | **DO** (W3) |

### A2 note — why the RS half is DO and the rest is not
`resolve.py` charges `file_mb` = the whole file. The engine's own log splits that file into
`ROCm0 6539.67 MiB` + `CPU_Mapped 322.07 MiB` (host), so the VRAM charge is **332.67 MiB too high**.
Two further terms are wrong in the other direction: `compute buffer = 410.28 MiB` measured against a
flat `COMPUTE_BUFFER_MB = 150` (**260.28 MiB too low**), and RS `149.62 MiB` charged **zero**. Net at
one slot: 332.67 over − 260.28 under − 149.62 uncharged = **77.2 MiB under-charged**. On Windows the
1200 MB desktop floor hides that; on Linux (400 MB) it can over-commit.

Only the **RS term** is unambiguously safe to add: it is a pure omission, it is computable from the
GGUF header, and the log matches the formula exactly. The `token_embd` subtraction is **not** safe to
apply universally — the fixture is a 1-slot, `kv_unified=false`, non-Rigma launch
(`prism-v.log:2163,2170,4495`), and whether `token_embd` stays host-resident is a property of the
fork/model, not of every model. The compute-buffer figure was deliberately lowered from 400 to 150
because it was obtained by differencing measured VRAM and already contains the draft head's buffers
(`resolve.py:14-28`) — raising it back would double-count. Both are therefore recorded here as
**NEEDS-GPU / OWNER-DECISION** (see Tier E) rather than silently re-guessed.

---

## Tier B — the harness seam (Rigma ↔ DSH / mcode)

| id | statement | cause | proposed resolution | evidence | risk | tag |
|---|---|---|---|---|---|---|
| B1 | `harness.kill_tree` hardcodes POSIX `ok=False` (so the DSH timeout message **always** cries "could not be confirmed dead" on Linux/macOS) and uses a bare `proc.kill()` (direct child only on POSIX) | integration's implementation vs `r3/harness` 15c802a | delegate to `tools._kill_tree` (AUDIT F35: killpg + poll); add `attempted`; POSIX returns `proc.poll() is not None` | R2 report: unique ~40 lines, rest of `r3/harness` is byte-identical or superseded | LOW | **DO** (W1) |
| B2 | DSH drops 27 of 51 session-event types silently, including `command/run|done`; the projector is wrapped in `except: pass` | `_dsh_runner.py:56-135` (`_STATE_EVENTS`, 21 types), `:305-331` (`_notice_text`, 6 substrings), `:491-496` (`except: pass`); `serve.py:2449-2451` has no `else` | project the unknown types to a generic notice instead of dropping; log the drop once per type | R5 report | MED | **DO** (W2) |
| B3 | `subagent/descriptor` reaches the UI but dies in the fold (no `childId`), so subagent activity is invisible | `subagents.ts:174-175` | carry `childId` through the fold | R5 report | LOW | **DO** (W2, frontend) |
| B4 | Elicitation (mcode `ask_user`) is auto-declined: `serve.py` passes `on_permission` but never `on_question` | `serve.py:2164`; checkpoint 27 §714 | pass an `on_question` handler that surfaces the question to the UI and returns the answer | R5 report | MED | **DO** (W3) |
| B5 | Three `AcpClient` methods have zero callers anywhere (`session_list`, `session_load`, `session_activate`) | `harness_mcode_acp.py:571/556/684` | wire session listing to a route + UI, or delete and record why | R5 report | LOW | **DO** (W3) |
| B6 | `mode_set` and `queue_update` have no UI; `usage_update.cost` is handled but not rendered; `acp_commands` is a static list that cannot be run | `acp:1028-1030`, `AgentState.tsx:345-378`, `:541-563`, `acp:1201-1207` | render cost, make commands runnable, add a mode control | R5 report | LOW | **DO** (W4, frontend) |
| B7 | S5c: `--alias`, `--reasoning-format deepseek`, a tools-conditional proxy parameter layer, and a streaming tool-call rescue translator are all absent | `models.py:408-452`; `/v1` proxy is a raw `aiter_raw` passthrough `serve.py:6930-6976`; rescue only on the buffered native path `serve.py:2752,3371` | add `--alias` and `--reasoning-format`; parse the `/v1` body and apply params only when `tools` is present; add the rescue only if a leak is observed | R5 report; checkpoint 28 §6a | MED | **DO** (W2 for alias/reasoning-format; rescue is gated on an observed leak) |
| B8 | DSH governance/approval audit trail has no `dsh-user-approval`/`dsh-permission-presets` row | checkpoint 30 O4; `data/dsh/agent-capabilities.patch.yml` | add the rows | R5 report | LOW | **DO** (W3) |
| B9 | `/goal` and `/feedback` native commands absent; control ops open one mcode process each | checkpoint 30 O10/O11/O13 | record as design work | R5 report | MED | **SKIP** (needs a live mcode turn; §0.1) |

**Branch `r3/harness`:** do **not** merge. It is a post-merge rewrite based on a stale base
(`5b5790b`); integration already merged the originals via `c16722a`. 6 merge conflicts, all
stale-base artifacts. Only B1 and a 9-line test fix are unique. Port them as fresh commits (W1),
then retire the branch. `r3/http`, `r3/sandbox`, `r3/fit`, `r3/memory`, `r3/vllm` are all fully
merged (0 unique commits).

---

## Tier C — missing levers (R4)

Premise corrections: `-b`/`-ub` and `-ngl` **are** already Rigma-producible (`models.py:329-330,
423-426`; the sweep sets batch 16384 / ubatch 2048 at `bench.py:481`) — they are unexposed in the UI,
not missing from the argv. `--checkpoint-every-n-tokens` **does not exist** at either pin; the real
flags are `--ctx-checkpoints`/`-ctxcp` and `--checkpoint-min-step`/`-cms` (Rigma emits only the
latter). `LLAMA_ATTN_ROT_DISABLE` exists at both pins as an env var only.

| id | flag | why | tag |
|---|---|---|---|
| C1 | `--ctx-checkpoints`/`-ctxcp` | tunes the hybrid rewind cost Rigma already manipulates via `--checkpoint-min-step 4096`; fork `arg.cpp:1687`, mainline `:1449` | **DO** (W3) |
| C2 | `--no-op-offload` | same class as the AMD-driver env toggles the sweep already trials (`bench.py:482-484`) — sweep config only | **DO** (W4) |
| C3 | `--reasoning-effort` | fork-only (`arg.cpp:3678`); mainline b9867 lacks it — gate on the detected engine | **DO** (W4) |
| C4 | `--kv-mean-center` | fork-only (`arg.cpp:2452-2459`), requires `--cache-type-k q4_0` and a per-model calibrated bias GGUF; Rigma excludes q4_0 for tools models (`bench.py:536-539`) | **SKIP** (needs the GPU to calibrate; nothing to gain here) |
| C5 | `--image-min/max-tokens` | Rigma's fit math does not count image tokens, so exposing it would silently invalidate the VRAM plan | **SKIP** (the fit would have to learn image tokens first) |
| C6 | `--tensor-split`/`-ot`/`-sm`/`-mg`, `--numa`, `--threads*`, `--mlock`, `--no-mmap`, `--defrag-thold`, sampler flags | single GPU, single socket; sampling is per-request (`sessions.py:236-238`); `--mlock`/`--no-mmap`/`--defrag-thold` are deprecated in the fork | **SKIP** (no effect on this machine) |
| C7 | `--context-shift` | KV shifting is unsupported on the DeltaNet hybrid (`models.py:443-445`) | **SKIP** (would corrupt the recurrent state) |
| C8 | `--cache-ram` | default 8192 MiB already on; raising it competes with MoE offload for system RAM | **SKIP** (leave the default) |
| C9 | LoRA attach (`--lora`/`--lora-scaled` + `/lora-adapters`) | Rigma has no LoRA code at all and the owner trains LoRAs | **OWNER-DECISION** (a feature, not a knob; scoped in OWNER-DECISIONS.md) |

---

## Tier D — benchmark honesty and UI/UX

| id | statement | cause | proposed resolution | evidence | tag |
|---|---|---|---|---|---|
| D1 | The benchmark measures an empty window with repetitive filler: a MoE routes to a narrow expert set, and there is no depth dimension, so "tok/s at 131K" says nothing about a filled window | `bench.py:24` `"The quick brown fox…" * (prompt_tokens // 8)`; `BenchResult` `:16-20` has no depth | varied synthetic text + an optional fill depth, recorded with every result | owner's 2026-08-23 measurement (`n_cpu_moe 18` crowned on filler, `n_cpu_moe 0` beat it by 24% on varied text); `findings-r3/37 §3` | **DO** (W3) |
| D2 | First-load configuration dialog (quant, ctx, KV, MTP, vision, save-as-default): backend + route exist, UI partial | `LaunchDefaults` `models.py:195`, `POST /api/models/{slug}/defaults` `serve.py:4582`; only `Sidecar.tsx:249` sends ctx/kv/vision | add the dialog; send quant/spec_type/backend too | R6 report S5a | **DO** (W4, frontend) |
| D3 | Chat reattach: in-page works, across a reload it does not (streams are in-memory, no GET stream route, no server-exposed per-session running flag); incremental persistence exists but the frontend never reads the `partial` flag | `chatStore.ts:686,830,914-916`; `serve.py:4702` `_streaming` not returned by `GET /api/sessions`; `serve.py:2937-2971` writes `partial` | expose a running flag + a stream reattach route; read the `partial` flag so a reload shows the tail | R6 report S5b | **DO** (W4, frontend) |
| D4 | 4 routes have no UI caller (`/api/settings` GET+POST, `/api/models/{slug}/reprobe`, `/rename`, `/api/backup` GET, `/api/restore` POST, `/api/mcp` GET, 3 `/api/methods` routes) | R3 report item 4 | decide per route: wire or delete; `/api/settings` in particular has no UI at all | R3 report | **DO** (W4) |
| D5 | No `*.test.tsx` in the repo, so the render layer is never browser-tested (F1 passed `tsc` + 434 tests and was still a live render bug) | checkpoint 33 §"does NOT establish" | add a first render test for the harness view | R2/R5 reports | **DO** (W4, frontend) |
| D6 | `/api/settings` docstring assumes a UI that does not exist; `probe.py:69` `_sum_other_processes` is dead in `src` (tests-only) | R3 report item 8 | fix the docstring; delete or use the dead helper | R3 report | **DO** (W4) |

---

## Tier E — NEEDS-GPU / OWNER-DECISION

| id | statement | why not now | the exact test to run | tag |
|---|---|---|---|---|
| E1 | Charge the host-resident `token_embd` against RAM instead of VRAM | needs a Rigma-launched log at `--parallel 2` to confirm which tensors stay host-side on this fork, and whether it is universal or model-specific | extend `findings-r3/37a-262k-verification-plan.md`: launch the 27B hybrid at ctx 65536/ubatch 512/`--parallel 2`, capture `load_tensors` buffer lines, compare against the fit's charge | **NEEDS-GPU** |
| E2 | Re-derive `COMPUTE_BUFFER_MB` from measurement | the 150 was deliberately differenced out of a measured total that already contained the draft head's buffers; changing it without a measurement re-introduces a double count | same launch, capture `sched_reserve: compute buffer size`, compare with the plan | **NEEDS-GPU** |
| E3 | 262K context on the 27B hybrid (already fully resident at q5_1) | needs the card | `findings-r3/37a-262k-verification-plan.md` (do not start a second plan) | **NEEDS-GPU** |
| E4 | vLLM phases 1–5 | on Windows `vllm_availability` refuses (`engines.py:288-291`), so `rigma up --engine vllm` exits 1; only unit-level work with fakes is possible | unit tests only; the live run needs Linux/WSL + a GPU | **NEEDS-GPU** |
| E5 | R3-9 `view_image` reads absolute paths outside the workspace with no grant; R3-10 `copy/move_files` destination can be any absolute path | security policy trade-off | see `OWNER-DECISIONS.md` | **OWNER-DECISION** |
| E6 | 13-2 prompt-injection → secret exfiltration: the run profile still defaults to `"all"` (`serve.py:2630,6262`) | changing a global default | see `OWNER-DECISIONS.md` | **OWNER-DECISION** |
| E7 | LoRA attach (C9) | a feature, not a knob; needs a scope decision | see `OWNER-DECISIONS.md` | **OWNER-DECISION** |
| E8 | A pre-fix RAG index still holds secrets until reindex (R3-11 residual) | auto-reindexing the owner's files is theirs to decide | see `OWNER-DECISIONS.md` | **OWNER-DECISION** |
| E9 | DSH ACP client (checkpoint 28 §320); answering `ask_user` needs a live mcode turn | §0.1 forbids a live harness turn | see `OWNER-DECISIONS.md` / `findings-r3/37a` | **NEEDS-GPU** |

---

## Tier F — already done; do NOT re-do

Verified present in the tree (R1/R6): R3-4 `confirm_exec` on the run route (`serve.py:6322`),
R3-7 `restart_run` via `_save_guarded`, R3-17 pack-folder credential denylist, 06R3-6
hardware-keyed calibration, 18-O2 compute buffer budgeted, `goal_create/patch/clear` wired
(`controlPlane.ts`), per-row queue ops wired, `/export` implemented, `_ok_of` resolved, DSH live
tests marked hardware, `_Run.hstate` deleted, and **all four open HIGH findings from run 1/2 closed
in-tree** (13-1 pinned transport `tools.py:1002-1105`; 04-8 whole-reply fence `tools.py:442`; 11-1
per-session drafts `chatStore.ts:1255-1269` + `drafts.ts`; 13-2 confined reads + credential denylist
+ `allow_outbound_post`, both grants default OFF).

Stale docs that must not seed future work: `FINDINGS.md` per-finding "open" prose (its banner
`:6-11` says run-1 evidence only), `RESUME-2026-09-21-arm.md` §11 ("22 open LOW findings", contradicted
by its own §8c), `HANDOFF-speedups.md` (`--parallel 1` and a key without hardware identity, both
superseded), and checkpoint 28 §2.1 (says DSH bridges 15 state events; the tree has 21 after R6-WORKFLOW).

## Tier G — SKIP, with the reason

| id | skipped | why |
|---|---|---|
| G1 | Merging `r3/harness` | stale-base rewrite; only ~53 unique lines; 6 conflicts. Port B1 + the test fix instead. |
| G2 | `--kv-mean-center`, `--image-min/max-tokens`, `--tensor-split`, `--numa`, `--threads`, `--mlock`, `--no-mmap`, `--defrag-thold`, `--cache-ram`, `--context-shift`, sampler flags | C4–C8 above: no effect on this machine, or actively wrong for a DeltaNet hybrid. |
| G3 | A streaming tool-call rescue translator | gated on an observed leak; upstream's fix is `--reasoning-format deepseek` (B7), so do B7 first. |
| G4 | Raising `COMPUTE_BUFFER_MB` back to 400 | would double-count the draft head's buffers (`resolve.py:14-28`). E2 instead. |
| G5 | "22 open LOW findings" from `RESUME` §11 | contradicted by that document's own §8c; the tree shows them fixed. |
