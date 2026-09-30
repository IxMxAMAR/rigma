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
| A1 | A chat turn whose final save loses all 3 `StaleWriteError` retries is reported as saved; the tail of the reply is never written and only the older `partial` checkpoint survives a reload | `serve.py:3740-3752`: `alive = True` before the retry loop, `break` on success, no `alive = False` on exhaustion; then `:3758` sets `_ckpt["saved"]=False, final=True` | track `saved` explicitly; on exhaustion keep `alive`/checkpoint unfinal and surface "could not save" | R3 report; code read | LOW (localised, testable) | **DONE** (W1) |
| A2 | The fit charges the whole GGUF to VRAM and charges the hybrid's recurrent state (RS) **zero**; a 27B hybrid at `--parallel 2` therefore over-commits by the RS buffer | `resolve.py:520` `file_mb = gguf.bytes / 2**20`; no RS term anywhere in `resolve.py`; `COMPUTE_BUFFER_MB = 150` (`:28`) used only at `:312` | charge `RS = (n_seq_max) × per-seq RS` computed from GGUF `ssm.*` fields; charge the host-resident `token_embd` only when the engine keeps it host-side | `.scratch/prism-v.log:4486-4487` (CPU_Mapped 322.07, ROCm0 6539.67), `:4643` RS 149.62, `:4669` compute 410.28, `:4672` splits 2; formula 48 layers → 144.00+5.62 MiB matches the log | MED — see note below | **DONE** (W1) |
| A3 | `_spilled` reports 6/64 CPU layers where the engine leaves 7 (the output layer `ngl` counts); the explorer's `offload_pct` and `speed` bucket inherit the off-by-one | `resolve.py:411` `(n - min(ngl, n)) / n` vs `_cpu_layers` `:381-382` `(n+1) - ngl` | make the dense branch use `_cpu_layers`/`n`, clamped to 1 | `resolve.py:369-382` docstring + the measured regression `ms/token = 0.28×gpu + 9.18×cpu` (R² 0.99992): 7 CPU layers → 80.5 ms predicted vs 79.4 measured; 6 → 71.3 | LOW | **DONE** (W1) |
| A4 | `**/` in `find_files`/`grep` patterns is not collapsed → a catastrophic-backtracking regex, 2.19 s **per file**, on a tool that auto-runs | `tools.py:2010` `_glob_re` | collapse repeated `**/` groups before compiling; bound the compiled pattern | `findings-r3/04-sandbox-r3.md:386` | LOW | **DONE** (W1) |
| A5 | `_engine_compat_note` swallows the engine's authoritative `check_engine` verdict and falls through to a stale model-side check → a loadable model can be reported "THIS ENGINE CANNOT LOAD THIS MODEL" | `cli.py:1048-1056` (contradicts its own comment at `:1040-1047`) | prefer the engine verdict; use the model-side check only when the engine cannot be asked, and say which was used | R3 report | LOW | **DONE** (W1) |
| A6 | Any `update_engines_manifest()` failure (offline, HTTP error, corrupt file) is printed as "no newer pin published" — indistinguishable from "already current" | `cli.py:429-434` returns a bare `False` for every failure | return a reason; print the real failure; distinguish offline from up-to-date | R3 report | LOW | **DONE** (W1) |
| A7 | An unknown run `profile` is silently coerced to `"all"` (full network + delete) instead of 400 | `serve.py:6324` | reject an unknown profile with 400 and name the allowed values | R3 report; `serve.py:2630,6262` default is already `"all"` | MED (behaviour change) | **DONE** (W3) |
| A8 | KV restore failure is swallowed but `kv_fp` is still written → the engine silently re-prefills a 4-minute prompt on every restart | `server_ops.py:816-820` | on restore failure, do not write `kv_fp`; log the reason and say so in the plan | R3 report | LOW | **DONE** (W3) |
| A9 | Idle auto-unload failure under `except Exception: pass` with no log — the card stays occupied and nothing says why | `serve.py:6919-6925` | log the failure with its reason; keep the idle state | R3 report | LOW | **DONE** (W4) |
| A10 | Workspace save `.catch(()=>{})` then an optimistic `setWorkspacePath` → the UI and the server disagree about the working directory | `frontend-v2/src/App.tsx:95-99` | surface the failure, do not set the optimistic path | R3 report | LOW | **DONE** (W5) |
| A11 | `/api/restore` applies settings and methods sequentially; a mid-loop `OSError` leaves a half-restored store and the client sees only a 500 | `findings-r3/01-http-r3.md:57` | stage the restore, then commit; report which stage failed | R1 report | MED | **DONE** (W3) |
| A12 | ~~The chat prompt queue is unbounded in RAM and silently popped on **any** unwind, including a turn exception~~ **Already fixed in-tree before the wave base:** R3-5 `9b8d983` (an ancestor of `e6e8c3d`) bounds the queue at `_QUEUE_MAX = 32` with a 429 at the door and makes `_release_claim(dropped=False)` default to KEEP, so only a reader stop / disconnect drops it | fixed in `9b8d983` | done — no W4-B commit made (nothing to fix) | verified by the W4-B verifier: `serve.py:4833-4838,4972-4988,5086-5103` + `tests/test_r3_prompt_queue.py` (5 tests) green at `e6e8c3d` | MED | **ALREADY DONE IN-TREE** (9b8d983) |
| A13 | VLLM-1: the KV-cache fingerprint omits the engine runtime (`FINGERPRINT_FIELDS` carries only `engine` = binary path), so a cache written by a different runtime is accepted | `kvcache.py:43-58`; `findings-r3/17-vllm-engine.md:22` | add the runtime/version to the fingerprint | R1 report | MED | **DONE** (W1) |
| A14 | `add_consolidated` decides the conflict verdict on a pre-`await` snapshot | `findings-r3/10-memory-r3.md:53` | re-read after the await, or take the lock across the decision | R1 report | MED | **DONE** (W2) |
| A15 | Method drafts are unbounded and have no DELETE route | `findings-r3/10-memory-r3.md:60` | bound the draft store; add `DELETE /api/methods/drafts/{id}` | R1 report | LOW | **DONE** (W2) |
| A16 | `list_models` swallows fit exceptions; the GGUF version check is floor-only; `params` is unbounded; `engine_log` attribution is wrong (06R3-7/8/9/10) | `findings-r3/06-fit-r3.md:111-157` | surface the fit error per model; bound `params`; fix the attribution | R1 report | LOW | **DONE** (W4) |

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
| B1 | `harness.kill_tree` hardcodes POSIX `ok=False` (so the DSH timeout message **always** cries "could not be confirmed dead" on Linux/macOS) and uses a bare `proc.kill()` (direct child only on POSIX) | integration's implementation vs `r3/harness` 15c802a | delegate to `tools._kill_tree` (AUDIT F35: killpg + poll); add `attempted`; POSIX returns `proc.poll() is not None` | R2 report: unique ~40 lines, rest of `r3/harness` is byte-identical or superseded | LOW | **DONE** (W3, rejected once then fixed) |
| B2 | DSH drops 27 of 51 session-event types silently, including `command/run\|done`; the projector is wrapped in `except: pass` | `_dsh_runner.py:56-135` (`_STATE_EVENTS`, 21 types), `:305-331` (`_notice_text`, 6 substrings), `:491-496` (`except: pass`); `serve.py:2449-2451` has no `else` | project the unknown types to a generic notice instead of dropping; log the drop once per type | R5 report | MED | **DONE** (W2) |
| B3 | ~~`subagent/descriptor` reaches the UI but dies in the fold (no `childId`), so subagent activity is invisible~~ **Corrected by the verifier:** the `subagent.started`/`subagent.finished` lifecycle already opens the rail row, so activity was never invisible — the descriptor only **renames an existing entry** (the fold's non-lifecycle tail never appends). The cause path and tag were also wrong: the fix point is backend `_dsh_runner.py:313`, not `subagents.ts` | fixed in `1ff09c8` | done | R5 report | LOW | **DONE** (W4) |
| B4 | Elicitation (mcode `ask_user`) is auto-declined: `serve.py` passes `on_permission` but never `on_question` | `serve.py:2164`; checkpoint 27 §714 | pass an `on_question` handler that surfaces the question to the UI and returns the answer | R5 report | MED | **DONE** server half (W4); UI is B4b |
| B5 | Three `AcpClient` methods have zero callers anywhere (`session_list`, `session_load`, `session_activate`) | `harness_mcode_acp.py:571/556/684` | wire session listing to a route + UI, or delete and record why | R5 report | LOW | **DONE** (W4) |
| B6 | `mode_set` and `queue_update` have no UI; `usage_update.cost` is handled but not rendered; `acp_commands` is a static list that cannot be run | `acp:1028-1030`, `AgentState.tsx:345-378`, `:541-563`, `acp:1201-1207` | render cost, make commands runnable, add a mode control | R5 report | LOW | **PARTIAL**: cost rendered (W5); commands in W6; mode control blocked on B6a |
| B7 | S5c: `--alias`, `--reasoning-format deepseek`, a tools-conditional proxy parameter layer, and a streaming tool-call rescue translator are all absent | `models.py:408-452`; `/v1` proxy is a raw `aiter_raw` passthrough `serve.py:6930-6976`; rescue only on the buffered native path `serve.py:2752,3371` | add `--alias` and `--reasoning-format`; parse the `/v1` body and apply params only when `tools` is present; add the rescue only if a leak is observed | R5 report; checkpoint 28 §6a | MED | **DONE** (W3) |
| B8 | DSH governance/approval audit trail has no `dsh-user-approval`/`dsh-permission-presets` row | checkpoint 30 O4; `data/dsh/agent-capabilities.patch.yml` | add the rows | R5 report | LOW | **DONE but INERT** (W4) - see B8b / OD-10 |
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
| C1 | `--ctx-checkpoints`/`-ctxcp` | tunes the hybrid rewind cost Rigma already manipulates via `--checkpoint-min-step 4096`; fork `arg.cpp:1687`, mainline `:1449` | **VERIFIED, HELD** (W4) |
| C2 | `--no-op-offload` | same class as the AMD-driver env toggles the sweep already trials (`bench.py:482-484`) — sweep config only | **DONE** (W4) |
| C3 | `--reasoning-effort` | fork-only (`arg.cpp:3678`); mainline b9867 lacks it — gate on the detected engine | **SENT BACK once**, fix re-verifying (W4) |
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
| D1 | The benchmark measures an empty window with repetitive filler: a MoE routes to a narrow expert set, and there is no depth dimension, so "tok/s at 131K" says nothing about a filled window | `bench.py:24` `"The quick brown fox…" * (prompt_tokens // 8)`; `BenchResult` `:16-20` has no depth | varied synthetic text + an optional fill depth, recorded with every result | owner's 2026-08-23 measurement (`n_cpu_moe 18` crowned on filler, `n_cpu_moe 0` beat it by 24% on varied text); `findings-r3/37 §3` | **DONE** (W4, rejected once then fixed) |
| D2 | First-load configuration dialog (quant, ctx, KV, MTP, vision, save-as-default): backend + route exist, UI partial | `LaunchDefaults` `models.py:195`, `POST /api/models/{slug}/defaults` `serve.py:4582`; only `Sidecar.tsx:249` sends ctx/kv/vision | add the dialog; send quant/spec_type/backend too | R6 report S5a | **DO** (W4, frontend) |
| D3 | Chat reattach: in-page works, across a reload it does not (streams are in-memory, no GET stream route, no server-exposed per-session running flag); incremental persistence exists but the frontend never reads the `partial` flag | `chatStore.ts:686,830,914-916`; `serve.py:4702` `_streaming` not returned by `GET /api/sessions`; `serve.py:2937-2971` writes `partial` | expose a running flag + a stream reattach route; read the `partial` flag so a reload shows the tail | R6 report S5b | **DO** (W4, frontend) |
| D4 | 4 routes have no UI caller (`/api/settings` GET+POST, `/api/models/{slug}/reprobe`, `/rename`, `/api/backup` GET, `/api/restore` POST, `/api/mcp` GET, 3 `/api/methods` routes) | R3 report item 4 | decide per route: wire or delete; `/api/settings` in particular has no UI at all | R3 report | **DO** (W4) |
| D5 | No `*.test.tsx` in the repo, so the render layer is never browser-tested (F1 passed `tsc` + 434 tests and was still a live render bug) | checkpoint 33 §"does NOT establish" | add a first render test for the harness view | R2/R5 reports | **DONE** (W5) |
| D6 | `/api/settings` docstring assumes a UI that does not exist; `probe.py:69` `_sum_other_processes` is dead in `src` (tests-only) | R3 report item 8 | fix the docstring; delete or use the dead helper | R3 report | **DONE** (W4) |

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
+ `allow_outbound_post`, both grants default OFF). **A12 is also already done**, found by the W4-B
verifier: the mid-turn prompt queue is bounded at `_QUEUE_MAX = 32` (a 429 at the door) and is popped
only on delivery or a reader stop — never on an unwind — since R3-5 `9b8d983`, an ancestor of the
wave base; `tests/test_r3_prompt_queue.py` pins both halves.

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

## Follow-ups opened by the verifiers (wave 1+2)

These were found by the independent verifiers, not by the implementers. Each is a real, recorded
gap that was left out deliberately to keep the landed commit coherent — none of them invalidates a
merged item. Ordered by severity.

| id | follow-up | where | why it was not in the landed commit |
|---|---|---|---|
| A2b | `_budget_rows` (display only) still omits the RS term, so an explorer budget row can read "fits" while the fit offloads a layer | `resolve.py` | display-only; the fit itself is correct. The explorer's `offload_pct` was fixed (A3) but the budget table was not. |
| A2c | `memtruth.py:474` still hard-codes `["--parallel","2"]` instead of `models.LAUNCH_PARALLEL` | `memtruth.py:474` | one line; the plan and the argv agree today (both 2), but the memory oracle would verify a different plan if the constant ever changed. |
| A2d | A pure-Mamba header, a header carrying an explicit `{arch}.attention.recurrent_layers` array, or a partial `ssm.*` geometry derives `recurrent_layers=0` (or charges `S` only) and gets no `rs=unknown` note | `gguf_meta.py`, `resolve.py` | needs a real header of that shape to test; flagged UNVERIFIED. Silent-zero is the failure mode to close. |
| A13b | For a **registered** engine (`engine_binary_for` prefers one at an arbitrary path; this machine has `prism-b10743-vulkan` and `prism-b10743-hip`) the fingerprint records the path but the version of the *pin*, so an in-place swap still evades it | `kvcache.py`, `server_ops.py` | the brief required "the same source the rest of the code uses"; measuring the launched `exe` (which `launch_fingerprint` already receives) is a wider change. |
| A4b | Inherited from base `7771ed3`, not from A4: a run of `?` collapses to one `[^/]`, so `??` matches one character instead of two | `tools.py` `_glob_re` | a real semantic regression against pre-R3-16; the A4 DP matcher faithfully reproduces it. Fix separately, with a fuzz test. |
| B2b | Every formerly-quiet DSH event type now surfaces as a notice, including bookkeeping that checkpoint 28 §2.1 called noise | `_dsh_runner.py`, UI | B2 mandates "an explicit notice, not a silent drop"; a compact/quiet rendering for bookkeeping types is a UX judgement worth making explicitly. |
| D1c | The default benchmark prompt changed size (2304 → 2048 words at a 2048 budget, −11.1 %) | `bench.py` | needed to make the text varied; recorded so existing calibration entries are not silently treated as comparable. |
| D1b | `depth > ctx` is not clamped or warned about | `bench.py`, `cli.py` | the user asked for that depth explicitly; clamping would be a product decision. |
| A5b | Retained false negative: a `Q2_0`=42 file on the pinned `b9867` is still not warned about, because the model-side check short-circuits before the engine is asked | `cli.py` | base behaviour, kept so that no `--verify --refuse` outcome moves. Changing it is an owner decision. |
| S2b | `engine_log._unknown_plan` returns `unexpected_splits=False` where `None` is meant; and `found=True` with zero buffer lines has no test | `engine_log.py:304-307` | harmless while `known=False` guards it, but a caller reading that key alone sees "no split problem". |
| A13c | `cli.py:1840` still says "a hash of THIRTEEN launch fields"; the tuple is now 14 | `cli.py:1840` | a comment; fixed in the test file but missed here. |
| S2c | `tests/test_engine_log_memory.py:14` says the real log "is 5 MB"; it is 357,442 bytes | test docstring | a comment. |
| A7b | `mcp_server.offered(prof=...)` has the same unknown-profile hole as A7 — no check that the value is one of the allowed set | `mcp_server.py` | a different file from A7's route; found by the implementer while grepping for profile readers. |
| A11b | `/api/restore`'s rollback is best-effort, and a crash mid-apply can still leave a partial store | `serve.py` | an all-or-nothing guarantee across a crash needs a journal and a recovery step on start, not an undo log; both limits are stated in the commit message and the 500 body. |
| A11c | `atomic_write_bytes`'s stated reason is measurably false: `atomic_write_text` opens with `newline="\n"`, which disables translation, so it round-trips CRLF bytes exactly | `atomicio.py` docstring + the A11 commit message | the helper is still a defensible byte-exact primitive; only its *reason* is wrong. Being corrected in wave 4. |
| A11d | `/api/restore`'s snapshot and rollback bypass `MemoryStore._xlock` (only the apply write holds it), so a concurrent memory write landing in the window is discarded by a rollback | `serve.py:1385-1409` | measured by the verifier with a race probe. Restore is rare and deliberate, but "memory keeps its lock" is true only of the apply phase. |
| A8b | `rigma up` records `kv_fp` but never calls `kvcache.restore`, so CLI launches always cold-start | `cli.py:2512` | adjacent to A8, not part of it; the whole point of the cache is skipped on the CLI path. |
| B7b | The `/v1` tools-conditional parameter layer injects all of `RUN_PARAMS`, including `max_tokens: 20000`, into an agent's request (the caller wins on conflict) | `serve.py` proxy | scoped to requests that carry `tools`; whether output length should be injected at all is an owner judgement. |
| A5c | `refuse=True` on `_verify_plan_or_explain` has **no production caller** — there is no `--refuse` CLI option on any tree — so the gate is test-reachable only | `cli.py:954` | recorded by the re-verification. Base behaviour is preserved in both directions; either add the option or drop the dead parameter. |
| A13d | The engine-identity memo is keyed on `(path, size, int(mtime))` with mtime truncated to whole seconds, so a same-size in-place swap within one second serves a stale identity | `engine_build.py:215-220` | pre-existing (R3-ENG-6), but now load-bearing for cache correctness. |
| A8c | A8's restore-refusal notice is real in the JSON body and the log, but **no frontend code reads the `/api/server/switch` response body** (`engineApi.switchTo` is typed `unknown`; every caller discards it), so the user-visible half is only stderr | `frontend-v2/src/lib/engineApi.ts:324-325`, `EngineSurface.tsx:102-113` | the backend half is correct and merged; making it visible needs a UI change (folded into the frontend wave plan). **DONE in W5** (`335dffe`). |
| **A17b** | **A17's split baseline is wrong and would cry wolf on every healthy launch.** `compare_plan(..., expected_splits=1)` and `tests/test_engine_log_memory.py:156` assert the owner's real log is "unexpected" — but that log IS healthy: `offloaded 65/65`, flash attention fused, `graph splits = 2`, because the token-embedding lookup runs on the host (`CPU_Mapped 322.07 MiB`). The baseline for an all-GPU or contiguous dense partial offload is **2**; a silent CPU attention fallback adds ~2 splits per attention layer (the real signal); and **MoE expert offload (`n_cpu_moe > 0`) legitimately produces many splits**, so the expectation must come from the PLAN and read "not comparable" where it cannot. | `engine_log.py` `compare_plan`, `test_engine_log_memory.py:156` | raised by **GUIDANCE.md entry 4 (12:20 UTC)**, which no wave had read — the orchestrator had to flag it in STATUS.md. Being fixed on `impl/w7b`. |
| **A17c** | **A17's VRAM divergence compares against a bare file size.** The fixture's 9275.57 MiB "actual" includes KV at ctx 65536, so comparing it to a bare file size reports a 32 % "divergence" that is just the KV cache. | `engine_log.py` | **GUIDANCE.md entry 5**. Must compare against the plan's OWN prediction for the same ctx/cache/slots, or report "not comparable". **DONE in W7** (`099e2c1`): the real log reads +2.63 %, and no prediction reads `not_comparable`. |
| **A17d** | **The VRAM axis is inert on the live surface.** The library contract is correct, but `/api/server/findings` passes no prediction, so it always reports `not_comparable`. Wiring needs `expected_vram_mb=` the running plan's own prediction at `serve.py:4326`/`:4173`, via a small `server_ops.planned_vram_mb(state)` helper derived from the fit's own code. | `serve.py:4326`, `:4173`, `server_ops.py` | Found by the A17b/A17c verifier. Assigned to `impl/w9a`. |
| **A17e** | **The derived split baseline still false-positives on a multi-GPU dense load.** `expected_splits` knows `ngl`/`n_layers`/`n_cpu_moe`/backend but not the **device count**, so a healthy all-GPU load split across two devices (CPU embedding + ROCm0 + ROCm1 = 3) is flagged `diverges`. Not reachable on the owner's single RX 9070 XT, but it is the same failure GUIDANCE 4 warned about, one device later. Fix by deriving the device count from the plan's tensor split (or from the load's device labels, disclosing the circularity), and reading `not_comparable` where it cannot. | `engine_log.py` | Found by the A17b/A17c verifier. Assigned to `impl/w9b`. |
| **A18c** | **The same wedge survives a *readable but unusable* `run.json`.** A18 guards the case where the load throws; a valid-JSON record with no `session_id` (e.g. `{}`) still dies at `run["session_id"]` and leaves the run `running` forever with its slot claimed — identical user-visible failure, same defect class. | `serve.py` `_run_loop`, `_load_run_for_loop` | Found by the A18 verifier. Assigned to `impl/w9a`. |
| **A18d** | **Three bounded nits on A18's release path.** (1) The "copy the unreadable bytes aside first" claim is not guaranteed — if `shutil.copy2` fails while `_atomic_write`'s retry succeeds, `run.json` is overwritten with **no** backup. (2) The backup name is second-granular and `not dst.exists()` silently skips, so two releases in one second keep only the first. (3) An unreadable **and** unwritable `run.json` still leaves it saying `running` (the slot IS released). | `serve.py` | Found by the A18 verifier. Assigned to `impl/w9a`. |
| **A11d** | **The restore lock does not actually serialise.** `serve._memory_store()` builds a **new** `MemoryStore` per request (`serve.py:5520`), so A11c's in-process `RLock` cannot serialise two concurrent restores; only the 10-second file lock does, and `_FileLock.acquire` then proceeds **unsynchronised** after giving up (`memory.py:327-330`). The lost-update fix therefore rests on A11c's undo-log narrowing. | `serve.py`, `memory.py` | Found by the A11c verifier. Assigned to `impl/w9a` (judge first: process-wide store, refuse-on-give-up, or document). |
| **A2d-budget** | **The budget table still shows a confident `rs_mb` where the geometry is unknown.** A2d-gap fixed the `explain` line; `_budget_rows` still emits a bare number for the both-present shape, and the Models page renders it as known. The budget row is the surface used to pick a quant. | `resolve.py` `_budget_rows` | Flagged by the A2d-gap implementer and verifier. Assigned to `impl/w9c`. |
| **B6c-mid** | **The runnable-command control cannot deliver mid-turn.** `chatStore.send` refuses mid-turn (`chatStore.ts:1072`, pinned by `chatStore.test.ts:713`) and the panel only exists during a streaming turn, so a click hits that refusal and is surfaced as a notice rather than swallowed. The server *would* queue it (`serve.py:5017`). **Product decision:** route the mid-turn case through `api.control(sid,"queue_enqueue",{text})` (UNVERIFIED that mcode resolves a queued `/name` as a command), or hide the control mid-turn. | `frontend-v2/src/chat/commands.ts`, `AgentState.tsx` | Found by the frontend wave-2 verifier. Needs an owner decision. |
| **W5F1a-res** | **Four A8c/D5 call sites are still deletable with the whole suite green:** `engine/EngineSurface.tsx:166`, `models/ModelsSurface.tsx:390` and `:646`, `chat/Sidecar.tsx:903`. | frontend | Found by the frontend wave-2 verifier; the implementer's "one residual" understated it. Assigned to frontend wave 3. |
| **A7b-note** | **A7b is a deliberate contract tightening:** a *present* `RIGMA_MCP_PROFILE` that is `""`, `"ALL"`, `"all "` or whitespace-padded now raises instead of being coerced to `all`. Fail-closed and identical to A7's route; both production callers guard with `except Exception`, so a bad env **de-registers** the MCP server rather than crashing. Worth one line in a release note. | `mcp_server.py` | Found by the A7b verifier. |
| A8d | A8's `rp.explain` half is a local list in the switch path — appended but never persisted or returned, so it is not independently user-visible | `server_ops.py:858` | same convention as the existing `step_down_notice`; recorded so it is not mistaken for a delivered channel. |
| **A18** | **The run loop's INITIAL load is unguarded: `_run_loop` does `run = _runs.load(run_id)` then `run["session_id"]`, and `_runs.load` swallows every exception and returns `None`.** A transient unreadable `run.json` at startup raises `TypeError: 'NoneType' object is not subscriptable`, the driver dies, and the run stays `running` forever with the slot claimed — exactly the failure the R3-RUN-5 comment at `serve.py:5516-5526` describes and fixes for the *inner* loop, but not for the first load. | `serve.py:5507-5508` | **Found by the full suite**: `tests/test_phase4_lifecycle.py::test_restart_reattaches_and_finishes` failed under suite concurrency with that traceback and passes in isolation. Fix: apply `_load_run_for_loop` (or its terminal-release path) to the initial load too. **High value — this is a latent "run wedged forever" bug, not a test artefact.** |
| B5b | `STANDARD_METHODS` still lists `session/load` and `session/list` after B5 deleted the two wrappers, which makes `test_every_standard_method_in_the_table_is_one_the_client_can_send` pass vacuously (its fallback filters out literals appearing anywhere in the file, including the table itself) | `harness_mcode_acp.py:69,71`; `test_acp_control.py:514-529` | the deletion was scoped to "the methods, nothing else"; decide whether the table is a wire-surface record or a client-capability record. |
| B7c | B7's `--alias` commit message says "additive — the path id stays", which is **false**: `--alias` REPLACES the served id (`server.cpp:161-164` inserts the path only when `model_alias` is empty), so `/v1/models` changes from the gguf path to the slug. It is not a compatibility fix — the verifier confirmed `/v1/chat/completions` never reads the request `model` in single-model mode, so `docs/AGENTS.md` is correct | `models.py` argv + the commit message | the alias still helps a client that validates against `/v1/models`; only the claim is wrong. |
| B7d | The `/v1` proxy treats ANY truthy `tools` as present, so a non-list `tools` (`"abc"`, `{"a":1}`) merges instead of falling back to the byte-identical passthrough | `serve.py` `_proxy_body` | harmless (the engine will reject the malformed body anyway) but short of the stated criterion; one-line fix to require a non-empty list. |
| B7e | A user-supplied `ComboFlags.alias` has no validator; llama.cpp splits it on commas into multiple aliases | `models.py` `ComboFlags` | auto-slugs are safe (`_slugify` strips non-alphanumerics), so only a hand-written alias can do it. |
| D1d | `as_measured`/`run_sweep`/`_log_rows` gate the `depth` key on `is not None`, so `depth=0` (already depth-less in `run_bench`, UNKNOWN in `measured_depth`) persists `"depth": 0` and `-5` persists `-5`; `"depth" in measured` is therefore not a valid test for `depth <= 0` | `bench.py` | latent — no consumer uses the `in` test; one-line fix to gate on `measured_depth`'s predicate. |
| D1e | `bench_text`'s docstring says "different budgets build different text", but the cache and seed key on `9*(n//8)`, so budgets 8 and 9 build identical bytes | `bench.py` | wording only. |
| D1f | `measured_filler` is written but nothing consumes it: `calibration_stale` does not invalidate a pre-marker (old repeated-sentence) entry, and `auto_calibrate` adopts a `calibrated` legacy entry without re-measuring, so an old-filler entry keeps applying its flags indefinitely | `bench.py`, `calibration` | a marker-only commit was the right scope; adding `measured_filler(entry) != FILLER_GENERATION` as a soft stale reason (~3 lines) is an owner decision, because it forces a re-measure. |
| B1b | `harness_mcode_acp.py`'s child is not detached, so on POSIX a cancelled ACP turn can orphan mcode's own subagents (it never calls `kill_tree`, so it cannot kill Rigma — pre-existing, outside B1) | `harness_mcode_acp.py:185` | B1 fixed the two Popens that reach `kill_tree`; this one is on a different path. |
| A18b | The full suite is not deterministic: `test_phase4_lifecycle.py::test_restart_reattaches_and_finishes` fails under suite concurrency and passes in isolation | `serve.py:5507` | it is a *symptom* of A18, not a flaky test — the run loop really does die on a transient read failure. Fixing A18 should make the suite deterministic again. |
| A16b | A16 made a broken fit distinguishable **in the payload** (`error` is the only discriminator), but the Models page does not read it: `RunsCell` keys on `ok`/`speed` and renders a broken fit exactly like a genuine no-fit, and the fallback omits `budget` so `budgetHint` shows the generic "does not fit this machine" | `ModelsSurface.tsx`, `engineApi.ts:125` (`Fit` has no `error`) | the payload arrives via `r.json() as Promise<T>`, a cast, so the extra key is ignored and `tsc` is unaffected — no rebuild needed to *tolerate* it, but the UI half is unwritten. Frontend follow-up. |
| A16c | A16's comment says llama.cpp prints `common_params_print_info:` once per process; the pinned PrismML build prints **two** adjacent markers (`prism-v.log:1-2`) | `engine_log.py` | harmless (adjacent, same banner) but the provenance sentence is wrong; the "last marker" rule still picks the right load. |
| A16d | The `{2,3}` GGUF version allow-list is a forward-compatibility policy: a future v4 engine's files stop importing until the parser is bumped | `gguf_meta.py` | reversible (one tuple edit) and the verifier judged it the right direction for a local tool, so it did not formally need the owner — but it deserves a one-line `OWNER-DECISIONS.md` note. |
| A16e | `_MAX_PARAMS = 1e15` is a judgement constant: ~1e12–1e15 still yields an absurd `params`, though no real file is near it (measured margin ~28,000×) and it is not evadable | `gguf_meta.py` | plus a dead `n = 0` before the per-tensor `try`. Cosmetic. |
| W5F1a | The frontend wave-1 fixes are wired but their **wiring** is untested: only pure functions and transport have assertions, so deleting the render line for the switch notice or the usage cost does not fail the suite | `frontend-v2/src/**` | confirmed by the verifier. `renderToStaticMarkup` is already proven in D5 and needs no new dependency, so real markup assertions are available — do this in the next frontend wave. |
| W5F1b | The rendered cost format `USD 0.0123` is an invented convention, not an established one in the tree | `chat/usageLine.ts` | cosmetic; pick the file's existing convention or ask the owner. |
| W5F1c | `ModelPicker.tsx` now keeps the modal open while a switch notice is pending; if the notice never clears the modal could strand | `chat/ModelPicker.tsx` | check for a clearing path and a timeout. |
| B8b | B8 lands a **disclosed-but-inert prerequisite**: both new rows are mounted correctly, but the approval trail still cannot fill (`includeRuntimeContext: false` suppresses the policy sentence; `danger-full-access` means nothing escalates; `dsh-permission-presets` needs a confining `ctx.shell` the closure does not ship) | `agent-capabilities.patch.yml`, recorded as **OD-10** | the file now says all of this; the false claim it used to make is corrected. Making it real needs an owner decision (`includeRuntimeContext: true` changes the system prompt every turn) plus a packaged sandboxed shell. |
| B2c | The projector-failure path still writes the **raw, uncollapsed, multi-line** exception message to the log; only the progress notice is capped | `_dsh_runner.py` | if §0.3 is meant to cover logs too, this is the remaining hole. |
| A9b | The A9 test is timing-based with a ~20 s ceiling (same shape as the pre-existing F14 keepalive test), so a false red under load would read as "not logged" | `test_audit_serve.py` | accepted, matching the file's existing convention. |
| C2b | `rigma sweep`'s help text enumerates the sweep axes and now omits `no-op-offload` | `cli.py:1714` | one line; skipped deliberately because `cli.py` was outside the item's file set. |
| C2c | `rigma plan` prints `rp.flags.model_dump()`, so the plan text gained `'no_op_offload': False` | `cli.py:962` | argv unchanged; no test asserts that line. Cosmetic, but the same class of issue as any new `ComboFlags` field. |
| C3b | C3's fork gate is version-blind: a registered PrismML build older than `87268f77` still declares types 142/143 and would be handed a flag it predates. Made **non-fatal** (one relaunch without the flag, logged), not correct | `server_ops.py`, `models.py` | registration carries only a `source` string, so a version check is not possible today. |
| B4b | B4 surfaces a question on the approval channel with `kind:"question"` + `awaiting:true`, but no frontend renders it: `governance.ts` `foldApproval` ignores `data.kind`/`question`/`schema`, so the row falls through to `GovernanceBlock`, which draws the Allow-once/Refuse buttons for every `awaiting` entry. Clicking one POSTs `{allow}` and the route now answers **409** ("this chat is waiting on a question, not a permission request"), surfaced as `lastError` — dead controls until the card ships, and every real `ask_user` declines after `QUESTION_WAIT_SECS`. | `frontend-v2/src/chat/governance.ts`, `AgentState.tsx:295-317`, `lib/api.ts:81-86` | the server half is complete and its event shape is what the card will read; the UI is a frontend-wave change and B4 is scoped to the server. Recorded in OD-8. |
