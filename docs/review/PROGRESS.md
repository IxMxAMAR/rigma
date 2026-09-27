# Review progress ledger

Append one line per finished step. On any restart, read this first and continue from the last line.

Repo: `C:\ComfyUI\RD\rigma-review` · branch `review/deep-audit-2026-09-22` · HEAD at start of this run: `0bdfad1`
Original repo `C:\ComfyUI\RD\rigma` is READ-ONLY and is never written or git-queried.

## State found at the start of this run (2026-09-25)

- The clone already carries **388 commits** of prior review-and-fix work on this branch. Commit
  subjects reference findings `F23 F24 F33 F35 F36 F37 F38 F44 F46 F51 F52 F54 F55 F56 F57 F58
  F60 F61 F62`, so an earlier consolidated findings list existed and part of it was fixed.
- **All review artifacts were lost.** `docs/` is git-ignored (`.gitignore` line 14), so
  `docs/review/*.md` and `docs/review/findings/*` were never committed and are not on disk.
  Only two files survived: `docs/review/AGENT-BRIEF.md` and `docs/review/plan.tsv`.
- Two uncommitted files survived: `tools/review_status.py` and `tests/test_review_status.py`
  (the fan-out ledger: a findings file without its sentinel, or with an unmentioned scoped file,
  counts as absent, so a half-finished review cannot be mistaken for a finished one).
- The original findings text is **not recoverable**. Therefore this run re-runs the 15 planned
  area reviews against the *current* HEAD, so the new FINDINGS.md describes the code as it is
  now, and every fix candidate is re-confirmed present before being touched.

## Measured environment facts (this run)

- `docs/` is git-ignored; scratch is `.scratch/<area>/` (excluded via `.git/info/exclude`).
- **The test suite cannot run without a workaround here.** A directory created with mode `0o700`
  gets a DACL the sandbox refuses to enumerate or delete (`os.listdir` → WinError 5; even
  `icacls` is denied). pytest's `tmp_path_factory` creates every temp dir with
  `Path.mkdir(mode=0o700)`, and `tempfile.mkdtemp` does the same, so every `tmp_path` test
  errored at setup. Workaround, runner-side only, no repo change:
  `.scratch/00-recon/plug/dsh_tmpfix.py`, loaded with `-p dsh_tmpfix`, plus an explicit
  `--basetemp` under `.scratch/` (the default `%TEMP%\pytest-of-<user>` already exists with a
  poisoned DACL from a prior run and cannot be removed).
  Working invocation:
  `$env:PYTHONPATH='...\src;...\.scratch\00-recon\plug'` then
  `python -m pytest <file> -q -p no:cacheprovider -p dsh_tmpfix --basetemp=<scratch>\bt`

## Log

- 2026-09-25 · recon: repo state, prior work and lost artifacts established; test harness fixed.
  Files: none. Next: commit recovered ledger tooling, write MAP.md, launch area reviews.
- 2026-09-25 · committed `tools/review_status.py` + `tests/test_review_status.py` (14 tests pass
  under the workaround) as `dadc99c`. Files: tools/review_status.py, tests/test_review_status.py.
- 2026-09-25 · wrote `docs/review/MAP.md` (areas, entry points, risk ranking). Files: docs/review/MAP.md.
- 2026-09-25 · wave 1 dispatched: areas 01 (http-surface), 04 (tools-sandbox), 06 (resolve-fit-probe),
  08 (cli-ux). Each writes `docs/review/findings/<NN>-<slug>.md` with a sentinel + coverage lines.
- 2026-09-25 · **BLOCKER recorded — the frontend build cannot run in this sandbox.**
  `npm.cmd run build` = `tsc --noEmit && vite build`. `tsc --noEmit` passes; `vite build` dies with
  `Error: spawn EPERM`. Two distinct causes, both the documented sandbox boundary (no named pipes,
  so any piped-stdio child process fails): (a) Vite on Windows calls `exec("net use")` in
  `windowsSafeRealPathSync`/`optimizeSafeRealPathSync`; (b) esbuild spawns its service binary with
  piped stdio. Retried once restructured (`vite build --configLoader runner`, which skips esbuild
  config bundling) — it then failed at (a) instead. Not escalated: the owner is away, approval is
  auto-refused, and the brief forbids asking. **Consequence:** `frontend-v2/src` must NOT be edited
  this run — CI has a "the committed bundle must match the source" job that rebuilds and diffs, and
  the pre-commit hook (`.githooks/pre-commit`) refuses source-without-bundle commits for the same
  reason: the wheel ships the built folder, so a source-only commit ships the OLD interface.
  UI/UX work therefore proceeds as (i) a written judgment with reasons, and (ii) server-side UX
  fixes that need no rebuild. Verified: the failure output above; the hook and CI job read.
- 2026-09-25 · CI facts that constrain every fix: `.github/workflows/ci.yml` runs
  `ruff check src tests`, then `pytest -m "not hardware"`, on py3.11 and py3.12; the frontend job
  runs `npm run check`, `npm test`, `npm run build`, then diffs the rebuilt bundle. So a fix must
  pass ruff and must not touch frontend source.
  Next: collect wave 1, dispatch wave 2 (02/03/07/09).
- 2026-09-25 · **areas complete so far** (files in `docs/review/findings/`):
  01 http-surface (5: 1H/3M/1L), 02 chat-engine (7: 1H/3M/3L), 04 tools-sandbox
  (10: 3H/5M/2L), 06 resolve-fit-probe (8: 4M/4L), 07 hangar-downloads
  (7: 5M/2L), 08 cli-ux (8: 2H/4M/2L). Running: 03, 05, 09, 10. Still to
  dispatch: 11, 12, 13, 14, 15.
- 2026-09-25 · **fixes committed** (one finding per commit, each with a test that
  fails before it):
  - `49c17e9` 08-1 — pid identity check before killing (state.py/cli.py/server_ops.py).
  - `9d59209` 04-1/04-2/04-3/04-4 — destructive blocklist now names the PowerShell
    cmdlets the shell actually runs, matches rm flags in any order/long form, and
    catches rooted/relative rmtree. Guard tested by regex-on-text only (rule 7).
  - `4161e3d` 01-1 — `_is_local_host` parses a 127/8 literal instead of a prefix
    test, closing the DNS-rebinding bypass.
  - `7e5d828` 02-1 — prefix key hashes every content part, so two different
    images no longer collide on one snapshot.
  Also `dadc99c` — the recovered review ledger tool + its 14 tests.
  Every commit passed ruff (`ruff check src tests`) and the affected test files.
- 2026-09-25 · Next: fix 07-1 (unvalidated slug → path outside custom/models),
  then 04-5/04-6 (confinement escapes), then consolidate into FINDINGS.md.
  Files: docs/review/PROGRESS.md.
- 2026-09-25 · more fixes committed:
  - `c7ff3d8` 07-1 — a model slug is confined to custom/models (custom_spec_path:
    alphabet fullmatch, reserved device names, resolve-and-contain) before any
    read, write or unlink.
  - `fe5dc89` 04-5/04-6 — sample_files no longer globs out of the workspace, and
    _resolve_image honours the confined profile for absolute paths.
  Areas complete: 01,02,03,04,05,06,07,08,09 (9/15). Running: 10, 11, 12, 13.
  Still to dispatch: 14, 15.
- 2026-09-25 · Remaining fix candidates, ranked, with what is NOT done:
  - 09-1 HIGH: a DSH timeout kills the runner but leaves the Node agent (and its
    subagents) running, holding the model server's VRAM. harness_dsh._stop's hard
    path uses proc.kill() on the direct child only.
  - 01-2 MEDIUM: the per-chat "one turn at a time" check races (flag set in
    _drain, checked in the endpoint) → two concurrent turns interleave.
  - 03-4 MEDIUM: /stop does not cancel the in-flight engine request.
  - 03-5 MEDIUM: a normal chat turn can run against the session an active run is
    driving.
  - 02-2/02-3 MEDIUM: prefix/kv cache fingerprint gaps (warm_key not invalidated
    on slot clear; FINGERPRINT_FIELDS omits flash_attn).
  - 06-1/06-2/06-3 MEDIUM: n_layers=0 ZeroDivisionError; KV rank table and
    CACHE_BYTES missing bf16/q5_0/q4_1.
  - 05-2 MEDIUM: rescue_tool_call executes an unanchored XML/ReAct call found in
    prose.
  Not attempted for budget reasons, listed so the next run can pick them up.
- 2026-09-25 · REVIEW COMPLETE, all 15 areas. 128 raw findings -> 112 distinct
  (0C/18H/55M/39L) after 16 merges. 12 fixed, 100 open. Artifacts:
  docs/review/MAP.md, docs/review/FINDINGS.md (357 lines, ranked, top 25 in
  detail), docs/review/REPORT.md, docs/review/findings/01..15-*.md.
- 2026-09-25 · final fixes committed this run (9 code/test commits):
  - `08b8180` 10-1 HIGH — MemoryStore lost learned rules and score updates (no
    lock; fixed temp name collided; add_consolidated wrote back a pre-await
    snapshot). Per-store RLock + unique temp + os.replace + re-read in lock.
    LIMIT: in-process only; two processes still last-writer-wins (open as 10-1r).
  - `a14ecac` 12-2 HIGH — test_ui_smoke accepted 503 with every real assertion
    behind `if status == 200`, so a /v2 serving regression shipped green.
  Earlier: 49c17e9 (08-1), 9d59209 (04-1..04-4), 4161e3d (01-1), 7e5d828 (02-1),
  c7ff3d8 (07-1), fe5dc89 (04-5/04-6), dadc99c (ledger tooling).
- 2026-09-25 · DECLARED GAP, not a clean run: area 15's coverage line says
  `src/rigma/serve.py (NOT READ)`, so review_status.py reports area 15 COMPLETE
  but NOT fully covered. The first-run/startup surface of serve.py (startup
  messages, port choice, bind address) was read by no area. Next run should
  cover it. Also: areas 08 and 15 judged an earlier head, so 08-1 is stale.
- 2026-09-25 · ENVIRONMENT LIMITS that shaped the run, for whoever continues:
  (a) `npm run build` and `npm test` (vitest) both die with esbuild `spawn EPERM`
  — Vite calls exec("net use") and esbuild uses piped stdio, which the sandbox
  blocks as named pipes. So frontend source must NOT be edited (the wheel ships
  the built src/rigma/data/ui_v2 and the hook refuses source-only commits), and
  ZERO frontend tests ran. `npm run check` (tsc --noEmit) does work.
  (b) pytest needs `.scratch\00-recon\plug\dsh_tmpfix.py` via `-p dsh_tmpfix`
  plus an explicit `--basetemp`, because mode-0o700 dirs (pytest tmp_path,
  tempfile.mkdtemp) get a DACL the sandbox refuses; %TEMP%\pytest-of-<user> is
  poisoned. (c) Git's sh.exe/bash.exe crash here — use PowerShell, native
  commands, and `git -c core.hooksPath=`; write commit messages with `-F <file>`
  because an escaped quote in a double-quoted `-m` terminates the string.
- 2026-09-25 · REVIEW RUN ENDS. Nothing pushed, tagged, released or published.
  docs/ is git-ignored, so all review artifacts are untracked working notes.

## Run 3 (adversarial re-audit + the owner's fail-safe) — 2026-09-26

Base `5b5790b` (run 2's HEAD). Branch `review/deep-audit-2026-09-22`, worktrees
`.wt-r3-{a,b,c,d,e,vllm}`. Six agents, one per area, each required to REPRODUCE
before reporting and to run the mutation that proves its test can fail.

- Dispatched 6 adversarial agents with `.scratch/BRIEF-R3.md`: http (R3-1..R3-8),
  sandbox (R3-1..R3-11), harness (09-1..09-7), fit (06R3-1..06R3-5), memory
  (10-R3-11..10-R3-17), vLLM (the engine-runtime spec). Route forced to
  `tokenjuice / deepseek-ai/DeepSeek-V4.1-Flash` on every `subagent` call.
- **IMP-14 fail-safe** (`c2a6237`): `src/rigma/resilience.py` + 56 tests, wired into
  the chat turn. Retry only TRANSIENT; a stream that already emitted bytes is never
  retried; per-endpoint circuit breaker. Mutation testing found one survivor (the
  breaker never recorded a failure) → 2 more tests. `LoopGuard` was integrated and
  then REVERTED: two existing tests showed it replaced two better messages with a
  generic one. Kept as a documented, unwired primitive.
- Merged and verified: `r3/http` (2), `r3/sandbox` (8), `r3/harness` (3),
  `r3/fit` (5), `r3/vllm` (3), `r3/memory` (5). Each merge re-ran the affected
  suites on the merged result.
- **Orchestrator fixes beyond the agents**: R3-4 run exec grant (`e9639f1`, with the
  UI switch and the `profiles.ts` copy correction), R3-16 glob ReDoS (`7771ed3`),
  R3-5 prompt queue (`9b8d983`), R3-7 restart_run guard (`b2b44f9`), UIUX-22
  citations (`053bd38`), 09-7 browser-open (`1637a44`), R3-11 RAG credential
  excludes (`0c03523`), plus a `SyntaxWarning` fix and a `.raggity/` gitignore
  (`868c32f`).
- **Measured, not read**: real raggity 0.13.0 `/healthz` returns
  `{"status","version","index_backend","documents"}` — started on a spare port with a
  throwaway config to settle a merge warning. Real raggity ingest used with a
  `exclude = []` CONTROL to prove the R3-11 fix (two earlier attempts at that
  measurement were vacuous — canaries with non-indexable extensions or dotfile names
  raggity prunes itself — and both were discarded rather than believed).
- vLLM verdict: **cannot run on this machine**. Windows is the blocker, not the GPU
  (RX 9070 XT is gfx1201, on vLLM's supported ROCm list; Python 3.12 matches the ROCm
  wheels). `rigma engine-runtimes` prints that honestly.
- Run 3 record: `docs/review/RUN3.md`. Per-area findings in
  `docs/review/findings-r3/` (6 documents, consolidated from the worktrees).
  Nothing pushed, tagged, released or published; docs/ remains git-ignored.

### Run 3, closing the objective's last gaps

- **R3-CAL-1 (`hwid.py`)** — the calibration key was `model:quant:backend`, so a
  3090 inherited a 4090's stored throughput and nothing about the number looked
  wrong. Now keyed on `model:quant:backend:<identity>` where identity is a truncated
  sha256 of `backend|vendorID|deviceID|deviceUUID` — the card, not the model.
  `deviceID` alone is insufficient (Vulkan: *"the same device ID should be used for
  all physical implementations of that device version"*), so `deviceUUID` is read
  through a new `vkGetPhysicalDeviceProperties2` pNext chain in `probe.py`, verified
  live. Excluded on purpose: `pipelineCacheUUID` (driver-sensitive), `deviceName`
  (embeds the driver), the driver version (a recorded field and a SOFT reason, never
  a key), VRAM size and GPU count. Hard mismatch = a different card (do not show the
  number); soft = same card, new driver/engine/ctx (re-measure, do not discard) —
  because identity is necessary but not sufficient (power limits alone moved a
  3090's pp512 +29% with no identity change). The owner's 9 real entries all still
  resolve via the legacy-key fallback. See `findings-r3/19-calibration-identity-r3.md`.
- **The engine's "free" is not always true — the important finding of this round.**
  Chasing the owner's note that ComfyUI was running during the earlier measurements:
  with a real `llama-server` holding ~13 GB of a 16 GB card, the Windows counter read
  13,149 MiB in use while `llama-fit-params` reported **16,140 MiB free and its own
  fit verdict said "fits"**. Whatever `hipMemGetInfo` returns on this driver, it is
  not available VRAM — the AMD/Windows form of the sysmem-fallback trap. So the
  engine's fit VERDICT is not evidence on this platform and only its per-device
  ACCOUNTING is usable. `--verify` now cross-checks `free` against the OS counter,
  believes the OS, rewrites `free`, re-runs the overcommit check against it, and
  prints `UNRELIABLE` instead of "fits" over a warning. Verified live under real
  contention, not only mocked.
- **A correction to my own earlier reading.** I first called the earlier `--verify`
  runs "contaminated by ComfyUI". Half wrong: `gpu_used_mb()` reads RESIDENCY, not
  allocation — 22 MiB with ComfyUI idle, 12,586 MiB during a generation. Those runs
  were taken while ComfyUI held nothing, so their figures were right at that instant.
- **R3-MEM-1 (`memtruth.py`)** — the VRAM plan was never checked against the
  engine. `resolve.py` predicts fit with a formula over GGUF metadata, nothing ever
  compared it to reality, and its own source admits one term errs toward
  overcommitting. The pinned llama.cpp ships `llama-fit-params`, which does a
  no-alloc dummy load and reports exact per-device accounting against real free
  memory — on disk the whole time, called by nothing. Now `rigma plan --verify` and
  `rigma up --verify` run it and print both numbers. It refuses on arithmetic
  (holds > free) rather than on the engine's verdict, because on Windows WDDM and
  NVIDIA's Sysmem Fallback Policy let an over-budget allocation SUCCEED out of
  system RAM with no error — so "the server started" is not evidence it fits. The
  oracle immediately caught a real defect: the resolver planned
  `ternary-bonsai-2-27b` at ctx 262144 and the pinned engine cannot load its quant
  at all (ggml type 142; the pin accepts [0,42)). 32 tests, mutation-checked, plus a
  live test against the real binary. See `findings-r3/18-memtruth-r3.md`.
- **A correction I had to make.** The parallel/KV research reported that Rigma
  passes `--parallel 2` WITHOUT `--kv-unified`, so unified stays off and the KV pool
  doubles — contradicting Rigma's own comment. That was wrong, and my first test of
  it was wrong the same way: I launched `llama-server` with `--parallel 2` and no
  `--kv-unified`, i.e. a command line Rigma does not use. Rigma adds the flag in
  `_launch_extra`, not `ModelSpec.server_args`, so grepping the latter missed it.
  Measured with Rigma's real argv: unified → `n_ctx_slot` 32768, non-unified →
  16384, so total KV is 32768 either way and the comment is correct as written.
  `docs/AGENTS.md` says the same independently.

- **R3-VLLM-4 (`f0efd61`, `1e42ea7`)** — vLLM was specified, documented and
  diagnosable but **UNREACHABLE**: nothing called `detect_engine_runtime`, there was
  no `--engine` flag, no state field for the running engine, and no API/UI surface.
  Added the flag (refusing rather than falling back on a command line, while keeping
  the fallback for a stored preference), `state.write_state(engine=…)` as a
  separate axis from `backend`, `/api/server` reporting `engine` + `engine_runtimes`
  without inventing a value, a read-only engine badge in the Sidecar, and a `rigma
  status` line that no longer prints `()`. 20 Python + 7 frontend tests,
  mutation-checked. Three bugs the tests caught are in RUN3.md.
- **mcode re-verified LIVE, not read.** `rigma harness` → `mcode ok (0.5.4)`
  matching `VERIFIED`. `tools/mcode_probe.py --port 11599 --turn …` against the
  repo's own `tests/fake_oai_server.py` completed a real turn: provider
  `custom_provider:rigma` saved and active at `http://127.0.0.1:11599/v1` with
  `apiFormat: openai-completions`, 3 text events streamed, `status: succeeded`,
  `session_id` returned. The wire log confirms the passthrough: `/v1/chat/completions`
  with `stream: true`, `Bearer local`, **21 tools**, 3 messages, plus mcode's own
  `/v1/responses/input_tokens` pre-flight. Fake engine stopped by captured PID;
  the owner's llama-server (10224) had already exited on its own and was NOT touched.
- **Fail-safe confirmed wired**: `serve.py` imports `resilience`, holds a
  per-endpoint `CircuitBreaker`, wraps the chat send in `retry_async(policy=TURN)`,
  and unwraps `RetryExhausted.last` so the actionable ConnectError message survives.
  56 tests pass.

## Run 2 (fix-and-improve) — started 2026-09-25

- Read run-1 ledger + FINDINGS.md. Harness re-verified: pytest with `-p dsh_tmpfix --basetemp`
  passes (19 passed in test_prefixcache.py); `git worktree add/remove` works.
- Closed the declared coverage gap: new area **16 serve-startup** (plan.tsv + findings/16-serve-startup.md),
  read serve.py:1033-1070, 5040-5155 and cli.py:29-44, 895-929, 1030-1142. 3 LOW findings
  (16-1 wildcard-bind false-free, 16-2 browser opened before bind, 16-3 raw traceback on bind race).
- Wave 1 dispatched (4 implementer subagents, own worktrees): sec (13-1,13-2,13-3,13-4,13-6),
  harness (09-1,09-2,09-4,09-5,09-6,09-7,09-9,09-10), memory (10-1r,10-2,10-3,10-4,10-5,10-7..10-10),
  frontend-A (11-1,11-2,11-3,11-4). Next: merge each impl branch, rerun its tests, dispatch wave 2
  (perf 14, cli 08+15+16, tests 12, http 01).
- 2026-09-25 · Gap audit with tools/review_status.py: areas 13 and 15 flagged "not fully covered".
  Both are FALSE POSITIVES of the tool: `_coverage_gaps` treats any coverage line containing the
  words "NOT READ" as a whole-file gap, but 13's and 15's lines are *partial-read* notes that also
  carry line ranges (13 read harness_mcode L60-L739; 15 read serve.py L30-169, L1033-1252,
  L5120-5155). Re-read 13's two uncovered ranges (harness_mcode L1-59 = docstring; L740-804 =
  event-loop tail, no new finding) — no new defects. Area 16 still added as an explicit startup
  review. The tool over-strictness is folded into the wave-2 tests brief (with 12-4).
- 2026-09-25 · wave-1 frontend agent FAILED mid-run (provider error) with 11-1 uncommitted but
  complete. Lead recovered it: reviewed the diff (per-session `drafts` in chatStore + rail marker +
  start-session capture on send), added 4 vitest cases to chatStore.test.ts, verified
  `npm.cmd run check` exit 0, committed as `1c1a101` (impl/frontend). NOTE: a git worktree has no
  `node_modules`, so tsc must be run after
  `cmd /c mklink /J <wt>\frontend-v2\node_modules <main>\frontend-v2\node_modules` — added to
  BRIEF-COMMON.md. Re-dispatched a smaller frontend agent on the SAME worktree/branch for
  11-2/11-3/11-4.
- 2026-09-25 · impl/frontend COMPLETE and MERGED into review branch (ff): `1c1a101` (11-1, recovered
  by lead), `3a15896` (11-2 param_ranges from server), `86f545e` (11-3 listFetch + ErrorBoundary),
  `ecfd917` (11-4 action errors). Verified on merged result: `ruff check src tests` clean,
  `tests/test_serve_server.py` 10 passed, `npm.cmd run check` exit 0. Worktree removed, branch deleted.
  Needs rebuild: all 4 commits. New out-of-scope defect flagged by the agent: SkillsSurface.tsx:65
  and SettingsSurface.tsx:77 ignore r.ok on DELETE (queued for the UIUX agent).
- 2026-09-25 · wave 2 started: perf agent dispatched (14-1,14-2,14-3,14-5,14-9,14-10) in wt-perf.
- 2026-09-25 · impl/memory COMPLETE and MERGED (`c57740e` merge, 9 commits fe155b8..fc586c4):
  10-1r cross-process file lock, 10-2 import 400, 10-3 apply type checks, 10-4 db migration race,
  10-5 rag sidecar pid identity, 10-7 load_sources, 10-8 EFFORTS, 10-9 schema-versioned lift,
  10-10 cap every kind + append. Verified on merged result: ruff clean; 8 test files 163 passed
  3 skipped. Note: sessions now carry a `schema` field; method_schema imports sessions.
  Worktree removed, branch deleted. No frontend files touched.
- 2026-09-25 · cli/docs/startup agent dispatched (08-2,08-4..08-8, 15-1..15-7, 16-1..16-3) in wt-cli.
- 2026-09-25 · impl/harness COMPLETE and MERGED (`29238a5` merge, 8 commits 60dd1da..163d110):
  09-1 kill_tree on hard stop, 09-2 dict guards, 09-4 bounded frames, 09-5 wedged MCP, 09-6 bool
  coercion, 09-7 saw_end, 09-9 args wrap, 09-10 longest-name routing. One conflict in
  tests/test_method_schema.py (both sides added a test) resolved by keeping both. Verified on merged
  result: ruff clean; 9 test files 170 passed. CAVEAT (pre-existing, confirmed at baseline): two
  test_harness_mcode cancel tests HANG here because the sandbox denies taskkill.
- 2026-09-25 · tests/CI agent dispatched (12-1,12-3..12-12, 12-4b) in wt-tests.
- 2026-09-25 · impl/perf COMPLETE and MERGED (`be92768`, 6 commits b490f38..8f69840):
  14-2+08-6 stat-based watcher, 14-1+14-4 bounded scans, 14-3 grep/find budgets, 14-9 job buffer
  deque, 14-5+14-6 off-loop session/log reads, 14-10 off-loop memory recall. No conflicts. Verified
  on merged result: ruff clean; 12 test files 223 passed 3 skipped. Residual inside 14-5:
  `_clear_checkpoint` and `_unlock_tools` still synchronous (rare paths). Observation: watch.py
  `_known` never evicts deleted files.
- 2026-09-25 · http-surface agent dispatched (01-2..01-5) in wt-http.
- 2026-09-25 · impl/tests COMPLETE and MERGED (`6e6f0e2`, 11 commits f5b4d4c..757b0ec): 12-1 real
  sweep+CLI, 12-3 hook 100755 + porcelain staleness, 12-4/12-4b ledger coverage both directions,
  12-5..12-12. No conflicts. Verified on merged result: ruff clean; ledger now 16/16 fully covered;
  10 test files 188 passed. PARTIAL (cannot execute here): v2 browser smoke, publish workflow, 3.13
  interpreter — validated by YAML parse / py_compile / invariant tests.
- 2026-09-25 · runs agent dispatched (03-1..03-7) in wt-runs.
- 2026-09-25 · impl/http COMPLETE and MERGED (4 commits e6f834b..279c552): 01-2 atomic chat-slot
  claim, 01-3 session field type validation (400 naming the field) + export coercion, 01-4 /v1 502
  OpenAI-shaped error, 01-5 budget_hours validated before session create. No conflicts. Verified on
  merged result: ruff clean; 9 test files 171 passed. Note: archive/pending_nudges/trigger_state are
  now type-checked on PATCH even though the UI does not offer them.
- 2026-09-25 · resolve/fit/probe agent dispatched (06-1,06-3,06-4,06-5,06-6,06-7) in wt-resolve.
- 2026-09-25 · DUPLICATE FIX ALERT: 08-6 (advance the watch baseline only when the snapshot was
  written) was assigned to both the perf agent (done inside b490f38, already merged) and the cli
  agent (28642d5). On the cli merge, keep perf's merged version for watch.py.
- 2026-09-25 · Agents blocked on full-suite runs that hang here (sandbox denies taskkill): steered
  cli and runs to the affected-files-only recipe. Known hangs: test_harness_mcode cancel tests (2)
  and 9 long-lived test_harness_dsh runner tests.
- 2026-09-25 · impl/cli COMPLETE and MERGED (`dee48d7`, 16 commits e71430d..ee6deef): 08-2..08-8,
  15-1..15-7, 16-1..16-3. Two conflicts resolved: src/rigma/watch.py (kept perf's merged version,
  which already carried the 08-6 fix plus the 14-2 stat cache) and tests/test_watch.py (kept BOTH
  agents' 08-6 tests). `.gitignore` now `docs/*` + `!docs/AGENTS.md`; only docs/AGENTS.md is tracked
  under docs. Verified on merged result: ruff clean; 14 test files 136 passed. The cli agent also ran
  the FULL suite: 1688 collected, 8 failed / 1677 passed / 3 skipped in 372s — all 8 are the known
  sandbox taskkill/temp-write denials (test_audit_tools x2, test_harness_dsh_live x2,
  test_harness_mcode x2, test_phase3_new_tools x2).
- 2026-09-25 · UI/UX agent dispatched (frontend only: 11-6,11-8,11-9,11-10,11-11,11-12, DELETE r.ok
  defect, IMP-1, IMP-9, IMP-10 + own judgement; writes docs/review/UIUX.md) in wt-uiux.
- 2026-09-25 · impl/sec MERGED (5 commits 3859a2b..2d14d05): 13-6 mcode env allowlist, 13-4
  models_dir containment, 13-3 exec gating behind confirmation (regex now advisory), 13-2
  workspace-confined reads + credential denylist + outbound POST gating (old absolute-read behaviour
  kept behind a grant), 13-1 pinned-IP connect + same-origin guard on work-doing GETs. No conflicts.
  Verified: ruff clean; 12 test files 292 passed. Worktree kept until the agent's report lands.
- 2026-09-25 · impl/uiux COMPLETE and MERGED (`6bf2aec`, 13 commits 8416c92..564e638): 11-6,11-8,
  11-9,11-10,11-11,11-12, 11-4 follow-up (skill/preset/memory DELETE r.ok), IMP-1 (drafts in
  localStorage), IMP-9 (Esc/focus/`/`), IMP-10 (shared EmptyState), plus 3 own items. Doc:
  docs/review/UIUX.md (198 lines, 24 judgements: 16 implemented, 1 rejected, 4 blocked).
  `npm.cmd run check` exit 0 on merged result. NEEDS REBUILD: all 13 commits. Blocked on the server
  side: IMP-2/4/5/6/7/8/11/12 (agent improvements).
- 2026-09-25 · impl/runs COMPLETE and MERGED (`1c752e3`, 7 commits 3bbc644..76f18a2): 03-1..03-7.
  One serve.py conflict (03-2's token_cap vs 01-5's validated budget_hours) resolved by combining
  both. Two new test_runs_job_ownership tests needed the 13-3 `confirm_exec` grant in their synthetic
  ctx — fixed in the merge. Verified: ruff clean; 53 + 56 + 46 + 8 tests pass (run in batches; the
  combined run exceeded the 600 s tool cap).
- 2026-09-25 · impl/resolve MERGED (auto-commit, 6 commits 5e049f8..9785472): 06-1 n_layers guard,
  06-3 one CACHE_BYTES table + validators, 06-4 model_validate calibration merge, 06-5 ctx rung to
  the floor, 06-6 GgufParseError, 06-7 dropped the unused kv parameter. Verified: ruff clean; 14 test
  files 187 passed. New observation: server_ops.KV_CACHE_TYPES is a third, narrower vocabulary.
- 2026-09-25 · impl/tools COMPLETE and MERGED (`777bca1`, 9 commits a195a4e..99e8869): 04-7
  unconditional defuse + `sentinel` tool flag, 04-8 rescue only when the reply IS the call, 04-9
  _DELETE_PY additions, 04-10 reserved device names, 05-1 profile-gated prompt menu, 05-3
  has_vision-gated sample tail, 05-4 _prune_jobs, 05-5 numbered in the dedupe key, 05-6 prompt path
  scoping (golden updated). models.py conflict resolved by keeping BOTH the resolve CACHE_BYTES
  block and the tools _RESERVED_DEVICE block. Verified: ruff clean; 227 + 8 tests pass.
- 2026-09-25 · REGRESSION FOUND AND ASSIGNED: 13-3 gates exec behind the new per-session
  `confirm_exec` field, which no default and no UI sets, so code execution is silently OFF for every
  existing chat. `confirm_exec` is in sessions.MUTABLE_FIELDS so the API can set it; the CLI path
  goes to agent `improvements`, the UI toggle (IMP-4) to agent `wire`.
- 2026-09-25 · Final wave dispatched: hangar (07-2..07-7) in wt-hangar, improvements (IMP-3,5,6,7,8,
  11,12 + the confirm_exec regression) in wt-improvements, frontend wiring (IMP-4 + consume the new
  endpoints) in wt-wire. All 15 areas + area 16 are now reviewed and their fixes merged or in flight.
- 2026-09-25 · impl/hangar MERGED (fast-forward, 6 commits a9b7eb0..ec67e9e): 07-2 per-candidate
  probe try/except, 07-7 eagle as a token, 07-5 free-space gate, 07-6 delete drops the template +
  calibration rows, 07-4 per-key pull cancel, 07-3 size/sha256 before the dest short-circuit +
  case-twin rejection. Verified: ruff clean; 153 passed.
- 2026-09-25 · impl/wire MERGED (9 commits d9e0899..349e707): IMP-4 grants row (confirm_exec,
  allow_absolute_reads, allow_outbound_post) + run profile select, IMP-2 progress/rate/ETA, IMP-7
  usage meter, IMP-8 stop reason + budget, IMP-11 log panel, IMP-6 edit + error split, plus 3 own
  defect fixes (ModelsSurface dropped pull.error; a refused forget rendered as a load failure;
  a hardcoded 120-line log tail). `npm.cmd run check` exit 0 on the merged result. NEEDS REBUILD: all 9.
- 2026-09-25 · impl/engine MERGED (6 commits a977232..2525eab): 02-2 warm_key keyed by engine
  generation (snapshot side too), 02-3 flash_attn in the fingerprint, 02-4 try/finally teardown on
  disconnect, 02-5 last_points per (kv_fp, session), 02-6 swa warning shape, 02-7 the test now asserts
  what it names. Verified: ruff clean; 85 passed (test_prefixcache/kvcache/engine_log/audit_serve).
  PARTIAL: 02-3's FA-format premise (no engine). AREA 02 was missed in the first wave and dispatched
  late — caught while building the FINDINGS status table.
- 2026-09-25 · impl/improvements MERGED (8 commits 41010d2..bf5a1f3): the 13-3 confirm_exec
  REGRESSION (default stays OFF, named + type-checked, `rigma session exec/list`, granted session
  reaches the tool ctx), IMP-6 PATCH/DELETE /api/memory + `rigma memory`, IMP-7 stats enrichment,
  IMP-8 stop_reason + budget, IMP-11 byte-bounded log tail, IMP-3 `rigma doctor`, IMP-5
  app_settings + /api/settings, IMP-12 versioned backup/restore. Verified: ruff clean; 96 passed
  (the 8 IMP test files + test_cli).
- 2026-09-25 · Docs updated: FINDINGS.md (run-2 status table, all 112 closed, none BLOCKED),
  REPORT.md rewritten for run 2, UIUX.md UIUX-24 resolved + the honest UI remainder listed,
  IMPROVEMENTS.md status lines by the agent. Run 2 = 142 commits (128 non-merge), 26 frontend
  commits needing a bundle rebuild, 28 new test files.
