# STATUS — Rigma improvement program, 2026-09-30

> **ORCHESTRATOR NOTE (14:50 UTC) — GUIDANCE 6/7 actioned; a SECOND deep review is in
> `.scratch/orchestrator/deep-review-2.md`.** GUIDANCE 6's nine findings (DR1–DR9) are all fixed and merged
> (DR1/DR3/DR8 `ed52928`, DR2/DR4/DR5/DR6 `c4d3541`, DR7 `a903474`). The second review, over the diff since
> the first, found a **regression introduced by a fix that had already passed its own verifier**: A17d's
> VRAM axis compares the engine's *device* VRAM against the plan's *whole-GGUF* prediction, so a healthy
> dense-spill or expert-offload load reports `plan_divergence` on **every** launch — the same
> false-positive-on-a-healthy-load class as the original A17 bug. That is DR2-1, assigned to `impl/w13a`.
> The two reviews are the reason to distrust a per-change verifier: in both, the test was shaped to the fix.


> **GUIDANCE 4+5 (posted 12:20 UTC) — READ AND ACTIONED in wave 7.** A17's split baseline is wrong
> (a healthy all-GPU load has `graph splits = 2`, because the token-embedding lookup runs on the host;
> `compare_plan`'s `expected_splits=1` and `test_engine_log_memory.py:156` therefore flag the owner's
> real, healthy log as "unexpected"), and the VRAM divergence must compare against the plan's own
> prediction for the same ctx/cache/slots, never a bare file size (the fixture's 9275.57 MiB includes
> KV at ctx 65536). Both are assigned to `impl/w7b` (launched 13:05 UTC). The live channel
> `.scratch/orchestrator/GUIDANCE.md` is now re-read before every wave, as the brief requires.

One line per item: id · commit · one-sentence outcome. Updated at every wave boundary.
Backlog with causes/evidence: `BACKLOG.md`. Owner decisions: `OWNER-DECISIONS.md`.
Hand-off and resume instructions: `HANDOFF.md`. Wave log: `.scratch/orchestrator/WAVES.log`.

Base: `review/deep-audit-2026-09-22` @ `7ea1827` (unchanged). Program docs first committed as `8de0434`.
Baseline full suite: **2935 non-hardware tests, all pass (exit 0)**; `ruff check src tests` clean.
**Full suite on the wave-1 merges: exit 0, no failures, 3 skipped.**
**Full suite at `e745820` (waves 1–3 merged): 1 failure — `tests/test_phase4_lifecycle.py::test_restart_reattaches_and_finishes`,
which passes in isolation. Its traceback is `serve.py:5508 TypeError: 'NoneType' object is not subscriptable`,
i.e. the real **A18** bug (a transient first read killing the run loop), not a flaky test.**
**Full suite at `eac509f` (waves 1–8 merged, A18 in): 3229 passed, 3 skipped, 5 deselected, 0 failed,
exit 0, 548 s.** The `test_phase4_lifecycle` test that exposed A18 now passes **in the full run**, which
is A18's acceptance criterion. The count is up from the 2935 baseline because every item added tests.

> **FINAL SUITE — blocked by the environment, not by the code (see `REC-1` in `BACKLOG.md`).** The
> acceptance run for waves 9–15 was started three times and each time the box was already carrying
> **two orphaned full-suite runs** (`PID 364`/`23672` from 16:33 UTC, `PID 22828`/`25412` from
> 16:46 UTC) plus **72 live `tests/fake_acp_server.py` children**; the runs froze at 55 % with the
> CPU flat, exactly as `REC-1` describes. The guard refused `taskkill`, and a process this session
> did not start is not its to stop, so the orphans could only be waited out. **They exited by
> ~17:00 UTC**, and a clean full-suite run on `f6a0874` was then started with the box otherwise idle
> apart from the owner's own `sage-amd` benchmark. Per-chunk evidence gathered while the box was
> contended: `tests/test_harness_mcode_acp.py` **61 passed in 17.16 s**, `test_acp_control.py` +
> `test_acp_modes_surface.py` + `test_harness_mcode_acp_turn.py` **62 passed in 58.06 s**, the tail
> 9 files **59 passed in 13.23 s**, `test_phase4_lifecycle` + `test_resolve` + `test_resolve_calibration`
> + `test_launch_batch_ngl` + `test_quant_quality` + `test_bench_sweep` **140 passed**, all exit 0,
> `ruff check src tests` clean, and the frontend gates at HEAD green (`npm run build` exit 0 with
> `src/rigma/data/ui_v2` **idempotent**, `tsc --noEmit` exit 0, `vitest run` **56 files / 656 tests
> passed**). The one number this session could not produce is the single-process whole-suite count.

Integration branch head: `f6a0874` (plus the docs commits that follow it).

> **Commit-identity note (2026-09-30).** Early in the run the orchestrator overrode the git author for
> its merge commits, and several implementers committed under a local identity
> (`rigma-program`, `rigma-impl`, `impl-w1c`, `w1f`, `impl-w2c`, `review`, `Rigma Implementer`). Per
> `GUIDANCE.md` every commit on this branch must carry the repository identity, because the branch may
> be published. A **metadata-only `filter-branch`** re-authored all 22 affected commits to
> `IxMxAMAR <officialamrendrasingh@gmail.com>`; **trees are byte-identical** (`git diff` between the
> old and new tips is empty) and `7ea1827` and all pre-program history are unchanged. Any hash written
> into a report before that rewrite is stale — the mapping for the merged commits is
> `d8f84ab→8de0434`, `e605134→4eaec9d`, `b2cc137→82e2bbe`, `cb4d221→30e711e`, `492a1b2→28ce163`,
> `a56c0e0→5df51a6`, `21d303b→8280d1c`, `5b9f149→f93ea8e`, `dee1c43→e922c41`, `c08945b→beda4f4`,
> `854bcaf→6c83ab5`, `f6f84a3→1e23169`, `84cf575→962c00e`, `9617529→9f2011d`, `277f945→b040751`,
> `b4895fb→f8dc5b8`, `8e412f6→ca83bc0`, `091cff1→698f45f`, `be80260→17ade5c`.
> `0ab5e3a` (A2) and `97b804c` (A3) were already correct and kept their hashes.

## Wave 21–22 — the GUIDANCE 14–15 escape (ODR-1b/ODR-1c) and the optional ODR-4/ODR-8/ODR-9 (2026-09-30, ~22:36–23:15 UTC)

The orchestrator's 22:36 UTC review of the merged ODR-1 found a remaining escape of the same class, and
GUIDANCE 14–15 assigned it; the optional ODR-4/ODR-8/ODR-9 were then taken because the clock allowed.
Both follow-on waves are merged and independently verified.

| id | commit | outcome |
|---|---|---|
| ODR-1b (HIGH, security) | `2e41045` (merged `1c57d77`) | `_do_transfer` vetted only the destination FOLDER through `_write_path`, so the per-file target `_free_name(dest, src.name)` was never checked and the FILE shapes were reachable: with the default home workspace, `write_file("x/.bashrc", …)` (allowed — not the home `.bashrc`) then `copy_files(paths=["x/.bashrc"], dest=".")` created `~/.bashrc`, which Git Bash sources on every start. Each target is now passed to `_refuse_persistence_write` before `copy2`/`move`; a refused move leaves its source in place. Verified PASS-WITH-NITS: the exact scenario refused for copy AND move; every file shape refused as a source basename, including `.config/git/config` and `.config/fish/config.fish` where the folder is not a dir shape and the per-file check is the sole defence; case/trailing-dot/8.3 variants refused; the `_free_name` rename is harmless, not a bypass; ordinary transfers unaffected. Base actually created `~/.bashrc` (7 failed); fix 101 passed. |
| ODR-1c (small) | `2e41045` (merged `1c57d77`) | `_persistence_path_reason` returned `""` (ALLOW) when `resolve()` raised; a path that cannot be checked is now refused with a clear message. The branch is not reachable end-to-end (an earlier `_ws_path` resolve already refuses a NUL), so it is defence-in-depth. |
| ODR-4 (MED) | `8752c98`+`28fba74`+`19a5d7d` (merged `bf35e05`) | `methods.save_user` is now atomic (temp + `os.replace`); the restore writes the prior bytes of every file it will overwrite/delete to `~/.rigma/restore-undo/` before applying, and a boot step replays an uncommitted journal / discards a committed one. The verifier found a MED defect (E1): with no commit record, a kill after the apply committed but before the clear made the next boot **roll back a committed restore**. `19a5d7d` adds a durable `committed: true` marker before the best-effort clear and makes replay discard a committed journal; re-verified PASS (the two deterministic reproductions now fail at the buggy assertion, a mid-apply kill still rolls back, the clear is inside the lock). |
| ODR-8 (LOW) | `eba1c76` (merged `bf35e05`) | New `rigma.writelock.WRITER_LOCK` (RLock) taken by `methods.save_user`/`delete_user` and `app_settings.save`/`replace`, held outermost by `_apply_restore` for the whole method+settings region (including the journal write and clear). Order documented `WRITER_LOCK -> MemoryStore._xlock`; verified acyclic and deadlock-free in the attempted orderings. |
| ODR-9 (LOW) | `355326b` (merged `bf35e05`) | `create_session` awaits `asyncio.to_thread(sessions.create, …)`, so the seed's `json_extract` over every session body no longer runs on the event loop. Verified same args/return, exceptions propagate, and a slow create does not stall a concurrent ticker. |

**Full suite at `1c57d77` (the final head; run once, alone, output to `.scratch/orchestrator/full-suite-wave22.txt`):**
**0 failed, 3 skipped, exit rc 0; 3594 test marks in the progress stream (≈3591 passed + 3 skipped), zero
`F`/`E`.** The terminal summary line was again lost to the PowerShell 5.1 `*>` redirect; the count is
reconstructed from the progress stream and confirmed by rc 0. Wave-20's baseline was 3577 marks / 0 failed
/ 3 skipped; the +17 are ODR-1b's 7 and ODR-4/8/9's 10. `ruff check src tests` clean on every branch.
**Zero regressions from all eleven merges tonight.**

Nits carried to `BACKLOG.md`: ODR-1b's `dest.mkdir` runs before the per-file check, so a refused copy can
leave an empty `~/.config/git` directory (smallest fix: vet all targets before `mkdir`); ODR-1c is
defence-in-depth only; a UNC spelling of the machine's own LAN IP is not mapped by `_local_unc_to_drive`
and needs the write grant; a corrupt restore-journal manifest is inert litter (never cleared).

## Wave 20 — the OD-2/OD-15 deep-review findings closed (2026-09-30, ~21:48–23:xx UTC)

A new Head Agent took over at 21:45 UTC after the previous session ended without actioning GUIDANCE 11–13.
Subagent capacity was CONFIRMED (a throwaway call returned `SPAWN-OK`); this tool instance refuses explicit
`provider`/`model` params, so plain calls inherit DeepSeek-V4.1-Flash. Three implementer worktrees
(`.scratch/wt-odr1`, `wt-odr36`, `wt-odr257`) plus one frontend agent in the MAIN tree. Every item got an
independent verifier; **the ODR-1 verifier FAILED the first implementation**, and the fix was re-verified.
One merge conflict (odr1 vs odr36, a duplicated comment above `_resolve_image`'s `_credential_path_reason`
call) was resolved by merging the two comments; the merged tree was AST-parsed and targeted-tested before
the merge commit.

| id | commit | outcome |
|---|---|---|
| ODR-1 (HIGH, security) | `2d58d55`+`562b20f` (merged `7ec8712`) | A **persistence-location write denylist** (`_persistence_path_reason`) on `_write_path` and every write caller of `_ws_path` (`write_file`, `edit_file`, `undo_last_change`); reads untouched; the `allow_absolute_writes` grant and `write_allowlist` cannot reach a persistence dir. Tested with the **product-default home workspace** (32 cases, monkeypatched env — the real Startup is never touched). **The first verifier FAILED it:** (FAIL-1, HIGH) Windows strips trailing dots/spaces but `resolve()` canonicalises them only for components that already exist, so `.gitconfig.` / `Documents/PowerShell./…` / `System32\Tasks.` escaped the lexical shape match; (FAIL-2, MED) `\\localhost\c$\…\Startup` and `\\?\UNC\…` escaped the drive-letter anchors with the grant. `562b20f` adds `_strip_trailing_dots` (per-component, Windows-only) and `_local_unc_to_drive` (local admin shares; remote shares deliberately untouched). Re-verified **PASS-WITH-NITS**: 52 passed, base 46 failed/6 passed, **no reachable bypass survives** (trailing dot+`..`, 8.3+dot, mixed separators, `.`/`..`, existing-file rewrite, `COMPUTERNAME`/`127.0.0.1` UNC variants all refused). Residual (owner call): a redirected/OneDrive `Documents` PowerShell profile is not matched — needs `SHGetKnownFolderPath`. |
| ODR-3 (MED) | `2ff154e` (merged `ce4a6d3`) | `move_files` gates the **SOURCE's parent** through `_write_path` (`_move_source_parent_gate`), so removing an out-of-workspace file needs the same write grant/allowlist as writing there; `copy_files` unchanged. Verifier: read-grant-only outside move refused with the original intact; both grant routes allow; `..`, junction, 8.3 and case handled; a spy shows the destination is checked once and the source parent once (one rule, no duplicate). Sample-mode tightening confirmed real and previously uncovered. |
| ODR-6 (LOW) | `3ba6568` (merged `ce4a6d3`) | `_resolve_image` resolves before the credential/state-dir/browser denylist and passes `_unlong(p)`, so `x/../<state-dir>/secret/x.png` and `Google\x\..\Chrome\Default\Cookies.png` are refused while a normal image still loads. The author's "UNVERIFIED" ≥260-char hardening was verified. |
| ODR-2 (MED) | `1527961`+`8eb9ba1` (merged `1c7bf42`) | `default_write_allowlist` checks `is_absolute` **before** `resolve` (the old order was dead) and floors the seed against filesystem roots, the home dir and its ancestors, `%APPDATA%` and its ancestors, and persistence dirs. The verifier found and closed a `\\?\`/`\\.\`/UNC alias escape and the `%APPDATA%`-unset fallback (`8eb9ba1`); 38-case battery, no regression, `Documents` stays blocked on purpose. |
| ODR-5 | `caa1c32` (merged `1c7bf42`) | `macros.forget_method()` + `_apply_restore` clears macro trust for every method id a restore **writes or deletes**, before the first write; untouched ids keep trust; a mid-restore failure leaves trust cleared for touched ids (safe direction, documented). Prefix safety proven; uses the existing trust machinery. |
| ODR-7 backend | `55d1f32` (merged `1c7bf42`) | `_apply_restore` returns `(before, after, stale_ids)`; `/api/restore` adds an additive top-level `deleted: [ids]`; existing fields unchanged. Nit (safe direction): a hand-named/already-gone id is over-reported. |
| ODR-7 frontend | `06a636e` (direct on the integration branch) | The card says "restoring replaces the memory, settings and methods; chats and documents are kept"; every "whole store"/"anything not in the file is gone" rendered claim is gone; the deleted ids are rendered (ids+count / "nothing was deleted" / "the server did not report…" for absent or malformed — never a fabricated `0`). Rebuilt `ui_v2` in the same commit. Verifier **PASS-WITH-NITS**: 17 tests, tsc clean; nits are two stale comments and `String()` coercion of malformed ids. |

**Not merged tonight** (first work next session): **ODR-4** (on-disk restore undo replayed at startup +
atomic `save_user`), **ODR-8** (one lock for method/settings writers held by the restore), **ODR-9**
(`asyncio.to_thread(sessions.create)`). They were not started — the security fixes and their verification
used the available clock. Their file:line, scenario and smallest fix are in `.scratch/orchestrator/deep-review-od.md`.

Nits carried to `BACKLOG.md`: the ODR-1 OneDrive PowerShell-profile gap; `_glob_under`/`_fuzzy_file` still
pass a `\\?\` path to the credential rule (safe today, fragile); `write_file ".env"` is still allowed while
`read_file ".env"` refuses; the ODR-5 trust read-modify-write is non-atomic (safe direction); the ODR-7
`deleted` over-report. Owner housekeeping: a verifier's scratch file `C:\nonexistent-odr1\a.txt` could not
be deleted (the guard blocks deletes outside the workspace) and needs the owner.

**Full suite at `7ec8712` (the final head; run once, alone, output to `.scratch/orchestrator/full-suite-wave20.txt`):
0 failed, 3 skipped, exit rc 0; 3578 non-hardware tests selected (`pytest --co`).** The terminal summary
line was lost to the PowerShell 5.1 `*>` redirect (a capture artifact this session also hit in wave 19), so
the count is reconstructed: the progress stream carries 3577 test marks with **zero `F`/`E`** and 3 skips,
and pytest exited 0. Before it, the merged-tree targeted run was **89 passed** (`test_audit_sec13` +
`test_sessions_store` + `test_restore_replace` + `test_tools_hardening`), and ODR-1's own file adds **52
passed** (base: 46 failed / 6 passed). Wave-19 baseline: 3513 passed, 3 skipped, 0 failed. **Zero
regressions from the six merges.** `ruff check src tests` clean on every branch; the frontend gates at
`06a636e` were build exit 0 / vitest 17 passed / `tsc --noEmit` exit 0.

## Wave 19 — the owner's decisions implemented (2026-09-30, ~20:30–21:30 UTC)

The owner said **"I will trust you on those Recommendations"**, so every recommendation in
`OWNER-DECISIONS.md` is ACCEPTED. This wave implements the contained ones and records the rest. Every
verdict below came from a FRESH independent verifier whose brief contained rule 13; every merge was
serial and conflict-checked (`ast.parse` + a conflict-marker grep). One conflict (dtests vs statguard,
both in `test_tools_hardening.py`) was resolved by keeping both tests.

| id | commit | outcome |
|---|---|---|
| dtests | `852592d` (merged `75d9cf5`) | The two full-suite failures are fixed at the source: `test_view_image_missing_file` and `test_compiled_spec_seeds_the_plan` now build their fixtures under `tmp_path`. Verifier independently reverted the hunks and reproduced both failures. |
| OD-2 | `fcc586f` (merged `0b84d57`) | The `copy/move_files` **destination allowlist** (option 1). `write_allowlist` is a session list seeded for a new session from the owner's existing session workspaces (SQLite `json_extract`, metadata only) + RAG source folders; `_write_path` accepts the workspace, an allowlisted root, or the blanket grant. Verifier: **32 escape probes, none escaped** (prefix sibling, `..`, junctions, UNC, 8.3, case, ≥260-char paths); seed leaked no prose. |
| OD-3 | `e60d7e9` (merged `3e519d8`) + UI `25eda5b` | A **true rebuild index** (option 2): `rag.rebuild_index()` stops the recorded sidecar, deletes the index directory, then re-ingests with the current credential denylist; `POST /api/rag/reindex` (202 / 409 busy / 400 no-raggity), owner-triggered only — no timer, no startup path. Verifier confirmed the true-rebuild ordering and the 10-5 sidecar identity check. |
| OD-15 | `d76f08d`+`9d940b5` (merged `e706b5a`) + card `f2acf81` | `/api/restore` **truly replaces** (option 1): settings reset (`app_settings.replace`), stale user methods deleted via the store's own `delete_user`, deletions folded into A11's rollback region. Verifier's own probe forced a 500 mid-restore and proved settings + every method file + memory **byte-identical**; the card copy now says the truth. |
| statguard | `3278b95`+`8e76911` (merged `8d5d17d`) | The **root cause** the D:-tests exposed: `anchor_spec`'s `parent.exists()` raise escaped `compile_mission`, serve swallowed it, and `run["spec"]` stayed `None` with `fallback_spec` never stored. Guarded, plus `tools._stat_ok`; and `view_image` no longer returns a **directory** named `x.png` as `IMAGE_SENTINEL`. |
| OD-1 / OD-7 / OD-8 / OD-10 / OD-11 / OD-14 | (no code change) | Recommendations are the status quo and are recorded as IMPLEMENTED; OD-1's profile chooser gained a pinning test in `25eda5b`. OD-4/OD-5/OD-6/OD-14-option-1 are SCHEDULED with reasons. |

Nits from all five verifiers are recorded in `BACKLOG.md`'s "Wave 19 nits" section; none blocks.

**Full suite at `3e519d8` (+ the docs commit): 3513 passed, 3 skipped, 5 deselected, 0 failed** in
521.13 s — the two `D:`-path failures that had stood since the base are gone. Baseline at the
pre-wave-19 `05e0398`: **2 failed, 3477 passed, 3 skipped, 5 deselected** in 494.54 s (the same two
`D:` tests). **The run is intermittent, and the record must say so:** of four HEAD attempts, two
stalled with the CPU flat and 36 `fake_acp_server` children alive at
`test_phase4_lifecycle.py::test_restart_reattaches_and_finishes` — the REC-1-class stall already
recorded below, and the same file's known low-rate flake (it passes alone 3/3 in ~1.5 s); a third,
run under a `faulthandler` stack-dump wrapper, was stuck in `test_download_resume.py` when the dump
fired at 120 s and then died with an access violation (`0xC0000005`), so that attempt is **not**
counted as a product signal (the wrapper is the likely contributor and was not used again); the
fourth, clean attempt completed as above. The stall is therefore **not attributable to wave 19 and
not cleared of it** — the base was run once. It is the strongest reason to re-test REC-1 next session.

| id | wave | commit | outcome |
|---|---|---|
| — | 0 | `8de0434` | Program docs written (BACKLOG / STATUS / HANDOFF / OWNER-DECISIONS); no source change. |
| A1 | 1 | `4eaec9d` (`82e2bbe`) | Chat-turn persist: a fully-failed save is no longer reported as saved; the failure reaches the user on the turn-level `event: notice` channel and is logged with its `StaleWriteError` reason. |
| A17/S2 | 1 | `30e711e` (`28ce163`) | The engine's own load log is parsed: model/KV/RS/compute buffers, host-vs-device split, `n_seq_max`, `graph splits`, and plan-vs-actual divergence. Unknown is reported as UNKNOWN, never 0. **Its first split baseline (1) and its VRAM comparison were both wrong and would have flagged the owner's healthy log — corrected in A17b/A17c (`099e2c1`), which the orchestrator had to flag via GUIDANCE 4+5 because no wave had read the live channel.** |
| A4 | 1 | `5df51a6` (`8280d1c`) | `**/` glob patterns whose groups interleave with a literal separator can no longer compile to a catastrophic-backtracking regex; a DP matcher handles them (1.5 s → 0.0002 s at the base worst case). |
| A2 | 1 | `f93ea8e` (`0ab5e3a`) | Fit: the hybrid's per-sequence recurrent-state buffer is charged (149.625 MiB/seq × `LAUNCH_PARALLEL` = 299.25 MiB at 2 slots), derived from the real GGUF `ssm.*` geometry and matching the engine's own logged `RS buffer size = 149.62 MiB`. |
| A3 | 1 | `f93ea8e` (`97b804c`) | `_spilled` counts the output layer, matching `_cpu_layers` (7/64, not 6/64), including the duplicate formula the explorer used. |
| A13 | 1 | `e922c41` (`beda4f4`) | The KV-cache fingerprint records the engine's measured build identity, not just its binary path, so an in-place binary swap invalidates the cache. |
| A5 | 1 | `6c83ab5` (`1e23169`, `9f2011d`) | The engine's authoritative compatibility verdict is no longer swallowed or replaced by the stale model-side verdict; the note names its source, and the `--verify --refuse` gate behaves exactly as base in both directions. |
| A6 | 1 | `6c83ab5` (`962c00e`, `9f2011d`) | Engine-manifest failures are distinguishable from "already current", and `update_engines_manifest()`'s boolean keeps its original meaning. |
| B2 | 2 | `b040751` (`f8dc5b8`) | An unknown-but-well-formed DSH session event is projected as an explicit notice plus a once-per-process warning, instead of being silently dropped. |
| A14 | 2 | `ca83bc0` (`698f45f`) | `add_consolidated` re-validates the conflict nomination after the gate's await instead of deciding on a stale snapshot. |
| A15 | 2 | `ca83bc0` (`17ade5c`) | Method drafts are bounded (`MAX_DRAFTS = 20`, live drafts pinned) and deletable (`DELETE /api/methods/drafts/{id}`). |
| A7 | 3 | `e6e8c3d` (`10590b1`) | An unknown run profile returns 400 naming the allowed set, instead of being silently coerced to the most permissive profile; the absent-key default (`"all"`, OD-1) is unchanged. |
| A11 | 3 | `e6e8c3d` (`3a55987`) | `/api/restore` snapshots and rolls back so a mid-loop failure cannot leave a half-restored store, and the 500 names the failing section. |
| B1 | 3 | `f0b78b3` (`eb79687`, `b0d3e1e`, `6dab549`) | Harness children are detached on POSIX and `tools._signal_tree` refuses to signal a group equal to Rigma's own, so a tree-kill can no longer kill Rigma itself. **Rejected once** (the first version still signalled our own group), fixed, re-verified. |
| D1/S3 | 4 | `dbb2110` (`c8b326a`, `79ae31b`) | The benchmark prompt is lexically varied at the **legacy** size — word-identical to the pre-change module at 512/2048/131072 — and carries an optional fill depth; the 0.15 tolerance is gone. **Rejected once** (the first version silently shrank the default prompt). |
| A8 | 3 | `e745820` | KV-restore refusal is loud (log + `explain` + the `/api/server/switch` notice). The implementer deliberately did **not** clear `kv_fp` — it is the key the slot is saved under, so clearing it would make the unreadable blob immortal; the verifier accepted that deviation and corrected the row's stated consequence. |
| B7 | 3 | `d53d65d` (`67e7a58`, `a92125e`) | The model slug is served as an engine `--alias` and the `/v1` proxy applies the tool-parameter layer only to requests that carry tools. **The verifier corrected two premises:** `/v1/chat/completions` never reads the request `model` in single-model mode (so `docs/AGENTS.md` is right and this is *not* a compatibility fix), and `--reasoning-format` was already the compiled default at both pins (a guard, not a fix). |
| B5 | 4 | `e2a9ce0` (`99d7fb7`) | The two genuinely-dead `AcpClient` session wrappers are deleted; `session_activate` was **kept** because a real-pipe test is its only caller. Recorded as OD-9. |
| A16 | 4 | `46b53f5` (`bd0eaa8`, `9859970`, `7717b22`, `a2aafef`, `a71d47d`) | A broken fit is distinguishable (`error`), an unknown GGUF version is refused loudly, `params` is bounded, and `findings` attributes to the **current** run (the marker is line 1 of all 16 real logs). |
| B3 | 4 | `e770ac0` (`1ff09c8`) | The subagent descriptor carries the child id the rail fold reads. **The verifier corrected the row's premise:** the lifecycle already opened the row, so this *renames* an existing entry rather than making activity appear. |
| A9 | 4 | `9c47346` (`a807187`) | A failed idle auto-unload is logged with its reason; the engine stays loaded and the idle state is unchanged. |
| A12 | 4 | — | **Already fixed in-tree** by `9b8d983` (R3-5) before the wave base: `_QUEUE_MAX = 32` with a visible 429, popped only on delivery. Independently confirmed, so it was recorded as already-done rather than re-implemented. |
| B4 | 4 | `9c47346` (`ef83a6e`) | mcode's `ask_user` rides the existing permission channel with a minted elicitation id and a 5 s bound that returns a real decline. The UI half is unwritten (**B4b**). |
| B8 | 4 | `a78a82e` (`ed7010e`, `2cb6bb4`, `ef9bf13`) | The approval/permission services are mounted, notices are one capped line, and two docstrings are corrected. The verifier measured that the rows are a **disclosed-but-inert prerequisite** (**OD-10**) and found a **false claim in the file**, now corrected in `ebd6998`. |
| C1 | 4 | _verified, held_ | `--ctx-checkpoints`/`-ctxcp` lever, verified correct and independently mergeable; held only because it shares a branch with C3. |
| C3 | 4 | _re-verifying_ | `--reasoning-effort` lever. **Rejected once:** the fork gate was re-derived at argv-build time, so on a first run the fork-only flag reached the pinned mainline binary and llama-server would have exited in argparse. Fixed by freezing the gate onto the plan at engine-choice time, with a non-fatal relaunch fallback. |
| C2 | 4 | `7f8991c` (`e78190e`) | `--no-op-offload` is trialled as a sweep axis; the OFF case emits nothing, so every pre-existing trial's argv is byte-identical. |
| D5 | 5 | `b63ff92` | The **first `*.test.tsx`** in the tree: `HarnessFacts` extracted and pinned with real markup assertions — no new dependency. |
| A10 | 5 | `036a5d3` | A refused workspace save is surfaced and no longer reported as saved. |
| A8c | 5 | `335dffe` | The switch notice the backend returns is typed and rendered instead of discarded. |
| B6b | 5 | `6361ca2` | The ACP usage cost the store already holds is rendered. |
| OD-10 | 4 | `45a907f` | The approval trail is mounted but inert, and the file now says why (false claim removed). |
| A18 | 6 | `622f6ef` (`1bcb4fe`) | **The wedge is closed.** The run loop's **first** load is guarded and retried, and a permanently unreadable `run.json` releases the slot with a terminal `interrupted` status + reason instead of leaving the run `running` forever with its slot claimed. Found by the full suite; the verifier reproduced the exact `serve.py:5602 TypeError` on base and confirmed the slot is released in every branch. |
| D2 (backend) | 6 | `622f6ef` (`3870586`) | Launch defaults gained a GET, `backend` is accepted, `null` clears all seven fields, and a `first_load` signal exists for the dialog. The registry-model refusal is unchanged by design (**OD-11**). |
| D3a | 6 | `622f6ef` (`2b81d63`) | A turn's `streaming` state is on the session list rows and the single-session body, so a reloaded chat can tell "still streaming" from "interrupted". |
| B7d | 6 | `622f6ef` (`92f0de0`) | The `/v1` proxy's tool-parameter layer requires a non-empty `tools` list; a non-list falls back to the byte-identical passthrough. |
| B5c/B6a | 6 | `ddd0dfc` | The ACP method table no longer advertises what it cannot do, its guard can fail, and the session's **real** advertised modes are surfaced for a UI control (B6d). |
| A13b/A13d/A2b/A2d | 6 | `50f24e7` | A registered engine is identified by the binary that will actually run (not the pin), the identity memo keys on nanosecond mtime, the explorer budget charges the recurrent-state term, and an unrecognised recurrent geometry reports `unknown` instead of a silent zero. |
| A17b/A17c | 7 | `099e2c1` | **GUIDANCE 4+5.** The split baseline comes from the plan — **2** for a healthy all-GPU dense load, because the token-embedding lookup runs on the host — MoE expert offload reads `not_comparable` instead of crying wolf, and the VRAM divergence compares against the plan's own prediction for the same ctx/cache/slots (+2.63 %, not the bogus +35 %). **The owner's real log now yields no finding.** |
| B1b/D4 | 7 | `30122c3` | The ACP child is detached **and** its group is signalled, so mcode's subagents die with a cancelled turn (POSIX; Windows byte-identical and pinned). D4 closed as **not redundant**: `POST /api/methods` is the only overwrite-by-id path, pinned by a test. |
| A5c/A11c/A7b | 7 | `773d39d` | `--refuse` is reachable on `plan`/`up` and can no longer be swallowed by the fallback ladder — **`typer.Exit` IS a `RuntimeError`**, so `up` was serving the next model after a refusal; the restore region holds the store lock; an unknown MCP profile is refused instead of coerced. |
| A2d-gap/A11b/A16e | 8 | `acc5c41` | The `rs=unknown` label now fires in the one shape where the count can be wrong (an explicit `attention.recurrent_layers` array **plus** `full_attention_interval`), the CRLF contract is pinned by test, and a dead assignment is gone. |
| W5F1a/A16b/B6c/D4a | 6 | `baa1178`…`4edea5e` | Frontend wave 2: the wave-1 render fixes are testable, a broken fit renders as broken, advertised ACP commands are runnable, and `/api/settings` has a UI. |
| A17e | 9 | `eac509f` (`420ff3d`) | The split baseline now counts one run per device, derived from the load's own device labels (`1 + n_devices`), and reads `not_comparable` when it cannot derive them. A two-device healthy load yields no finding; the real single-device log is byte-identical. **Residual (recorded):** a whole device silently dropping out shrinks the label set with it, so it is invisible from a log-only surface — and the plan carries no device count to compare against. |
| A2d-budget/A2e | 9 | `a23dc7b` (`1ed4940`, `99fc00c`) | The budget row carries `rs_unknown` beside an **unchanged** `rs_mb` charge, so the surface used to pick a quant can say the recurrent-state term is an estimate. A2e closed as **cosmetic** with the arithmetic pinned (`round(X+299.25)−round(X)=300`; the "exact delta" fix would be worse, because `over_mb`'s sign is the over/headroom verdict). **The wire half only** — the Models page still does not render `rs_mb`, so a frontend follow-up is needed before the item is user-visibly closed. |
| W5F1a/B6d/B4b/D3b | 9 | `703337d` `3190295` `68b8edc` `f3861b9` `3c94cd8` | Frontend wave 3: the four remaining A8c/D5 call sites are asserted (each with a delete-the-line proof); a mode control offers the session's **own** advertised modes (never hardcoded, "unknown" when absent); mcode's question is a real form that submits `{requestId, answer}` instead of Allow/Refuse; a reloaded streaming chat no longer says "interrupted" (rail dot + bounded poll, and a poll cannot race a turn this tab owns). |
| C10 | 10 | `e852cb1` (`dc99464`) | `-b`/`-ub`/`-ngl` become settable per model, validated, and the fit still rules: a larger `ubatch` is re-priced, `ngl` is clamped with a note (never refused — the value outlives the machine state it was chosen in), negatives and `ubatch > batch` are 400s, and with **no** override every existing plan is byte-identical (`sha256 31226A7C…406DD3`, reproduced independently). PASS-WITH-NITS; the nits became C10-cli/N2/N3 and **DR2-3**. |
| C11 | 11 | `703abdc` (`8be9faf`, `3e282ae`) | **Rejected once, then fixed.** A sweep could crown `LLAMA_ATTN_ROT_DISABLE` on tokens/sec alone and persist it to calibration, after which `resolve()` applied it to **every** later launch — silently trading quantized-KV quality (+0.19 % rot ON vs **+2.42 % OFF**) for speed. Now a tools-capable sweep drops the axis and `crowned_row` refuses such a row by default; the non-tools sweep still trials and records it. The A17c VRAM constant is also corrected to the whole GGUF file (`9048.34` MiB, +227.23 MiB / +2.51 %). |
| DR1/DR8/DR3 | 11 | `ed52928` (`42ebeb1`, `8e130b1`, `faae1f9`) | GUIDANCE 6's three MED findings. `run.json` whose **read** failed is never overwritten with a 4-field stub (so `restart_run` returns **200** and reattaches instead of a permanent 409 "the run's chat session was deleted"); the first load runs inside the run loop's `try/finally`; a turn that could not be saved does no post-save housekeeping (prefix snapshot / title / auto-compact) so a concurrent writer's messages survive. **Residual (DR1-residual, `impl/w13a`):** when the read failed *and* the pointer was readable, the run is left saying `running` with no driver and is **invisible to the boot reaper**, so only Stop clears it. |
| DR2/DR4/DR5/DR6 | 11 | `c4d3541` (`d086426`, `5b47b4f`, `df010da`, `1cc924e`) | **DR2 (the owner's platform):** on Windows the ACP stop killed only the `mcode.cmd` shim, leaving the node agent and its subagents alive; the stop now runs `taskkill /F /T` on the tree first (descendants only, so Rigma is unreachable). **DR4:** a reaped leader no longer hides its process group. **DR5:** `?` in a glob is one character each (`??` no longer collapses). **DR6:** grep's content regex runs under a wall clock in a child, so a pathological pattern cannot wedge the server. |
| DR7 | 12 | `a903474` (`cf5de39`) | An engine whose `--version` cannot be measured is identified by its **own file** (`+file:<size>:<mtime_ns>`), not by the pin's identity, so an in-place swap of an unmeasurable build no longer restores the old build's KV cache. A measurable pin is byte-identical to the pre-fix fingerprint; the only orphaning shape is a pin that was itself unmeasurable, and the owner's KV store holds **zero** entries, so today's impact is zero. |
| DR2-3/C11-read/C10-nits/C10-cli | 12 | `81db56f` (`af0be83`, `f06cc47`, `23ec10c`, `2feacd3`) | **DR2-3:** the compute buffer is charged from the engine's **own** measured term (`.scratch/prism-v.log:4669` `compute buffer size = 410.28 MiB` at ubatch 512), not from the differenced-out 150 — the old base under-reserved by **1041.12 MiB** at ubatch 2048, which is the unsafe direction; ≤512 is byte-identical. **C11-read:** a quality-degrading env lever is refused at calibration **merge** time, closing the read path the C11 fix left open. **C10 nits:** a calibrated placement override is now visible in the plan's note, and the `ngl` clamp is surfaced at launch. **C10-cli:** the CLI can set the three flags. |
| W5F4-fix | 11 | `d2bd06c` `f2c3f48` `03c1680` `f3ba511` `8fc4087` | Frontend wave 4's one FAILED commit fixed: the restore confirmation claimed `/api/restore` "replaces the whole store … anything not in the file is gone", but the route **replaces memory and merges settings and methods** (proved by the verifier's TestClient probe: a second method and a setting both survived). The copy and four comments now state the truth and the two-step gate is intact. Plus four new pins (the per-quant load trigger, the client's `streaming` read against a real payload, `budgetHint`'s broken-fit branch, the Sidecar tooltip) and the dead `done` prop / empty-payload branch removed. |
| W5F5 | 12 | `1d5607c` `f3803a6` `41cf991` `cd2d076` `81b15ca` | Frontend wave 5. **DR2-4:** the launch dialog no longer invents a native context (`262144` fallback gone) — an unknown window offers only the model default and says so, and a stored context above a **known** window says the launch will clamp it (the real clamp is `server_ops.py:754`, not the `resolve.py` line the review cited). **C10 frontend:** batch/ubatch/ngl are in the dialog with the fit's own answer and the server's 400 shown verbatim. **DR2-5:** the cross-language `streaming` guard reads the live payload instead of counting source substrings. The backup card no longer promises a restore reproduces the file. |
| W5F5b | 12 | `f038121` | Frontend wave 5's one FAILED commit fixed: the fit's `ngl: 99` **"all layers" sentinel** was rendered as a literal count, so a fully-resident model read *"the fit places 99 of 64 layers on the GPU"* — the only committed fixture was the partial `ngl: 59`, which is why nothing caught it. Now "all N layers"/"every layer" with a fully-resident test. Plus the ubatch price basis, the refusal attributed to the **server**, the below-floor ctx note, and the test pinning the real 400 body. |
| DR2-1/DR2-2/DR1-res | 13 | `5241a44` (`23e0df5`, `181b7f8`, `62459bd`, `f3de580`) | **DR2-1 — the regression deep review 2 found.** A17d's VRAM axis compared the engine's *device* VRAM against the plan's *whole-GGUF* prediction, so a healthy 33-of-65-layer spill reported `plan_divergence` **on every launch** (6017.1 vs 9037.7, −33.4 %). `compare_plan` now decides comparability from the engine's own `offloaded N/M` + `CPU` model buffer (`engine_log.weights_are_device_resident`), and a non-device-resident load is `not_comparable`/`not_device_resident` and **never fires** — while the same low figure with `offloaded 65/65` still diverges and still fires. **DR2-2:** the plan prediction uses **one** registry (was a chimera of the argument's weights and a global KV registry: `4193.82` vs the right `7777.82`), memoised for 20 s (`Registry.load()` measured **6.34 ms**, `planned_vram_mb` 12.80 ms cold → 0.011 ms warm, two polls 4 → 1 parses), and off the event loop. **DR1-residual:** the reaper is now module-level and sweeps **every** run directory, not just `active()`, so a run left `running` with no driver is reconciled at boot and `restart_run` returns 200 — while an **unreadable** `run.json` is still left byte-identical. |
| W13-B | 13 | `866bf7b` (`eefb205`, `80e9094`) | The C11 verifier's two nits, both real: the sweep's quality gate read only the **per-trial override**, so a lever the plan already carried contaminated every trial — it now reads the **effective** env; and the lever match was **exact-case**, which on the owner's Windows platform is a real hole (MSVC's `getenv` is case-insensitive — established from the pinned `llama-kv-cache.cpp`, Microsoft's docs, and a ctypes probe of `ucrtbase.dll`), now matched case-insensitively on **both** the write and the read path. Two independent verifiers; second pass says MERGE as-is. |
| W14-A | 14 | `5b4bcd5` `5806031` | **OD-13 (option 1, the recorded recommendation):** a stop control for a turn this tab did **not** start, reusing the real `POST /api/sessions/{sid}/stop` — a *request* that re-reads the server's `streaming`, so `stopped:false` says "already finished" rather than claiming success and no tab can set its own stopped state. **A2d-budget UI:** the recurrent-state term now has **three** provenance states — measured (unlabelled), `rs_unknown: true` (the estimate wording), and **absent** ("the server did not say whether this is a measurement or an estimate"), which closes the item's UI half. |
| W14-B | 14 | `009730d` (`d92996e`, `222662f`, `9ee1af2`) | The 400 body no longer blames the engine — the **server** refuses the illegal batch pair and the engine merely clamps (verified against the pinned `llama-context.cpp:207`: `n_ubatch = std::min(n_batch, n_ubatch)`, unconditional, so an illegal pair is accepted and clamped, not refused). Pydantic's `"Value error, "` no longer reaches the user. `hangar.py:347`'s bare 2048 became `MIN_NATIVE_CTX`. |
| W14-E | 14 | `f6a0874` | The **flaky** `test_phase4_lifecycle.py::test_a_pause_does_not_burn_the_clock` (failed ~1 run in 10, proved pre-existing at base by 20 runs) — diagnosed as a **test race, not a product bug** (the test fabricated the pause state and released the held engine turn *before* injecting it, so a loop not yet past its first pause check stalled at iteration 0 with a bogus "over 60 minutes" reason); the test now injects while the turn is held, and the Head Agent's own 30-run loop is **0 failed / 30**. Plus `serve.py:5092`'s stale provenance comment and the registry `Combo.flags` env bypass at `resolve.py:1445-1446`. |
| W14-F | 14 | `02785bf` `7b0db3b` `cefd6fd` `97da7b5` | Frontend nits: the stale `REFUSAL_400` fixture that claimed to be the real body, the remote-stop request outliving its turn, absent `stopped` read as `false`, the absent-`rs_unknown` wording pinned at the real tooltip, and the legacy `data/ui/panels.js` context floor. |
| DR3-1/DR3-3 | 15 | `12be5cd` `85abd8a` | Deep review 3's two frontend findings: the launch dialog's native-context fallback took the **RUNNING** model's window (DR2-4 survived one fallback later), and a remote-stop request pinned to an **unnameable** turn never expired, leaving the control disabled across later turns. |
| W15-A | 15 | `8977ea8` (`8ba7ca0`, `6fd370a`) | **DR3-2:** the sweep's `q4_0` guard was **override-only** while the `env` guard beside it read the effective env — the same asymmetry DR2-6 was just fixed for, one lever over. A plan already carrying q4_0 (a calibration row: `cache_type_k`/`cache_type_v` are `_CALIBRATION_PLACEMENT_KEYS`) therefore trialled every config on a tools-capable model with the cache the guard exists to avoid. All three checks now read the effective flags, and a plan that already carries q4_0 **says so in a warning** instead of silently shrinking the sweep. **DR3-4:** the boot sweep reconciled **every** run directory, including a run another **live process** drives — reachable because the CLI's own "pass a different --port" invites a second instance on one `RIGMA_HOME`; `runs.save` now stamps the writing process (pid + create time, the same pair the engine/UI pid checks use) and the sweep skips a DIFFERENT live process while still reaping everything else, so DR1-residual stays closed. **Verdict provenance: Head-Agent-sourced and single-sourced** — the resumed session cannot spawn subagents (`SubagentDepthError`), so the proofs (224 targeted passed, ruff clean, both delete-the-line demonstrations) were run in-head and `verify-w15a.md` says so. |

| W16-A | 16 | `f51dfe0` (`15089bd`) | **W13B-1:** a sweep trial's flags were built twice — once by `run_sweep` and once by `_effective_flags`/`_effective_env` — so the two could drift and the child could be launched with flags the sweep never trialled. One construction (`bench._trial_flags`) is now the only source, and the seam test pins the CALLER of every call, not just the tail of the built list. |
| A8b | 16 | `b2f6a14` (`134efb9`) | **A8b:** `rigma up` cold-started the KV cache on **every** launch — the restore path existed but was never wired. The CLI now restores the saved cache and says so. |
| REC-1/OD-16 | 16 | `e26b6c4` (`2cb3abb`) | **REC-1/OD-16 (option 3, the recorded recommendation):** one FULL suite at a time, enforced by a per-checkout lock in `tests/conftest.py` — a second concurrent full suite exits **rc 4** with a clear message instead of corrupting the other's timings; named-file runs are **not** locked. Proved end-to-end (rc 4, rc 0 targeted, rc 0 revived-after-kill, dead-owner takeover). |
| A2d-kv | 16 | `c373516` (`180528d`, `aa7ed80`) | **A2d-kv:** the KV charge was presented as a measurement when the geometry was only partly known. A `kv_geometry_unknown` flag now marks the estimate honestly; the first fix flagged **every** pure-Mamba spec, and `aa7ed80` narrowed the derived clause with a `head_dim > 0` discriminator and a real control. |
| REC-1b | 16 | `d61dfb7` (`f19395c`) | **REC-1b:** REC-1's own review found the suite lock guarded only ONE of the **six** literal fake-server ports (`11594`–`11599`); the other five are now behind the same lock. |
| A16c/D1d | 16 | `c8aac9d` (`f2a6858`, `08dc15d`) | **A16c:** the run-start comment was wrong — a log begins with **two** `common_params_print_info:` markers per process, so a parser that treats the first as the run start truncates it. **D1d:** `bench._recorded_depth` became ONE predicate at all four write sites. |
| OD-12 server | 16 | `17bfbe8` (`acdd5b5`) | **OD-12 (option 1):** a question that expires now says so — the server emits `approval/decided` with `decision:"expired"` instead of leaving the UI spinning; `"answered"` was added in `9167aa6`. |
| A8b nits | 16 | `70608d8` (`9697163`) | **A8b-diag:** a refused KV restore was swallowed — the CLI now logs a warning and prints one `restore_failure_note`. |
| DR4-2/DR4-3 | 16 | `573025a` (`4766a94`, `2446c26`) | **DR4-2:** the driver-identity check did not execute on the path that mattered (a live pid with a different create time still looked like the recorded driver). **DR4-3:** an UNKNOWN identity wedged a run; it now fails open with a reason. |
| W5F5B-N3 | 17 | `ae97585` (`66ec122`) | **W5F5B-N3:** `MIN_LAUNCH_CTX = 2048` became the single owner of the launch floor (the inline literal and the `MIN_NATIVE_CTX` duplicate are gone; the latter is now an alias). |
| B7e/B2c | 17 | `f3bd020` (`16cd943`, `37f008c`) | **B7e:** a `ComboFlags.alias` containing an ASCII comma silently became two names; the validator now rejects it (construction-only, and that scope is now a documented note). **B2c:** a failure log line could contain a newline; `_dsh_runner._forward` now logs `_one_line(exc)`. |
| DR2-1-residual | 17 | `cb92962` (`bad1b4f`) | **DR2-1-residual:** the plan recorded the placement it was ASKED for, not the one it used — `state.json` now carries an additive `placement: {ngl, n_cpu_moe}` and the VRAM axis compares the engine's device figure against the device-side prediction. |
| B4b-schema | 17 | `873b90c` + `75960df` | **B4b-schema:** the question form's `required` check counted UI ROWS while the answer builder DROPPED blank rows, so a required container could be submitted absent (`answer:{}`). The fix makes `problemFor` ask `answerFor` whether the container would survive — one judge of emptiness. Verified PASS-WITH-NITS (22 shapes + a 4000-trial fuzz: 198 violations before, 0 after; the guard is single-judged by mutation). |
| test-hard | 17 | `4f477c9` (`01cb07d`, `7221f17`, `6388e77`) | Three tests that could not have caught the bug they name: **DR42-n1** (the create-time COMPARISON was unpinned — a bare `create_time(); return True` left all eight tests green), **SUITELOCK-n1** (`started_at == 0.0` was treated as DEAD while `conftest` itself writes `0.0`, so a lock written after a psutil failure was immediately stealable — for a lock the safe direction is ALIVE), **W16A-1** (the seam test pinned only the helper's name, not the flags the child was launched with). |
| resid | 17 | `ea822d4` (`ad36384`, `6da0460`) | **CTXFLOOR-n4:** the ctx-floor test FAKED `perform_switch`, so it never proved the raise-to path used the same floor; the inline raise-to is now a pure `_raised_launch_ctx` that `perform_switch` calls, with a sibling test. **B7E-n1:** the alias validator's construction-only scope is now a provenance NOTE plus a test. **W13B-3 deferred with no commit** — and the verifier independently confirmed the deferral: the naive fix (read `os.environ`) keeps **0 of 8** configs. |
| OD12-n2/W15A-n2 | 17 | `175f1bd` (`863dddd`, `f088d2b`) | **OD12-n2:** `serve.py` kept ONE approval slot per SESSION, so a second concurrent question overwrote the first and the first handler's `finally` popped the second's slot — the first answer 409'd and the still-live second question became unanswerable. A `_questions` dict keyed by REQUEST ID fixes it. **W15A-n2:** `auto_calibrate` on a tools-capable `q4_0` plan wrote NO entry, so the warning re-fired every load; it now caches "nothing to calibrate" (`calibrated:true`, no `flags`, `measured:{}`). |
| B4b-required | 17 | `75960df` | The required-container defect above, closed (see B4b-schema). |
| DR21R-n1 | 17 | `57adf3e` (`b16ca96`) | **DR21R-n1:** the VRAM axis's two sides sat on different bases — the engine's device figure includes the compute/reserve buffers while `memtruth.planned_mb` excludes them — so a healthy DEEP spill reported `plan_divergence`. The plan-side prediction now carries `resolve.compute_buffer_mb` (the charge DR2-3 already reserves against) **uniformly**, not gated on `_spilled > 0`; the slack was not widened. Verified PASS-WITH-NITS: the helper matches an independent first-principles figure to 0.000000 at six spill depths. |
| regr | 17 | `dd1fd60` (`a3a0db4`, `dd1fd60`) | The two narrow regressions the OD12-n2/W15A-n2 verifier named: **OD12N2-n1** (question slots became request-id-keyed and the route stopped comparing to `sid`, so session A's answer was accepted through session B's route — restored with the base's 409 body) and **OD12N2-n2** (registration moved inside the `try` so a raising `call_soon_threadsafe` cannot leak a slot). **W15AN2-n1** (the "nothing to calibrate" cache omitted the plan flags, so a later calibratable plan was skipped forever — the entry now records its decision BASIS and re-evaluates it on read). |

| UBATCH-n1 | 18 | `57bde6e` (`7d40ea5`) | The launch flags the CLI builds are now **validated before they are used**. `cli.py` merged `--batch`/`--ubatch`/`--ngl` into `rp.flags` with `model_copy(update=...)` (which does not validate) after checking only against the MODEL's stored defaults, so a `batch` that arrived from a **calibration row** (`resolve._apply_calibration`) survived with an illegal `ubatch > batch` pair — llama.cpp clamps silently, so the plan charged the requested ubatch while the engine ran the smaller one. Same class as DR21RN1-n1, one path over. The fix re-validates the resulting `ComboFlags` and refuses with the same message and exit code (one owner: `models.batch_pair_error`). The implementer probed the **combo** route the brief named, found it genuinely unreachable, and found the live one instead. Verified PASS-WITH-NITS; the verifier **corrected the commit's own arithmetic** (the tolerance is `max(512, 15%·expected)`, not a flat 512, so the false positive needs a plan under ~5470 MiB — CLIVAL-n1). |

Legend: `_in flight_` = implementer working · `_verifying_` = verifier running · `_merged_` = on the
integration branch · `_rejected once_` = sent back to the implementer with the verifier's finding.

## Verification notes worth carrying forward

- Every merged item's new test was **executed against the unmodified base** by an independent
  verifier and observed to fail there, then pass on the branch. Verdicts: `.scratch/orchestrator/verify-*.md`.
- **Wave 16 closed the single-sourced gap.** Waves 14–15 had been written *and* checked by the same
  Head Agent (the 15:55 resume could not spawn subagents). Every verdict merged after that power cut
  was handed to a **fresh independent verifier**, each also answering rule 13. **There are no
  single-sourced verdicts left in the tree.** One of them — A2d-kv — **FAILED** its re-verification
  (the derived `kv_geometry_unknown` clause flagged every pure-Mamba spec) and had to be fixed at
  `aa7ed80` and re-verified before it could stay.
- **Wave 17's own fixes were caught by the verifiers that checked them.** Two regressions introduced
  by fixes merged the same hour — `OD12N2-n1` (moving question slots to request-id keys silently
  dropped the cross-session scoping check) and `W15AN2-n1` (the cached "nothing to calibrate"
  decision omitted the flags that made it a no-op) — were found by the independent verifiers, fixed on
  `impl/regr` (`b6b04dd`), and re-verified PASS. That is the program's whole premise working.
- **FULL SUITE at `57bde6e` (the final head): 2 failed, 3477 passed, 3 skipped, 5 deselected** in
  531.86 s, and run **three times** (485 s, 546 s, 532 s) with identical results. Both failures are
  **PRE-EXISTING and ENVIRONMENTAL**, reproduced **identically at the base `7ea1827`** in a throwaway
  worktree: `test_tools_hardening.py::test_view_image_missing_file` and
  `test_autonomous_run.py::test_compiled_spec_seeds_the_plan` both use `D:/...` paths, and **`D:` is a
  BitLocker-locked drive on this host** (absent from `Get-PSDrive`), so Windows returns
  `[WinError -2144272384] This drive is locked by BitLocker` instead of "no such file". Neither file
  was touched by this program. **Zero regressions from the whole program — 59 merges and 202 other
  commits on top of the base `7ea1827`** (19 of those merges landed in the evening session alone).
- **A stall worth recording so it is not misread.** Twice a full-suite run sat at **0 CPU for over a
  minute** with **36 `fake_acp_server` children still alive** — the exact REC-1 symptom. In both cases
  the run's stdout was being buffered rather than drained; the identical command with its output
  **drained live** or **redirected to a file** completed **3 of 3 times** with the same result. Most
  likely an artifact of this session's output capture (a full pipe buffer blocking the writer), not a
  Rigma deadlock — recorded rather than dismissed, because the six-port fix's guarantee deserves a
  re-test under load and a future session should not mistake the artifact for a new bug.
  **Wave 19 correction: the "artifact of output capture" reading above is too strong.** Two further
  **file-redirected** HEAD runs stalled the same way (CPU flat, 36 `fake_acp_server` children, at
  `test_phase4_lifecycle.py::test_restart_reattaches_and_finishes`), so the stall is **real and
  intermittent**, not merely a buffered-pipe effect. It reproduced only at HEAD in the wave-19 window,
  but the base was run once and completed, and the stall is on the pre-existing REC-1 list — so it is
  neither attributed to nor cleared of wave 19. Next session: re-test REC-1 directly.
- **A flake in `test_phase4_lifecycle.py`.** Run six times in isolation it passed 12/12 five times and
  once failed `test_restart_reattaches_and_finishes` (the A18 acceptance test). In every full run that
  file passed 12/12. So it is a low-rate flake, not a regression — but it is the same class as the
  W14-E flake and should be watched.
- A2's charge moved two existing expectations on purpose (`test_launch_cache_ceiling`: ngl 58→55 and
  "7 of 64"→"10 of 64"; `test_quant_quality`: ngl 52→53). The verifier recomputed both by hand and
  agreed they are the exact consequence of the charge, not weakened assertions.
- A5/A6's first verification was PASS-WITH-NITS; the three nits (a shifted `--verify --refuse`
  outcome, the changed boolean meaning, a lost annotation) were sent back and fixed in `9f2011d`,
  then re-verified independently. That re-verification also found that **`refuse=True` has no
  production caller on any tree** — there is no `--refuse` CLI option — so the gate is test-reachable
  only. Base behaviour is preserved regardless.
- A13's verifier found a **gap the implementer did not record**: `engine_identity(backend)` keys on
  the compute backend, but `engine_binary_for` prefers a *registered* engine at an arbitrary path
  (this machine has two, `prism-b10743-vulkan` and `prism-b10743-hip`). For a registered-engine
  launch the path is recorded but the version is the pin's, so an in-place swap still evades the
  fingerprint. Not a regression (base was path-only) and the brief's "same source" requirement is
  met; carried as follow-up **A13b**.
- A7's verifier confirmed the route survey is exhaustive (only `serve.py:6340` reads a body
  `profile`; `run_profile` is not in `sessions.MUTABLE_FIELDS`) and that the `mcp_server` twin is real
  but correctly out of scope (**A7b**).
- A11's verifier confirmed the undo log is genuinely all-or-nothing at the first section, the first
  method, the last method and memory — but measured that the concurrency window is **not** exempt for
  memory (the snapshot/rollback bypass `MemoryStore._xlock`; only the apply write holds it), and that
  `atomic_write_bytes`'s stated reason in the commit message is **false** (**A11b**, **A11c**).
- D1's verifier measured the default benchmark prompt shrinking 2304 → 2048 words and the test
  tolerance being fitted around it (**D1c**), and that the docstring overclaims the MoE-router effect.
- `pwsh` is not installed; the shared semaphore script had a `param()` block that made `-m` unpassable
  (`powershell.exe -File` bound it as a parameter name). Fixed: the script reads `$args` and appends
  `-m "not hardware"` itself when no marker is given, so §0.1 can no longer be skipped by accident.
- **`git stash` is banned.** It is repository-wide, not per-worktree: one agent's `stash push` was
  consumed by another agent's `pop`, which then applied a foreign stash into a third worktree
  (`wt-w1e`), where a stray `memory.py` had to be reverted by hand. No work was lost. Always commit
  on your own branch instead.
