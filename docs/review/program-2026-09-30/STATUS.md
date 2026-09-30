# STATUS — Rigma improvement program, 2026-09-30

One line per item: id · commit · one-sentence outcome. Updated at every wave boundary.
Backlog with causes/evidence: `BACKLOG.md`. Owner decisions: `OWNER-DECISIONS.md`.
Hand-off and resume instructions: `HANDOFF.md`. Wave log: `.scratch/orchestrator/WAVES.log`.

Base: `review/deep-audit-2026-09-22` @ `7ea1827` (unchanged). Program docs first committed as `8de0434`.
Baseline full suite: **2935 non-hardware tests, all pass (exit 0)**; `ruff check src tests` clean.
**Full suite on the wave-1 merges: exit 0, no failures, 3 skipped.**

Integration branch head: `e6e8c3d`.

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

| id | wave | commit | outcome |
|---|---|---|---|
| — | 0 | `8de0434` | Program docs written (BACKLOG / STATUS / HANDOFF / OWNER-DECISIONS); no source change. |
| A1 | 1 | `4eaec9d` (`82e2bbe`) | Chat-turn persist: a fully-failed save is no longer reported as saved; the failure reaches the user on the turn-level `event: notice` channel and is logged with its `StaleWriteError` reason. |
| A17/S2 | 1 | `30e711e` (`28ce163`) | The engine's own load log is parsed: model/KV/RS/compute buffers, host-vs-device split, `n_seq_max`, `graph splits`, and plan-vs-actual divergence (+2275.57 MiB / +32.51 % on the real log). Unknown is reported as UNKNOWN, never 0. |
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
| B1 | 1 | _rejected once, fixed, re-verifying_ | `harness.kill_tree` delegating to `tools._kill_tree` would `killpg` Rigma's own process group on POSIX (harness children are not detached). Fixed in `8afc625` by detaching on POSIX plus a never-signal-our-own-group guard in `tools._signal_tree`. |
| D1 | 2 | _sent back once_ | The benchmark's prompt is lexically varied and carries an optional fill depth — but the verifier measured that the "unchanged default path" is in fact 11.1 % shorter and the test's 15 % tolerance hides it, so it is back with the implementer. |
| A8 | 3 | _verifying_ | KV-restore refusal is loud (log + `explain` + the `/api/server/switch` notice) instead of silent. The implementer deliberately did **not** clear `kv_fp`: it is the key the slot is saved under, so clearing it would make the unreadable blob immortal. |
| B7 | 3 | _verifying_ | `--alias` and `--reasoning-format deepseek` are emitted, and the `/v1` proxy applies the parameter layer only when `tools` is present. |
| A16 | 4 | _in flight_ | `list_models` fit exceptions, the GGUF version check, the unbounded `params`, and `engine_log` attribution (06R3-7/8/9/10). |
| A9 | 4 | _in flight_ | Idle auto-unload failure is logged with its reason instead of `except Exception: pass`. |
| A12 | 4 | _in flight_ | The chat prompt queue is bounded and is no longer popped on an unwind. |
| B4 | 4 | _in flight_ | mcode `ask_user` elicitation gets an `on_question` handler on the existing permission channel instead of being auto-declined. |
| C1 | 4 | _in flight_ | `--ctx-checkpoints`/`-ctxcp` lever. |
| C3 | 4 | _in flight_ | `--reasoning-effort` lever (fork-gated if it is fork-only). |
| B8 | 4 | _in flight_ | `dsh-user-approval` / `dsh-permission-presets` rows in the DSH capability patch. |

Legend: `_in flight_` = implementer working · `_verifying_` = verifier running · `_merged_` = on the
integration branch · `_rejected once_` = sent back to the implementer with the verifier's finding.

## Verification notes worth carrying forward

- Every merged item's new test was **executed against the unmodified base** by an independent
  verifier and observed to fail there, then pass on the branch. Verdicts: `.scratch/orchestrator/verify-*.md`.
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
