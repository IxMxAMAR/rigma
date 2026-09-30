# STATUS — Rigma improvement program, 2026-09-30

One line per item: id · commit · one-sentence outcome. Updated at every wave boundary.
Backlog with causes/evidence: `BACKLOG.md`. Owner decisions: `OWNER-DECISIONS.md`.
Hand-off and resume instructions: `HANDOFF.md`. Wave log: `.scratch/orchestrator/WAVES.log`.

Base: `review/deep-audit-2026-09-22` @ `7ea1827`; program docs committed as `d8f84ab`.
Baseline full suite: **2935 non-hardware tests, all pass (exit 0)**; `ruff check src tests` clean.

Integration branch head after wave 1+2 merges: `854bcaf`.

| id | wave | commit | outcome |
|---|---|---|---|
| — | 0 | `d8f84ab` | Program docs written (BACKLOG / STATUS / HANDOFF / OWNER-DECISIONS); no source change. |
| A1 | 1 | `e605134` | Chat-turn persist: a fully-failed save is no longer reported as saved; the failure reaches the user on the turn-level `event: notice` channel and is logged with its `StaleWriteError` reason. |
| A17/S2 | 1 | `cb4d221` | The engine's own load log is parsed: model/KV/RS/compute buffers, host-vs-device split, `n_seq_max`, `graph splits`, and plan-vs-actual divergence (+2275.57 MiB / +32.51 % on the real log). Unknown is reported as UNKNOWN, never 0. |
| A4 | 1 | `a56c0e0` | `**/` glob patterns whose groups interleave with a literal separator can no longer compile to a catastrophic-backtracking regex; a DP matcher handles them (1.5 s → 0.0002 s at the base worst case). |
| A2 | 1 | `5b9f149` | Fit: the hybrid's per-sequence recurrent-state buffer is charged (149.625 MiB/seq × `LAUNCH_PARALLEL` = 299.25 MiB at 2 slots), derived from the real GGUF `ssm.*` geometry and matching the engine's own logged `RS buffer size = 149.62 MiB`. |
| A3 | 1 | `5b9f149` | `_spilled` counts the output layer, matching `_cpu_layers` (7/64, not 6/64), including the duplicate formula the explorer used. |
| A13 | 1 | `dee1c43` | The KV-cache fingerprint records the engine's measured build identity, not just its binary path, so an in-place binary swap invalidates the cache. |
| A5 | 1 | `854bcaf` | The engine's authoritative compatibility verdict is no longer swallowed or replaced by the stale model-side verdict; the note names its source. |
| A6 | 1 | `854bcaf` | Engine-manifest failures are distinguishable from "already current", and `update_engines_manifest()`'s boolean keeps its original meaning. |
| B1 | 1 | _rejected once_ | `harness.kill_tree` delegating to `tools._kill_tree` would `killpg` Rigma's own process group on POSIX (harness children are not detached). Back to the implementer with the required `start_new_session` + self-pgid guard. |
| B2 | 2 | _verifying_ | An unknown-but-well-formed DSH session event is projected as an explicit notice plus a once-per-process warning, instead of being silently dropped. |
| D1 | 2 | _verifying_ | The benchmark's prompt is lexically varied (seeded, deterministic) and carries an optional fill depth, so a MoE is not placed on repetitive filler and "tok/s at N ctx" records how deep the window actually was. |
| A8 | 3 | _in flight_ | KV-restore failure: stop writing `kv_fp` so the next restart does not silently skip a 4-minute prefill. |
| A14 | 3 | _in flight_ | `add_consolidated` no longer decides its conflict verdict on a pre-`await` snapshot. |
| A15 | 3 | _in flight_ | Method drafts are bounded and can be deleted. |
| B7 | 3 | _in flight_ | `--alias` / `--reasoning-format deepseek` and a tools-conditional `/v1` parameter layer. |
| A7 | 3 | _in flight_ | An unknown run profile is rejected instead of being silently coerced to the most permissive profile. |
| A11 | 3 | _in flight_ | `/api/restore` is staged so a mid-loop failure cannot leave a half-restored store. |

Legend: `_in flight_` = implementer working · `_verifying_` = verifier running · `_merged_` = on the
integration branch · `_rejected once_` = sent back to the implementer with the verifier's finding.

## Verification notes worth carrying forward

- Every merged item's new test was **executed against the unmodified base** by an independent
  verifier and observed to fail there, then pass on the branch. Verdicts: `.scratch/orchestrator/verify-*.md`.
- A2's charge moved two existing expectations on purpose (`test_launch_cache_ceiling`: ngl 58→55 and
  "7 of 64"→"10 of 64"; `test_quant_quality`: ngl 52→53). The verifier recomputed both by hand and
  agreed they are the exact consequence of the charge, not weakened assertions.
- A5/A6's first verification was PASS-WITH-NITS; the three nits (a shifted `--verify --refuse`
  outcome, the changed boolean meaning, a lost annotation) were sent back and fixed in `9617529`,
  then re-verified independently.
- A13's verifier found a **gap the implementer did not record**: `engine_identity(backend)` keys on
  the compute backend, but `engine_binary_for` prefers a *registered* engine at an arbitrary path
  (this machine has two, `prism-b10743-vulkan` and `prism-b10743-hip`). For a registered-engine
  launch the path is recorded but the version is the pin's, so an in-place swap still evades the
  fingerprint. Not a regression (base was path-only) and the brief's "same source" requirement is
  met; carried as follow-up **A13b** in `BACKLOG.md`.

## Known follow-ups opened by wave 1+2 (recorded in BACKLOG.md)

`A2b` `_budget_rows` still omits the RS term (display only) · `A2c` `memtruth.py:474` still
hard-codes `["--parallel","2"]` · `A13b` registered-engine identity · `A4b` inherited `?`-run
collapse (`??` matches one character) · `S2b` `engine_log` unknown-load `unexpected_splits` is
`False` where `None` is meant · `B2b` the formerly-quiet DSH event types now all surface (a quiet
bookkeeping set may be wanted).
