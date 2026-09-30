# STATUS — Rigma improvement program, 2026-09-30

One line per landed item: id · commit · one-sentence outcome. Updated at every wave boundary.
Backlog with causes/evidence: `BACKLOG.md`. Owner decisions: `OWNER-DECISIONS.md`.

Base: `review/deep-audit-2026-09-22` @ `7ea1827`.
Baseline full suite: **2935 non-hardware tests, all pass (exit 0)**; `ruff check src tests` clean.

| id | wave | commit | outcome |
|---|---|---|---|
| — | 0 | _pending_ | Program docs written (BACKLOG / STATUS / HANDOFF / OWNER-DECISIONS); no source change. |
| A1 | 1 | _in flight_ | Chat-turn persist: a fully-failed save is no longer reported as saved. |
| A2 | 1 | _in flight_ | Fit: the hybrid's recurrent-state buffer is charged (per sequence × launch slots). |
| A3 | 1 | _in flight_ | `_spilled` counts the output layer, matching `_cpu_layers` (7/64, not 6/64). |
| A17/S2 | 1 | _in flight_ | The engine's own load log is parsed: model/KV/RS/compute buffers, `n_seq_max`, `graph splits`, plan-vs-actual divergence. |
| A4 | 1 | _in flight_ | `**/` glob patterns can no longer compile to a catastrophic-backtracking regex. |
| B1 | 1 | _in flight_ | `harness.kill_tree` delegates to `tools._kill_tree`; POSIX no longer always reports failure. |
| A5 | 1 | _in flight_ | The engine's authoritative compatibility verdict is no longer swallowed. |
| A6 | 1 | _in flight_ | Engine-manifest failures are distinguishable from "already current". |

Legend: `_in flight_` = implementer working · `_verifying_` = verifier running · `_merged_` = on the
integration branch · `_rejected_` = dropped after two verifier rejections.
