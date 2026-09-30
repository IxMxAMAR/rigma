# Independent verification — OD12N2-n1/n2 (a3a0db4) + W15AN2-n1 (dd1fd60)

Worktree `.scratch/wt-vfy-regr` on `vfy/regr` @ `dd1fd60`, base `175f1bd`.
Both commits re-read in full; `.scratch/orchestrator/verify-od12n2.md` read.
No GPU, no engine launch, no model load, no live harness turn — fakes only.
Probes `.vfy/probe_n1.py`, `.vfy/probe_n2.py`, `.vfy/probe_n2b.py`,
`.vfy/probe_cost.py` (deleted; `git status` clean, HEAD unchanged, nothing
committed/merged/pushed). 13-file regression batch: 208 passed. `ruff check src
tests`: All checks passed!

```
ITEM: OD12N2-n1/n2 (a3a0db4)
VERDICT: PASS
FAILS-WITHOUT-FIX: yes — `git checkout 175f1bd -- src/rigma/serve.py` with the
  two new tests present: `2 failed, 8 deselected`, matching the implementer's
  claim exactly — cross-session `assert 200 == 409` with the HIJACK body
  (`{"ok":true,...,"answer":{"path":"HIJACK"}}`) and the leak test
  `assert {...} == {}` (the slot left behind). Restored to HEAD each time;
  `git status --porcelain` empty.
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_b4_question_channel.py -o addopts= -k "another_session or raising_ask_publish" --tb=short
TEST OUTPUT (before fix):
  collected 10 items / 8 deselected / 2 selected
  tests\test_b4_question_channel.py FF                                     [100%]
  __ test_an_answer_for_another_session_is_refused_and_does_not_reach_the_turn __
  tests\test_b4_question_channel.py:447: in test_an_answer_for_another_session_is_refused_and_does_not_reach_the_turn
      assert hijack.status_code == 409, hijack.text
  E   AssertionError: {"ok":true,"requestId":"q-fb86b080","allow":null,"answer":{"path":"HIJACK"}}
  E   assert 200 == 409
  _____________ test_a_slot_does_not_survive_a_raising_ask_publish ______________
  tests\test_b4_question_channel.py:501: in test_a_slot_does_not_survive_a_raising_ask_publish
      assert questions == {}, (
  E   AssertionError: the slot leaked: a question whose ask never published is still registered and could be answered
  E   assert {'q-47242035'... unset>, ...}} == {}
  2 failed, 8 deselected, 1 warning in 1.41s
TEST OUTPUT (after fix):
  collected 10 items / 8 deselected / 2 selected
  tests\test_b4_question_channel.py ..                                     [100%]
  2 passed, 8 deselected, 1 warning in 1.01s
RULE-13 STATE CHECKED: (1) a PERMISSION asked on A answered through B's route —
  409 `{"error":"this chat is not waiting on a permission request"}` and the
  slot was NOT woken; through A's own route 200 `{"ok":true,...,"allow":true}`
  and the turn received `PERM_ALLOW=True`. The permission path is untouched and
  still per-session. (2) TWO questions on the SAME session answered through that
  session — both slots carried `sid == A`, both answers 200, both bodies reached
  the turn (`ONE` and `TWO`). (3) a slot injected with NO `sid` key — 409, slot
  not woken (fail-closed). (4) an empty session id — `/api/sessions//approval`
  is a 404 (not routable); a `%20` sid 409s. (5) a CORRECT-sid answer after the
  turn ended — 409 `{"error":"this chat is not waiting on a question"}`.
BREAKS: nothing
NITS:
  - The cross-session refusal uses the QUESTION wording, not the true
    pre-regression base's (9167aa6) shared-branch permission wording. Status
    (409) and body shape (`{"error": <str>}`) match; the wording change is
    documented in the commit's DELIBERATELY NOT DONE.
  - A question slot lacking `sid` (reachable only by hand-injection — the old
    in-memory shape cannot survive a process restart) is now refused. Fail-closed,
    but it is the one shape the updated
    `test_an_answer_cannot_win_after_the_expiry_has_been_claimed` had to be
    taught about.
  - `_questions` is still never cleared at app shutdown (pre-existing, carried
    over from `_approvals`).
REASON: The diff is two hunks; `_answer_permission`, `QUESTION_WAIT_SECS = 5.0`,
  the per-question `slot["lock"]` expiry claim and the exactly-once
  `approval/decided` block are outside every changed line (the only changed
  serve.py lines are the slot dict, the indentation move of the
  registration+publish into the `try`, and the route's new `sid` comparison), so
  the prior fix's two-concurrent-questions behaviour is preserved and the two
  named regressions are closed. Every RULE-13 state I could invent still behaves:
  the permission round-trip is unchanged, same-session concurrency still works,
  and the new check fails closed on a slot that names no session.
```

```
ITEM: W15AN2-n1 (dd1fd60)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes — `git checkout 175f1bd -- src/rigma/bench.py` with the
  two new tests present: `2 failed, 37 deselected`, matching the implementer's
  claim exactly (`AssertionError: the now-calibratable plan was not swept` /
  `assert []`; `TypeError: is_calibrated() takes 3 positional arguments but 4
  were given`). Restored to HEAD; `git status --porcelain` empty.
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_bench_sweep.py -o addopts= -k "rechecked_when_the_plan_changes or is_calibrated_gate_rechecks" --tb=short
TEST OUTPUT (before fix):
  collected 39 items / 37 deselected / 2 selected
  tests\test_bench_sweep.py FF                                             [100%]
  ____ test_a_cached_nothing_to_calibrate_is_rechecked_when_the_plan_changes ____
  tests\test_bench_sweep.py:966: in test_a_cached_nothing_to_calibrate_is_rechecked_when_the_plan_changes
      assert launched, "the now-calibratable plan was not swept"
  E   AssertionError: the now-calibratable plan was not swept
  E   assert []
  _____ test_the_is_calibrated_gate_rechecks_a_cached_nothing_to_calibrate ______
  tests\test_bench_sweep.py:1000: in test_the_is_calibrated_gate_rechecks_a_cached_nothing_to_calibrate
      assert bench.is_calibrated("m", "Q4", "vulkan", q4.flags) is True
  E   TypeError: is_calibrated() takes 3 positional arguments but 4 were given
  2 failed, 37 deselected in 2.34s
TEST OUTPUT (after fix):
  collected 39 items / 37 deselected / 2 selected
  tests\test_bench_sweep.py ..                                             [100%]
  2 passed, 37 deselected in 1.98s
RULE-13 STATE CHECKED: (1) a HARDWARE change after a `no_calibrate` entry — the
  stored entry's `hardware.device_id` moved to `0x9999`: `is_calibrated(q4)` went
  True -> False (hard_mismatch) and `auto_calibrate` re-swept (2 fake launches)
  and re-measured the row. (2) a `no_calibrate` entry followed by a real sweep
  that FAILS (every launch raises) — `crowned_row` returns None so nothing is
  saved; the stale `calibrated/no_calibrate` row survives and BOTH loads
  re-swept. (3) `prune_calibration` with ONLY `no_calibrate` entries (three
  distinct identities for one model) — all three kept, none dropped; `prune` does
  not read `no_calibrate` at all. (4) `expected_tg("m","Q4","vulkan")` -> None and
  `serve.measured_rates()` -> (0.0, 0.0) on a `no_calibrate` entry;
  `resolve._apply_calibration` leaves the plan byte-identical (origin still
  `calculator`). (5) the added cost: `is_calibrated(model, quant, backend, flags)`
  on a cached `no_calibrate` entry makes 2 `Registry.load()` calls (both
  `_tools_capable` and `_capabilities` re-read and re-parse every
  model/combo/gpus JSON) and measured 3.2-4.9 ms/launch vs 0.068 ms for the
  flags-less call, i.e. ~3-5 ms added to every launch while the entry exists.
  (6) supersede in BOTH directions: measured -> no-op save removes the `flags`
  key and writes `no_calibrate` with `measured={}`; `is_calibrated` then False
  for a calibratable plan and True for a q4 plan; a later real measurement
  restores `flags`/numbers and removes the `no_calibrate` key.
BREAKS: nothing
NITS:
  - `is_calibrated`'s `no_calibrate` branch returns before `entry.get(
    "calibrated")` is read, so an entry carrying `no_calibrate` but NOT
    `calibrated` reports True. Unreachable in-tree — both writers that pass
    `no_calibrate` (bench.py:1144 legacy-adopt, bench.py:1182 no-op) also pass
    `calibrated=True` — but it is a latent short-circuit, verified directly by
    construction.
  - No migration for an f088d2b-era no-op row (`calibrated: true`, no `flags`,
    no `no_calibrate`): such a row still suppresses a later calibratable plan
    forever, because `_no_calibrate_expired` keys on `no_calibrate`. Unreachable
    on this machine: `~/.rigma/calibration.json` has 9 entries, all 8
    `calibrated` rows carry a `flags` key, and none has the no-op shape.
  - A `no_calibrate` entry whose re-sweep fails is retried on every load while
    still carrying `calibrated: true` (state 2 above). Correct for the two
    loaders because they pass `flags`, but a flags-less caller would still skip.
  - The added ~3-5 ms per launch is new cost on the hot path, not a regression.
REASON: The fix does exactly what the commit says — `_sweep_would_drop_all`
  re-derives the same predicate `run_sweep` drops with, a still-no-op plan is a
  cache hit (asserted by the passing new test: second q4 load launches nothing
  and re-fires no warning), a now-calibratable plan sweeps and overwrites the
  decision in place, and the `flags`/`no_calibrate` pops make each direction of
  supersede total. Every reader of `flags` (`resolve._apply_calibration:220,242`,
  `auto_calibrate._apply:1128`, `server_ops._measured_placement:392`) gates on a
  truthy `flags`, and the no-op row has none, so it can never be read as a
  measured placement; `prune_calibration` is untouched and keeps the same single
  identity-keyed row it always did.
```

OVERALL: PASS — both commits reproduce their claimed before/after output byte for
byte, close the two regressions named in `.scratch/orchestrator/verify-od12n2.md`,
break nothing across the 13 named files (208 passed), and leave `ruff` clean; the
reservations are one latent `is_calibrated` short-circuit, an unreachable
f088d2b-era migration gap, and ~3-5 ms of new per-launch cost.

## Prose

**Item 1 (a3a0db4).** The commit is two hunks in `serve.py` plus the test file. I
reverted `src/rigma/serve.py` alone to `175f1bd` with both new tests present and
got `2 failed, 8 deselected` with the exact bodies claimed, then restored. At
HEAD the two new tests pass. Changed-line inspection of
`git diff 175f1bd HEAD -- src/rigma/serve.py` (50 lines, 2 hunks) confirms
`_answer_permission`, `QUESTION_WAIT_SECS`, the `with slot["lock"]` expiry claim
and the exactly-once `approval/decided` emit appear in NO changed line — only the
slot dict gains `"sid": sid`, the registration+publish block moves inside the
`try` (indentation only; the order register -> publish -> wait is preserved, so a
route call can only arrive after the slot exists), and the route gains
`if slot is not None and slot.get("sid") != sid: slot = None`. The route's 409 is
the existing `{"error": "this chat is not waiting on a question"}` path, so
status and body shape match; only the wording differs from the true
pre-regression base `9167aa6`, whose single shared branch said "…not waiting on a
permission request" — the commit documents that choice. My independent probe
drove the real route with a fake ACP driver: a permission on A refused 409 through
B's route without waking the slot and accepted 200 through A's route with
`PERM_ALLOW=True` reaching the turn; two questions on one session both answered
200 with both bodies in the turn; a `sid`-less slot refused fail-closed; a
correct-sid answer after the turn ended 409; an empty sid 404.

**Item 2 (dd1fd60).** Reverting `src/rigma/bench.py` alone to `175f1bd` gave
`2 failed, 37 deselected` exactly as claimed; restored. `_sweep_would_drop_all`
re-derives the same predicate `run_sweep` drops with (`_drop_uncalibratable` is
now the single filter, called from both), and `auto_calibrate` passes
`quick_configs` with the identical `caps=_capabilities(slug)`, so the decision
basis and the sweep cannot drift. I enumerated every reader: the three appliers
(`resolve._apply_calibration`, `auto_calibrate._apply`, `server_ops.
_measured_placement`) all gate on a truthy `flags`; the number readers
(`serve.measured_rates`, `server_ops.expected_tg`, `bench._recorded_depth`,
`_recorded_filler`) all use `entry.get("measured") or {}`; `prune_calibration`
reads only `hardware`/`date` and never `no_calibrate`. A `no_calibrate` row
carries no `flags` key and `measured={}`, and I confirmed by probe that both
directions of supersede are total (`save_calibration(no_calibrate=...)` pops
`flags`; a `no_calibrate=None` save pops `no_calibrate`). The one real
reservation is the latent `is_calibrated` short-circuit; the f088d2b migration gap
is real in principle but absent from the owner's 9-entry calibration.json.

**Item 3.** The 13 named files in one process: `208 passed, 1 warning in 15.50s`
(the warning is the pre-existing starlette/httpx deprecation). This reconciles
with the implementer's three batches (41 + 57 + 110, overlapping
`test_bench_sweep.py` once).

**Item 4.** Provenance comments are extensive and cite
`verify-od12n2.md` line ranges. No dead code: `_drop_uncalibratable` (2 call
sites), `_sweep_would_drop_all` (2), `_no_calibrate_basis` (1),
`_no_calibrate_expired` (1) are all used. No new `try/except`; no new
dependency. The optional `flags` is documented in the `is_calibrated` docstring
("Without `flags` there is nothing to re-check against, so the decision
stands") and is safe because the only two production callers
(`cli.py:2651`, `server_ops.py:1002`) both pass the plan's flags; every other
caller is a test asserting the coarse behaviour.

**Not verified.** The `loop.call_soon_threadsafe` leak window was exercised only
through the test's monkeypatch of `BaseEventLoop.call_soon_threadsafe`, not by
actually closing a loop at shutdown. No frontend was built or run. The 3-5 ms
cost was measured with the real registry on this machine; `Registry.load` is
uncached, so the count (2 per call) is the durable fact and the millisecond
figure is machine-dependent. No GPU work, no engine launch, no model load, no
live harness turn was performed at any point.
