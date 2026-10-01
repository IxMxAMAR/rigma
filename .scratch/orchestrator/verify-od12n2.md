# Independent verification — OD12-n2 (b1f4569) + W15A-n2 (e1d6f0e)

Worktree `.scratch/wt-vfy-od12n2` on `vfy/od12n2` @ `e1d6f0e`, base `34e53d8`.
Re-derivation probes `.vfy/probe_od12n2.py` and `.vfy/probe_w15an2.py` (deleted;
no GPU, no engine launch, no live harness turn — fakes only). 10-file regression
batch: 212 passed. `ruff check src tests`: All checks passed! No source file was
left modified; no commit/merge/push.

```
ITEM: OD12-n2 (b1f4569)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes — `git checkout 34e53d8 -- src/rigma/serve.py` with the
  new test present: 1 failed, 7 deselected (`assert 409 == 200`), matching the
  implementer's claim exactly. Restored to HEAD (git status clean).
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_b4_question_channel.py -o addopts= -k "two_concurrent" --tb=short
TEST OUTPUT (before fix):
  tests\test_b4_question_channel.py F                                      [100%]
  tests\test_b4_question_channel.py:388: in test_two_concurrent_questions_each_keep_their_own_slot
      assert first.status_code == 200, first.text
  E   AssertionError: {"error":"that request is no longer the one being waited on"}
  E   assert 409 == 200
  FAILED tests/test_b4_question_channel.py::test_two_concurrent_questions_each_keep_their_own_slot
  1 failed, 7 deselected, 1 warning in 3.20s
TEST OUTPUT (after fix):
  collected 45 items / 43 deselected / 2 selected
  tests\test_b4_question_channel.py .                                      [ 50%]
  tests\test_bench_sweep.py .                                              [100%]
  2 passed, 43 deselected, 1 warning in 3.66s
  (the same command without -k, 10 files: 212 passed, 1 warning in 61.42s)
RULE-13 STATE CHECKED: (1) a question on session A answered through session B's
  route — base 409, HEAD 200 and the answer reached A's turn (`VIA-B`); the fix
  drops the per-session check the comment at serve.py:5395-5398 says exists.
  (2) an expiry while another question is live — base: q1's `finally` popped q2's
  slot, so after q1 expired q2 answered 409 ("not waiting on a permission
  request") and BOTH were reported `expired`; HEAD: both slots live, q1 200, q2
  200, decided {q1:answered, q2:answered}, both answer bodies in the turn.
  (3) the same request id answered twice — HEAD 200 then 409, one `decided`;
  base 200 then 200 with the second answer winning (HEAD is strictly better).
  (4) aborted turn: reasoned, not executed — the handler blocks in
  `Event.wait(QUESTION_WAIT_SECS=5.0)` and always reaches its `finally`; the only
  leak window is the `loop.call_soon_threadsafe` between registration and `try`.
BREAKS: src/rigma/serve.py:5721-5724 — a QUESTION slot is now global (keyed by
  request id only) and the route never compares it to `sid`, so an answer for
  session A is accepted (200) through session B's route; base refused it (409).
  Explicitly contradicts the pre-existing comment at serve.py:5395-5398 ("Keying
  by session is what lets the route refuse an answer that belongs to a different
  chat, which a global slot would have silently accepted"). Low severity: the id
  is 32 random bits shown only on A's own SSE stream and the answer still lands on
  A's question, so it is a dropped scoping check, not a misdirected answer.
NITS:
  - The slot is registered at serve.py:2633 but the `try` starts at :2647; the
    only statement between is `loop.call_soon_threadsafe` (:2637-2646). If that
    raises (closed loop at shutdown) the slot leaks permanently — unlike
    `_approvals`, which the next permission overwrites. Pre-existing shape for
    permissions; new for a dict that is never overwritten.
  - An `answer` with an empty/missing `requestId` now 409s for questions
    (serve.py:5722); base could answer a pending question by session alone. Safer,
    but a behaviour change not mentioned in the message.
  - `_questions` is never cleared at app shutdown (same as `_approvals`).
REASON: The bug is independently re-derived end to end: on the base the FIRST
  answer is refused 409 and the first handler's `finally` then pops the SECOND's
  slot, leaving a live question unanswerable — both then report `expired`; at HEAD
  both answers are 200, each with its own `approval/decided` and the right id. The
  diff is 5 hunks in serve.py, `_answer_permission` and `QUESTION_WAIT_SECS=5.0`
  are outside every hunk, and `_approvals` remains the permission-only store.
  The only reservation is the dropped cross-session scoping check.
```

```
ITEM: W15A-n2 (e1d6f0e)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes — `git checkout 34e53d8 -- src/rigma/bench.py` with the
  new test present: 1 failed, 36 deselected (`assert False is True` for
  `is_calibrated`), matching the implementer's claim exactly. Restored (clean).
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_bench_sweep.py -o addopts= -k "caches_nothing_to_calibrate" --tb=short
TEST OUTPUT (before fix):
  tests\test_bench_sweep.py F                                              [100%]
  tests\test_bench_sweep.py:892: in test_auto_calibrate_on_a_q4_tools_plan_caches_nothing_to_calibrate
      assert bench.is_calibrated("m", "Q4", "vulkan") is True
  E   AssertionError: assert False is True
  E    +  where False = <function is_calibrated at 0x...>('m', 'Q4', 'vulkan')
  FAILED tests/test_bench_sweep.py::test_auto_calibrate_on_a_q4_tools_plan_caches_nothing_to_calibrate
  1 failed, 36 deselected in 0.83s
TEST OUTPUT (after fix):
  collected 45 items / 43 deselected / 2 selected
  tests\test_b4_question_channel.py .                                      [ 50%]
  tests\test_bench_sweep.py .                                              [100%]
  2 passed, 43 deselected, 1 warning in 3.66s
  (the same command without -k, 10 files: 212 passed, 1 warning in 61.42s)
RULE-13 STATE CHECKED: (1) a HARDWARE change after the cached entry — re-sweeps:
  `is_calibrated` False, the q4_0 warning re-fires, nothing launches, a second
  entry is written under the new identity (hard_mismatch / identity-key miss).
  (2) the flags-absent entry hitting every reader — `_measured_placement` -> {},
  `resolve._apply_calibration` returns the plan with q4_0 unchanged and no
  KeyError, `expected_tg` -> None, `measured_rates()` -> (0.0, 0.0) but the header
  reads `expectedTg` (None) so it renders "No calibrated expectation" — benign.
  (3) the state the test does NOT cover and the fix gets wrong: after the
  "nothing to calibrate" entry is cached for identity A, `auto_calibrate` on a
  plan whose KV is no longer q4_0 (same model:quant:backend, same hardware) does
  NOT sweep — 0 launches, no `flags` entry written; the base swept it (3 launches,
  a real `flags` entry). The cache key (model:quant:backend:identity) omits the
  plan flags that made the decision a no-op.
BREAKS: src/rigma/bench.py:1080-1099 — the `not rows` gate writes
  `calibrated: true` under a key that does not describe the plan, so once cached,
  the first-load tune is permanently skipped for that model+quant+backend on that
  hardware (the callers gate on `not is_calibrated`, server_ops.py:922-923 and
  cli.py:2651). Recoverable only through `force_calibrate`/recalibrate. Narrow:
  requires the user to change the KV cache type away from q4_0 after a first load.
NITS:
  - The commit says the number readers report "no number rather than a zero that
    looks measured", but `serve.measured_rates` (serve.py:1232-1233) literally
    returns 0.0; only `server_ops.expected_tg` returns None. The UI is saved by
    `expected_tg`, not by `measured_rates`.
  - `calibrated: true` now means both "a config was measured" and "nothing was
    measurable"; `is_calibrated` cannot distinguish them.
REASON: The fix does what the commit says — a single entry with `calibrated:true`,
  no `flags`, `measured:{}`, `is_calibrated` True and the warning fires once — and
  the flags-absent entry is genuinely not mis-read as a measured placement by any
  of the three appliers or the two number readers. The reservation is that the
  cache key omits the plan flags that produced the decision, so it also suppresses
  a later, legitimate sweep for the same model on the same hardware.
```

OVERALL: PASS — both fixes reproduce exactly as claimed and their new tests fail
without them, with two narrow named regressions: OD12-n2 drops the cross-session
scoping check for questions, and W15A-n2's cached "nothing to calibrate" decision
is keyed too coarsely and suppresses a later calibratable plan.

## Prose

**What I did.** Worktree `.scratch/wt-vfy-od12n2` (`vfy/od12n2` @ `e1d6f0e`).
Read both full commits and both finding docs. Ran each new test at HEAD, then
reverted only that commit's source file (`git checkout 34e53d8 -- <file>`) with
the tests present, captured the failure verbatim, and restored (`git status`
clean). Ran the ten named files in one process: 212 passed. `ruff check src
tests`: clean. Wrote two throwaway probes (`.vfy/`, deleted) that drive the real
route and the real `auto_calibrate` with fakes only — no GPU, no engine, no model,
no live turn.

**OD12-n2 diff vs message.** Five hunks in `serve.py`: the `_answer_question`
docstring, `_approvals[sid] = slot` -> `_questions[request_id] = slot`, the
`finally` pop -> `_questions.pop(request_id, None)`, the new `_questions` dict,
and the route's lookup. `_answer_permission` (2555-2591) and
`QUESTION_WAIT_SECS = 5.0` (524) are outside every hunk, so the permission branch
is byte-identical in wording and expression: for `answer is None` the route still
executes `slot = _approvals.get(sid)` and still 409s with "this chat is not
waiting on a permission request". `_approvals` is written/popped only by
`_answer_permission` (2572/2588) and read only by the permission branch (5724);
`test_acp_approval.py:63,71,72`'s source assertions still hold. My probe
re-derived the bug independently: base -> q1 answer 409, registry holds only q2,
q1's cleanup pops q2, q2 answer 409, both `expired`; HEAD -> both 200, both
`answered`, both answer bodies in the turn, one `decided` per id.

**OD12-n2 what it breaks.** The per-session scoping check for questions is gone
(serve.py:5721-5724): answering session A's question through session B's route is
200 at HEAD and 409 at base. The code comment at 5395-5398 says keying by session
is exactly what prevents that. Severity is low (the id is 32 random bits delivered
only on A's own stream, and the answer still reaches A's question), but it is a
real behaviour change not mentioned in the commit message. Other states: a
same-id double answer is now 200 then 409 (base: 200 then 200, last wins) — an
improvement; an aborted turn does not leak because the handler always reaches its
`finally` within `QUESTION_WAIT_SECS`.

**W15A-n2 diff vs message.** One hunk in `bench.py` and the new test. The entry
is `save_calibration(key, {}, flags=None, calibrated=True, ...)`, which writes
`calibrated:true`, `measured:{}` and no `flags` key. I enumerated every applier and
number reader: appliers `bench._apply` (1056), `resolve._apply_calibration` (214),
`server_ops._measured_placement` (392) all gate on `entry.get("flags")` /
`entry.get("flags") or {}`; the only unguarded `entry["flags"]`
(`resolve.py:236`) is inside the `entry.get("flags")` gate. Number readers
`serve.measured_rates` (1232), `server_ops.expected_tg` (91),
`bench._recorded_depth` (230) / `_recorded_filler` (241) all use `.get("measured")
or {}` or `entry.get("measured", entry)`. Probe: `_measured_placement` -> {},
`expected_tg` -> None, `resolve._apply_calibration` unchanged, no KeyError. So it
cannot be mis-read as a measured placement.

**W15A-n2 what it breaks.** `not rows` in `auto_calibrate` is a weaker condition
than the comment's "tools-capable plan already carrying q4_0": it is true whenever
no trial ran, and `quick_configs` never returns empty, so in practice it is the
tools/q4 guard (or the quality-lever drop) — the comment is accurate. But the
cache key is `model:quant:backend:identity` and does not include the plan flags
that made the decision a no-op, so after one q4_0 first load the model is
`is_calibrated` True and the callers' `not is_calibrated` gate (server_ops.py:922,
cli.py:2651) skips `auto_calibrate` forever. Probe: same identity, plain
(calibratable) plan -> HEAD 0 launches, no flags entry; base 3 launches and a real
`flags` entry. Recoverable via `force_calibrate`/recalibrate, and it needs a plan
change after the first load, so I rate it narrow rather than a hard fail.

**Not verified.** The exact reachability of the cross-session POST through the
shipped UI (the request id is not shown to another session, so it needs a
hand-made client) and the `_questions` leak window (reasoned from the
registration/`try` gap, not executed). No frontend was built or run; the UI claim
rests on reading `telemetry.ts:12-13` and `store.ts:33-35`.
