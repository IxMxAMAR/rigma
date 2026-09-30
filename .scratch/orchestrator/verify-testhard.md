# Independent verification — impl/test-hard (DR42-n1, SUITELOCK-n1, W16A-1)

Branch `vfy/testhard` @ `6388e77` (impl/test-hard), base `573025a`, integration HEAD `f3bd020`.
Worktree `.scratch/wt-vfy-testhard`. No commit/merge/push; no doc edits; no engine/GPU/model.
Mutations applied and reverted one at a time; `git diff -- src/` empty and `git status` clean after every revert.

```
ITEM: DR42-n1 (01cb07d)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes (verified)
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_run_loop_first_load.py -k "different_create_time or boot_reaper or still_reaped or unknown_driver"
TEST OUTPUT (before fix): mutation = src/rigma/state.py:73 `return abs(psutil.Process(pid).create_time() - want) < 1.0` -> `psutil.Process(pid).create_time(); return True`
  FAILED tests/test_run_loop_first_load.py::test_the_boot_reaper_reconciles_a_live_pid_with_a_different_create_time
  E  AssertionError: a live pid with a DIFFERENT create time was read as the recorded driver, so a recycled pid still makes a dead driver look alive
  E  assert True is False            (fails at tests\test_run_loop_first_load.py:804, the intended assert)
  1 failed, 8 passed, 23 deselected, 1 warning in 0.70s   [EXIT=1]
  OLD BODY (base 573025a file, same mutation, same -k): 0 failed — the only failure is the new test;
  a mixed run of new+base files selected by the same -k gave "1 failed, 16 passed, 46 deselected"
  (8 old tests passed, 8 other new-file tests passed). Implementer's "1 failed, 8 passed, 23 deselected" reproduced verbatim.
TEST OUTPUT (after fix): .........  [100%]
  9 passed, 23 deselected, 1 warning in 0.69s   [EXIT=0]
RULE-13 STATE CHECKED: (a) stamp create time in the FUTURE (real + 100_000) — test still PASSED:
  driver_is_live_elsewhere=False, status=interrupted, so no wedge. (b) pid recycled by a process created
  sub-millisecond/<1s away (stamp real - 0.5) — the committed test FAILS there because the code's 1.0 s
  tolerance reads the live pid as the SAME driver (run skipped, not reconciled). This is the code's
  tolerance, not a test bug; the committed test cannot cover it without contradicting the tolerance.
BREAKS: nothing caused by this commit. (Pre-existing flake, see NITS.)
NITS:
  1. The row's claim "a recycled pid cannot make a dead driver look alive" has a 1.0 s blind spot:
     a pid recycled by a process whose create time is within 1.0 s of the recorded time is read as the
     same driver (rule-13 probe b). Pre-existing tolerance in src/rigma/state.py:73, not introduced here.
  2. Pre-existing timing flake in this file, unchanged by the branch and reproduced at base: see BREAKS note in the W16A block / item 3.
REASON: The named mutation is caught by the new test at the exact comparison assert, and the base test
body stays green under that same mutation, so the create-time COMPARISON is now load-bearing; src is
byte-identical after revert. The only residue is the pre-existing 1.0 s tolerance, which the row's prose
overstates.
```

```
ITEM: SUITELOCK-n1 (7221f17)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes (verified)
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_suite_lock.py -k "unreadable_create_time or psutil_failure"
TEST OUTPUT (before fix): mutation = tests/conftest.py `if recorded <= 0.0: return True` -> `return False`
  FAILED tests/test_suite_lock.py::test_an_unreadable_create_time_does_not_make_a_live_owners_lock_stealable
  E  assert False is True
  FAILED tests/test_suite_lock.py::test_a_lock_written_after_a_psutil_failure_is_not_immediately_stealable
  E  AssertionError: a lock written after a psutil failure was immediately stolen
  2 failed, 1 passed, 23 deselected in 0.54s   [EXIT=1]
  (third test, the dead-pid bound, passes under the mutation by design)
  OLD BODY (base 573025a tests/test_suite_lock.py, same mutated conftest, whole file): 23 passed in 2.58s [EXIT=0]
TEST OUTPUT (after fix): 26 passed in 1.85s   [EXIT=0]   (base file had 23 tests; +3)
RULE-13 STATE CHECKED: (a) a lock whose owner is a LIVE process with a NEGATIVE create time
  (`started_at = -5.0`): `_owner_is_alive(os.getpid(), -5.0) is True` and `_acquire_suite_lock` refuses —
  honoured, PASS (the `<= 0.0` branch covers negatives). (b) two full suites started within the same
  second: the writer records psutil's float create_time, so the second acquire compares the live owner's
  real time against it and matches (<1.0 s), so it is refused — the safe direction; PASS by inspection of
  tests/conftest.py:212 and the rc-4 e2e test.
BREAKS: nothing.
NITS:
  1. "it cannot wedge forever" is slightly overstated. A GONE pid is taken over (psutil raises ->
     conftest.py:198-201 -> False), so a normal crash cannot wedge. But a stale lock carrying 0.0 whose
     pid has since been RECYCLED by a long-lived unrelated process is honoured until that process exits
     (no finalizer releases it). This is the accepted trade: a visible rc-4 refusal (and named-file runs
     still work) versus silently starting the two concurrent full suites REC-1 is about. I judge ALIVE
     the right direction for a LOCK; only the commit's no-wedge guarantee is too strong.
  2. Direction derived independently: stealing from a LIVE owner reproduces the exact failure the lock
     exists to prevent (two full suites, shared fixed port, 72 orphaned fake_acp_server processes);
     refusing at worst delays a full-suite start, visibly, and never affects targeted runs. ALIVE-for-0.0
     is correct. GONE pid still taken over is pinned by
     test_an_unreadable_create_time_on_a_dead_pid_is_still_taken_over (passed).
  3. Scope confirmed: `git show --numstat 7221f17` = tests/conftest.py (+27/-5), tests/test_suite_lock.py
     (+77/-0); `git diff 573025a..impl/test-hard -- src/` is empty. NO src change.
REASON: The pre-fix direction is caught by two new tests and the base test body is green under the same
mutation; the direction is independently defensible for a lock and a gone pid is still reclaimed. The only
residue is the overstated no-wedge claim for a recycled live pid.
```

```
ITEM: W16A-1 (6388e77)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes (verified, all three named mutations)
TEST COMMAND: powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_bench_sweep.py -k "same_way"
TEST OUTPUT (before fix): OLD body = base 573025a test_bench_sweep.py seam test, run under each mutation:
  A `_effective_flags` -> `return plan_flags.model_copy(update=override)`:
    FAILED tests/test_bench_sweep.py::test_the_sweep_and_its_guards_build_a_trial_the_same_way
    E  AssertionError: _effective_flags must read the trial through bench._trial_flags ...  (tests\test_bench_sweep.py:983)
    1 failed, 1 passed, 70 deselected in 0.55s   [EXIT=1]   (old body: 1 passed)
  B `_effective_env` -> `return getattr(plan_flags.model_copy(update=override), "env", None) or {}`:
    E  AssertionError: _effective_env must read the trial through bench._trial_flags ...  (line 986)
    1 failed, 1 passed, 70 deselected in 0.50s   [EXIT=1]   (old body: 1 passed)
  C `_trial_flags` -> `return plan_flags`:
    E  AssertionError: the flags the child was launched with are not bench._trial_flags's return value  (line 992)
    1 failed, 1 passed, 70 deselected in 0.53s   [EXIT=1]   (old body: 1 passed)
TEST OUTPUT (after fix): 1 passed, 35 deselected in 0.54s   [EXIT=0]; whole file: 36 passed in 2.46s [EXIT=0]
RULE-13 STATE CHECKED: (a) config list of length ZERO — the NEW test FAILS spuriously:
  `by_caller.get("run_sweep")` is None and `None == []` is False (tests\test_bench_sweep.py:981), even
  though run_sweep is correct (no configs -> no calls, no launches). The old tail-slice body passed there
  (`built[0:] == []`). This is the residual shape-fragility the commit message says it did not change.
  (b) `_trial_flags` called from an unnamed caller: the test pins caller names literally, so a future
  helper/rename fails it — intended for a seam pin, not a defect.
BREAKS: nothing.
NITS:
  1. Zero-length configs (rule-13 a): the caller assertions assume each caller is present, so
     `by_caller.get(...) == expected` fails when expected is `[]`. Use `by_caller.get(k, []) == expected`
     to keep the strict pin while surviving the empty sweep. Not reachable from the committed test
     (configs is hardcoded length 2), so no false red today.
REASON: All three named mutations are caught at the intended assertion and the old tail-only body is
demonstrably green under each, so the guards and the launched flags are now pinned; the change is purely
additive in assertion strength. The one residue is a spurious-failure shape at zero configs.
```

## Item 2 — diff vs message

- `01cb07d`: message says a real foreign sleeper stamped `real_create_time - 10000`, asserting
  `driver_is_live_elsewhere is False` and `serve._reconcile_orphaned_runs` -> `interrupted`; diff is exactly
  `_foreign_sleeper()` (real `subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])`),
  `state._create_time(proc.pid)`, `_stamp_driver(rid, proc.pid, real - 10_000.0)`, the `is False` assert,
  the reconcile call, the `interrupted` assert, and a `finally: _reap(proc)`. Matches. Numstat +39/-0.
- `7221f17`: message says create time read first (gone -> False), then recorded parsed; `recorded <= 0.0`
  with a live pid -> True; missing/None/garbage still taken over; three tests. Diff matches line for line.
  Numstat conftest +27/-5, test +77/-0. **SUITELOCK direction decided independently** (see NIT 2 above):
  ALIVE-for-0.0 is right for a LOCK; GONE pid still taken over; change confined to test infrastructure,
  `src/` untouched.
- `6388e77`: message says record `inspect.currentframe().f_back` caller, group by caller, assert loop +
  both guards once per config in order, and assert the launched trial's `flags` equal the helper's return.
  Diff matches. Numstat +44/-12, only tests/test_bench_sweep.py.

## Item 3 — breakage / lock health

- Combined `tests/test_run_loop_first_load.py tests/test_suite_lock.py tests/test_bench_sweep.py`:
  **94 passed, 1 warning, EXIT=0** (reproduced twice on the clean tree). Named-file runs are not
  suite-locked — exit 0, not the rc-4 refusal; also pinned by `test_several_named_files_are_not_a_full_run`,
  `test_marker_followed_by_a_named_file_is_not_a_full_run`, `test_targeted_run_is_never_locked_even_by_a_held_lock`.
- Lock's own tests: `tests/test_suite_lock.py` -> **26 passed in 1.85s, EXIT=0**. rc 4 (second full run
  refused with the REC-1 sentence), rc 0 (targeted run and revived run after the owner is killed), and
  dead-owner takeover are all exercised by the end-to-end test; the dead-owner release is pinned by
  `test_releasing_lets_the_next_run_start`. KeyboardInterrupt: **not directly exercised** — there is no
  such test, and a real SIGINT to a full-suite subprocess is not cheap on Windows. The release path is the
  session fixture's `finally` and `_release_suite_lock` never calls `_owner_is_alive`, so the conftest
  change cannot affect it.
- One combined run flaked: `tests/test_run_loop_first_load.py::test_an_unreadable_first_load_releases_the_slot`
  (line 203) and on another run `::test_an_empty_object_run_json_releases_the_loop_slot` (line 400) — a
  run-loop slot-release timing race. Both test bodies and `src/` are unchanged by this branch; the same
  file at base `573025a` flaked **1/8** whole-file runs (new file 3/~16), and each test passes in isolation.
  **Pre-existing, not caused by these commits.**

## Item 4 — style / dependencies

- Real processes, not fabricated identities: DR42 uses a real foreign sleeper + its real pid with a
  deliberately different stamp; SUITELOCK uses a real dead `subprocess.Popen` and the real live self pid.
  W16A is a seam test (monkeypatched helpers), which is appropriate and spawns nothing.
- No weakened assertion: W16A-1 replaces one tail-slice assert with four asserts (three caller pins +
  the launched-flags pin) and keeps the old intent (`run_sweep` builds through the helper). Strictly
  additive.
- No `@testing-library` (grep: no matches). No new dependency: only stdlib `inspect` added;
  `pyproject.toml` / `requirements*.txt` unchanged between base and impl/test-hard.
- `ruff check src tests`: **All checks passed! [EXIT=0]**.

OVERALL: PASS — all three named mutations are caught by the new tests and demonstrably were not caught by
the extracted base test bodies, `src/rigma/bench.py` and `src/rigma/state.py` are byte-identical after every
revert, and the only residues are two test-shape nits (W16A zero-configs, DR42 1.0 s tolerance) plus one
pre-existing timing flake in an unchanged test.
