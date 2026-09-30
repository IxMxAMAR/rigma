# verify-resid — independent verification of `impl/resid` (ad36384, 6da0460) and the W13B-3 deferral

Worktree `.scratch/wt-vfy-resid` @ `vfy/resid` (base `f3bd020`, tip `6da0460`); integration `review/deep-audit-2026-09-22` @ `cb92962`. Worktree left CLEAN (`git status --porcelain` empty, `git diff HEAD` empty). No engine, model, GPU or live harness turn; every pytest run appended `-m "not hardware"` via the shared semaphore script.

---

ITEM: CTXFLOOR-n4 (ad36384)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes
TEST COMMAND: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_hangar_api.py -k "below_the_floor" --tb=short` (run from `.scratch\wt-vfy-resid`)
TEST OUTPUT (before fix): `src/rigma/server_ops.py` checked out from `f3bd020`, new test present:
```
F                                                                        [100%]
__________ test_a_ctx_below_the_floor_is_raised_to_the_launch_floor ___________
tests\test_hangar_api.py:328: in test_a_ctx_below_the_floor_is_raised_to_the_launch_floor
    assert server_ops._raised_launch_ctx(1, 32768) == floor
E   AttributeError: module 'rigma.server_ops' has no attribute '_raised_launch_ctx'
FAILED tests/test_hangar_api.py::test_a_ctx_below_the_floor_is_raised_to_the_launch_floor
1 failed, 16 deselected, 1 warning in 0.64s
```
Mutation (`return max(floor, min(int(ctx), int(native_ctx)))` -> `return min(int(ctx), int(native_ctx))`, restored after):
```
E   AssertionError: assert 1 == 2048
1 failed, 16 deselected, 1 warning in 0.66s
```
Both match the commit message verbatim.
TEST OUTPUT (after fix): `1 passed, 16 deselected, 1 warning in 0.70s`
RULE-13 STATE CHECKED: I wrote `.scratch/orchestrator/vfy-resid-probe.py` (no engine; imports `rigma.models`/`rigma.server_ops` only). **(a) exhaustive equivalence** — the old inline `max(MIN_LAUNCH_CTX, min(int(ctx), native_ctx))` vs the helper over a 90-pair grid of `ctx in {-100,-1,0,1,2047,2048,2049,4096,32768,999999} x native in {-1,0,1,512,2047,2048,2049,32768,999999}`: **0 mismatches**. **(b) `native_ctx == floor`** `helper(4096, 2048) == 2048`; **`native_ctx < floor`** `helper(4096, 2047) == 2048`, `helper(1, 512) == 2048`; **NEGATIVE ctx** `helper(-5, 32768) == 2048` (raised, not rejected — the route's 400 is the front door); **`ctx == 0`** `helper(0, 32768) == 2048`; **monkeypatched floor of 0** `helper(1, 32768, floor=0) == 1`, `helper(0, ..., floor=0) == 0`, `helper(-5, ..., floor=0) == 0` — the raise-to becomes a **no-op**, and the helper body contains **no `/`, `//` or `%`**, so nothing divides. `floor=None` reads the live `server_ops.MIN_LAUNCH_CTX` at call time (rebinding it to 4096 moves `helper(1,32768)` to 4096; restoring returns 2048).
BREAKS: nothing. `tests/test_models.py tests/test_hangar_api.py tests/test_repo_invariants.py tests/test_hangar.py tests/test_server_ops_ctx.py` -> **109 passed, 1 warning in 23.81s**; `ruff check src tests` -> **All checks passed!**. Merge-conflict check at integration HEAD `cb92962`: `src/rigma/cli.py` has exactly one `from . import kvcache as _kvcache` (line 2618) and all three uses (`launch_fingerprint` 2681, `restore` 2684, `restore_failure_note` 2703); an AST scope check proves all four lines are inside the single `def up` (2154–2765), so the launch path cannot `NameError`. `tests/test_cli.py tests/test_launch_records_fingerprint.py` -> **52 passed in 4.68s**.
NITS: (1) the structural half of the new test is a substring check (`"_raised_launch_ctx(" in body`), not an AST call check, so it does not pin the argument order — harmless here because `max(floor, min(int(ctx), int(native)))` is symmetric, a swap is behaviour-neutral. (2) the helper adds `int(native_ctx)` where the old inline used `spec_full.native_ctx` bare; for the typed `ModelSpec.native_ctx: int` field this is a no-op, but it is a (harmless) widening. (3) the `floor` parameter exists only as a test seam — production never passes it. None is material.
REASON: The test genuinely fails on a clean `f3bd020` revert of the source with the test kept, exactly the `AttributeError`/`1 failed, 16 deselected` the commit quotes, and the mutation reproduces `assert 1 == 2048`; the extraction is behaviour-preserving on a 90-pair equivalence grid including `native_ctx < floor`, and `perform_switch` genuinely calls the helper (`server_ops.py:826`) with no behaviour change at/above the floor.

---

ITEM: B7E-n1 (6da0460)
VERDICT: PASS-WITH-NITS
FAILS-WITHOUT-FIX: yes
TEST COMMAND: `powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_models.py -k "construction_only" --tb=short` (before), `... tests/test_models.py -k "alias"` (after)
TEST OUTPUT (before fix): `src/rigma/models.py` checked out from `f3bd020`, new test present:
```
tests\test_models.py:194: in test_the_alias_guard_is_construction_only_by_design
    assert "CONSTRUCTION-ONLY (B7E-n1)" in src
E   assert 'CONSTRUCTION-ONLY (B7E-n1)' in 'from __future__ import annotations\n\nimport re\n...'
FAILED tests/test_models.py::test_the_alias_guard_is_construction_only_by_design
1 failed, 11 deselected in 0.56s
```
The failure is on line 194, the source-note assert — i.e. the two behavioural asserts above it (`ComboFlags(alias="a,b")` raises; `model_copy`/attribute assignment bypass) **passed on the old source**, confirming the bypass is real and pre-existing, exactly as claimed.
TEST OUTPUT (after fix): `2 passed, 10 deselected in 0.43s`
RULE-13 STATE CHECKED: **(a) pydantic `model_validate` from a dict carrying a comma**: `ComboFlags.model_validate({"alias": "a,b"})` -> `ValidationError` (2 errors); `model_validate({"ctx": 4096, "alias": "a,b"})` -> `ValidationError` (1 error) — it **rejects**. **(b) `ComboFlags(**{"alias": "a,b"})`** -> `ValidationError` (2 errors) — it **rejects**. For contrast the documented bypasses still hold: `ComboFlags(ctx=4096).model_copy(update={"alias": "a,b"}).alias == "a,b"` and `flags.alias = "a,b"` both succeed; `ComboFlags.model_config.get("validate_assignment")` is `None` (off).
BREAKS: nothing (same 109-passed run above; ruff clean).
NITS: (1) the "NOTHING sets `alias` after construction" invariant is grep-verified today but **not regression-pinned**: the companion test *asserts the bypass works*, so a future production writer using `model_copy(update={"alias": ...})` would keep it green while the note silently rots. (2) the provenance assertion is a whole-file substring (`"CONSTRUCTION-ONLY (B7E-n1)" in src`), so it would still pass if the note moved away from the validator. Both are low-value and consistent with the repo's existing source-check style.
REASON: I re-did the grep independently — in `src/`, `alias` is the field (`models.py:470`), the validator (`:524`), the note, and the single reader `served_as = self.flags.alias or self.model_slug` (`:703`); there is no `.alias =`, no `"alias":` dict key, no `setattr(..., "alias", ...)`. Every variable-dict merge site is safe: `as_overrides()` (`models.py:331`) omits `alias`, `sweep_configs`/`quick_configs` override keys are a fixed literal set with no `alias` (`bench.py:838-892`), `server_ops._measured_placement` returns only `ngl`/`n_cpu_moe` (`:393-407`), and the one untrusted merge `resolve._apply_calibration` re-validates through `ComboFlags.model_validate` (`resolve.py:254`), so a hand-edited comma alias there is rejected too. The note is accurate and the bypass is genuinely unreachable in production.

---

ITEM: W13B-3 (deferred)
VERDICT: PASS (deferral is correct)
FAILS-WITHOUT-FIX: n/a — no commit; the *naive* fix's failure mode is verified by probe
TEST COMMAND: `& C:\ComfyUI\RD\rigma\.venv\Scripts\python.exe C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\vfy-resid-w13b3-probe.py` (PYTHONPATH = worktree `src`; no engine)
TEST OUTPUT (before fix): n/a
TEST OUTPUT (after fix): n/a — probe of the proposed fix:
```
lever exported : 1
all configs    : ['baseline','fa-off','kv-q8','kv-q4','batch-big','coopmat-off','no-op-offload','attn-rot-off']
TODAY gate keeps  : ['baseline','fa-off','kv-q8','kv-q4','batch-big','coopmat-off','no-op-offload']
NAIVE os.environ gate keeps: []
naive empties the sweep: True
no lever: today == naive: True 7 kept
```
RULE-13 STATE CHECKED: the process env — `LLAMA_ATTN_ROT_DISABLE=1` exported, then the real `bench.sweep_configs` + `bench._effective_env` + `bench._carries_quality_env_lever` predicates evaluated with (today) the plan's env and (proposed) `{**os.environ, **eff}`. Today the tools-capable gate keeps 7/8 (drops only `attn-rot-off`); the naive fix keeps **0/8**. Without the lever both gates agree (7 kept).
BREAKS: nothing (no commit). `runtime.py:460-461` is verbatim `if plan.flags.env: popen_kw["env"] = {**os.environ, **plan.flags.env}`, confirming the mechanism.
NITS: none blocking. The backlog row (`docs/review/program-2026-09-30/BACKLOG.md:277`) is severity **LOW**, status **recorded**, and already classifies this as a "measurement caveat" with the write path closed.
REASON: Deferring is right. The gate cannot see an exported lever (confirmed), and `verify-w13b2.md` NIT 3's independent conclusion is now independently reproduced: a gate that read `os.environ` would drop **every** config on a tools-capable model whenever the owner has the lever exported (0/8), leaving no measurement at all — strictly worse than the caveat, since the row records only the override (never the ambient lever) and the eventual plain launch runs under the same exported env, so a crowned config stays representative. If the owner ever wants the gate aware of the ambient env, the correct fix is **not** the naive read: it is delta/ambient-aware (drop a config only when the trial introduces a lever *beyond* the ambient baseline) or a loud runtime warning that the ambient env carries a quality lever. Neither exists today and neither is test-backed, so it is a new behaviour change and an owner decision — exactly the kind of item this deferral is for.

---

OVERALL: PASS — both commits fail cleanly before their fix, pass after, are behaviour-preserving/reachability-accurate, break nothing, and the W13B-3 deferral is correct because the only obvious alternative empties the sweep.

## Prose

**1. Fails without the fix — confirmed for both, to the exact counts the commits claim.** CTXFLOOR-n4: reverting only `src/rigma/server_ops.py` to `f3bd020` with the new test kept gives `1 failed, 16 deselected` and `AttributeError: module 'rigma.server_ops' has no attribute '_raised_launch_ctx'`; the `max(floor, ...)` removal gives `assert 1 == 2048`, `1 failed, 16 deselected`. B7E-n1: reverting only `src/rigma/models.py` to `f3bd020` gives `1 failed, 11 deselected`, failing at the source-note line 194 while the two behavioural asserts pass — the commit's parenthetical is true. Both sources were restored immediately; the worktree is clean.

**2. Does each fix do what its commit says?** CTXFLOOR-n4: yes. The helper's arithmetic is identical to the old inline over a 90-pair grid including `native_ctx < floor`, negative `ctx`, `ctx == 0` and `native_ctx == floor`; `perform_switch:826` calls it; no behaviour changes at/above the floor (the committed test pins `floor`, `floor+1`, the native cap, and a native window below the floor). B7E-n1: yes. The note's factual claims hold and the bypass is genuinely unreachable — every writer is the constructor or a merge that re-validates, and the only reader is `models.py:703`.

**3. What does it break?** Nothing. The five named files pass (109 passed), the two CLI files pass (52 passed), `ruff check src tests` is clean. The `_kvcache` import at integration HEAD `cb92962` survived the merge resolution: one import at `cli.py:2618`, three uses at 2681/2684/2703, all provably inside `def up`, so the launch path cannot `NameError`.

**4. Style.** Provenance comments are present and accurate; the CTXFLOOR fix is a pure helper rather than a widened monkeypatch (the faked `perform_switch` seam was *not* loosened); no existing assertion was weakened (both commits only add tests); no dead code (the new helper is called in production and by the test). The only style-adjacent weaknesses are the two nits above: substring structural checks and the unpinned "no post-construction writer" invariant.

**Rule 13.** For CTXFLOOR I put the code in five states the committed test does not: `native_ctx == floor`, `native_ctx < floor`, negative `ctx`, `ctx == 0`, and an explicit `floor=0` (the raise-to becomes a harmless no-op; the helper divides by nothing). For B7E I put it in two: `model_validate` from a dict carrying a comma (rejects) and `ComboFlags(**{"alias": "a,b"})` (rejects). I also probed the W13B-3 process-env state end to end with the real gate predicates.
