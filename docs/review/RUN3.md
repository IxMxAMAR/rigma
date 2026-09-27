# Run 3 — the deep audit, second pass

Area `r3` · fork `review/deep-audit-2026-09-22` · base `5b5790b` (run 2's HEAD) ·
48 commits. Every finding is **reason / cause / resolution**, and every fix has a
test whose pre-fix failure was measured.

## The four open recommendations, closed

Run 3 ended with four recommendations researched but not implemented. All four are
now resolved, and two of them changed shape once the pinned build was measured
rather than read:

1. **Close the loop on VRAM prediction → DO NOT BUILD.** The research rejected it on
   evidence: Windows exposes no per-process GPU memory, AMD's own tracker has a case
   where `amd-smi` moved 1 MB for a 2 GB allocation, and `--fit` returns different
   answers on the same machine before and after a sleep/resume (llama.cpp#26401), so
   a persisted delta would be applied to a baseline that had already moved. What was
   built instead is `memtruth.py` — a per-launch oracle that reads the live device.
2. **Parse the engine's own memory log → done, with a correction.** The structured
   `--log-jsonl` path is preferred and implemented, but the pinned b9867 does **not**
   have the flag (verified: it answers `invalid argument`), so the stderr table
   parser is the verified path and the JSONL path is forward-looking and labelled as
   never having run against an emitting build. The KV log line was renamed four times
   in under two years, which is why the table parser is pinned by a live test.
3. **The `--parallel 2 --kv-unified` decision → the comment was RIGHT and the
   research was wrong.** Measured with Rigma's real argv: unified gives
   `n_ctx_slot` 32768, non-unified 16384, so total KV is 32768 either way and `-c` is
   the total across slots. My first test of this was wrong the same way the research
   was — I launched `llama-server` without the flag, i.e. a command line Rigma does
   not use, because Rigma adds it in `_launch_extra` rather than in
   `ModelSpec.server_args`.
4. **Key calibration by GPU identity → implemented** (`hwid.py`), including the part
   the research was explicit about: identity is **necessary but not sufficient**
   (power limits alone moved a 3090's pp512 +29% with no identity change), so a
   different card is a *hard* mismatch and a new driver is *soft*.

## The finding that came from the owner noticing ComfyUI was running

Asked whether the earlier physical measurements were invalidated by ComfyUI, the
answer turned out to be "not the ones you would expect, and something worse." The
`free` readings were fine — `gpu_used_mb()` reads **residency**, not allocation, so
it read 22 MiB with ComfyUI idle and 12,586 MiB during a generation, and the runs in
question were taken while it held nothing.

But under deliberate contention the engine itself was caught lying about free memory:

```
# Windows counter: 13,149 MiB dedicated in use of 16,304 MiB
common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed
```

**16,140 MiB free on a card with 3,155 MiB free, and its own fit verdict said
"fits".** So on ROCm/Windows the engine's fit **verdict** is not evidence and only
its per-device **accounting** is usable. `--verify` now cross-checks against the OS
counter, believes the OS, and prints `UNRELIABLE` instead of "fits" over a warning.

This is also why the standing advice to agents in `docs/AGENTS.md` now says that on
Windows a successful start is not evidence that a model fits.

## Verified final state

* **2410 tests, 0 failures, 0 errors, 3 skipped** in 348 s (`-m "not hardware"`,
  totals read from `--junit-xml`).
* `ruff check src tests tools` clean; **174 vitest tests** across 17 files pass;
  `tsc --noEmit` clean.
* 48 commits on the review branch; working tree clean.
* **The original repo is untouched** (no file modified since 2026-09-25) and the
  `upstream` push URL is still `DISABLED-no-push-from-fork`.
* Verified live against the real pinned engine and the real card: the fit oracle
  returns real per-device accounting; Vulkan enumeration returns the real
  `deviceUUID`; the owner's 9 existing calibration entries all still resolve.
* **Not verified, and named as such:** NVIDIA/CUDA paths (no NVIDIA hardware here),
  the JSONL parser against a real emitting build, and the AMD gfx target as a key
  component.

Run 2 closed 112 findings across 16 areas. Run 3 exists because run 2's own
artifacts named work it had left open, because two run-2 fixes turned out to be
incomplete, and because a *review* that only reads code cannot find the class of
bug that only appears when two subsystems disagree. So run 3 was adversarial:
six independent agents, each with a worktree, each required to reproduce a finding
before reporting it and to run the mutation that proves its test can fail.

## Why six areas, and not sixteen

Run 2's 16 areas were re-walked, and the six with the worst
`open / fixed` ratio got a dedicated adversarial pass:

| Area | Agent branch | Fixed | Open | Headline |
|---|---|---|---|---|
| HTTP / API / runs | `r3/http` | 2 | 6 | a boolean grant accepted the string `"false"` and granted it |
| sandbox / tools | `r3/sandbox` | 9 | 9 | a **junction** escaped every walker |
| harness / MCP / CLI | `r3/harness` | 4 | 3 | the `/v1` byte passthrough is genuinely byte-for-byte |
| fit / resolve / hangar | `r3/fit` | 5 | 5 | `model_copy(update=…)` bypassed a validated field, again |
| memory / RAG / methods | `r3/memory` | 5 | 2 | a stored memory could forge a section of the system prompt |
| vLLM engine runtime | `r3/vllm` | 3 | 4 | vLLM **cannot run on this machine**; the honest answer is the deliverable |

Two findings the agents reported as *not fixed* were closed by the orchestrator
after checking their reasoning, and both are worth reading as method notes rather
than as fixes:

  * **R3-11** was deferred because the fix looked like a product decision (refuse the
    source, or exclude the credentials?). Excluding is not a product decision, and
    the second half of the agent's own suggestion was the whole answer. The
    interesting part is that closing it needed a **control measurement**: see below.
  * **09-7** was deferred for a real reason (a daemon thread writing to stderr) that
    turned out not to be the interesting half — 15 s is simply shorter than the cold
    start the helper exists for. Fixed in `1637a44`.

Plus one thing no agent was asked for and the orchestrator did, because it was the
owner's explicit request: a fail-safe against the failure modes that had already
cost this project work.

## The findings that matter most

### A silently granted capability, with a UI showing the opposite — R3-1 (HIGH)

`POST /api/sessions/{sid}` type-checked exactly one mutable field (`confirm_exec`,
added by run 2's 13-3). Every other boolean accepted a quoted value, and
`bool("false")` is **True**. Measured at the parent: `PATCH
{"allow_absolute_reads": "false"}` answered **200**, stored the string, and
`tools._absolute_reads_allowed` read it as a **GRANT** — reads outside the
workspace. Meanwhile `grants.ts` honours only a literal `true`, so the UI
*displayed* the grant as off. The same shape applied to `allow_outbound_post`,
`allow_code`, `use_tools`, `auto_compact`, `one_action` and `carry_reasoning`.

This is the third time this exact mistake was found — 09-6 fixed it for macros,
13-3 for `confirm_exec` — and each fix had covered only the field it was written
for. The fix is structural: every `MUTABLE_FIELD` now declares a type, a test
asserts the declaration is exhaustive, and a switch must be a real bool (`1` is
refused too). 67 test failures at the parent.

### A junction escaped every walker — R3-1 (HIGH, sandbox)

`os.walk(followlinks=False)` and `is_symlink()` only recognise *name-surrogate*
reparse points. A Windows **junction** is not one, so every walker descended into
it: `grep` returned the **content** of files outside the workspace (including under
the `confined` profile, while `read_file` refused the same path), `find_files`
named them, the watcher recorded them, `pack_folder` packed them, and
`undo_last_change()` with no argument **wrote** to a file outside the workspace
through the link. The fix prunes reparse directories and tests containment on the
**resolved** path.

### The credential denylist was one dropped character deep — R3-2 (HIGH, sandbox)

`_fuzzy_file` did the denylist check and *then* substituted the denied sibling:
`read_file(".en")` returned `.env`, `read_file("id_rs")` returned `id_rsa` — each
with a note **naming the file it had used**, which made it a discovery oracle as
well as a disclosure. `move_files(paths=["id_rs"])` laundered the key out.

### A stored memory could forge a section of the system prompt — 10-R3-12 (HIGH)

`memory` text is injected into the agent **system prompt** and the driving user
message. The advisor-reflection path bypasses `clean_rule`, and newlines were
intact, so a rule derived from a model's own output could open a `### NOTES`
section or a bare `SYSTEM:` line. Injection now flattens to one bounded line; the
store is left unsanitised so the Memory panel still shows the real text.

### Two denial-of-service hangs, both reachable from the model — HIGH ×2

  * **10-R3-11**: `memory.looks_like_raw_trace`'s `_FILENAME`/`_CALL_SYNTAX` were
    quadratic. `store.add("A" * 200_000)` took **125 s**, synchronously on the run
    loop. A leading `\b` makes it 0.004 s.
  * **R3-16**: `_glob_re` translated each `**/` into its own `(?:.*/)?`, and
    nesting those is exponentially ambiguous. MEASURED on a 24-deep path: `**/`×10
    = 2.19 s, **×12 = 50–61 s**, once per file in the walk, synchronously on the
    event loop. Collapsing a run of `**/` into one group is semantically
    **identical** — `.*` already spans `/` under `(?s:)` — and takes it to
    0.0035 ms. Equivalence is asserted by brute force over 17,640
    (pattern, path) pairs against the pre-fix translation, not argued.

### The turn path had no idea a build had drifted — 09-4 (MEDIUM)

`VERIFIED` existed and `rigma harness` reported drift, but nothing on the **turn**
path compared anything, so installing a newer mcode produced a normal-looking turn
on a build nobody had measured. Drift is now a notice in the turn (and a non-zero
exit from `rigma harness`, so `rigma harness && …` is no longer a green light).

### An autonomous run could never be granted code execution — R3-4 (MEDIUM)

`start_run` set `allow_code=True` while `confirm_exec` stayed OFF, and **no surface
could turn it on**: an active run's chat is filtered out of `/api/sessions`, and
the only grants UI is the open chat's Sidecar. So `run_shell`/`run_python`/
`start_job` were refused for *every* run, and the refusal told the user to enable a
setting they could not reach. Worse, `profiles.ts` asserted the opposite — every
profile sentence that does not withhold execution said code execution "still
works", so `all` promised a capability it did not have and `confined` differed from
`all` by accident. Fixed in all three places: a validated, default-OFF
`confirm_exec` on `POST /api/runs`; a separate explicit switch in the launcher; and
sentences that no longer claim what a profile does not grant.

### The credential denylist had a second door, and the index was behind it — R3-11 (MEDIUM)

The 13-2 denylist is enforced on the **tool read** path. `rag.add_source` took any
path with **no validation at all**, and everything under it was embedded into the
local vector index — so adding a home directory, or Documents, put `.env`,
`.ssh/id_rsa`, `.git-credentials` and browser cookie DBs into the index, where
`search_my_documents` (a `safe=True`, **auto-run** tool) retrieved them with no
confirmation and no human in the loop.

The fix derives raggity's `exclude` list from the *same* tuples the read path uses,
so a denied pattern is excluded from indexing by construction — a second
hand-maintained copy of a denylist is exactly how R3-2 and R3-11 both happened.

**Closing this needed a control measurement, and two of my attempts at it were
vacuous.** I ingested a canary corpus twice — once with `exclude = []`, once guarded
— and then *asked each index* whether the canary was retrievable. My first two
attempts produced "clean" results that proved nothing: the canaries had
non-indexable extensions (`.json`, `.pem`, none) or dotfile names raggity prunes on
its own, so nothing was ever in scope. With indexable extensions the control showed
8 of 10 files indexed and five canaries retrievable; the guarded run showed 2 (the
real documents) and none.

That measurement then found a gap the derived list alone did **not** close.
`**/credentials` and `**/*.pem` are exact basenames, so an appended extension
defeats them: `credentials.md`, `my.api_key.md`, `credentials.json.md`,
`token_api_key.txt` and `server.pem.txt` were **all still indexed and all still
retrievable**. The reachable case is a README about key rotation. Every pattern now
also gets a `.*` variant.

Named rather than implied: an index built **before** this fix still holds what was
already embedded. Regenerating the config changes what a future ingest includes; it
does not remove existing rows, so a user who added a folder containing credentials
should reindex.

### The fail-safe (owner request, IMP-14)

Every engine call was a single attempt, so a `llama-server` still loading a 13 GB
model, a socket the OS dropped mid-prefill, or a 502 from a restarting upstream
ended the turn with "the engine is unreachable" about an engine that came back two
seconds later.

`src/rigma/resilience.py` adds `classify` (TRANSIENT / PERMANENT / CANCELLED,
erring toward PERMANENT because that direction is visible), `Policy` +
`retry_async` with **two** limits (attempts *and* a total sleep budget),
`CircuitBreaker` (per endpoint, single-probe after cooldown, doubling cooldown so a
flapping endpoint settles into a slow poll), and `LoopGuard`.

THE ONE RULE, stated where the retry is installed: **a stream that has already
emitted bytes cannot be retried**, because the replay would show the answer twice —
a corruption that reads as a model bug and is ours.

Two things about this are worth recording as much as the feature:

  * **`LoopGuard` is a tested primitive that is deliberately NOT wired in.** It was
    integrated, and two existing tests caught that it *replaced two better
    messages* — "the model spent this turn reasoning" and "stopped after its tool
    calls" — with a generic one. Wiring it in was a regression in message quality.
    It was reverted, and its docstring now says so, naming the two tests.
  * **Mutation testing found a survivor.** 10 deliberate breaks were caught, but
    "the breaker never records a failure" initially survived, because every test
    drove the breaker by hand. A breaker that stays closed through a real outage is
    the whole bug, so two tests now drive failures through `retry_async`.

## Two run-2 fixes that were incomplete

  * **06R3-1**: run 2's 06-4 fix validated a field *after* `model_copy(update=…)`
    had already bypassed it — and the sibling call site was missed. Same class, one
    line apart.
  * **R3-2**: run 2's 03-5 fix guarded the run loop's *driving-line* write only.
    Three other whole-row session saves in the same loop had no `base_rev`,
    reproduced with a real run against a scripted engine: a concurrent message is
    erased. The `finally` one is the worst — it clears `run_id`, so the realistic
    second writer is the owner typing the instant the run stops.

## What run 3 deliberately did NOT change

The mandate said not to force a change whose margin is small. These are recorded as
**cleared** or **rejected**, with the evidence, so nobody re-spends the effort:

  * `/v1/{path}` **is** a byte-for-byte passthrough, streaming included (re-measured,
    not read). The one plausible way the mcode backend could be broken end-to-end
    was measured against real `llama-server` b9867: mcode sends its ~12 KB
    instruction block as a `developer`-role message and llama.cpp maps it to
    `system`. It must not be rewritten.
  * `rag.sidecar_health` requiring a raggity field is safe: the **real** raggity
    0.13.0 `/healthz` returns `{"status","version","index_backend","documents"}` —
    verified by starting it on a spare port with a throwaway config, not by reading
    code.
  * vLLM **cannot run on this machine**. Windows is the blocker, not the GPU (the
    RX 9070 XT is gfx1201, explicitly on vLLM's supported ROCm list, and Python 3.12
    is exactly what the ROCm wheels require). What the owner asked for was vLLM
    "specced out as a backend", and that is what it is: an honest verdict, a
    selectable engine runtime, and no forced install.
  * UIUX-5 (accent tokens differ from the constitution's stated hexes) stays an
    **open question for the owner**: one of the two documents is out of date and
    only the owner can say which.

## The engine runtime became selectable — R3-VLLM-4 (the run's last gap)

Worth recording separately, because it is the difference between a spec and a
feature. `r3/vllm` shipped `detect_engine_runtime`, `vllm_argv`,
`launch_vllm_server` and `rigma engine-runtimes` — all correct, all tested, and
**none of it reachable**: nothing called `detect_engine_runtime`, there was no
`--engine` flag, no state field recorded which engine was running, and no API or UI
surface to read. `engine-runtimes` was a report about a road with no on-ramp.

The design point it turns on: `detect_engine_runtime` **falls back** to llama.cpp
with the reason recorded, which is right for a stored preference (a saved vLLM
choice must not make `rigma up` unrunnable where it used to work) and **wrong for a
command line** — someone who types `--engine vllm` and gets a working UI has been
told vLLM works, which is the same class of lie as serving the CUDA wheel to an AMD
card. So an explicit request that cannot be honoured is now an error naming the
reason and what does work, while the fallback stays for the stored path, and tests
pin both so neither is "simplified" into the other.

Three bugs the new tests caught, all in my own work:

  1. `state._write_record` is a **fixed key list**, so adding `engine` to
     `write_state`'s signature was not enough — the field was silently dropped from
     disk and `read_state()["engine"]` raised `KeyError`.
  2. I defaulted `engine` to `"llamacpp"`. That would make every record written by a
     path that launches no engine (UI-only `rigma up`, `perform_unload`) assert
     llama.cpp, and would **overwrite** the engine a vLLM launch had recorded. It
     defaults to `None` now — "not recorded" — and `/api/server` passes it through
     without inventing a value, so the UI badge shows nothing rather than guessing.
     A wrong engine label is worse than no label.
  3. My test guessed the dataclass field name (`d.engine`; it is `d.runtime`). The
     test was wrong, not the code.

`rigma status` had the same shape of flaw: a vLLM launch has no quant, so it printed
`running: Qwen/Qwen3-8B ()  up 3 min` — empty brackets exactly where the fact that
explains them belongs.

## Method, and the mistakes it caught

Every agent was required to: reproduce before reporting; state what it could NOT
execute; run the mutation that proves its test fails; and never claim a green it
had not measured. That last rule is why this run's record includes the agents'
own discarded work — one agent threw away a test that could not fail when its fix
was reverted, rather than commit a test that passes for the wrong reason.

It also caught the orchestrator's own errors, which are recorded here because a
review that hides its reviewer's bugs is not a review:

  * the `LoopGuard` integration above (reverted);
  * **a regression the retry layer introduced**: `RetryExhausted` wraps the real
    fault, so `isinstance(e, httpx.ConnectError)` stopped matching and the most
    actionable message Rigma has ("the engine is unloaded — load it again")
    silently degraded to a generic "failed after 3 attempts". The handler now looks
    *through* the wrapper. This is the one thing a retry layer must not do;
  * two bugs in the new citations code, both found by its own tests:
    `setdefault` does not replace an explicit `None`, and `str(None)` had become
    the filename `"None"`.

## Verification

Measured on the final committed tree (`1e42ea7`), not asserted:

  * **Python: `2336 tests, 0 failures, 0 errors, 3 skipped` in 365 s** (JUnit XML
    totals, `-m "not hardware"`). `ruff check src tests tools` clean. Run 2's
    baseline was 1987 passed, so run 3 added 349 tests.
  * **Frontend: 174 vitest tests pass (17 files), `tsc --noEmit` clean.** The UI
    bundle is rebuilt and committed, and `git status` on `src/rigma/data/ui_v2/` is
    empty — the shipped artifact matches the source, which `.githooks/pre-commit`
    enforces anyway.
  * **Working tree clean**; 44 commits since run 2's HEAD.
  * **mcode verified LIVE**, not read: a real 0.5.4 turn against the repo's own fake
    engine completed (`status: succeeded`, 3 events, provider saved and active). The
    wire log is in `findings-r3/09-harness-r3.md`.
  * One whole-suite run hung past 8 minutes with the parent process's CPU flat and no
    failure reported. The same suite then passed four times, so it is
    **intermittent, not a regression** — consistent with the sandbox refusing
    `taskkill` on the process-tree tests. `pytest-timeout` is not installed, so a hang
    has to be capped from outside. Recorded rather than smoothed over, because "the
    suite is intermittently flaky" is exactly the kind of thing a review should not
    hide.
  * Method note: `pytest -q` suppresses the final `N passed` summary line on this
    build, so the totals above come from `--junit-xml` rather than from reading the
    tail of the console output. A first attempt to read them from the console
    reported nothing and looked like a failure; it was not.

Per-area findings: `docs/review/findings-r3/` (six documents, one per agent).
Ledger: `PROGRESS.md` (Run 3 section). UI items: `UIUX.md`.
