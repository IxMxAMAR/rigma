# R3 adversarial review — consolidated findings and fixes

Date: 2026-09-27
Fork: `C:\ComfyUI\RD\rigma-review`, branch `review/deep-audit-2026-09-22`
Method: eight adversarial review subagents, each given one subsystem and told to
reproduce every claim by calling the real code. Each was run on
`tokenjuice / deepseek-ai/DeepSeek-V4.1-Flash`, the only authorized route.

The instruction was "seems like there are invisible problems, and some really
serious ones". There were. This document records what was found, what was fixed,
what was deliberately left, and — most importantly — the **method error** that
let the serious ones hide behind a green test suite.

---

## 0. Why the existing suite did not catch any of this

Every reviewer ran the full suite. It was green before each review and green
after: **2502 Python tests, 0 failures, 3 skipped**, plus 179 frontend tests,
`tsc --noEmit` clean and `ruff` clean.

So none of the 60-odd findings below was a regression a test would have caught.
Three structural reasons, all of which recur across the findings:

1. **The test asserted the bug as correct.** `test_audit_sec13.py` pinned "an
   absolute destination is a write and keeps its old behaviour"; 
   `test_autonomous_run.py` asserted a run with pending steps reaches `done`;
   `test_phase5_mcp.py` asserted that `allow_code` alone reaches an MCP server.
   A test that encodes the defect is worse than no test: it makes the defect
   look intentional and protects it from review.
2. **The test exercised a *different* path than the one it named.** The spill
   test checked the file existed using `pathlib`, never `read_file` — so "the
   model has a way back to this data" was asserted without ever being
   exercised. The autonomous-run tests drove the completion gates through
   scripts that always did enough work to pass them.
3. **The filter was tested on the input the model *chose*, not the output it
   *resolved*.** The credential denylist was verified against the literal
   filename, so it was green on `read_file(".env")` and absent on
   `read_file(".en*")` and on `grep`. This is the single most repeated shape in
   the whole review.

The last one generalizes into the rule worth keeping: **test the boundary, not
the spelling of the boundary.** A security check is only as good as the set of
inputs it is applied to, and the interesting inputs are the ones the check does
not see.

---

## 1. Fixed — tool surface and confinement

### R3-TOOL-1 · `grep` and `find_files` read credential files  [high]

The credential denylist was applied by `read_file`, `list_directory`,
`sample_files` and `view_image` — every tool that takes a **name** the model
chose. The two tools that take a **pattern** applied nothing.

```
grep {"pattern": "SECRETKEYMATERIAL"}          -> .ssh/id_rsa:2: SECRETKEYMATERIAL
grep {"pattern": "API_KEY", "glob": "**/.env"} -> .env:1: OPENAI_API_KEY=sk-...
read_file(".env")                              -> refused
```

The default workspace is `Path.home()`, so `~/.ssh/id_rsa` was one grep away,
and the content could then leave through `fetch_url` — an auto-run GET, which
the outbound-POST grant does not gate.

**Fix:** the denylist moved into `_iter_workspace_files`, the one walker `grep`
and `find_files` share, so both are covered and any walker-based tool added later
inherits it. `_credential_path_reason` is now applied where the files are
enumerated rather than where the request is parsed.

### R3-TOOL-3 · `read_file` with a glob dumped credential files  [high]

Same root cause, different mechanism. `_read_path` ran the denylist on the
*literal* string; a glob does not exist on disk, so it passed, and `_glob_under`
tested containment and nothing else.

```
read_file(".env")  -> refused
read_file(".en*")  -> (your pattern '.en*' matched one file: .env)sk-live-DEADBEEF
read_file("sec*")  -> secret.pem contents
```

`_glob_under` now takes an optional `ctx` and filters the **resolved hits**
through the denylist, so a pattern that resolves only onto credentials reports
"matched no files" instead of returning their contents.

### R3-TOOL-4 · `move_files`/`copy_files` wrote anywhere on disk  [high]

`_write_path` allowed any absolute destination outside the workspace, gated only
by the default-ON `allow_code`:

```
move_files {"dest": "...\\Start Menu\\Programs\\Startup", "paths": ["evil.bat"]}
  -> moved 1 file(s)
```

`write_file` — the weaker capability — has always been pinned to the workspace by
`_ws_path`. So the model reached unrestricted file placement through the sibling
tool. A planted `.bat`/`.lnk` in Startup is persistence, not data loss.

**Fix:** a new, dedicated session grant `allow_absolute_writes`, defaulting OFF
and deliberately **not** reusing `allow_absolute_reads` — reading a file and
replacing one are different risks. Added to `MUTABLE_FIELDS`, `_FIELD_TYPES`,
the session defaults, the MCP grant readers, all three product `ctx` builders
(`serve.py` ×2, `_macro_tool_ctx`), and the UI (`grants.ts`, so it is visible
and changeable rather than server-only).

### R3-TOOL-5 · `view_image` ignored the absolute-read grant  [medium]

```
view_image({"path": "C:\\outside\\shot.png"})   -> sentinel, bytes base64'd in
view_images({"folder": "C:\\outside"})          -> refused
list_directory({"path": "C:\\outside"})         -> refused
```

The one read path that skipped `allow_absolute_reads`. A prompt-injected model
could pull any image off the disk into the conversation, and from there out
through `fetch_url`.

**Fix:** an absolute image outside the workspace now needs the grant, using the
same `_inside_workspace` predicate `_read_path` uses. When **no** workspace is
set there is no boundary to escape, so the grant is not required — refusing
there would confine nothing and would only break a caller that deliberately runs
without a workspace. Every ctx the product builds sets one.

### R3-TOOL-6 · `edit_file` had no size cap  [medium]

`read_file` refused a 20 MB file; `edit_file` read it whole, then read it again
for the undo snapshot.

**Fix:** `_EDIT_MAX_BYTES` (8 MB), the same ceiling, named separately because the
two are different budgets that happen to agree.

### R3-TOOL-7 · the spill recovery path could not be read  [medium]

Oversized results were written to `rigma_home()/results` and the model was told:

```
Read more with: read_file path="C:\Users\...\.rigma\results\grep-a32438e4.txt"
```

which the read gate refuses without `allow_absolute_reads`. It worked only when
the workspace happened to contain `rigma_home` — i.e. the default home
workspace. Every project workspace and every autonomous run (which sets its own)
lost the data with an instruction that could not succeed, and nothing said the
missing piece was a grant.

**Fix:** spill into `<workspace>/.rigma-results/`, and name the file with a
**relative** path. `.rigma-results` was added to `IGNORE_DIRS` so Rigma's own
scratch output does not appear in the model's searches or get watched for undo.

### R3-TOOL-2 · MCP tools bypassed `confirm_exec` and `no-delete`  [high]

The `mcp__` branch checked `allow_code` only. Since `allow_code` is ON by default
and `confirm_exec` is OFF, a configured MCP shell or filesystem server was
**strictly easier to reach** than `run_shell`, and `exec_decision` — the single
decision point the other three exec tools share — was never consulted.
`no-delete` was not honoured either.

**Fix:** the branch now requires `_exec_confirmed(ctx)` (so a library embedder
that never sets the field keeps the old behaviour, while the product default is
safe), and applies `_text_refusal` to the call's arguments so a `no-delete` run
cannot reach deletion through a server.

---

## 2. Fixed — persistence and state

The theme here is one torn write turning into permanent loss, because every
loader in this project treats an unreadable file as "empty".

### R3-STORE-1 · calibration.json was not atomic  [high]

A truncated file loaded as `{}`; the **next** save wrote that `{}` back plus one
entry — destroying every other measured model. Fixed with a new
`src/rigma/atomicio.py`: temp file beside the target, unique per write, fsync
before the rename.

### R3-STORE-2 · `prune_calibration` evicted other models  [high]

It grouped by **hardware identity alone**, so `keep_per_identity=1` kept one
entry per GPU *across every model on it*. Calibrating model B evicted model A,
and because `save_calibration` prunes on every save, two models on one card
oscillated and neither ever stayed calibrated. The grouping key is now
`(model:quant:backend, identity)`.

### R3-STORE-3 · `state.json` was the only store with no tmp+replace  [medium]

A torn state.json reads as "nothing running", so `rigma up` starts a second
engine and the live one is orphaned — unkillable via `rigma stop`, holding its
VRAM and port. Now atomic.

### R3-STORE-4 · `update_state` raised `KeyError('engine')`  [medium]

`_FIELD_DEFAULTS` omitted `engine` while `_write_record` did `rec["engine"]`, so
`perform_unload()` on a state.json written before the field existed raised —
**after** the engine had already been killed, leaving a record claiming a live
engine. Now `.get("engine")` and named in the defaults.

### R3-STORE-5 · `os.replace` itself is not concurrency-safe on Windows  [found by testing]

Found while writing the regression test for the unique temp name, not by
reasoning: four threads writing one path produced
`PermissionError(13, 'Access is denied')`. A unique temp name stops two writers
clobbering each other's *file*, but not two writers renaming onto the same
*destination* at the same instant. `atomic_write_text` now retries that specific
transient errno with bounded backoff.

**This is the single most useful thing the review produced**, because it is a
bug in the *fix* for a different bug, and only a test that actually ran the
concurrent case could have found it.

---

## 3. Fixed — the autonomous run loop

### R3-RUN-1 · a run reported "mission complete" having done nothing  [high]

When the mission compiler falls back — `parse_spec` returns None for any reply
that is not a valid spec object, a documented and deliberate path — the plan
stays **empty**, so `pending` is empty; and the fallback step's verification is
`type: none` with no artifact, so `missing` is empty. Both completion gates were
therefore vacuous, and the only remaining gate merely forced one extra turn that
a second `task_complete` satisfied.

```
STATUS done | HALT task_complete (verified) | ITER 1 | PLAN [] | ACTIONS 2
```

Two `task_complete` calls, no file written, no verification tool ever run. The UI
rendered "finished — the mission is complete".

**Fix:** an evidence gate. A completion claim is only accepted if the run
performed at least one successful non-bookkeeping tool call, counted over the
**whole run** (so a run that did its work earlier and is now tidying up can still
finish). Otherwise the run ends `stalled` with "claimed the mission was complete
without doing any work".

Note the first attempt at this fix was wrong and the suite caught it: seeding the
fallback step into the plan made every subsequent `manage_plan add` number around
it (#2 instead of #1), breaking the model's own step references. The gate is the
right fix; the plan seeding was not.

### R3-RUN-2 · after 2 challenges the run was written done with steps pending  [medium]

```
STATUS done | CHALLENGES 2 | PLAN [{pending}, {pending}]
```

`missing` and `pending` were both skipped once the challenge budget was spent,
and the summary listed only BLOCKED steps — so the pending ones were invisible.

**Fix:** when the budget is spent with work outstanding the run ends `stalled`
with "STOPPED SHORT", naming the missing deliverables and the pending steps.

### R3-RUN-3 · the spill directory grew without bound  [medium]

One full-size file per oversized tool result, never pruned; a long run polling
`job_output` every turn left thousands. Now bounded by count (200) **and** total
bytes (256 MB), because either alone can be defeated.

### R3-RUN-4 · `budget_hours: NaN` produced a six-minute run  [low]

`nan <= 0` is False and `max(0.1, min(nan, 48.0))` is 0.1. Now rejected with
`math.isfinite`.

### R3-RUN-5 · one unreadable run.json stranded the run slot  [medium]

`runs.load` swallows every exception and returns None, so "I could not read my
own state" was indistinguishable from "the run was deleted", and the loop simply
broke. The run then claimed to be running with no driver: pause and inject
answered 409 "run has no driver", restart 409 "run is running", a new run 409 "a
run is already active". Only Stop could clear it. Now retried briefly, and a
still-unreadable state releases the slot with `interrupted`.

---

## 4. Fixed — CLI and harness

### R3-CLI-1 · `rigma harness --json` exited 0 on drift  [medium]

The `as_json` branch returned above the drift check. The comment beside that
check records that this command "used to exit 0 either way — so
`rigma harness && ...` was a green light"; the fix had landed only in the human
branch. A script parsing the JSON is exactly the caller that cannot see the
printed word `DRIFTED`. Extracted to `_exit_on_drift` and called from both.

### R3-CLI-2 · `--port 0` / `--port 70000` crashed  [low]

`bind` raises `OverflowError`, not `OSError`, for a port outside 0–65535, and
every call site iterates `(port, port - 1)`. Now caught, plus an explicit range
check in `up()` that names the range.

### R3-HARN-1 · DSH silently ignored the permission setting  [high]

`harness_dsh.run_turn` accepts `permission` and drops it by design (DSH's
confinement is its own bundle's business), but the UI rendered the
"off — no tools at all" selector for every non-native backend. A user could arm
a safety setting and have nothing happen, with nothing on screen saying so.

**Fix:** the adapter declares `honours_permission=False`; the UI renders the
selector only for a backend that applies it, and shows a sentence instead
otherwise. An older server that does not send the field reads as honouring it,
which keeps the control rather than silently removing it.

### R3-HARN-2 · the disclosure explaining that was unreachable  [medium]

The `unsupported` list — including "the sandbox pins danger-full-access, so a
confined profile does not survive the seam" — was rendered only for backends that
are **unusable**. Choosing a working external agent hid the entire disclosure,
which is the normal case. Now shown for the selected backend.

---

## 5. Fixed — model acquisition

### R3-HANG-1 · two specs sharing a filename destroyed each other's file  [high]

A model file's identity on disk is its **filename**; a pull is keyed on
`(slug, file)`. `mmproj-F16.gguf` ships in the registry twice and quantiser
filenames collide routinely, so two pulls shared one `models/<file>` and one
`.part`, and two threads interleaved into it. Both reported success while one
model's multi-GB file was replaced by the other's — and the fit math then planned
against the registry's byte count for a file that was no longer that model.

**Fix:** a pull whose destination is already claimed by a *different* spec is
refused with an explanation. Refused rather than renamed, because the destination
name is what the spec, the loader and the fit math all look up.

---

## 6. Deliberately not changed

These are real findings that were left alone, with the reason.

- **`~/.rigma/results` is also never cleaned** — covered by R3-RUN-3 for the
  workspace case; the home fallback inherits the same pruning.
- **The default workspace is the whole home directory** (`serve.py`,
  `sessions.py`), so the tools are on across everything the user owns. Changing
  this is a product decision about what Rigma *is*, not a bug fix, and the owner
  has not asked for it. Recorded here so it is a known position rather than an
  oversight.
- **`runtime.ensure_model` trusts any file at `models/<file>`** with no size or
  hash check, so `rigma up` can launch on a truncated file. Real, but fixing it
  properly needs `sha256` to have a producer (`GgufFile.sha256` is `null` for all
  142 live entries, so the hash gate in `_install_download` is dead code). That
  is a two-part change and belongs in its own pass.
- **The engine binary is executed with no pinned checksum on first install**, and
  `_fetch` follows redirects to any host. `engines.json` carries no hash at all.
  Fixing this means publishing per-asset digests, which is a release-process
  change, not a code change.
- **`no-network` filters tool *names* only**, so with `confirm_exec` granted a
  `no-network` run can still reach the network through `run_shell`. Whether the
  profile promises "no network tools" or "no network I/O" is the owner's call.
- **A run started with an empty workspace silently uses the home directory.**
  Same class as the default-workspace item above.
- **`server_ops.expected_tg` only reads the legacy calibration key**, so the
  engine-room health verdict is permanently "unknown" on any install without
  pre-R3-CAL-1 rows. Cosmetic; no data loss.
- **`prune_calibration`'s remaining cap** keeps one entry per
  `(model, quant, backend, identity)`. Two entries for the same model on the same
  card cannot coexist, which is intended.

---

## 7. Verification

Every fix above is covered by a test that fails against the code before it.

- `tests/test_r3_adversarial_fixes.py` — 22 new tests, one per fixed finding,
  each written to fail on the old behaviour. The credential, write-grant and
  spill tests were all confirmed red against the pre-fix code before being
  accepted as green.
- `tests/test_autonomous_run.py` — the completion-gate tests were rewritten
  because the old ones **asserted the defect**; a new test pins that a run which
  did nothing cannot report done.
- `tests/test_phase5_mcp.py` — the MCP grant tests were updated, and two new
  tests pin the `confirm_exec` requirement and the `no-delete` case.
- `tests/test_audit_sec13.py`, `tests/test_audit_tools.py`,
  `tests/test_r3_image_credentials.py` — the tests that encoded the old
  behaviours were updated to assert the new boundary *and* to still exercise the
  capability through the grant, so nothing was silently dropped.

Full suite, frontend suite, `tsc --noEmit` and `ruff` results are recorded in the
commit messages for this batch.
