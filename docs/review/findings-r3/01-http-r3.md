# 01-http-r3 — HTTP / API / run surface, review run 3

fork: r3/http  base: 5b5790b ("Rebuild the UI bundle for the run-2 frontend changes")
worktree: `.wt-r3-a`

This run audited the run-2 *fixes* as new code paths, plus the parts of the
surface area 01 (run 2) declared it had not read. Two findings are fixed and
committed; the rest are declared open with a reproduction, because a
speculative patch on a data-integrity path is worth less than a stated gap.

## Fixed and committed

### R3-1 [HIGH] Every mutable session field except `confirm_exec` accepted a quoted boolean, and `bool("false")` is True
- **Where:** `src/rigma/sessions.py:119-150` (`_FIELD_TYPES`, `validate_field_types`), reached from `src/rigma/serve.py:1538-1541` (`update_session`)
- **Trigger:** `POST /api/sessions/{sid}` with any of `use_tools`, `use_rag`, `allow_code`, `auto_compact`, `one_action`, `carry_reasoning`, `allow_absolute_reads`, `allow_outbound_post` set to a *string* — `"false"`, `"0"`, `"no"`, `"off"`, `""`. Also `{"authors_note_depth": "abc"}` / `{"max_tool_rounds": "abc"}`.
- **Consequence:** the write answered **200** and stored the string; every reader uses `bool(...)`, so the string form of "no" turned the switch **ON**. Two of those switches are the 13-2/13-3 grants, so `{"allow_absolute_reads": "false"}` **granted** reads outside the workspace and `{"allow_outbound_post": "false"}` **granted** exfiltration-capable POSTs — and `{"allow_code": "false"}` left code access on. Measured, at the parent commit:
  ```
  use_tools            -> HTTP 200  stored='false' bool()=True
  use_rag              -> HTTP 200  stored='false' bool()=True
  allow_code           -> HTTP 200  stored='false' bool()=True
  auto_compact         -> HTTP 200  stored='false' bool()=True
  one_action           -> HTTP 200  stored='false' bool()=True
  carry_reasoning      -> HTTP 200  stored='false' bool()=True
  confirm_exec         -> HTTP 400  stored=False   bool()=False
  allow_absolute_reads -> HTTP 200  stored='false' bool()=True
  allow_outbound_post  -> HTTP 200  stored='false' bool()=True
  authors_note_depth   -> HTTP 200  stored='abc'   bool()=True
  ```
  The same value is **displayed as off**: `frontend-v2/src/chat/grants.ts:70` (`readGrants`) honours only a literal `true`, and its own test asserts a string is not a grant (`grants.test.ts:20`). So the UI showed the grant unchecked while the server enforced it — a grant the user cannot see is a grant they cannot revoke. `authors_note_depth: "abc"` then raised in `sessions.build_messages`'s `int(...)` (`sessions.py:583`) and `max_tool_rounds: "abc"` in `serve._round_cap` (`serve.py:998`).
- **Cause:** AUDIT 01-3 added the write-time type guard but 13-3 gave exactly one name in it a real type (`confirm_exec`). Every other boolean stayed in `MUTABLE_FIELDS` and out of `_FIELD_TYPES`, so `update_session`'s copy loop (`serve.py:1573-1575`) passed them through. Run 2 had already solved this twice — `macros._as_bool` (09-6) and `method_schema`'s `isinstance(st[f], bool)` check — and never applied it to the HTTP surface. 13-2's two grants were also left out of `_SESSION_DEFAULTS` while 13-3's `confirm_exec` was named there, so a chat predating them loads with no key at all (latent: the product only used `.get()`).
- **Fix:** implemented. `_FIELD_TYPES` now covers every mutable field; a switch must be a real `bool` (`1` is refused too — `isinstance(True, int)` is True, so a naive int branch would have accepted it and `tools._exec_confirmed` reads a truthy value as a grant) and a count a real `int`; the documented `null` → empty-value coercion is preserved. `_SESSION_DEFAULTS` gained the two missing grants. A structural test asserts `set(MUTABLE_FIELDS) <= set(_FIELD_TYPES)` so the next field added to the PATCH surface cannot inherit the hole.
- **Verified:** `tests/test_r3_session_field_types.py` — **67 failures** at the parent commit (one parametrized case per switch × quoted form, plus the count, null and grant cases), all passing after. The two R3 test files together:
  ```
  & C:\ComfyUI\RD\rigma\.venv\Scripts\python.exe -m pytest tests/test_r3_session_field_types.py tests/test_r3_run_session_writes.py -p no:cacheprovider -p dsh_tmpfix --basetemp=...\.scratch\00-recon\bt-a
  68 passed, 1 warning in 23.44s
  ```
  `ruff check src tests tools` → `All checks passed!`
- **Commit:** `74c40c4`

### R3-2 [HIGH] The run loop's post-turn writes were the last whole-row session saves with no `base_rev`
- **Where:** `src/rigma/serve.py` — tool-result append (was L4794), `run_profile` flip (was L4803), `finally` cleanup (was L5129); now `_save_guarded` / `_append_guarded` at L4431-4480
- **Trigger:** any second writer to the run's chat session landing between the loop's load and one of those writes — a rename, a `POST /api/sessions/{sid}` PATCH, or the owner typing the instant the run stops. The section's own comment says it takes "ONE load for the whole post-turn section" (L4751-4755), and there are two `await`s (`_runs.append_action`, `_runs.log_tool_action`) between that load and the tool-result write.
- **Consequence:** the whole-row replace erases the concurrent message. Measured with a real run against a scripted engine (marker stored by a second writer at the loop's own tool-result write):
  ```
  E  AssertionError: the run loop's tool-result save erased a message another writer
     stored in between; messages now: ["New mission — ...", "→ manage_plan",
     "TOOL RESULT manage_plan: added step #1: step one", ...]
  ```
  The `finally` write is the worst of the three: `_run_loop`'s `finally` clears `mission`/`run_id`, which is exactly what makes the chat reappear in the rail (`serve.py:1432-1438` hides only the *active* run's session), so the realistic second writer is the owner typing into the chat the moment the run ends.
- **Cause:** AUDIT F1 built `base_rev` for precisely this, and run 2's 03-5 fix applied it to the loop's *driving-line* write only ("AUDIT 03-5: save with base_rev. This whole-row write used to be unconditional, so a message that landed in the session while the driving line was being built ... was silently dropped", `serve.py:4570-4574`). The three saves in the same iteration were left unconditional.
- **Fix:** implemented. `_save_guarded` reloads on a lost race and re-applies the caller's mutation to the fresh row; `_append_guarded` appends to the fresh row so a retry cannot duplicate the entry; both return the row that actually persisted so the loop's later reads (`run_profile`, `_last_trace`) see what is stored. The `finally` write keeps its `mission`/`run_id` clear on whatever row won.
- **Verified:** `tests/test_r3_run_session_writes.py` fails at the parent commit (message above) and passes after. The test also asserts the write the loop issued *carried* a `base_rev`, so it cannot pass by luck. Neighbours: `tests/test_autonomous_run.py`, `test_serve_sessions.py`, `test_sessions.py`, `test_serve.py`, `test_audit_serve.py`, `test_runs_session_guard.py`, `test_runs_control_guard.py`, `test_run_tools.py`, `test_run_stop_reason.py`, `test_auto_compact.py`, `test_backup_restore.py`, `test_app_settings.py`, `test_macros_interpreter.py`, `test_methods_api.py` → **296 passed**.
- **Commit:** `589a677`

## Open findings (reproduced, not fixed)

### R3-3 [MEDIUM] `/api/restore` applies settings and each method as it goes, so a mid-loop write failure leaves a half-restored store
- **Where:** `src/rigma/serve.py:5339-5350`
- **Trigger:** a backup document whose method writes fail part-way — a full disk (`OSError` 28), a permission fault, or any `OSError` on the Nth method. Reproduced by failing the second `methods.save_user` call.
- **Consequence:** the endpoint's docstring promises "Nothing is written until the WHOLE document has passed", and the validate pass does hold — but the *apply* pass does not. Measured:
  ```
  HTTP 500 Internal Server Error
  settings idle_unload_minutes: 99.0 (was 12, doc said 99)
  methods now: ['aaa']
  memory now: ['before']
  ```
  Settings from the failed restore are live, one of two methods landed, and memory was never replaced. The client sees only a 500 and has no way to learn what applied.
- **Cause:** validation is all-or-nothing but the writes are sequential and unguarded. `app_settings.save` (`app_settings.py:91`) is atomic per file, `methods.save_user` (`methods.py:598`) is a plain `write_text`, and `MemoryStore.restore` (`memory.py:492`) is last. Nothing stages or rolls back.
- **Fix (not implemented):** stage every method to a temp file and rename only after all are written, capturing the previous bytes of each target so a failed rename can be rolled back; only then replace settings and memory. I did not implement it: `save_user` has no staging seam, and a partial rollback written untested on the only copy of the user's learned rules is worse than the current state. Recommend it as its own scoped change.
- **Verified:** `.scratch/00-recon/r3probe3.py`, output above. `tests/test_backup_restore.py` (5 tests) passes — its `test_restore_validates_the_whole_document_first` covers only the *validation* failures, which is why this survived.

### R3-4 [MEDIUM] `confirm_exec` is off for every autonomous run and no surface can turn it on, so a run's `run_shell`/`run_python`/`start_job` are permanently refused
- **Where:** `src/rigma/serve.py:5229-5236` (`start_run`'s `sess.update(use_tools=True, allow_code=True, ...)` — `confirm_exec` is not named and defaults False), enforced at `src/rigma/tools.py:3744-3749`
- **Trigger:** start any run from the UI and let it reach a step that needs to spawn a process.
- **Consequence:** `exec_decision` returns `("running shell commands or code needs explicit confirmation for this chat ... Enable 'confirm execution' on the session to allow it")`, and for a run there is no session to enable it on: `start_run`'s chat is hidden from `GET /api/sessions` for as long as the run is active (`serve.py:1432-1438`), and the only grants UI is the Sidecar fieldset for the *open* chat (`frontend-v2/src/chat/Sidecar.tsx:823-850`, fed by `readGrants`). The run's session is not openable from the run view either (nothing in `frontend-v2/src` reads a run's `session_id`; `Sidecar.tsx:957` is the method-draft path). So a coding mission that needs `run_python` loops on a refusal whose own text names an action the user cannot perform.
- **Cause:** run 2's 13-3 grant ("give a chat a way to grant execution", `41010d2`) was wired into the chat Sidecar and the CLI (`cli.py:98-134`), and `start_run` — which predates the grant — sets `allow_code=True` and never `confirm_exec`. `tools._exec_confirmed` deliberately treats a missing `confirm_exec` as "fall back to `allow_code`" for library embedders, but every product ctx sets it explicitly, so the run's explicit `False` wins.
- **Fix (not implemented):** a design decision, not a one-liner — either `start_run` sets `confirm_exec` from a per-run request field (with the run view offering the toggle, which means exposing the run's `session_id` to it), or the refusal text for a run points at the run's own control instead of a session field it cannot reach. Flagging for the owner rather than choosing.
- **Verified:** by reading `start_run` (no `confirm_exec`), `tools.exec_decision`/`_exec_confirmed`, the `/api/sessions` filter, and every `readGrants`/`GRANTS` call site in `frontend-v2/src`. I did **not** execute a live run that calls `run_shell` (no engine on this box) — the refusal text and the unreachability are both by inspection, and are the reason this is open rather than patched.

### R3-5 [MEDIUM] The mid-turn prompt queue is unbounded, is never persisted, and is silently discarded when the running turn fails
- **Where:** `src/rigma/serve.py:4058-4066` (accept), `4120-4129` (drain), `4047-4056` (`_release_claim`)
- **Trigger:** POST `/api/sessions/{sid}/chat` while `sid in _streaming`. The accept branch appends `message` to `_queued[sid]` with no length or size bound and answers 200 `{"queued": N}`. Then either (a) keep posting, or (b) let the in-flight turn raise.
- **Consequence:** (a) `_queued[sid]` grows without bound — the messages are full request bodies held in process memory, so a scripted client or a stuck UI tab can grow the server's RSS arbitrarily, and none of it is on disk. (b) `_release_claim` runs `_queued.pop(sid, None)` from `_drain`'s `finally`, and that `finally` fires on *any* unwind including an exception out of `_llm_turn` — so every queued prompt is dropped at the exact moment the user most expects the queue to survive, with the only signal being that the reply never comes. The `_ack` event already told them "queued behind the running reply".
- **Cause:** the queue is in memory on purpose (the comment at `3940-3944` explains why it is not in the session row), but the accept path has no cap and the release path treats "this turn is over" and "this turn died" identically.
- **Fix (not implemented):** cap `len(_queued[sid])` and the accepted message length (400 on overflow, so the client learns), and on the failure path re-queue rather than pop — or at minimum emit an SSE `info` event naming how many queued prompts were dropped. Small, but it changes client-visible behaviour, so it belongs in its own change with its own test.
- **Verified:** by reading the three blocks. Not executed: I did not drive a failing `_llm_turn` through the real route (needs a stubbed engine and a concurrent client); the accept-branch unboundedness is unconditional by inspection.

### R3-6 [LOW] Three consecutive lost races silently discard a whole chat turn's messages
- **Where:** `src/rigma/serve.py:3006-3024` (`_llm_turn`'s merge loop)
- **Trigger:** `_merge_and_save` raises `StaleWriteError` on all three attempts (`for _attempt in range(3)`).
- **Consequence:** the loop exits with `alive` still `True`, then sets `_ckpt["saved"], _ckpt["final"] = False, True` and carries on as if the turn had been stored. The reply and its tool trace exist only in the SSE stream the client already consumed; the transcript never gets them and nothing logs that they were dropped. Contrast `_append_guarded`/`_save_guarded`, which return the persisted row, and `serve.py:307-308`, which at least logs a warning.
- **Cause:** the retry loop has no `else:` for exhaustion — only the `alive is False` (deleted session) case is handled.
- **Fix (not implemented):** after the loop, if the last attempt raised, log at warning and yield an SSE `notice` telling the user the reply was not saved. Three races is rare, which is why this is LOW, but the failure is silent and the fix is two lines.
- **Verified:** by reading the block; not executed (would need three interleaved writers).

### R3-7 [LOW] `restart_run` writes the run's session with no `base_rev` either
- **Where:** `src/rigma/serve.py:5730-5734`
- **Trigger:** `POST /api/runs/{rid}/restart` while anything else writes the run's chat between the `sessions.load` at L5672 and the `sessions.save` at L5734.
- **Consequence:** the same lost update R3-2 fixed, in the last remaining run-path whole-row save. The window is short (both calls are synchronous on the loop, and the only `await`s between them are the plan reconciliation), which is why it is LOW — but `sess["mission"]`/`sess["run_id"]` are then re-asserted from a stale snapshot.
- **Cause:** R3-2's sweep stopped at `_run_loop`; `restart_run` is the other writer of the same session.
- **Fix (not implemented):** route it through `_save_guarded`. Not done because `_save_guarded` is a closure inside `build_app` defined *after* `restart_run` in source order — it is in scope at call time, but I did not want to add an untested cross-reference in the same commit as the R3-2 sweep. Trivial follow-up.
- **Verified:** by reading L5672-5734.

### R3-8 [LOW] `GET /api/runs/{rid}/log` still reads the whole `progress.md` on the event loop
- **Where:** `src/rigma/serve.py:5382-5389`
- **Trigger:** `GET /api/runs/{rid}/log` for a run whose `progress.md` has grown large (one line per tool action, `runs.append_progress` writes 600-char `done`/`next` fields per call).
- **Consequence:** the whole file is read and decoded synchronously inside the handler, so it stalls the event loop for the duration — the exact class IMP-11 fixed for the engine log by introducing `runs.read_tail_bytes` and `server_ops.log_tail_bounded`, and the exact class 14-5 fixed for session loads by threading them. The run's own `/api/runs/{rid}` and `/api/runs/active` already bound it via `_runs.get_log_tail` (64 KB), so the two endpoints disagree about the same file.
- **Cause:** IMP-11 changed `server_ops.log_tail_bounded` and left this route alone.
- **Fix (not implemented):** `await asyncio.to_thread(_runs.read_tail_bytes, path)` and return the bounded text (or keep the full body behind an explicit `?full=1`). Not done because both UIs read the log through `/api/runs/{rid}`'s `log_tail`, so this route has no in-repo consumer — only tests — and changing its response shape is a public-API decision, not a bug fix.
- **Verified:** by reading the route and both callers (`frontend-v2/src` has no `/log` call for runs; `src/rigma/data/ui/panels.js:1068-1077` uses `/api/runs/active` and `/api/runs/{id}`). Not measured — I did not build a large `progress.md`.

## Considered and cleared (this run)

- **01-2's atomic claim is real.** `serve.py:4045-4071`: `cancel = threading.Event()`, `_release_claim`, the `if message and sid in _streaming` test, and `_streaming.add(sid)` / `_cancels[sid] = cancel` are one unbroken synchronous stretch — the only `await`s in the handler are `_ensure_loaded` and the two `sessions.load`/`validate` calls *above* the test, and the `_drain` generator now only releases. No path claims twice, and the `except BaseException` at 4098 releases on a failed request. The `background=BackgroundTask(_release_claim)` net is identity-guarded (`_cancels.get(sid) is cancel`), so a late finaliser cannot release a successor's claim.
- **03-5's 409 is reachable from every HTTP turn entry point.** `/api/sessions/{sid}/chat` is the only route that starts a chat turn; `/v1/chat/completions` is a pure passthrough (`serve.py:5770-5816`) that never touches a session; the run's internal turn goes through `_drain_turn` directly and so never needs the guard; `_macro_drive_turn` and `_external_turn` are reached only from `chat_turn` (which checks) or from a run (which owns the session). The run-finishes-while-a-turn-is-in-flight case is benign: the check is a snapshot, and the worst outcome is that the chat turn proceeds against a run that has just stopped, which is a normal turn.
- **The `base_rev` reload-and-re-append does converge** for the driving-line write (`serve.py:4575-4587`) and for `_llm_turn`'s merge (`2968-3018`): both reload and re-apply, and `sessions.save` bumps `session[REV_KEY]` so a second save in the same turn can pass it again. The gap was the three sibling saves (R3-2), not the pattern.
- **SSE teardown closes its generators.** `_drain`'s `finally` releases the claim; `_llm_turn`'s `except BaseException` checkpoints, abandons the round's tool tasks (`_abandon_tasks`, synchronous by design under `GeneratorExit`) and re-raises; `_drain_turn` cancels the in-flight `__anext__` task before `aclose` (AUDIT 03-4) and swallows the `aclose` failure. `_external_turn` persists through a guarded save. The `/v1` streaming branch closes the upstream response in its `finally`.
- **01-4's `/v1` error shape is complete for the failure it named:** `client.send` is wrapped and `aread` is wrapped, both answering the OpenAI-shaped 502. A non-`HTTPError` from `request.body()`/`build_request` is still uncaught, but neither raises for a normal body.
- **The byte-bounded engine-log tail cannot be bypassed by size.** `runs.read_tail_bytes` seeks before reading and drops the first partial line; a single line longer than the bound is dropped whole, which is the documented intent. It reads `server_ops.log_path()`'s newest `server-*.log`, which is the current log for a running engine (the filename changes per launch).
- **`db.py`:** `upsert_session`'s guarded path is a single `UPDATE ... WHERE id=? AND rev=?` under `BEGIN IMMEDIATE` with `rowcount` as the test, so the F1 guard is genuinely atomic; a missing row counts as lost, so a deleted chat is never resurrected. `connect()` opens per call, sets WAL/`synchronous=NORMAL`, and the two `_MIGRATIONS` are `ADD COLUMN` only. A concurrent double-migration is handled by swallowing exactly `duplicate column name`. A failed migration raises out of `connect()` (loudly, not half-migrated), and `_initialised` is only set after the commit.
- **`app_settings`:** `validate` allowlists against `DEFAULTS` (an unknown key is a 400, so a backup cannot set a field outside the allowlist), `_coerce` refuses NaN and clamps, `save` merges into the loaded dict and writes temp-then-`replace`. `/api/settings` and `/api/restore` both go through `validate` first.
- **`memory.clean_rows`/`restore`:** `restore` re-validates and writes through `_write_all` (unique temp name + `os.replace`) under both the in-process RLock and the cross-process file lock, so the memory half of a restore is atomic. A crafted row cannot name a path — rows carry no path field.
- **Static serving:** `/v2/assets/{name}` rejects `/`, `\` and `..`; `/ui/{name}` serves only the `_UI_FILES` allowlist; `_v2_file` still carries its stale "resolve + prefix check" comment but `importlib.resources.joinpath` has no filesystem-escape primitive, and the only caller passing a user string rejects the traversal characters first. Unchanged from 01's "Improvements".

## Could not check

- No engine was ever running (safety rule), so nothing here is an end-to-end live turn: R3-4's refusal and R3-5's queue drop are by inspection, and R3-3/R3-6/R3-7/R3-8 were reproduced at the function/route level with a `TestClient` or a real run against a scripted in-process engine.
- `methods_api.register` (`serve.py:5572`) adds its own routes to the same app; `methods_api.py` was not read in full this run.
- The frontend was read for the grants wiring only (`grants.ts`, `Sidecar.tsx:800-850,945-960`, `api.ts`); no `npm run check`/`npm test` was run because no frontend source was changed.
- `src/rigma/data/ui_v2/` (the committed bundle) was not rebuilt — no frontend source changed, so no rebuild was required.
- The full suite was not run (6 minutes, forbidden); the 296 tests listed under R3-2 are the affected files plus every neighbour that touches session writes, run control, sessions and settings.

<!-- coverage: src/rigma/serve.py L280-330, L776-1000, L1282-1810, L1947-2450, L2900-3240, L3325-3490, L3938-4180, L4280-5140, L5217-5760, L5770-5824 (read; not read linearly — routes enumerated by grep then read handler by handler), src/rigma/sessions.py (read fully, 790 lines after edit), src/rigma/runs.py (read fully, 552 lines), src/rigma/app_settings.py (read fully, 122 lines), src/rigma/db.py (read fully, 282 lines), src/rigma/memory.py L171-360, L486-502 (read), src/rigma/tools.py L1590-1680, L3690-3770, kill_jobs_for_run (read), src/rigma/methods.py L520-630, methods_dir/_method_file/_shape_errors (read), src/rigma/macros.py L285-334, _as_bool (read), src/rigma/method_schema.py L155-199, SETTINGS_FIELDS/CARRY_FIELDS (read), src/rigma/server_ops.py log_path/log_tail/log_tail_bounded/_TAIL_LINE_BYTES (read), frontend-v2/src/chat/grants.ts (read fully), frontend-v2/src/chat/grants.test.ts (read fully), frontend-v2/src/chat/Sidecar.tsx L800-850, L945-960 (read), frontend-v2/src/api.ts L20-45 (read), frontend-v2/src/lib/logTail.ts (read), src/rigma/data/ui/panels.js L1068-1080 (grep) -->
<!-- END 01-http-r3 -->
