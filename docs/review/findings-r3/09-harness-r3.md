# 09 harness + MCP + CLI — run 3

fork: r3/harness  base: 5b5790b
scope: the agent-harness seam (`harness.py`, `harness_dsh.py`, `harness_mcode.py`,
`_dsh_runner.py`), the MCP pair (`mcp_client.py`, `mcp_server.py`), `macros.py`,
and the CLI (`cli.py`) plus the `/v1` proxy in `serve.py`.

Mandate: audit run 2's fixes 09-1, 09-2, 09-4, 09-5, 09-6, 09-7, 09-9, 09-10
adversarially, then find what nobody looked at.

## What run 2's fixes actually do — re-measured, not read

| fix | verdict |
|---|---|
| 09-1 kill_tree on hard stop | **called** with the right pid, but the OUTCOME was thrown away → 09-3 below |
| 09-2 dict guards | true for the OUTER object only; one level down still fatal → 09-2 below |
| 09-4 bounded frames | holds. Oversize frame is skipped, framing stays in sync, a >16 MB unterminated frame ends the stream (documented, and `saw_end` then reports it) |
| 09-5 wedged MCP | holds, and recovers: a config change restarts it (`p_mcp_wedge.py` — call 0/1 time out, call 2 is `unavailable`, after the config edit it answers `ok`) |
| 09-6 bool coercion | holds. `_shape_errors` requires real bools; `_as_bool` is only a second line |
| 09-7 saw_end | holds (`no end event, exit 0` → `mcode exited 0 without completing the turn`) |
| 09-9 string `args` | holds |
| 09-10 longest-name routing | holds |

## Findings

### 09-1 [HIGH→FALSE] The `/v1` byte passthrough IS byte-for-byte, streaming included
- **Where:** `src/rigma/serve.py:5770-5816`
- **What I checked:** the mandate's HIGH-if-true hypothesis — that the arm's
  instructions are being rewritten on the way to the model.
- **How:** `.scratch/09-harness/p_wire.py` runs the REAL app under uvicorn on a
  real port and reads the WIRE with a raw socket (no `TestClient`, whose httpx
  decoder decompresses and masks exactly this). Upstream is a raw socket server
  that emits known bytes: `\r\n` line endings, no space after `data:`, a UTF-8
  sequence split across chunks, `Content-Encoding: gzip` in one case.
- **Result:** request body identical; response body de-chunked is byte-identical
  to what the upstream sent, in both the plain and the gzip case, with and
  without the client sending `Accept-Encoding`. The passthrough holds.
- **A trap worth recording, because it produced a wrong answer first:**
  `TestClient` (and any httpx response object) transparently decompresses, and
  `r.content` then reports PLAIN bytes while `r.headers["content-encoding"]` still
  says `gzip`. `p_min.py` "proved" a passthrough break that does not exist. Only
  `aiter_raw()` at the proxy, and the raw socket, showed the truth. Any future
  test of this must use one of those two.

### 09-2 [MEDIUM] An mcode event with a malformed NESTED object kills the whole turn
- **Where:** `src/rigma/harness_mcode.py:585-634` (`map_event`), reached from the
  read loop at `harness_mcode.py:817-867`
- **Trigger:** one NDJSON line whose `item`, `item.toolCall` or `error` is a
  string, a list or a number — e.g. `{"type":"item.started","item":{"id":"c",
  "type":"tool_call","toolCall":"bash"}}`
- **Consequence:** `AttributeError` out of `map_event`, caught by `drive_turn`'s
  defensive handler as `mcode stream failed: 'str' object has no attribute 'get'`.
  The turn ends there. Measured (`p_mcode_drive.py`): a turn that had already
  yielded `hello ` came back as `[('text','hello '), ('error', 'mcode stream
  failed: …')]` — the rest of the stream, including the final answer, is thrown
  away. On a versioned wire format this is the quiet-degradation class `VERIFIED`
  exists for.
- **Cause:** run 2's 09-2 guard was applied to the outer JSON value
  (`if not isinstance(obj, dict): continue`) and not to what it contains.
  `map_event` did `item.get(...)`, `call.get(...)`, `err.get(...)` on whatever
  was nested.
- **Fix:** implemented — `item`, `item.toolCall`, `turn.failed.error` and
  `exec.completed.result` are shape-checked. A malformed `item`/`toolCall` costs
  one event; a malformed `error` still reports that the turn failed rather than
  raising.
- **Verified:** `pytest tests/test_harness_mcode.py -k "malformed or leak or
  per_turn"` → 4 passed. Before the fix, `test_a_malformed_nested_object_is_skipped_not_fatal`
  and `test_one_malformed_event_does_not_end_the_turn` both FAILED with the
  AttributeError above; after, the reply survives and no error event appears.

### 09-3 [MEDIUM] A failed tree kill was reported as a successful one
- **Where:** `src/rigma/harness.py` `kill_tree`, `src/rigma/harness_dsh.py:368-407`
  (`_stop`), `harness_dsh.py:277-330` (`_read_events`)
- **Trigger:** a DSH turn hits its timeout and `taskkill /F /T /PID <n>` fails —
  which this very sandbox does, and which the brief lists as the reason ~9 runner
  tests hang here.
- **Consequence:** the user is told `DSH turn timed out after Ns and was killed.`
  `kill_tree` had swallowed the failure and returned nothing, so nothing on the
  path could tell. `proc.kill()` still takes the Python runner, so the turn ends
  normally — while the Node agent and every subagent it started keep running
  against Rigma's model server and hold VRAM. Nothing reports it. That is the
  precise case the mandate asked about.
- **Cause:** `kill_tree` was a `-> None` function whose `except (OSError,
  subprocess.TimeoutExpired): pass` made its one failure mode unobservable.
- **Fix:** implemented — `kill_tree` returns whether the TREE is known to be gone
  (`False` on POSIX, where there is no job object and it genuinely cannot know;
  `False` when taskkill fails or times out; `True` when the child was already
  gone). `_stop` records it on the run. `_read_events` now kills *before* it
  speaks, so the timeout message is chosen from the real outcome:
  `and was killed` when it worked, and
  `— and the process tree could NOT be confirmed dead, so the DSH agent and any
  subagents it started may STILL BE RUNNING and holding the model server` when it
  did not.
- **Verified:** `pytest tests/test_harness_dsh.py` → 24 passed. Before the fix,
  `test_a_failed_tree_kill_is_reported_not_assumed` could not even be written
  (there was no return value to assert on) and the message was unconditional.
  On the REAL platform, with a real process tree and no mocking
  (`.scratch/09-harness/p_killtree.py`): `taskkill` is refused here,
  `kill_tree` returns **False**, the child is still dead from the fallback
  `proc.kill()`, and `tasklist` no longer shows the pid. That is precisely the
  case that used to be reported as a successful tree kill.
- **Also fixed here, same class:** the kill now delegates to `tools._kill_tree`,
  which already solved this properly (AUDIT F35) — it polls the process instead
  of trusting taskkill's exit code, which is 0 even when the tree walk misses a
  re-parented grandchild, and it uses `killpg` on POSIX rather than reaching
  only the direct child. Two tree-kill implementations disagreeing about what
  "killed" means was its own hazard.
- **Residual, reported not fixed:** the PID-recycling window is real but tiny —
  `_stop` guards with `poll() is None`, so a child that has already exited is
  never handed to `taskkill`; the remaining window is inside `kill_tree` between
  that poll and taskkill's own open (`p_cli_open.py`). Closing it needs a job
  object / `CREATE_NEW_PROCESS_GROUP` at spawn time, which is a design change,
  not a fix.

### 09-4 [MEDIUM] `VERIFIED` drift was invisible on the turn path (the mcode 0.5.4 question)
- **Where:** `src/rigma/harness_mcode.py:92` (`VERIFIED = "0.5.1"`),
  `src/rigma/harness.py` `conformance`, `src/rigma/cli.py:641-690`,
  `src/rigma/serve.py:1743-1754`
- **The question asked:** when the installed build is not the verified build, does
  Rigma warn, refuse, or silently proceed?
- **Answer, proven:** **it silently proceeds.** `conformance()` computes
  `drift` correctly (tests at `tests/test_harness_seam.py:559-592` prove the
  three states), but its only consumers are the `rigma harness` command and the
  menu's `?check=1`. `harness.resolve()` — the one gate every external turn
  passes through — never looks. Neither does `drive_turn`. Installing
  `@minimax-ai/code@0.5.4` therefore produces turns that look entirely normal.
- **Consequence:** the failure `VERIFIED` was invented for. A `schemaVersion`
  bump that renames an item type drops tool calls from the transcript while the
  reply still arrives; nothing in the turn says so. `harness.py`'s own comment
  calls `drift` "the build this adapter was measured against"; a build that is
  never compared is a comment.
- **Fix:** implemented, and it does not refuse — the user's own install is their
  call, and a refusal would be Rigma overriding it. `drive_turn` now compares
  once per process and yields ONE `notice` in the transcript:
  `mcode 0.5.4 is not the build this adapter was measured against (0.5.1). Tool
  calls can go missing from the transcript while the reply still arrives — run
  `rigma harness` for the detail.` Unknown (`""`) and agreement stay silent, so
  the warning cannot become noise.
- **Verified:** `pytest tests/test_harness_mcode.py -k drift` → 3 passed. The
  fake CLI gained `--version` (defaulting to `0.5.1`) so this is exercised
  through the real `drive_turn` and the real subprocess, not a stub.
- **Not done, deliberately:** `harness.resolve()` still does not refuse a drifted
  backend. Making it refuse needs a subprocess per turn or a cached version, and
  refusing would be the wrong default — this is a warning, not a gate.

### 09-5 [MEDIUM] Two CLI commands reported success they had not established
- **Where:** `src/rigma/cli.py:1267-1290` (`stop`), `cli.py:641-690` (`harness`)
- **Trigger:** (a) `rigma stop` when the state record's pids no longer match the
  processes they name (recycled pid / crash left a record behind);
  (b) `rigma harness` on a DRIFTED backend.
- **Consequence:** (a) printed `stale state — nothing was killed` and exited
  **0**, so `rigma stop && rigma up` walked straight past a stop that never
  happened and no script could tell the two apart. (b) printed `DRIFTED - …` and
  exited **0**, so `rigma harness && deploy` was a green light on a build whose
  event schema nobody has checked — the one thing that command exists to catch.
- **Cause:** both printed the honest sentence and returned no status.
- **Fix:** implemented — `stop` exits 1 when nothing was killed (0 for the honest
  "not running", 0 for a stop that worked); `harness` exits 1 when any row is
  `drift is True`. `unverified` (drift `None`) stays 0 on purpose: nobody can say
  anything changed, and treating "unknown" as failure is how a real signal gets
  ignored.
- **Verified:** `pytest tests/test_cli.py tests/test_harness_seam.py` → all pass.
  `test_stop_says_so_in_its_exit_code_when_it_stopped_nothing` and
  `test_harness_exits_nonzero_when_the_build_drifted` both FAIL before the change
  (`assert 0 == 1`) and pass after. The pre-existing
  `test_the_conformance_command_says_drift_out_loud` asserted `exit_code == 0`
  and was updated — it was pinning the lie.

### 09-6 [LOW] The adapter signature guard retyped `serve.py`'s keyword set by hand
- **Where:** `tests/test_harness_dsh.py:269-298`
- **Trigger:** someone adds a keyword to the `adapter.drive_turn(...)` call in
  `serve.py:1855-1864` without updating the guard's literal dict.
- **Consequence:** the guard — whose entire purpose is the DSH adapter shipping
  unable to run a turn through the product — silently stops covering the new
  keyword. An adapter that cannot accept it raises `TypeError`, which
  `_external_turn._pump` catches and reports as a failed turn, so the arm looks
  broken rather than mismatched. This is the failure the guard's own docstring
  describes, one level up.
- **Cause:** `passed = {...}` in the test is a copy of the call site.
- **Fix:** implemented — the test parses `serve.py` with `ast` (the precedent is
  `tests/test_launch_records_fingerprint.py`) and collects the real keyword set
  from every `drive_turn` call, then asserts the guard still has a subject.
- **Verified:** `pytest tests/test_harness_dsh.py -k adapter` → passed; ruff
  clean. The guard now covers all 11 names the call site passes and would catch a
  12th.
- **Corrected mid-audit:** my first version of this asserted the guard omitted
  `timeout`. It does not — `serve.py` never passes `timeout`, and both adapters
  therefore run on their 1800 s default. The hand-copy problem is real; the
  "missing timeout" framing was wrong and is retracted here.

### 09-7 [LOW] `rigma up` can silently never open the browser
- **Where:** `src/rigma/cli.py:1304-1331` (`_open_when_listening`), called at
  `cli.py:1443` and `cli.py:1660`
- **Trigger:** the UI port does not accept a connection within the 15 s deadline
  (a slow first import, a busy disk, a cold model load).
- **Consequence:** the daemon thread reaches its deadline and returns. No
  browser, no message, no retry. Measured (`p_cli_open.py`): returns after 0.62 s
  for a 0.5 s deadline with `opened == []` and prints nothing. The URL is printed
  by `up`, so the user is not stranded — but they are told `chat UI: http://…`
  and the automatic open they expected just does not happen.
- **Cause:** the poll loop's `while` simply ends.
- **Fix:** FIXED by the orchestrator as `1637a44`. The report above deferred it
  because "it writes to stderr from a daemon thread that can outlive the terminal's
  `Ctrl+C`" — that risk is real, but it is not the interesting half. **15 s is
  simply shorter than the case this helper exists for:** a cold start is uvicorn
  importing FastAPI, building the app, and spawning a 13 GB model load, which is the
  exact slow first run F16-2 was written about. So the automatic open failed
  precisely when the user was most likely to be waiting for it. The deadline is now
  180 s (outlasting a real cold start) and reaching it prints why on **stderr**
  (stdout is what a scripted caller reads). The thread stays a daemon, so Ctrl+C
  still ends it; a torn line during interpreter shutdown is a better trade than the
  silence it replaces.
- **Test-first:** the "says so when it gives up" case fails against the old code
  (empty stderr). The default is also pinned as a contract, because the failure mode
  is a number nobody looks at again.
- **A vacuous test of my own, recorded so the method is honest:** my first attempt at
  pinning the deadline drove a fake clock past 15 s and **passed against the unfixed
  code** — it passed `timeout=120.0` itself, so it was asserting its own argument.
  Discarded and replaced with one that pins the real default.
- **Verified:** `.scratch/09-harness/p_cli_open.py` output quoted above.

## Hypotheses the mandate raised that are NOT bugs — with the evidence

Stated so nobody re-spends the budget on them:

- **`/v1` is not byte-for-byte** — FALSE. See 09-1. It is, streaming included.

## mcode re-verified LIVE by the orchestrator (not read)

The agent could not run the real mcode CLI. The orchestrator did, against the repo's
own `tests/fake_oai_server.py`, which was written for exactly this:

```
rigma harness                       -> mcode ok (0.5.4)   [VERIFIED = "0.5.4", drift False]
python tools/mcode_probe.py --port 11599 --turn "…"
  ensure_provider -> pid='custom_provider:rigma' err=''
  providers: custom_provider:rigma  baseUrl=http://127.0.0.1:11599/v1
             active=true  apiFormat=openai-completions
  real turn: 3 text events streamed
  state: session_id=mvs_fb7a… resumed=false status=succeeded
         usage inputTokens=8 outputTokens=3
```

And the wire log, which is the evidence for the passthrough claim:

```
{"path": "/v1/chat/completions",            "stream": false, "tools": 0,  "messages": 1}
{"path": "/v1/responses/input_tokens",      "stream": null,  "tools": 0,  "messages": 0}
{"path": "/v1/responses/input_tokens",      "stream": null,  "tools": 21, "messages": 0}
{"path": "/v1/chat/completions",            "stream": true,  "tools": 21, "messages": 3}
```

Three things this settles that fixtures cannot: the installed 0.5.4 still saves and
selects a custom OpenAI-format provider against a live endpoint; its event stream is
still the shape the adapter was written for; and it really does send its own
`/v1/responses/input_tokens` pre-flight (with the 21-tool roster) before a streamed
turn — which is why the fake server has a `--no-count` flag. `Authorization: Bearer
local` confirms the key path. The fake engine was stopped by captured PID; the
owner's own llama-server was not touched.

- **`macros.substitute` can expand recursively / blow up exponentially / read a
  file or the environment** — FALSE. `re.sub` is a single pass, so replacement
  text is never rescanned: `{{a}}`→`{{b}}`→`{{a}}` resolves to the literal
  `{{b}}` and stops (`p_macros.py` P1/P2/P6). 5,000 placeholders resolve in
  1.1 ms (P3). `_resolve` consults only `ctx`'s four builtins and `ctx["vars"]`;
  `{{HOME}}` does not even match `PLACEHOLDER` (`[a-z_]+`) and `{{env:PATH}}`
  returns `None`, left verbatim (P4/P5).
- **The mcode item-id memo is process-wide** — FALSE. `seen` is built inside
  `drive_turn` (`harness_mcode.py:811`), so it is per turn; a repeated id in a
  later turn re-emits (`p_mcode_fuzz.py`). The module-level dict in the file is
  `_VERIFIED`, which is the provider-id cache, keyed by
  `(base_url, model, ctx, max_tokens)`.
- **`mcp_client`'s wedge detector never fires / a wedged server never recovers** —
  it fires (2 consecutive timeouts → `_failed`) and it DOES recover on a config
  change (`p_mcp_wedge.py`). The `carried`/`stale` comment at
  `mcp_client.py:361-364` is accurate, not stale.
- **`mcp_server` exposes a weaker confinement than the in-process surface** — not
  demonstrated. All four roster tools go through `run_tool` with the same `needs`
  gate (`undo_last_change` is `needs="code"`, satisfied only by the explicit
  `RIGMA_MCP_ALLOW_CODE=1` the registrar sets). `RIGMA_MCP_PROFILE` is never set
  by `_mcp_spec` — so `mcp_server.profile()` is always `"all"` in production and
  its "second execution path that skipped the profile gates would be a hole"
  docstring describes a gate that cannot currently fire. Measured
  (`p_mcpsrv.py`): even with `RIGMA_MCP_PROFILE=confined`, the offered roster is
  unchanged (`remember`, `recall`, `undo_last_change`) because none of them is
  `kind="exec"` or in `_NETWORK_TOOLS`. It is dead code, not a hole — worth
  deleting the env read or wiring it, but it grants nothing today.
- **The signature guard only covers dsh** — FALSE. It covers `dsh` and `mcode`,
  which are the only two backends with adapters; `native` has none by design
  (`_ADAPTERS` has no `native` entry, and `serve.py` runs `native` through
  `_llm_turn`). See 09-6 for what was actually wrong with it.
- **`mcp_client` drops a reply whose JSON-RPC id is not an int** — real by
  inspection (`_ID_RE` only matches `\d+`, `_replies` is keyed by int) but not
  reachable: the only server Rigma ships is `mcp_server.py`, which echoes the
  request id verbatim, and a third-party server answering a string id would be
  answering an id Rigma never sent.

## Could not execute

- **The real DSH and mcode CLIs.** Forbidden by the task, and 09-1's tree kill
  cannot be proven end-to-end here anyway: `taskkill` is refused in this sandbox.
  Everything above is against the fakes in `tests/` and raw sockets.
- **The ~9 `test_harness_dsh` runner tests the brief says hang.** They are
  INTERMITTENT here, and the hang is environmental, not a fix of mine. The same
  `pytest tests/test_harness_dsh.py` completed 24 passed in ~2 s four times on
  separate basetemps, and the whole suite (1,100-odd tests) completed exit 0
  twice. One later whole-suite run then hung past a 10-minute cap with no
  failure — consistent with the brief's `taskkill`-denial explanation, since
  `taskkill` is refused on this box (`p_killtree.py`). I could not reproduce it
  on demand and am not claiming to have fixed it.
- **`pytest --timeout`** is not available (`pytest-timeout` is not installed in
  this environment), so a hang cannot be bounded from inside pytest; the
  whole-suite run has to be capped from outside.
- **`test_harness_dsh_live.py`** — skipped by name (needs a real checkout).
- **The full suite's `test_audit_serve.py::test_a_chats_archive_is_never_trimmed`**
  failed once under a whole-suite run with `-x` and PASSES in isolation. Not
  caused by anything here (my diff touches no compaction path); reported as
  suspected inter-test pollution, not chased.

<!-- coverage: src/rigma/harness.py (read fully), src/rigma/harness_dsh.py (read fully), src/rigma/harness_mcode.py (read fully), src/rigma/_dsh_runner.py (read fully), src/rigma/mcp_client.py (read fully), src/rigma/mcp_server.py (read fully), src/rigma/macros.py (read fully), src/rigma/method_schema.py (read fully), src/rigma/methods.py L433-600 (read), src/rigma/cli.py L1-50,630-700,1065-1105,1262-1450,1600-1672 (read), src/rigma/serve.py L1735-1764,1800-2000,5770-5818 (read), src/rigma/state.py L30-80,175-224 (read), src/rigma/tools.py L685-747,1700-1746,2048-2129 (read), tests/test_harness_dsh.py (read), tests/test_harness_mcode.py (read), tests/test_harness_seam.py (read), tests/test_cli.py (read), tests/test_cli_browser.py (read), tests/test_serve.py (read), tests/test_launch_records_fingerprint.py L15-59 (read) -->
