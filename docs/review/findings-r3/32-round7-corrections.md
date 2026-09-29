# Round 7 — the control plane becomes operable, and the audits that corrected it

Branch `review/deep-audit-2026-09-22`. Base for this round: `a215017`.

Three adversarial subagents were run against round 6/7 (all `deepseek-ai/DeepSeek-V4.1-Flash`
via `tokenjuice`). Between them they filed 2 HIGH, 3 MEDIUM and 3 LOW findings against the
round-7 code, plus 7 stale claims and a **standing-order breach**. Every finding below was
re-verified by hand before acting, and two of the HIGH findings turned out to be **wrong in
their stated mechanism while pointing at a real defect**. Both corrections are recorded,
because the reason a fix is needed matters as much as the fix.

---

## 1. The standing-order breach (the most serious thing this round found)

`tests/test_harness_dsh_live.py` guarded itself with

```python
pytestmark = pytest.mark.skipif(not harness_dsh.available(), ...)
```

which **opens** the module exactly when DSH is installed. DSH *is* installed on this machine
(`C:\AI\deepseek-harness\python\sdk\src` and `dsh.CMD` both exist), so `harness_dsh.available()`
returned `True` and `pytest -m "not hardware"` — the command used all round, and the one CI
runs — collected **2 tests that spawn the real `dsh` CLI and run a real agent turn**. Only the
model is fake.

VERIFIED, before the fix:

```
harness_dsh.available() = True
pytest tests/test_harness_dsh_live.py -m "not hardware" --collect-only -q
  -> tests/test_harness_dsh_live.py: 2
```

The standing order for this work is that no model is loaded and no live turn is run. A guard
that opens on installation is the opposite of a guard, and the `hardware` marker — the
project's existing convention, excluded by every documented test command — was missing.

VERIFIED, after the fix:

```
pytest tests/test_harness_dsh_live.py -m "not hardware" --collect-only -q  -> (nothing)
pytest tests/test_harness_dsh_live.py -m hardware     --collect-only -q  -> 2
```

The live turn now happens only when asked for by name. **This is a correction to my own earlier
claim that the suite ran no live turns.**

---

## 2. HIGH-2 — a real path containing a space broke every control operation

`drive_control` did `shlex.split(exe, posix=(os.name != "nt"))` on a value that `bin_path()`
returns **unquoted**. On Windows that is not a command line, it is a path:

```
shlex.split(r'C:\Program Files\nodejs\mcode.cmd', posix=False)
  -> ['C:\\Program', 'Files\\nodejs\\mcode.cmd']      # 2 tokens, want 1
```

So `argv[0]` became `C:\Program` and every control operation failed with a 502 — while the
`exec` turn path, which never splits, worked fine on the same value. The audit's second half
also verified: the quoted workaround cannot be used, because `bin_path()` existence-checks the
raw string and returns `None` for `'"C:\...\mcode.cmd"'`. **There was no working configuration
for a spaced path.**

Fixed by making `_control_argv` treat an existing file as ONE token, and splitting only when
the string is not a file — which is the test-double case (interpreter + script) and nothing
else. `acp` is appended unless it is already the **last** argument; checking "anywhere" would
let a directory merely named `acp` suppress the subcommand.

---

## 3. HIGH-1 — reported as "persistence never fires"; the truth is a race

The audit stated the fake's `--state-file` never gets written because `stop()` closes stdin and
calls `terminate()` with no wait, so `atexit` cannot run. It reported 3 failures in
`tests/test_acp_control.py` and 2 in the `--record` tests.

**This did not reproduce.** Measured 6/6 through `drive_control`:

```
run 0..5: ok=True  file_exists=True  {"sessionId": "mvs_race", "mode": "default", ...
```

and the `--record` tests passed 18/18. So the stated mechanism is wrong — the child does exit
on EOF and flush before the signal lands.

**But the race is real**, and that is exactly what a race looks like from one side: `stop()`
closed stdin and terminated in the same breath, so whether the child flushed depended on the
scheduler. Fixed by giving the child a bounded grace period (`wait(timeout=min(2.0, timeout))`)
before `terminate()`. This also removes the same flakiness from the pre-existing `--record`
channel, which shares the mechanism.

The audit's *diagnosis* was wrong and its *prescription* was right. Both are recorded.

---

## 4. MEDIUM-3 — a request body could drive ANY mcode session

Every client method builds `{"sessionId": self.session_id}` then `body.update(params)`, so a
caller-supplied `sessionId` silently **replaced** the resumed session. The route's stated
invariant is that the backend session comes from the chat row. VERIFIED:
`control_op_error("goal_create", {"objective": "x", "sessionId": "other"})` returned `''`.

Fixed at the seam where the session id is chosen (`body.pop("sessionId", None)`), not at the
eight call sites: the resumed session is the operation's **context**, not its input.

**And this exposed a protocol error of my own.** `delegation_stop` required a `sessionId` —
which was not a member id at all. MEASURED from mcode's own bundle
(`run-acp-command-JPZMIXGP.js`, `@minimax-ai/code` 0.5.4):

```js
e.app.onRequest("mcode/session/delegation/stop", xt, async({params:i}) => {
    let s = r(i.sessionId), a = await mo(e.runtime, s);
    return {receipt: await e.runtime.stopDelegation(a)}
})
```

`i.sessionId` is resolved as the **root** session and the whole tree is stopped. There is no
member id. So:

* the operation table now requires **no** params;
* the fake was corrected — it had invented a member-scoped stop, and **a double more capable
  than the server is the same defect as one less capable**;
* the UI's per-child "stop" button was **removed**. It said "stop this child" and would have
  stopped every child. It is now a form operation labelled "stop all delegated work", and
  `test_acp_control.py` asserts the per-member table is gone.

The fake also gained a `named` ledger recording every `sessionId` a request names, because its
state is one blob keyed by nothing — so before that, a client that honoured the override and
one that ignored it were **structurally indistinguishable**. The fix had to be made observable
before it could be tested.

---

## 5. MEDIUM-4 — the module broke its own rule about declaring capabilities

`initialize`'s docstring says *"Declaring a capability we cannot actually serve would be worse
than not declaring it"*, and then the capability dict declared:

```python
"fs": {"readTextFile": True, "writeTextFile": True},
```

with **no handler for `fs/read_text_file` or `fs/write_text_file` anywhere in the package**.
mcode plans around a declared capability, so Rigma was telling it to ask for file access and
then refusing. Removed rather than implemented: it would give the agent file access through a
second path beside the sandbox policy that already governs its tools — a security decision, not
a parity gap.

Second half: with no `on_request`, **every** server request was answered `{"result": None}`,
which is a valid JSON-RPC envelope but not a valid *answer* — `session/request_permission`
expects an `outcome` envelope and `elicitation/create` expects an `action`. The module already
had the right decline shapes; they were simply not wired to that path. Now they are.

---

## 6. MEDIUM-5 / MEDIUM-6 — consistency and budget

* `drive_control` resumed **without** `cwd` while `drive_turn_acp` sent it, so one session was
  resumed with a workspace and without one depending on the entry point. Fixed.
* The route advertises a one-minute timeout, and `drive_control` spent the full minute on each
  of **three** requests (~185s with `stop()`), all inside one `asyncio.to_thread` drawn from the
  default executor — so concurrent control requests could starve unrelated `to_thread` routes.
  The budget is now shared across the operation (`max(5.0, timeout/3)`).

A file-wide `timeout=timeout` → `timeout=each` substitution leaked `each` into `stop()` and
`drive_turn_acp`, where it is not defined. Caught by `ruff` (F821), reverted, and redone scoped
to `drive_control`'s body with an AST check that the name is only used where it is defined.

---

## 7. LOW — validation, bounds, and an orphan

* `session_id` was checked only for non-emptiness and then placed in a protocol frame verbatim;
  now bounded and control-character-rejected.
* `_FRAME_MAX` bounded what the server may send **us**; nothing bounded what we send **it**.
  Now bounded, because an unbounded write to a child's stdin surfaces as a hang.
* `harness_mcode.py` carried an orphaned tail of the **retracted** sentence — *"works — a NEW
  chat, because retrying this one cannot succeed"* — inside the docstring of the function whose
  whole job is to offer the way out of that dead end.
* `serve.py`'s lazy import kept, but its stated reason corrected: the deferral buys nothing for
  *this* request (the harness is already known to be mcode), and an ImportError here is a 500,
  which is the honest outcome.

---

## 8. The stale-claim audit — 7 confirmed, all corrected

| # | Location | Claim | Reality |
|---|---|---|---|
| A1 | `serve.py` route docstring | "Only the built-in can run a turn today" | All three are `runnable=True`; reaches `/openapi.json` |
| A2 | `harness.py` `Harness` docstring | used DSH as "probeable but not wired up" | DSH **is** wired |
| A3 | `harness_dsh.py` | "`VERIFIED` is Deliberately EMPTY" | `VERIFIED = "0.1.6-alpha.2"` |
| A4 | `api.ts` | "including the ones that cannot run a turn yet" | there are none |
| A5 | `harness_mcode_acp.py` | "the drift guard in the tests covers both" | no such guard existed |
| A6 | `harness.py` `HarnessError` | "unknown, not installed, or not implemented" | nothing is unimplemented |
| A7 | `harness_mcode.py` | orphaned fragment | see §7 |

A3 is the dangerous one: `conformance` computes drift as `(have != want) if (have and want)
else None`, so the non-empty value is what makes DSH drift detection **work**. Anyone
"correcting" the code to match that comment would have silently disabled it.

**A5 was fixed by writing the guard, not by deleting the sentence.** Three tables
(`STANDARD_METHODS`, `EXTENSION_METHODS`, `EXTENSION_NOTIFICATIONS`) had exactly one occurrence
each — their own definition — while the code claimed a guard covered them. They are correct
tables encoding wire facts, so the missing guard was the defect. `tests/test_acp_control.py`
now checks: the operation table against the dispatch chain, the dispatch targets against the
client's real methods, the extension table against the methods actually sent, and every
declared notification against `map_acp_update`'s source.

---

## 9. `map_acp_update`'s union count, and two new variants

The docstring said "the protocol's union has 13 members". VERIFIED against the SDK actually
resolved in the DSH checkout (`@agentclientprotocol/sdk@1.4.0` via `.pnpm`): it declares **15**,
adding `compaction_update` and `compaction_summary_chunk`.

Handled rather than dropped, because Rigma's **DSH** adapter already renders the same fact
(`compaction/` → the `compaction` event) — leaving them unmapped would render an identical
event on one backend and vanish on the other. VERIFIED: mcode 0.5.4's own bundle mentions
neither string, so this is forward-compatibility, not a live path.

The count is now stated **as of a version** rather than as a constant, and a test fails if a
bare count is reintroduced, because the number moves with the SDK.

---

## 10. UI: the discarded fields, and the last two HTTP-only operations

Three fields arrived from the server and were never rendered:

* `AcpDelegation.members[].errorMessage` — a failed child showed only the word "failed", while
  the queue two blocks down already rendered its own failure reason. The panel was inconsistent
  with itself about the same kind of fact.
* `AcpCommand.description` — bare `/names` a user had to guess at.
* `AcpConfigOption.name` / `.options` — the panel printed the raw protocol id while the server's
  display name sat unused, and discarded the list of values the server will accept.

`config_set` was then reachable only by raw HTTP; it now has a **select per option**, built from
the server's own list, because the values are a closed set and a typed value would be refused.
`mode_set` is **declared, not wired**, and the reason is recorded in the operation table: its
valid ids arrive in `session/new`'s response, which Rigma does not store, so there is nothing
for a dropdown to read. Hardcoding `plan`/`default` was rejected — it would silently stop
matching the server.

---

## 11. Dead code, honestly triaged

The audit named 7 dead constants. Three of them are **documentation, not dead code**, and were
kept with a note saying so — `TIERS` names the closed vocabulary `_TIER`'s values come from,
`T_REMIND_SECS` is the reference value the turn-based cadence was derived from, and
`VLLM_DOC_SERVE_ARGS_URL` sits in a block of three doc URLs of which the other two are used.
Removing those would have deleted the reason a reader can tell a deliberate choice from an
oversight.

`STANDARD_METHODS` and `EXTENSION_NOTIFICATIONS` were genuinely unreferenced and **no longer
are** — §8's drift guard references them.

`DEFAULT_ENGINE_RUNTIME` was the one real redundancy (a second name for `LLAMACPP`) and is
gone. `EMPTY_GOAL` in `goal.ts` claimed "exported for the tests" while no test imported it;
`EMPTY` was read by nothing either, so both are gone — a test written to justify an unused
export pins nothing.

Unreachable branches in `harness.py`, `serve.py` and `Sidecar.tsx` were **left in place**: they
guard states that are impossible *today* because every backend is wired, and they are what makes
the code safe if that stops being true. Removing a guard because it is currently unreachable is
how it stops being safe.

---

## 12. Gates

| Gate | Result |
|---|---|
| Python suite (before this round) | 2881 tests, 0 failures, 0 errors, 3 skipped |
| Frontend `npm run check` | clean |
| Frontend `npm test` | 434 tests, 29 files, all passing |
| `ruff check src tests` | clean |
| Bundle rebuilt | `index-BS03M442.js` + `index-UJCZ5dob.css` |

---

## 13. NOT VERIFIED — do not read as done

* No model was loaded, and no live mcode session was driven. The standing order holds.
* The DSH live turn was **not run** after being marked `hardware` — it is now excluded, which
  is the point, but it also means the marker is unexercised.
* ACP queue item `status` enum: still UNDETERMINED.
* Whether mcode replays a pending permission request on a resumed session: unknown.
* Whether a real mcode session survives a resume the way the fake models it: INFERRED from
  `session/resume` answering the same id.
* `compaction_update` / `compaction_summary_chunk`: mapped, but mcode never emits them, so the
  branch is unexercised by anything real.
* The control panel, the session-settings selects, the per-row buttons and the delegation
  panel have still never been seen in a browser. The render layer is covered by typecheck plus
  pure-function tests; there is no `*.test.tsx` in the project at all.
