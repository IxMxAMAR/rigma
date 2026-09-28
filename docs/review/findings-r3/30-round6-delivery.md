# Round 6 — the harness control planes, and a silent MCP failure

**Tree:** `C:\ComfyUI\RD\rigma-review`, branch `review/deep-audit-2026-09-22`.
**Scope:** make everything the two harnesses offer reachable from Rigma's UI, and find
what was being dropped on the way.

Round 5 established what the harnesses CAN do. Round 6 is the delivery: the ACP control
plane wired end to end, the events that were being dropped, and one defect that had been
making a whole capability silently absent.

## Commits

| Commit | What |
|---|---|
| `a7caded` | R6-ACP + R6-OKOF — an ACP client for mcode, and the tool-status enum resolved |
| `890b970` | R6-ACP-UI — mcode's control plane in the panel, and the drift guard that found a real bug |
| `e2528c3` | R6-ACP-TURN — a chat turn can run over ACP, and its approval answer reaches the panel |
| `c7bdd65` | docs — round-6 checkpoint |
| `2dc8b82` | R6-ACP-SEAM — the ACP wire is selectable, and cannot silently fall back |
| `67b412e` | docs — the seam, and the flaky test fixed rather than tolerated |
| `c6fac40` | R6-ACP-APPROVE + R6-EXPORT — a permission prompt can be answered, and `/export` exists |
| `781e276` | R6-HARNESS-GAPS — the DSH events that were dropped, and mcode's background tasks |
| *(this)* | R6-MCP-INSERT — the generated MCP patch was silently rejected by DSH |

## 1. The ACP control plane, end to end

mcode's Agent Client Protocol surface was reachable but unused, which meant the queue,
steering, plan mode and an ANSWERABLE permission prompt were all invisible in practice.
Now: a session field selects the wire, `sessions.MCODE_TRANSPORTS` validates the value,
the seam asks the adapter for a `drive_turn_acp` capability rather than testing its module
name, and if the field says `acp` while the capability is missing the turn ERRORS rather
than running over `exec`. Running it over `exec` anyway would be a turn the user did not
ask for.

`exec` stays the default, and the selector is gated on mcode.

## 2. A permission prompt can be answered

The last piece, and the one with real teeth. Over `exec` mcode has no interaction host, so
when `smart` decided to ask nobody could answer and mcode's own guard then refused to
start another session — the chat was blocked PERMANENTLY (R5-MCODE-DEADEND). Over ACP the
server asks and BLOCKS, so the answer has to come from the user.

The handshake crosses two threads — the ACP client's reader thread blocks inside the
request handler, the answer arrives on the event loop from an HTTP route — and they meet
on a `threading.Event` plus a slot, like the existing `_cancels` map. Three rules fall out,
each a way the obvious implementation is wrong:

- **The wait is BOUNDED.** An unbounded wait wedges the TRANSPORT, not the turn.
- **A timeout is `cancelled`, not a denial** — the protocol's own word for "no answer". A
  timeout is not a decision the user made.
- **The slot is keyed by SESSION and cleared in a `finally`**, so an answer for one chat
  cannot resolve another's, and a later click cannot answer a question that is gone.

The route refuses a non-boolean `allow` (`"false"` is truthy in Python, and granting
permission from a typo cannot be taken back) and refuses a mismatched `requestId` with 409
(a stale card must not decide a different question than the one on screen).

`awaiting` is what arms the button, and deliberately NOT "no outcome yet": a DSH ask with
no decision is pending in the audit trail but can never be answered over that wire, so a
button there would promise what the connection cannot do.

## 3. R6-MCP-INSERT — a silent failure, found by composing the tree

**This is the most important finding of the round.** `harness_dsh.mcp_patch_file` wrote a
BARE row. A DSH patch list is a list of OPERATIONS, and a bare row is read as a REPLACE —
so DSH answered

    dsh: [<tmp>\rigma-dsh-mcp.yaml] patch: entry "mcp-rigma" not found

dropped the overlay, and **exited 0**. Rigma's own MCP tools therefore never reached a DSH
turn, while `harness.py`'s capability menu said they were mounted. Nothing failed; the
tools were simply absent.

It survived because the file had been checked as YAML (it parsed) and as a string (the rows
were there), and the MCP server's own handshake had been verified by hand — none of which
is the claim that DSH ACCEPTS the overlay. The check that settles it needs no model:

    dsh --profile sdk-minimal --dump-config --patch <file>

`--dump-config` composes the profile tree and exits, so it does not touch the standing
order. With `insert:` the row appears; without it the command warns and the row is gone.
`tests/test_dsh_mcp_patch.py` now pins both the shape and the composed tree.

## 4. What was being dropped, and two findings that were NOT defects

Six round-5 findings (F5–F9, mcode #8) were re-checked by an adversarial sub-agent, which
REFUTED parts of them — including parts of my own reasoning. Two were not defects at all.

- **F7 — workflow/PTC reduced to an opaque string. TRUE, and worse.** `tool-workflow/*` was
  not a state event, so it fell to the notice path and matched there only by ACCIDENT: the
  filter tests for the substring `"tool"`, and `"tool-workflow"` contains it. All four
  lifecycle events reached the UI as `"session.event tool-workflow/agent-start"`, and
  `serve.py` had no arm for the names either. Fixed across all three hops, with a
  `WorkflowBlock`; the fold rebuilds DSH's own structure (`runId` joins the four, `seq`
  pairs an agent's start with its OWN end, so out-of-order finishes get right verdicts).
- **F8 — `web_fetch` omitted for a reason that only covers SEARCH. TRUE.** The key belongs
  to the search provider; `dsh-web-fetch-http` has "No credentials to check". A capability
  was absent for a constraint that does not apply to it. Mounted, with `search: false` so
  `web_search` is not registered against a provider that does not exist.
- **F9 — the subagent tool/provider pairing was not pinned. TRUE, in the test.** It asserted
  only that each `provider` was truthy, so SWAPPING spawn and fork passed while inverting
  the tools' meaning.
- **F5 — PARTLY, and my earlier reasoning was WRONG.** I had written that the descriptor's
  `label` is the provider's name. DSH says it is "the child's durable creation label" — it
  IS a child's name, so folding it would be USEFUL. What blocks it is the missing child id.
  Comment and test fixture corrected; the fixture had `label == provider`, a shape DSH does
  not produce, so it confirmed the error instead of catching it.
- **F6 — CANNOT FIRE.** The only appender is gated on `modelSelectionSettings === true`,
  which defaults false and which Rigma's rows do not set. Adding it would create a branch
  nothing can reach, which is worse than absence because it reads as coverage. Declared.
- **mcode #8 — PARTLY.** A chip appeared but nothing made it a task, so `details.task_id`
  was discarded and a quiet turn was indistinguishable from a hung one. A background result
  that STARTS a task now emits a row. **The trigger is the START, not the task id** — my
  first attempt keyed on the id and a PRE-EXISTING test failed, correctly: it was pinned
  from a real capture whose status was `"completed"`, and a finished job is not a live
  agent. The fix went to the change, not the test. A second half was needed too:
  `auto_promoted` is a RUNNING state, but the fold treated unknown statuses as ends, so a
  promoted command would have drawn as completed.

## 5. Gates at this checkpoint

- Python: **2850 tests, 0 failures, 0 errors, 3 skipped** (`.scratch/full35.xml`).
- Frontend: **404 tests across 28 files**; tsc clean.
- ruff clean across `src`, `tests`, `tools`.
- Bundle rebuilt and referenced from `index.html`, superseded assets removed.
- DSH profile tree composed with `--dump-config`: exit 0, empty stderr, `mcp-rigma` and all
  four web/workflow/subagent rows present.

## 6. NOT VERIFIED — the boundary, stated plainly

- **No model was loaded**, per the standing order. No DSH turn, no mcode turn, no chat.
- **No `mcode acp` session has been driven against a live engine.** The ACP client, the
  turn driver and the approval handshake are tested against a real subprocess
  (`tests/fake_acp_server.py`) and by inspection of the route — not against mcode.
- **No permission prompt has been answered end to end** in a browser. The rules are pinned;
  the round-trip through a real engine is not.
- **The workflow panel, the ACP panel and the durable panel have never been seen in a
  browser.** There is no `*.test.tsx` in the project, so the render layer is covered by
  typecheck plus reducer and formatter tests.
- **No capture contains ACP tool status 2**, so "2 means success" rests on the enum's names
  and the `isError` predicate rather than on an observed success.
- The `tool-workflow/*` field names come from DSH's own `dispositions.ts`, not from a
  capture, so a run has not been observed producing them.
- `dsh --dump-config` proves the overlay is ACCEPTED. It does not prove the MCP server
  starts under `dsh-mcp-client`, which is the separate claim the handshake test covers on
  its own.

## 7. Next, in the order I would take it

1. **`/goal` as a Rigma-native command.** `harness_mcode_acp` defines
   `goal_create`/`goal_patch`/`goal_clear` with ZERO callers and no route exposes them, so
   a Rigma-native `/goal` is implementable exactly as `/compact` was. The written reason it
   was not done is still valid for passing through DSH's own `/goal` and has gone stale as
   a blanket justification. Investigated, deliberately not done in this round.
2. **`/feedback`** — Rigma has no feedback mechanism of any kind: no route, store, file or
   log. That is missing BACKEND, not a missing command, so it is recorded rather than faked.
3. **Drive the ACP wire against a live mcode**, which needs the standing order lifted. That
   is the one step that would turn several NOT VERIFIED lines above into verified ones.
