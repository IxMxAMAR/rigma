# Round 7 — the control plane becomes invocable

**Tree:** `C:\ComfyUI\RD\rigma-review`, branch `review/deep-audit-2026-09-22`.
**Scope:** finish what round 6 left readable-but-untouchable, and correct two places
where the product told a stuck user something false.

## The finding this round turns on

`AcpClient` has defined mcode's whole control surface since `a7caded` — the goal, the
queue, steering, the delegation tree, plan mode, the configOptions — and **nothing called
any of it.** Every one of those arrived as a NOTIFICATION, so the panel could draw the
queue, the goal and the delegation tree, and a user could not touch any of them.

A control plane you can only watch is not a control plane. Measured before the fix:

| method | call sites |
|---|---|
| `goal_get` / `goal_create` / `goal_patch` / `goal_clear` | **0** |
| `queue_enqueue` / `queue_update` / `queue_delete` / `queue_steer` | **0** |
| `steer` / `activate` / `set_mode` | **0** |
| `delegation_get` / `delegation_stop` | **0** |
| `set_config_option` | 1 (the turn's own permission mode) |

So "carrying the goal control plane" was **not** done. It is now.

## 1. `drive_control` — one operation, one session, one process

`harness_mcode_acp.drive_control(op, params, exe=…, session_id=…)` opens a short-lived
client, `initialize`s (declaring `_meta`, because the answer and the notification are
**separately gated** — without the declaration every call succeeds and no `goal_update`
ever arrives), `session/resume`s the chat's own session, performs one operation and stops.
It never raises, for the same reason `probe_surface` never raises: it is called by a route
that is drawing a panel.

**The operation names are an allowlist** of 15 literals. An HTTP body supplies the name, so
the mapping is what stops a request from naming an arbitrary protocol method — including
`session/prompt`, which would be a model turn smuggled through a control route.

**Why not the turn's own connection.** That would be better — no second process, no
question about session ownership — but the turn's connection lives inside
`drive_turn_acp`'s worker thread, and reaching it from an HTTP route needs a new
cross-thread control channel. That is a larger change than this, so the cost is stated
rather than hidden: one mcode process per control action, which is what the capability
menu's probe already does.

## 2. `POST /api/sessions/{sid}/control`

Resolves the chat, then refuses in three specific ways rather than doing nothing:

- **404** for an unknown chat.
- **409** when the chat is not mcode, or when its transport is `exec` — and the body
  carries the transport so the UI can name what to change. `exec` projects one turn and
  holds no session, so there is nothing to steer and no queue to add to.
- **400** for an unknown operation, **before any process is opened**. A bad request is a
  client mistake, and paying for an mcode launch to discover it would also report it as
  the wrong kind of error.

It passes the **backend session id** (`harness_sessions.mcode`), not the chat id — passing
the chat id would resume a session mcode has never heard of.

## 3. Two places the product told a stuck user something false

`_interaction_dead_end`'s docstring said the permanent block was "not fixable here"
because "mcode's ACP path" was unavailable and "ACP needs `mcode login`". **Both claims are
false, and both had already been corrected elsewhere**: `c9f9f4a` established that ACP
needs no login — the earlier "blocked on authentication" was a probe bug, where stdin was
attached to a *file* so mcode hit EOF before answering — and `a7caded` shipped the ACP
client. The correction reached the docs and never reached this function.

The user-facing sentence repeated it: *"mcode's own TUI or ACP client, which Rigma cannot
drive. Start a NEW chat."* So a user whose chat had just died was told to abandon it, when
the remedy was one selector away. A stale sentence is not cosmetic here — it is the
difference between losing a conversation and switching a dropdown.

Both now say what is true: retrying cannot succeed, **switch the transport to `acp`**,
which *is* an interactive host. Whether mcode replays the pending request on a resumed
session is **not verified**, and the docstring says so rather than implying it.

## 4. `smart` + `exec` is broken, not risky

`smart` asks before risky actions. On `exec` there is nobody to ask, so mcode raises the
request, the turn fails, and its guard then refuses to start **any** later session in that
chat — the chat is dead permanently. That lived only in a tooltip, which is not where
anyone looks before choosing, and the two selectors that have to agree were free to
disagree silently.

The permission option said "a turn can fail", which understates it: the turn is
recoverable, the **chat** is not. There is now an inline warning, decided by a pure
`permissionTrap(harness, permission, transport)` helper so the condition is testable —
this project has no `*.test.tsx`, so UI decisions are covered by pure functions plus
typecheck.

## 5. Three gaps in the test double, found by DRIVING it

The same lesson as R6-MCP-INSERT, one level down. All three were invisible to reading:

1. **`goal/patch` was advertised and never handled**, so the real client's `goal_patch`
   got `-32601 Method not found`. A method the fake cannot answer is a method **no test can
   cover** — and the control plane is exactly what the standing order leaves untestable
   any other way.
2. **`delegation/stop`** — the same, and it is the only way to stop a delegated child, so
   the one destructive operation on that surface had no test.
3. **`delegation/get` answered `{"delegations": []}`.** The real server answers
   `{"snapshot": {schemaVersion, rootSessionId, members}}`, which is what the UI's
   `AcpDelegation` is built from — so the fake agreed with a shape nothing produces, and a
   client reading `delegations` would have passed here and shown nothing in production.

And a fourth, which is the important one:

4. **The fake kept its state in memory**, so a resumed session looked brand new. A control
   route opens its **own** process per operation, so this would have let a broken route
   pass: `goal_create` succeeds, `goal_get` reports no goal. `--state-file` now models the
   real store, and `test_a_goal_survives_the_process_that_set_it` is the test that catches
   it.

The first attempt at that fix sited the save at the top of the request loop — which
**records the state as of before the request**, so creating a goal and exiting wrote
`goal: null`. A save that runs and records the wrong thing is worse than no save, because
it looks like evidence. It moved to `atexit`, where the fake already reports its refused
config values.

## 6. The UI

`ControlPanel.tsx`, under the transport selector — because that selector is what decides
whether the control plane is available at all. Operations: set a goal, pause/resume it,
clear it, queue a message, steer the running turn, activate the session. Each has a hint;
every operation is a **mutation**, so the result is always shown, because a button that
appears to do nothing is indistinguishable from one that failed.

When it is unavailable it renders **the reason** rather than disabled buttons. `exec` is
the default transport, so the unavailable state is the one most users are in, and a panel
that vanished would leave them with no idea the control plane exists.

The per-row actions (`queue_delete`, `queue_steer`, `delegation_stop`) are **not** in the
form: they act on an id the live panel has and a form does not, so they belong on the rows
that carry those ids. That is a declared gap, not an oversight.

## 7. Gates

- Python: **2875 tests, 0 failures, 0 errors, 3 skipped** (`.scratch/full37.xml`).
- Frontend: **426 tests across 29 files**; tsc clean.
- ruff clean across `src`, `tests`, `tools`.
- Bundle rebuilt to `index-B8halxLY.js` + `index-BW4GRTgZ.css`, referenced from
  `index.html`, superseded assets removed.

## 8. NOT VERIFIED — the boundary, unchanged

- **No model was loaded**, per the standing order. No DSH turn, no mcode turn, no chat.
- **No live `mcode acp` session has been driven.** Everything here is exercised against
  `tests/fake_acp_server.py`, a real subprocess over real pipes — not against mcode.
- **The panel has never been seen in a browser.** There is no `*.test.tsx` in the project,
  so the render layer is covered by typecheck plus pure-function tests.
- **Whether mcode replays a pending permission request on a resumed session** is unknown,
  which is why the recovery message offers the switch without promising the replay.
- **Whether a real mcode session's state survives a resume the way the fake's now does** is
  inferred from `session/resume` existing and answering the same id. The fake models it;
  mcode has not been observed doing it.
- The **per-row control actions have no UI**, as above.

## 9. Next

1. **Per-row actions** — delete/steer a queued message, stop a delegate — on the rows that
   already carry the ids.
2. **The turn's own connection** for control operations, removing the extra process.
3. **Drive the ACP wire against a live mcode**, which needs the standing order lifted. That
   is the one step that would turn several NOT VERIFIED lines above into verified ones.
