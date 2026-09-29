# Round 7 corrections, second pass — the UI audit

The audit that produced this file was run against `05fdfc9`/`d71f3d5` by a subagent told to
be adversarial and to find what was WRONG. It found fifteen things. Three mattered.

The pattern worth naming: **I shipped two features whose UI could not work, and my own tests
did not catch it because they tested the layer below.** The `goal_patch` operation was
correct on the server and reachable from the panel; the button was just wired to send
nothing. The settings selects rendered and posted correctly; nothing could ever put a value
in them. Both passed `tsc`, `ruff`, 434 frontend tests and 2889 Python tests.

## F1 — HIGH — an unstable selector default would loop the render

`useChat((s) => selectStreaming(s)?.acpConfig ?? [])` returns a **fresh `[]` on every call**
whenever no turn is streaming. zustand 5.0.14 passes that selector straight to
`React.useSyncExternalStore`, which re-invokes `getSnapshot` after commit and forces a
re-render whenever the result is not `Object.is`-equal to the rendered value. That is the
documented *"The result of getSnapshot should be cached to avoid an infinite loop"* case.
`ControlPanel` is mounted for **every open chat**, so it would have fired for every chat
with no live turn — most of the time, and always at the moment the panel is first opened.

Fixed in `1db3dbc` by defaulting to a module constant (`?? NO_CONFIG`) **outside** the
selector. The audit also caught that my fix's comment said "ONE frozen array" while the
value is not `Object.freeze`d; the comment was corrected rather than the code, because
nothing mutates it and a freeze would only convert a future mistake into a throw.

**The audit read the shipped bundle to confirm this** (`??[]` present at `05fdfc9`, `??ib`
after the fix). That is the right way to check a claim about what shipped, and it is why I
am recording the finding rather than my intent.

## F2 — HIGH — "pause or resume the goal" did nothing and reported success

`ControlOp.arg` was `"text" | "objective"`, and `controlParams` returned `{}` for any
operation without an `arg`. So `goal_patch` — declared with no `arg` — sent **no params at
all**. The server accepted the empty patch (its table requires none), merged nothing, and
returned the goal **unchanged**; `controlResultText` then printed `goal is now <the old
status>`. The button was a no-op that reported the unchanged value as its outcome, which is
precisely the failure `controlPlane.ts`'s own header says the design exists to prevent.

The server-side test showed the call the UI could not make: `op("goal_patch", exe, sid,
status="paused")`.

Fixed by giving an operation a **closed set** as well as free text: `ControlOp.choices`
(`{param, values, labels}`), a `<select>` in the panel, `controlParams(spec, arg, choice)`
returning `{status: <choice or first value>}`, and a single `controlReady(spec, arg,
choice)` that the button and the send path both consult so they cannot disagree. The statuses
offered are `active`/`paused`, labelled "resume (active)" / "pause (paused)".

## F3 — HIGH — a successful settings change reverted, because nothing could update it

The selects read the live turn's `acpConfig`, which is written **only** by the `acp_config`
SSE event, which is produced **only** by a `config_option_update` notification — and a
control operation runs on its **own short-lived mcode process** whose notifications nobody
reads (`AcpClient(..., on_event=None)` → `_noop`). So the change succeeded, the panel said
"done", and the select snapped back to the old value and stayed stale.

Fixed on the client: `setAcpConfigOption` in the chat store, called **after** the server
confirms, so it records an accepted fact rather than an optimistic guess. It builds a new
array and new option objects, because the value is handed to React through a zustand
selector and mutating in place would leave the reference identical and the select would not
re-render.

**The honest limit:** this records what the user chose, not what the server's list now
contains. If a server clamps or rewrites a value, the panel shows the requested one until
the next turn's handshake corrects it. Closing that properly means forwarding notifications
from the control connection, which is a larger change than this round should carry.

## F4 — MEDIUM — the settings block was gated on a list nothing forwarded

The block renders when `config.length > 0`, and `config` is empty until a
`config_option_update` arrives. But the **handshake already answers with the option list**
(`session_new`/`session_resume` store `res["configOptions"]`) — nothing forwarded it as an
event. So on any server that does not push an update, the feature added in `05fdfc9`
rendered nothing at all.

Fixed in `drive_turn_acp`: after the resume/new, the client's `config_options` are emitted
as an `acp_config` event **unconditionally, including when empty** — skipping an empty list
would leave the previous turn's options on screen, which is stale settings presented as
current.

**And the reason no test caught it:** the fake never emits `config_option_update`. The
feature depended on a message that no test produced. Two tests now cover it, and the second
one pins the empty case specifically.

While writing those tests I asserted `e.data["event"]` and both failed: `TurnEvent.event` is
a **top-level** field ("the backend's own event name") with the payload in `data` beside it.
My test was wrong, not the code. Left in the record because guessing at a shape and being
corrected by the run is the only thing that made it certain.

## F6 — MEDIUM — `hasSession` never refreshed, so the panel stayed blocked

`mcodeSession` was set only inside the effect keyed `[currentId]`. But the backend session
does not exist until the first turn has run, **and a turn does not change `currentId`** — so
the panel kept rendering "no mcode session yet; it appears after the first turn" for the
life of the chat. The transition its own copy described could never be observed, and the
session-settings controls sit behind that gate.

Fixed with a streaming → idle edge effect that re-reads `harness_sessions`, as a **separate
minimal fetch** rather than widening the existing effect: that one also reloads the sampling
fields and the system prompt, and re-running it on every turn end would wipe what the user is
typing into them.

## The rest

| # | Sev | What | Disposition |
|---|-----|------|-------------|
| F5 | MED | An empty `currentValue` rendered `<option value="">` with no label — a selected option displaying nothing, so "unset" was indistinguishable from "unknown" | Fixed: an unset option now says `(not set)` |
| F7 | LOW | A falsy `value` counts as MISSING server-side, so an option with an empty value would be refused | Noted. Surfaced honestly through `j()`'s throw; reachability needs such an option |
| F8 | LOW | `String(c.name \|\| c.id)` rendered the literal text "undefined" and collapsed several options onto the duplicate React key `"undefined"` | Fixed: a positional fallback key and an honest label |
| F9 | LOW | The options list checked `length > 0` then filtered empty values out of what it printed → a literal `" ()"` | Fixed by extracting `configOptionValues()` into the pure module, which makes the empty case testable |
| F10 | LOW | `rowActionReady` checked only the id, so a failed or completed queue item still drew "steer" | Fixed: `RowAction.statuses`, and an unreadable status counts as not-ready |
| F11 | LOW | Three comments still described the per-child stop that `05fdfc9` removed | Fixed — same defect class as everything else this round |
| F12 | LOW | The cross-language pin read only `controlPlane.ts`, so `config_set` — the one op `05fdfc9` added — was outside it | Fixed: it now reads `ControlPanel.tsx` too, **and** checks the reverse direction (every op the UI can send must be in the server's allowlist) |
| F13 | LOW | `config_set` and `delegation_stop` both reported only "done" | Fixed: "setting changed", and a session-wide sentence that names how many tasks stopped |
| F14 | BENIGN | The fix's comment said "frozen"; it was not | Comment corrected |
| F15 | BENIGN | Reported as an empty panel box when a plan has neither `content` nor `uri` | **DID NOT REPRODUCE.** `hasPlan` at `AgentState.tsx:426-427` already requires a string `content` or `uri`, which is exactly the guard the finding asked for |

## Set difference, both directions (READ from both tables)

- Server `_CONTROL_OPS`: **15**. UI-reachable: **10** — `CONTROL_OPS` (7) + `config_set` +
  `queue_steer`/`queue_delete`.
- **Server allows, UI cannot reach (5):** `goal_get`, `queue_list`, `delegation_get`,
  `mode_set`, `queue_update`. The three reads are shown through notifications instead;
  `queue_update` is declared in `ROW_OPS` with no `RowAction`; `mode_set`'s absence is
  documented server-side and blocked on storing `availableModes`.
- **UI offers, server would refuse: the EMPTY SET.** Every operation the panel can send is
  allowlisted with its required params satisfied — and F12's new reverse-direction check now
  enforces that rather than leaving it to inspection.

## What this round does NOT establish

- **No browser has rendered any of this.** There is still no `*.test.tsx` in the project, so
  the render layer is covered by typecheck plus pure-function tests. F1 is the proof that
  typecheck plus pure tests does not cover render behaviour: it passed both.
- `members[].errorMessage` is rendered and **nothing in the repo produces it** — the fake
  never populates `members` at all, no test mentions it, and the type was declared with no
  cited source. The whole delegation-members block is untested.
- Whether mcode's `session/set_config_option` response carries the updated `configOptions`
  is unknown (the fake returns `{}`). If it does, F3's local record can be replaced by the
  server's own answer.
- No live mcode session was driven and no model was loaded. The `hardware` marker added in
  `d71f3d5` is therefore still unexercised.
