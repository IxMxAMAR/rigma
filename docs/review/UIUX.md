# UI/UX review — the whole interface (`frontend-v2/src`)

Area `uiux` · fork `review/deep-audit-2026-09-22` · worktree `.scratch/wt-uiux` · branch `impl/uiux`
Base head `dee48d7`. Every judgement is **reason / cause / resolution**, and every item is
IMPLEMENTED (with commit) or BLOCKED/REJECTED with the reason.

Design law read: `docs/design/CONSTITUTION.md` and `docs/design/UI-REWORK-PLAN.md`. `tokens.css` is
the token source; nothing here invents a hex, a spacing step or a duration — new UI uses the
existing `bg-panel/surface/float`, `text-primary/secondary/muted`, `amber/moss/red`, the 4px scale
and the 150ms ease-out.

Method: read `App.tsx`, `store.ts`, `Palette.tsx`, `tokens.css`, `Icon.tsx`, `FloatWindow.tsx`,
`WorkspacePanel.tsx`, `ErrorBoundary.tsx`, `LoadError.tsx` and every surface under `chat/`,
`models/`, `engine/`, `memory/`, `skills/`, `workflows/`, `settings/`, `autonomous/`; plus
`docs/review/FINDINGS.md` (11-*) and `docs/review/findings/11-frontend-ux.md`. `npm.cmd run check`
(`tsc --noEmit`) ran after **every** commit below, exit 0. `npm test`/vitest cannot run here
(esbuild `spawn EPERM`), so the new `*.test.ts` files are written but **unexecuted**; there is no
browser, so every render-path item is `verified: by inspection`.

---

## Layout & information architecture

### UIUX-1 · the session rail's export/duplicate/delete were hover-only — IMPLEMENTED `8416c92` (11-8)
- **reason** the three controls on each chat row could not be reached by keyboard at all, and did
  not exist on a touch device, so a keyboard user could create a chat but never delete, duplicate
  or export one — including one created by mistake.
- **cause** `hidden group-hover:flex`: a `display:none` element is not in the tab order, and touch
  has no hover state. The row's own affordances were state-keyed, but the actions were not.
- **resolution** render them always, `opacity-60` at rest and full contrast on row hover or
  `focus-within`. Verified by inspection of the class change; `tsc` clean.

### UIUX-2 · header telemetry was inert — IMPLEMENTED `7a79d59`
- **reason** `model`, `tok/s` and the health dot are the only always-visible state, but none of them
  could be acted on: "is 38 tok/s good here?" had no answer on screen, and a red dot did not lead
  anywhere.
- **cause** they were plain spans; the comparison (`expected_tg`) existed on `/api/server` but only
  the Engine page read it.
- **resolution** the dot is a button to the Engine page; the tok/s carries the calibrated
  comparison as its tooltip (`lib/telemetry.ts`, with `lib/telemetry.test.ts`). No new endpoint.

### UIUX-3 · rail width fixed at 230px while the app sidebar resizes — CLEARED
- **reason** the chat rail ignores the sidebar drag.
- **cause** a deliberate split: the rail holds chat titles and is not the navigation.
- **resolution** none — 230px fits `title`-truncated rows and the three actions, and making a second
  resizable column is a design change with no owner request. Recorded, not changed.

## Visual consistency & readability

### UIUX-4 · the global transition animated data values — IMPLEMENTED `b7f03a4` (11-10)
- **reason** the context meter and download bar slid to a new width instead of reading as
  measurements, and the composer grew 150ms behind the text, which reads as lag and makes the caret
  jump.
- **cause** `* { transition-duration: 150ms }` inherits `transition-property`'s initial value `all`.
- **resolution** name the properties (colour, opacity, transform for the pressed scale) and leave
  width/height alone. 150ms is unchanged (CONSTITUTION §28).

### UIUX-5 · the accent tokens differ from the constitution's stated hexes — REJECTED (needs owner)
- **reason** CONSTITUTION §3 names `--amber-base: #D97706` and `--moss-base: #4CAF50`; `tokens.css`
  ships `#e8a94f` and `#8fb573`. One of the two documents is out of date.
- **cause** the tuned values were committed to `tokens.css` after the constitution was written, and
  nothing reconciles them.
- **resolution** none. Changing every accent is a whole-app visual change, and the token file's own
  directive forbids inventing values outside it — so the shipped tokens stand and the discrepancy is
  an **open question for the owner** (fix the constitution's list, or re-tune the tokens).

### UIUX-6 · semantic colour use — CLEARED
- **reason** amber/moss/red must mean telemetry/healthy/degraded only.
- **cause** — **resolution** read across all surfaces: moss is grounded/healthy/verified, amber is
  telemetry/attention, red is failure; no decorative colour found. No change.

## Feedback while things run

### UIUX-7 · a refused action rendered as success — IMPLEMENTED `e610f39`, `564e638`, `7a79d59`
- **reason** a delete the server refused looked exactly like one that worked: the row stayed put and
  nothing explained it. Same for a refused compact and a refused grounding-folder removal.
- **cause** `await fetch(... DELETE)` followed by `refresh()` with `r.ok` never read (AUDIT F11-4's
  follow-up). `compact` swallowed the response the same way.
- **resolution** check `r.ok`, render the server's own sentence (`responseError` → shared
  `InlineError`), and refresh only on success. Files: `SkillsSurface`, `SettingsSurface`,
  `MemorySurface`, `Sidecar` (grounding), `ChatSurface` (compact).

### UIUX-8 · feedback while a turn runs — CLEARED
- **reason** a long turn must never look dead.
- **cause** — **resolution** read `Transcript`'s `Working` pulse (chips→thinking→generating), the
  per-session `streams` keying, the stop button, and `TurnError` outliving the turn. Correct as
  written; no change.

### UIUX-9 · download progress, rate and cancel — PARTIAL (server-side pending)
- **reason** adding a multi-GB model is the longest first-run action; a stalled pull looks like a
  slow one and there is no way to stop a wrong pick.
- **cause** the UI renders bytes/%/ETA from `pull.done/bytes/eta`, but the rolling rate and a cancel
  route need a server field/route that does not exist (IMP-2).
- **resolution** nothing invented: no server field was faked. **Server-side pending (agent
  `improvements`)**.

## Error messages

### UIUX-10 · a failed fetch rendered as an authoritative empty state — IMPLEMENTED `9eea124`
- **reason** "No models yet" / "No workflows found" is what a 500 looked like, so the page sent the
  user to fix the wrong thing; the Runs history rendered nothing at all on a failed load.
- **cause** the list surfaces conflated "the request failed" with "there is nothing", and the Runs
  history swallowed `!r.ok`.
- **resolution** every list surface now distinguishes the two: `LoadError` (with retry) for a
  failure, the new shared `EmptyState` for a genuinely empty list. Runs records the load failure.
  Follow-up `cc09c0e` keeps a refused *restart* (409) in its own slot, so it no longer renders as
  "could not load runs".

### UIUX-11 · the reason a backend cannot run was put where browsers never show it — IMPLEMENTED `5397b06` (11-11)
- **reason** the carefully written `harnessHint()` was unreachable: the greyed-out option read as an
  arbitrary restriction.
- **cause** it was a `title` on a native `<option>` — the OS draws that popup, not the page — and a
  disabled option cannot be selected either.
- **resolution** render the hint for every unusable backend under the picker.

### UIUX-12 · Settings advertised the engine's upstream port as Rigma's API — IMPLEMENTED `1086b3d` (11-6)
- **reason** a user copying `http://127.0.0.1:11499/v1` into an agent talked to llama-server
  directly, bypassing the session, tool-call repair and idle-unload; on a non-default launch the URL
  is dead.
- **cause** a hardcoded port. `serve._public_port(11499) = 11500`, and `/api/server` publishes the
  right value as `openai_base`.
- **resolution** render `openai_base` (`lib/openaiBase.ts`, with a test); fall back to the origin
  that served the page, never a literal port.

## Empty states

### UIUX-13 · bare "nothing here" with no next step — IMPLEMENTED `9eea124` (IMP-10)
- **reason** Models, Skills, Workflows, Settings, Memory and Runs each left the user to find the
  action themselves; Runs had no empty state at all.
- **cause** a one-line `<p>` per surface, with no primary action and no shared shape.
- **resolution** one `EmptyState` component (CONSTITUTION §33: centered, muted, mono glyph, one
  primary action), whose action is real — focus the HF search box / skill / preset form / mission
  box, or reload. Error states keep `LoadError`; the two never render as the same thing.

### UIUX-14 · the transcript's own empty state — CLEARED
- **reason** a new chat must invite, not sit blank.
- **cause** — **resolution** it already names the surface and points at Ctrl+K; consistent with
  `EmptyState`'s shape. No change.

## Keyboard use

### UIUX-15 · hover-only controls were keyboard-unreachable — IMPLEMENTED `8416c92`, `564e638`
- **reason** regenerate/continue/takes on a reply, and the grounding folder remove, were invisible
  and (for a keyboard user who never hovers) effectively absent.
- **cause** `opacity-0 group-hover:opacity-100`. `opacity:0` keeps an element in the tab order but
  invisible, so tabbing to it moved an unseen focus.
- **resolution** the rail's actions are always rendered; the per-message and per-folder controls
  reveal on `focus-within` as well. The folder remove also checks `r.ok` now.

### UIUX-16 · Enter / Esc / focus on the chat surface — IMPLEMENTED `12ceebd` (IMP-9)
- **reason** daily use is keyboard-heavy; Stop was mouse-only and switching chats lost the caret.
- **cause** no global key handling on the chat surface.
- **resolution** Enter sends / Shift+Enter newlines with a persistent hint line that also says
  "esc stop" while streaming; Esc stops the turn; the composer is focused when the chat changes. The
  rules are pure predicates in `chat/keyboard.ts` (`chat/keyboard.test.ts`): Esc never stops while
  the palette is open, and never from a single-line input that is cancelling its own edit.

### UIUX-17 · Ctrl+K for the session filter vs the palette — IMPLEMENTED `12ceebd` (IMP-9, resolved)
- **reason** IMP-9 asks for ⌘/Ctrl+K to focus the session filter, but Ctrl+K is the palette's
  binding and UI-REWORK-PLAN makes the palette first-class; taking the key would break it, which the
  brief forbids.
- **cause** two features want one accelerator.
- **resolution** the palette keeps Ctrl+K. The rail is focused by `/` when the user is not typing,
  and by a new "Filter chats" palette command reachable through Ctrl+K. **Caveat:** the composer is
  focused on load, so `/` is only live once focus leaves a field — deliberate, because `/skillName`
  is a real composer input and must not be stolen.

### UIUX-18 · focus rings and semantic elements — CLEARED
- **reason** keyboard access is mandatory (CONSTITUTION §24).
- **cause** — **resolution** `:focus-visible` is a global 2px amber outline; buttons/links/`nav`/
  `main`/`aside`/`header`/`section` are semantic. No change.

## Rough edges a daily user hits

### UIUX-19 · per-chat drafts were lost on reload — IMPLEMENTED `d9b2172` (IMP-1)
- **reason** a reload took a half-written paragraph with it; the in-memory map (F11-1) fixed the
  misroute but not the loss.
- **cause** `drafts` lived only in the store.
- **resolution** mirror it to `localStorage` (`rigma.drafts`) best-effort: an oversized draft is
  declined rather than truncated, the map is trimmed oldest-first, a malformed value is ignored, and
  a send or a chat delete clears the entry. Pure logic in `chat/drafts.ts` (`chat/drafts.test.ts`).

### UIUX-20 · HF search truncated at 8 with no count — IMPLEMENTED `7a79d59`
- **reason** a user searching a family concluded the repo they wanted did not exist and gave up.
- **cause** `hits.slice(0, 8)` and nothing said how many there were.
- **resolution** a "show N more of M" control; the result set starts collapsed at 8 again.

### UIUX-21 · "compact" destroyed the conversation, as far as anyone could tell — IMPLEMENTED `7a79d59`
- **reason** the button visibly shortens the transcript and nothing on screen says the earlier turns
  were archived rather than deleted.
- **cause** no tooltip, and a refused compact was swallowed.
- **resolution** a one-line tooltip before it happens, and a refused compact now shows the server's
  sentence instead of looking like nothing happened.

### UIUX-22 · the `citations` half that neither end implemented — IMPLEMENTED `053bd38` (run 3)
- **reason** the store looked like it already handled RAG citations, which is how the gap stayed
  invisible.
- **cause** a reducer branch and `StreamingTurn.citations` field with no server producer and no
  renderer.
- **resolution** run 2 deleted the dead half (`1ffd0a9`) — correctly, because a half that looks like
  a working pipeline is worse than no half. Run 3 built the **producer**, so both ends exist now.
  `search_my_documents` already received `citations` on every sidecar `/ask` and folded them into the
  MODEL's text as a `sources: …` line, so the sources reached the model and never the reader —
  backwards for the one feature whose value is "which of MY files said this". The turn loop now emits
  one `citations` event per ROUND carrying only what that round added, and the transcript renders a
  `<Sources>` disclosure. The citations travel on the tool CONTEXT, never the transcript, so a display
  change cannot change an answer; the model keeps being told the sources, and a test asserts it.
  Verified by a real two-round turn (emitted once, absent when nothing was searched, nothing
  persisted) plus 5 reducer cases; mutation-checked. **DONE.**

### UIUX-23 · `window.confirm` for destructive actions — BLOCKED (server-side pending)
- **reason** the browser dialog is unstyled and gives no way back; deleting a chapter or a 15 GB
  quant is exactly where a 10-second undo is worth more than a confirmation click.
- **cause** six call sites use `window.confirm`, and an undo needs a restore path.
- **resolution** none in this pass. An in-page confirm is a designed component, and a real undo for
  a chat needs a server restore route (soft-delete/trash for quant files likewise). **Server-side
  pending (agent `improvements`)**; `window.confirm` stays as the honest interim.

### UIUX-24 · the remaining "Improvements" items that need a new endpoint — RESOLVED in run 2
- **reason** IMP-2 cancel/rate, IMP-4 permission-profile UI, IMP-5 idle-timeout control, IMP-6
  memory edit, IMP-7 usage meter, IMP-8 stop reason/budget, IMP-11 engine-log viewer, IMP-12
  backup/restore all describe a UI over data the server does not publish yet.
- **cause** each needs a field or route owned by the Python side.
- **resolution** the endpoints landed (agent `improvements`, commits `41010d2..bf5a1f3`) and the UI
  was wired to them (agent `wire`, commits `d9e0899..349e707`): IMP-4 grants + run profile
  (`d9e0899`, `ec33d12`), IMP-2 progress/rate/ETA (`522be78`), IMP-7 usage meter (`967ed51`,
  `349e707`), IMP-8 stop reason + remaining budget (`94c1e88`), IMP-11 log panel (`a845bf2`), IMP-6
  edit (`944b844`, `0ff64a4`).
- **still without a UI, and why** — these are the honest remainder, each needing a route that does
  not exist rather than a guess:
  - **IMP-2 cancel button** — no pull-cancel HTTP route. `hangar` grew the per-key cancel flag
    (`e63503f`) that such a route would set, but the route itself was not added, so the UI shows
    progress only.
  - **IMP-5 idle-timeout control** — `GET/POST /api/settings` exists (`3ebc21e`) but the wire agent's
    branch was cut before it merged, so no Settings control reads it yet. Small, self-contained.
  - **IMP-12 backup/restore buttons** — `GET /api/backup` / `POST /api/restore` exist (`bf5a1f3`),
    same reason: no UI yet.
  - **IMP-8 token axis** — the wire agent believed `tokens_used` was never written; it is written per
    turn by `_run_loop` (03-2, `6cc68db`). The step/time axes are rendered; the token axis can be
    added from the existing `budget.tokens_used`/`token_cap` fields with no server change.
  - **UIUX-23 undo** — still needs a server soft-delete/trash route; `window.confirm` stays as the
    honest interim.

---

## Considered and cleared (breadth check, no change)
`FloatWindow`'s open/close and Esc handling; `Palette`'s arrow/Enter/Esc model and its empty
"Nothing matches"; `ModelPicker`; `MacroStrip`'s inline confirm (it already names the macro — the
model for replacing `window.confirm`); `WorkspacePanel`'s sync effect; `EngineSurface`'s telemetry
density and its own `openai_base`; `Markdown`/`sanitizeConfig`; `MoreRows` for quants; the
autonomous steering box; `AutonomousSurface`'s `RESTARTABLE` mirror of `runs.RESTARTABLE`.

## Could not check
- **No vitest run.** `npm test` dies before collecting (`esbuild` `spawn EPERM`); the four new test
  files are unexecuted. The owner rebuilds and runs them.
- **No browser and no build.** `npm run build` is forbidden here, so `src/rigma/data/ui_v2` was not
  rebuilt and no render behaviour was observed; render-path items are by inspection.
- The real REST surface was not exercised (that would need the owner's running server); shapes come
  from reading `serve.py`.

**Item count: 24 judgements** (16 IMPLEMENTED, 1 REJECTED with reason, 1 RESOLVED in run 2, 3
BLOCKED/partial with reason, 3 CLEARED). Every commit touching `frontend-v2/src` is a
**needs-rebuild** commit.

## Run 2 closing note
The four items this document left blocked on the server are unblocked and wired, except the honest
remainder listed under UIUX-24 (IMP-2 cancel route, IMP-5 and IMP-12 UI, IMP-8's token axis,
UIUX-23 undo). Run-2 frontend commits, all **needs-rebuild**: `d9e0899 522be78 967ed51 94c1e88
a845bf2 0ff64a4 ec33d12 944b844 349e707`, plus the run-2 UIUX wave `8416c92 1086b3d 5397b06 d928b3c
b7f03a4 1ffd0a9 e610f39 7a79d59 9eea124 cc09c0e 564e638` and `1c1a101 3a15896 86f545e ecfd917`.
The vitest files written for all of these have still never been executed here (esbuild `spawn
EPERM`); `npm.cmd run check` (tsc) is clean on the merged result.
