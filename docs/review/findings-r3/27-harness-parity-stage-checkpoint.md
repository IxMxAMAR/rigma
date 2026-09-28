# 27 — Harness capability parity: stage checkpoint

*Written for resumption in another session. State as of commit `a95069b` on
`review/deep-audit-2026-09-22`, fork `C:\ComfyUI\RD\rigma-review`. The original
repo `C:\ComfyUI\RD\rigma` is read-only and was clean at `0bdfad1`.*

**Goal.** Achieve harness capability parity: inventory every capability DSH and
mcode expose, determine which are unreachable from Rigma and *why*, implement
them with UI, tests and commits.

---

## 1. What is already true (measured this round, not recalled)

### 1.1 DSH's session event vocabulary is 51 types; Rigma bridges 9

`SessionEventMap` in `packages/core/session/src/types.ts:269-407` declares **13**
core event types. Plugin packages add the rest by module augmentation; the
authoritative *complete* list is the disposition table in
`packages/session/session-format-v0-to-v1/src/dispositions.ts`, which enumerates
**51**.

Rigma's runner (`src/rigma/_dsh_runner.py`) recognises exactly:

| what | event types |
|---|---|
| state (passed through whole) | `goal/change`, `todo/write`, `plan/mode`, `subagent/descriptor`, `subagent/catalog` |
| usage | `assistant/message` |
| tool lifecycle | `tool/call`, `tool/result` |

That is **9 of 51**. The other 42 never reach the UI. The ones that look
user-facing and are worth triaging first:

| event | what it would tell a user |
|---|---|
| `approval/asked`, `approval/decided`, `approval/policy` | **the agent is asking permission** — the security-relevant one |
| `sandbox/mode`, `permission/preset` | which confinement is actually in force |
| `session/title` | the server's own title for the chat |
| `model/selection` | a mid-session model change |
| `compaction/start\|end\|summary\|prune` | compaction progress and what it dropped |
| `command/run`, `command/done` | a slash command ran |
| `hook/invoked`, `hook/result` | hooks fired |
| `team/*` (5 types) | multi-agent collaboration |
| `tool-workflow/run-*`, `agent-*` | workflow progress |
| `agent-preset/selected`, `subagent/model-selection-policy`, `schedule/change`, `feedback/record`, `llm/retry*`, `agent/inbox/spliced`, `request/*`, `session/end-seed`, `step/*`, `turn/*`, `user/message`, `system/message`, `assistant/attempt`, `assistant/chunk`, `tool/code-dispatch*`, `session-log-deepseek/delivery-accepted`, `session/title-llm-request`, `web/deepseek-search-llm-request` | internal bookkeeping, or already covered |

### 1.2 The DSH wire cannot enumerate anything, and cannot answer an approval

`HarnessSdkRequestMap` in `packages/sdk/protocol/src/types.ts:115-119` has
exactly **three** methods:

```
initialize        session/prompt        shutdown
```

Consequences, and these are structural rather than oversights:

- **No enumeration RPC.** There is no way to list sessions, goals, todos,
  subagents or skills. They can only be observed as they stream past inside
  `session.event`. Any "list my subagents" UI must be built from the stream.
- **No approval response.** `approval/asked` and `approval/decided` exist as
  *notifications* only. Approval is decided by the policy engine, not by the
  client, so the most Rigma can honestly do is **display** the policy and the
  audit trail — not prompt the user. Presenting a clickable "Allow?" button would
  be a lie about the transport.
- **`sdk-minimal` also has no browser tooling**; the four top-level notifications
  are exactly `session.event`, `session.status`, `subagent.started`,
  `subagent.finished` (`types.ts:106-112`), verified verbatim.

### 1.3 DSH's slash commands are unreachable by transport

DSH ships three command packages:

| package | command |
|---|---|
| `@deepseek-ai/dsh-command-goal` | `/goal` |
| `@deepseek-ai/dsh-command-compact` | `/compact` |
| `@deepseek-ai/dsh-command-feedback` | `/feedback` |

`parseCommand` (`packages/interaction/commands/src/index.ts:125`) recognises
`/^\/[a-z][a-z0-9_-]*/`, and the only caller is `CommandRuntime.execute`
(`:367`), which is decorated **`@Remote`** — a web-layer remote service. It is
not on the stdio wire, so Rigma cannot invoke these commands however it words a
prompt. **Rigma's capability patch mounts no command row at all.**

The honest route to parity here is for *Rigma* to implement the command surface
in its own UI and back each command with the capability that is mounted —
`/goal` is already reachable, because `goal` + `tool-goal` are mounted. That is a
Rigma feature, not a harness bridge.

### 1.4 mcode: only `exec` is used; the entire ACP surface is unreached

`src/rigma/harness.py:330-332` records the decision deliberately: *"use exec's
versioned NDJSON stream, and treat ACP as a later upgrade — ACP would additionally
mean implementing the ACP client side."* No `mcode/...` method appears anywhere in
`src/rigma/`.

So every ACP-only surface is unreachable today, including the goal RPCs
(`mcode/session/goal/{get,create,patch,clear}`), delegation, plan, usage,
permissions and `session/fork`. Rigma reaches mcode's goal/todo state only by
scraping `item.toolCall.output.details` out of the `exec` stream (added in
`53ba130`).

### 1.5 Rigma already has 8 UI surfaces and permission modes

`frontend-v2/src/store.ts:18-27`: `chat`, `autonomous`, `models`, `engine`,
`memory`, `skills`, `workflows`, `settings`. Permission modes exist and are
persisted per session (`chatStore.test.ts:361-382` exercises `smart` / `off`
round-tripping to the server). So "permissions have no UI" would be **false** —
do not re-report it.

### 1.6 CORRECTION: the `macro_*` events are NOT dead

I first wrote here that `macro_step` and `macro_done` were dead reducer cases
because nothing in `serve.py` emits them. **That was wrong**, and the way I was
wrong is worth recording: I grepped for `_sse(...)` in `serve.py` and concluded
from its absence. There is a **second SSE emitter** — `methods_api.py:229` defines
its own `emit` using `sse(data, event)`, and the macro path uses it:

- `macros.py:264` — `await emit("macro_step", {...})`
- `methods_api.py:239-241` — `await q.put(sse({...}, "macro_done"))`

Macros are a live feature (`macros.py`, the Workflows surface, and
`chatStore.test.ts`'s macro tests all agree), so the reducer cases are correct and
there is nothing to clean up. **One grep for one helper name is not a search for a
capability.** Gap 7 of §3 is withdrawn.

---

## 2. Method — and a correction

Two DeepSeek-V4.1-Flash subagents were launched to produce "exhaustive
inventories" of DSH and mcode. **They were the wrong call and were stopped.** The
standing order is to delegate *bulk* work, not to re-derive facts that are one
`grep` away, and everything decisive in §1 had already been read directly:

- the 51 event types are one regex over `dispositions.ts`;
- the three wire methods are one line of `HarnessSdkRequestMap`;
- the command packages are one directory listing.

Two earlier agents with the same brief (`e5dcec72`, `f0324665`) were also `ready`
with no artifact on disk, so there was no reusable base either. **Do not relaunch
them.** If a genuinely bulk question appears later — for example reading mcode's
minified ACP client to recover its full method list — delegate *that*, with the
specific question named, rather than re-running a generic inventory.

## 3. Ranked gaps to implement (provisional — confirm against §2)

Ranked by (user value × honesty of the fix), highest first.

1. ~~**Approval and confinement visibility (DSH).**~~ **DONE** — see §6. The six
   governance events are bridged, rendered, and tested. `session/title` came along
   with them.
2. ~~**Server-side chat title (DSH).**~~ **DONE** — `session/title` is bridged and
   the store adopts it over its own guess.
3. ~~**Rigma-side slash commands.**~~ **DONE** — see §7. Six commands, every one
   backed by something Rigma can already do. `/goal` is deliberately NOT among
   them yet: setting a persistent goal needs a route Rigma does not have (see
   §7, "still open").
4. ~~**Compaction visibility (DSH).**~~ **DONE** — see §8. Four events bridged,
   folded as a bracket, and rendered beside Rigma's own compaction lines.
5. **mcode ACP client — MEASURED, and BLOCKED on authentication.** See §9. The
   surface is real and much larger than stream-json (14 methods plus a goal and a
   delegation control plane), but every session method refuses with
   `-32000 Authentication required`, and this machine has no credentials. Not
   buildable against an API that cannot be exercised.
6. ~~**`session/fork` for DSH.**~~ **WITHDRAWN — it does not exist.** See §10.
   Verified: `session/fork` appears nowhere in the 51-entry event vocabulary, nowhere
   in `HarnessSdkRequestMap`, and nowhere in the whole DSH package. DSH forks by
   seeding an event log through a HOST-side API (`ctx.sessions.create(id, { seed })`),
   not over the wire. The `subagent_fork` tool forks a *subagent*, not a chat.
7. ~~**Clean up the dead `macro_*` reducer cases**~~ **WITHDRAWN** — they are not
   dead; see the correction in §1.6.

---

## 4. Non-negotiables carried from the standing orders

- Work only in `C:\ComfyUI\RD\rigma-review`. The original repo is read-only.
- Every subagent names `provider: tokenjuice`, `model:
  deepseek-ai/DeepSeek-V4.1-Flash`.
- Full backend suite: `PYTHONPATH=src;.scratch\00-recon\plug`, `-p no:cacheprovider
  -p dsh_tmpfix`, with `--basetemp` under `.scratch`. The `dsh_tmpfix` plugin is
  **required** for any `tmp_path` test (it drops `mode=0o700`, which the DSH
  sandbox denies enumerating).
- Frontend gates: `npm run check` (tsc), `npm test`, then `npm run build` — the
  build output is committed under `src/rigma/data/ui_v2/`.
- A declared gap beats a fake green. Do not present a display-only surface as
  interactive, and do not claim a capability is bridged because a test passes on
  a hand-written payload.

## 5. Last verified suite state

2633 backend tests collected, 2630 passed, 0 failed, 3 skipped. Frontend 21 files
/ 248 passed, `tsc --noEmit` clean, `ruff check src tests tools` clean, bundle
rebuilt byte-identically.

---

## 6. Done this round: the governance bridge

**Commits.** `a95069b` was the starting point. The governance bridge is the commit
that follows it.

### What was unreachable, and why

DSH records its own governance as session events, and every one of them is
**`log-only`** in DSH's own words: durable, replayable, and *never part of the
model transcript*. Rigma's runner recognised nine event types and none of these,
so a user could not see that an agent had asked for permission, what it was told,
or how confined it was.

| event | payload (quoted from DSH) | source |
|---|---|---|
| `approval/asked` | `{id, toolName, callId?, reason?}` | `interaction/user-approval/src/types.ts:44` |
| `approval/decided` | `{id, outcome}` — `allowed-once\|rejected\|cancelled\|unavailable` | `.../types.ts:55` |
| `approval/policy` | `{policy, source?}` | `interaction/user-approval/src/index.ts:33` |
| `sandbox/mode` | `{mode, source?}` — `read-only\|workspace-write\|danger-full-access` | `sandbox/sandbox-policy/src/session-mode.ts:33` |
| `permission/preset` | `{preset}` | `interaction/permission-presets/src/index.ts:57` |
| `session/title` | `{title, messageSeqs, source}` | `session-format-v0-to-v1/src/dispositions.ts:86` |

### The honest boundary, stated in the code

**This is display-only, and that is a property of the transport rather than a
shortcut.** `HarnessSdkRequestMap` (`packages/sdk/protocol/src/types.ts:115-119`)
has exactly `initialize`, `session/prompt`, `shutdown`. There is **no
approval-response method**, because approval is decided by the policy engine, not
by the client. So no "Allow?" button exists, and `chat/governance.ts` says why in
its header comment — a clickable control there would misrepresent what the
connection can do.

What it *is*: an audit trail. `foldApproval` pairs each `decided` onto its own
`asked` by `id`, because two rows per decision would double the list and separate
a question from its answer. An **orphaned** decision is kept rather than dropped —
a refusal whose question predates the turn is still a refusal. A second decision
for the same id is a NEW row, not a silent rewrite, because the first verdict is
what actually happened.

### Files

| file | change |
|---|---|
| `src/rigma/_dsh_runner.py` | six events added to `_STATE_EVENTS`, with the display-only boundary explained |
| `src/rigma/serve.py` | `approval` (one name, DSH name carried alongside), `sandbox`, `permission_preset`, `session_title` |
| `frontend-v2/src/chat/governance.ts` | NEW — `foldApproval`, labels and tones; the boundary documented |
| `frontend-v2/src/chat/AgentState.tsx` | `GovernanceBlock`, below a rule, because confinement is about the connection rather than the turn's subject |
| `frontend-v2/src/chat/chatStore.ts` | `governance` field + four reducer cases; `session_title` adopted |
| `tests/test_dsh_capabilities.py` | 7 tests, payload shapes quoted from DSH |
| `frontend-v2/src/chat/governance.test.ts` | 18 tests |

### A bug the tests caught

`foldApproval` stripped the `approval/` prefix with `.replace(/^approval\//, "")`,
which is a **no-op when the prefix is absent** — so `{event: "goal"}` became a
trail entry called "goal". Now the prefix is *required*, not merely stripped.

### An existing test that correctly conflicted

`test_internal_chatter_is_still_dropped` listed `session/title` among the
bookkeeping to drop. That is no longer true and the test was **narrowed with a
reason**, not deleted — the other four names in it are still genuinely noise.

### Still open

Gaps 3–7 of §3 are untouched: Rigma-side slash commands, compaction visibility,
the mcode ACP client, `session/fork` verification, and the dead `macro_*` reducer
cases (§1.6).

---

## 7. Done this round: Rigma's own slash-command surface

### Why Rigma must own this

DSH ships `/goal`, `/compact` and `/feedback`. **Rigma can reach none of them,
whatever it types.** `parseCommand` (`interaction/commands/src/index.ts:125`)
recognises the syntax, and its only caller is `CommandRuntime.execute` (`:367`),
decorated **`@Remote`** — a web-layer remote service. The SDK stdio wire exposes
exactly `initialize`, `session/prompt`, `shutdown`, so there is no method to
invoke a command with. Sending `/compact` as prose would merely ask the model to
discuss compacting.

So this is a **Rigma** feature and the code says so, in `chat/commands.ts`. Every
command is backed by something Rigma can actually do:

| command | backed by |
|---|---|
| `/compact` | `POST /api/sessions/{sid}/compact` — already existed, never surfaced in chat |
| `/permission <off\|smart\|full>` | `setPermission`, already persisted per session |
| `/new` | `newChat()` |
| `/stop` | `stop()` |
| `/skills` | `setSurface("skills")` |
| `/help` | built from the roster, so it cannot drift from the menu |

### The correctness-critical part

The parser is deliberately **stricter than DSH's own** regex, and the strictness
is the feature. DSH uses `/^\/([a-z][a-z0-9_-]*)(?=$|[\t\n\r ])/`; this adds a
**known-name** requirement on top. Swallowing a legitimate message is the worst
thing a command box can do — it is indistinguishable from the app being broken —
so `/usr/bin/python is slow`, `/compaction is slow` and `/goal ship it` are all
just text.

### One place I was wrong, caught by the test

I first wrote `commandQuery` to close the menu the moment the name was exactly
complete, and asserted that in a test. **The test failed and the test was
right.** `/compact` is precisely when the user is looking at the menu, one
keystroke from Enter; closing there would take it away at the worst moment. The
menu now closes on the **space** — the moment the user has committed and moved to
the argument. The docstring records why.

### Interaction details that are easy to get wrong

- **Keyboard-first.** Up/Down move the highlight, Tab or Enter complete, Escape
  closes. The arrows and Enter are claimed **only while the menu is open**, so
  they keep their normal meaning in every ordinary message.
- **`onMouseDown`, not `onClick`,** on each row: the textarea's blur fires before
  a click lands, so `onClick` would never run and the command would silently do
  nothing.
- **Escape appends a space rather than clearing the draft.** Dropping what the
  user typed in order to dismiss a hint would be hostile.
- **A shrinking list cannot leave the highlight past its end** (`Math.min`), which
  would make Enter run nothing.
- **A `notice` channel, separate from `lastError`.** "compacted" is not a failure,
  and routing it through the red banner would train the user to read red as
  noise. It is also separate from a turn's `notices`: a command can run with no
  turn in flight, and a turn's stream is replaced when the next one starts, which
  would erase the message before it could be read.
- **`/compact` is refused locally as well as by the server** (which returns 409
  mid-reply) — the fold and the turn's save would race and one would erase the
  other, so no request is made at all.

### Files

| file | change |
|---|---|
| `frontend-v2/src/chat/commands.ts` | NEW — roster, strict parser, `planFor`, `helpText` |
| `frontend-v2/src/chat/commands.test.ts` | NEW — 24 tests, the strictness cases first |
| `frontend-v2/src/chat/ChatSurface.tsx` | `CommandMenu`, interception in `submit`, keyboard handling, notice banner |
| `frontend-v2/src/chat/chatStore.ts` | `compactChat()` and the `notice` channel |
| `frontend-v2/src/lib/api.ts` | `compactSession()` |

### Still open in this gap

- **`/goal` is not implemented.** Setting a persistent goal needs a route Rigma
  does not have; the model can set one through `tool-goal`, but a *user* cannot.
  That is the same missing-goal-surface item already recorded in the R3 findings,
  and it is the honest next step rather than a command that quietly does nothing.
- `/export` and `/feedback` are not implemented. `/export` has a route
  (`GET /api/sessions/{sid}/export`); `/feedback` has no Rigma equivalent at all.

---

## 8. Done this round: DSH's compaction lifecycle

**Why it was invisible.** Rigma's *native* compaction already reports itself — the
server emits `masked`, `housekeeping` and `compacted`, and the transcript draws all
three ("compacted 12 messages into the summary", `Transcript.tsx:310-322`). A DSH
turn had none of that: DSH reports its compaction as four session events and Rigma
recognised none of them, so **a long turn busy summarising its own context looked
exactly like a turn that had hung.**

| event | payload | source |
|---|---|---|
| `compaction/start` | `{compactionId, turn, sourceCommandId?}` | `compaction/compaction/src/types.ts:24` |
| `compaction/summary` | `{compactionId, summary: ContentBlock[], shadowedRange, shadowedSeqs, shadowedTokenCount, provider, model, …}` | `…/types.ts:34` |
| `compaction/prune` | `{shadowedRange, shadowedSeqs, shadowedTokenCount}` | `…/types.ts:82` |
| `compaction/end` | `{compactionId, turn, sourceCommandId?, error?}` | `…/types.ts:72` |

**Folded, not appended.** `start`…`end` is a **bracket** paired by `compactionId`, so
appending the four as four lines would report "compaction started" forever if the
end never arrived. `compaction/prune` carries **no id of its own** — it is a
model-free replacement priced by the metering event before it — so it is attributed
to the currently-open compaction via a tracked `openId`. A prune is *added* to the
summary's count rather than replacing it.

**What the UI shows.** An amber "compacting the context — this can take a while"
while a bracket is open, and afterwards "compacted 7 messages · 12,345 tokens into a
summary", with the model that wrote it. A failure is the one thing a reader must not
miss — the context did **not** shrink — so `compactionLine` leads with it and the
line renders in red.

**Payload passed through whole.** The server does not reshape it: flattening
`summary` (a `ContentBlock[]`) or `shadowedSeqs` (a list) server-side would stop the
fold being able to *count* what it was given.

**Same prefix bug, avoided by memory.** `foldCompaction` requires the `compaction/`
prefix rather than merely stripping it — the exact bug already fixed once in
`governance.ts`, where `{event: "goal"}` became a trail entry called "goal". A test
now pins it in both modules.

**Files.** `chat/compaction.ts` (new), `chat/compaction.test.ts` (new, 20 tests),
`_dsh_runner._STATE_EVENTS` (+4), `serve.py` (`compaction` SSE arm),
`chatStore.ts` (field + reducer), `Transcript.tsx` (rendering).

---

## 8b. Done this round: DSH's LLM retry

**Not a gap anyone had listed — found by asking which of the remaining 37 event
types were worth bridging, rather than assuming the list was exhausted.**

`dsh-llm-retry` is a dependency of **`sdk-minimal` itself** (`bundle/sdk-minimal/
package.json`), so it is always loaded and always firing. It is not an opt-in
capability. When a request to the model fails and DSH schedules another attempt it
writes two events and says nothing on any surface Rigma reads:

| event | payload | source |
|---|---|---|
| `llm/retry` | `{retryId, turn, step, provider, mode, policyKey, retry, delayMs, failure: {message, code, status?, providerRetryAfterMs?, requestId?}, maxRetries?}` | `llm/llm-retry/src/types.ts:16` |
| `llm/retry-started` | `{retryId, turn, step, retry}` | `.../types.ts:35` |

Against a local llama-server that stalls, OOMs or returns a malformed tool call,
the visible effect was **a turn that sat there producing nothing** — the same
"frozen turn" class as the compaction gap, and the one place a user most needs to be
told what is happening.

**Two modes, and the difference is not cosmetic.** `normal` is bounded by
`maxRetries`, so "attempt 3 of 5" is true. `always` is **unbounded** and its payload
has **no `maxRetries` field at all** — which is why the fold models the absence as
`null` and the line says "attempt 3, no limit". Reading it as a number would make the
UI claim "attempt 3 of 0".

`llm/retry-started` marks the wait over but does **not** erase what happened: the
line switches to "retried" rather than vanishing the moment it becomes true. A
started event naming a retry we never saw scheduled is ignored rather than invented.

**Files.** `chat/retry.ts` (new), `chat/retry.test.ts` (new, 16 tests),
`_dsh_runner._STATE_EVENTS` (+2), `serve.py` (`llm_retry` arm), `chatStore.ts`
(field + reducer), `Transcript.tsx` (rendering), 6 backend tests.

**Still unbridged, and why that is fine.** Of the 51 event types, Rigma now bridges
15 as structured state plus tools and usage through their own paths. The rest are
bookkeeping (`step/start`, `request/header`, `agent/inbox/spliced`), transport
plumbing (`session-log-*/delivery-accepted`, `web/deepseek-search-llm-request`), or
belong to plugins this profile does not mount (`team/*` comes from
`experimental/agent-team`, which the patch does not include; `tool-workflow/*`,
`hook/*`). Bridging those would add noise, not capability.

---

## 8c. Round 3: the transport matrix, and DSH's OTHER transport

**The consolidated answer now lives in
[28-harness-transport-matrix.md](28-harness-transport-matrix.md)** — read that first;
it supersedes the scattered per-section notes here for the "what is reachable and why"
question. Two findings from this round are worth stating in the checkpoint itself.

### The root cause of every display-only caveat

`HarnessSdkNotificationMap` (`sdk/protocol/src/types.ts:106-112`) is **notifications
only** — the server half of the wire has no `id` and expects no reply. The client half
is three methods, none of which answers anything. So there is **no way for Rigma to
respond to DSH at all**: not to an approval, not to a question. Every display-only
caveat in this document is a consequence of that one fact, and a clickable Allow
button would misrepresent the connection.

### DSH has a SECOND transport that IS bidirectional, and it needs no login

`@deepseek-ai/dsh-acp` is a real ACP server ("Automation-only Agent Client Protocol
server for driving DeepSeek Harness agents over JSON-RPC stdio"), and an `acp` profile
already exists at `~/.dsh/profiles/acp`. Probed for real
(`tools/dsh_acp_probe.py`, `tools/dsh_acp_permission_probe.py`):

- `initialize` answers; `sessionCapabilities: {close, list, resume}`
- `session/new` returns a real session with **`model`** (a grouped catalog of eight
  provider routes) and **`reasoning_effort`** (`off`/`low`/`high`/`max`) options
- `session/set_config_option` **switched the model** to tokenjuice and returned the
  complete resulting state; a `session/prompt` then ran to `{"stopReason":"end_turn"}`

**So Rigma could have real per-chat model and reasoning-effort selection, which the SDK
wire cannot do at all.**

It is NOT a substitute for the SDK wire, though: ACP explicitly omits DSH presentation
data — **plans, commands, terminals, modes, fork, deletion** — and keeps raw deltas,
**retry attempts** and presentation data off the wire. The two transports are
complementary.

`session/request_permission` is advertised ("one-shot allow/reject choices; your client
can answer automatically") but was **not observed**: a workspace write, and even a
write *outside* the workspace, both succeeded with no prompt, because DSH's default
policy is permissive. Advertised and policy-gated, not demonstrated.

**Not built, deliberately.** An ACP client is a feature-sized piece of work (a second
transport, a session-id mapping, a new UI), and it would be an ADDITION rather than a
replacement. It is written up with the evidence in §3 of the matrix so the decision is
made on facts.

---

## 9. Gap 5, measured: mcode's ACP surface is real and blocked

The earlier note said ACP "would additionally mean implementing the ACP *client*
side". That was true but undersold it. Probed directly (`tools/mcode_acp_probe.py`,
`tools/mcode_acp_methods.py`, `tools/mcode_acp_notifications.py`) by speaking ACP
`initialize` to `mcode acp`:

```
agent: minimax-code 0.5.4          protocolVersion: 1
loadSession: true
sessionCapabilities: {list, fork, resume, close}
mcpCapabilities: {http: true, sse: true}
promptCapabilities: {image: false, audio: false, embeddedContext: false}

ADVERTISED METHODS (14):
  session/activate
  mcode/session/activate
  mcode/session/steer
  mcode/session/queue/{list,enqueue,update,delete,steer}
  mcode/session/goal/{create,get,patch,clear}
  mcode/session/delegation/{get,stop}

ADVERTISED NOTIFICATIONS (4):
  mcode/session/current_session_update
  mcode/session/delegation_update
  mcode/session/goal_update
  mcode/session/queue_update
```

**This is a genuinely larger surface than the stream-json path Rigma uses.** Note
`mcode/session/goal/*` — a complete goal control plane, and `goal_update` to push it
— and `mcode/session/delegation/*`, which is subagent control. Rigma currently
scrapes `exec --output-format stream-json`, which cannot reach any of it.

**And it is blocked.** `session/fork` answered:

```json
{"jsonrpc":"2.0","id":2,"error":{"code":-32000,
 "message":"Authentication required: Run `mcode login` and try again."}}
```

Every session method needs it. `~/.minimax/auth/prod/cn/mcode-public/` contains
**only `auth.lock` files and no credentials**, so this machine is not signed in, and
`mcode login` opens a browser to sign in to a MiniMax account. `initialize` and the
method *advertisement* work unauthenticated; nothing else does.

**Conclusion: do not build this yet.** It is the largest remaining piece of work, and
it would be written against an API that cannot be exercised even once — which is
precisely the "declared gap beats a fake green" rule. It becomes worth doing when the
user signs in; the three probe scripts are committed so the work starts from measured
facts rather than a guess.

---

## 10. Gap 6, withdrawn: `session/fork` does not exist

Recorded because the wrong version was in this document. Verified three ways:

- `grep fork` over `session-format-v0-to-v1/src/dispositions.ts` (the authoritative
  51-entry list) — **no match**;
- `grep fork` over `sdk/protocol/src/types.ts` (`HarnessSdkRequestMap`) — **no match**;
- `grep 'session/fork'` over the whole `packages/` tree — **no match**.

Forking exists, but as a **host-side API**: `ctx.sessions.create(id, { seed })` seeds
a new session from an existing event log (`core/session/src/index.ts:443-529`). That
is the harness's own composition path, not something the stdio wire exposes. The
`subagent_fork` *tool* forks a subagent, not a chat, which is what made this look
reachable.
