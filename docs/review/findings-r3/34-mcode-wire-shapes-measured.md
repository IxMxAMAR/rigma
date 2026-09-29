# mcode's delegation and queue shapes — MEASURED

This file exists because the previous one could only say "declared, not verified". Every
field below was read out of **mcode 0.5.4's own shipped code**, so the frontend's types are
now checkable against a source rather than against my inference.

Two lessons are worth recording alongside the facts, because both cost real time.

**A single-file search lies.** I first searched only
`chunks/run-acp-command-JPZMIXGP.js` and found `errorMessage`, `backgroundTaskId` and
`failedReason` **zero** times. That looked like three invented fields. Searching all 270
`.js` files found `errorMessage` 279 times, `backgroundTaskId` 8, `failedReason` 34. mcode
is code-split, and the mappers live in a different chunk from the ACP entry point. A
negative from one file is not a negative.

**A guessed shape is how a destructive control lies.** `delegation_stop`'s result sentence
read `receipt.stopped`. That field does not exist, so the arm fell through to its own
fallback and reported *"all delegated work stopped"* for **every** outcome — including the
one where nothing was running. I wrote that arm in the commit immediately before this one,
by guessing.

## The delegation member — `chunks/chunk-M5QJG5VT.js`

The mapper is `Se(e)`:

```js
function Se(e){return{sessionId:e.sessionId,parentSessionId:e.parentSessionId,
  ...e.agentName?{agentName:e.agentName}:{},
  ...e.title?{task:e.title}:{},
  status:D(e.status),
  ...O(e.purpose),
  ...h("createdAtMs",e.createdAt),...h("updatedAtMs",e.updatedAt),
  ...e.errorMessage?{errorMessage:e.errorMessage}:{}}}
```

So `AcpDelegationMember` is **correct and complete** — all seven fields it declares are
produced. Notes that matter for the UI:

- **`task` holds `title`.** The rename is real, and the frontend comment already said so.
- **`agentName`, `task`, `backgroundTaskId`, `errorMessage` are CONDITIONAL** — emitted only
  when present. That is why a search for them can come back empty even though they are real.
- **`backgroundTaskId` comes from the PURPOSE**, not from a field:
  `O(e)` returns `{backgroundTaskId: e.purpose.slice("local-background-task:".length)}` only
  when `purpose` starts with that literal. The frontend declares it and **never renders it** —
  so a background-task delegation shows no link to its task. Real, available, unused.
- **`parentSessionId` IS present on every member.** The tree is drawn **flat** anyway. The
  data to draw it as a tree has been arriving the whole time.

### The status vocabulary — now SETTLED

`D(e)` normalises every raw status onto exactly six values:

```js
n==="queued"||n==="pending"||n==="created"                         -> "queued"
n==="running"||n==="started"||n==="active"||n==="busy"
  ||n==="processing"||n==="in_progress"                            -> "running"
n==="finished"||n==="completed"||n==="succeeded"||n==="success"
  ||n==="idle"                                                     -> "completed"
n==="error"||n==="failed"||n==="lost"                              -> "failed"
n==="aborted"||n==="interrupted"||n==="cancelled"||n==="canceled"
  ||n==="stopped"                                                  -> "stopped"
otherwise                                                          -> "unknown"
```

So `queued`/`running`/`completed`/`failed`/`stopped`/`unknown` is the complete set, and the
frontend comment that claimed exactly that is **verified**. **`pending` is an INPUT, never an
output** — and the row-action table I wrote last commit used `["queued","pending"]`, which
could never match the second value. mcode's own TUI filters for `queued` and `paused`:

```js
this.items.filter(r=>r.status==="queued"||r.status==="paused")
```

`paused` is the real second actionable status. Corrected to `["queued","paused"]`.

## The queue item — `chunks/chunk-E2AN54L4.js`

The mapper is `Tve(a)`:

```js
function Tve(a){return{itemId:a.itemId,sessionId:a.sessionId,status:a.status,
  ...a.source?{source:a.source}:{},
  ...a.reviewRequest?.scope==="local_changes"?{reviewRequest:{scope:"local_changes"}}:{},
  ...a.content!==void 0?{content:a.content}:{},
  ...a.attachments?{attachments:structuredClone(a.attachments)}:{},
  ...a.modelInfo?{modelInfo:{...a.modelInfo}}:{},
  ...a.createdAt!==void 0?{createdAt:a.createdAt}:{},
  ...a.expiresAt!==void 0?{expiresAt:a.expiresAt}:{},
  ...a.startedAt!==void 0?{startedAt:a.startedAt}:{},
  ...a.finishedAt!==void 0?{finishedAt:a.finishedAt}:{},
  ...a.failedReason?{failedReason:a.failedReason}:{}}}
```

`AcpQueueItem.failedReason` is **real** — mcode's TUI renders it as
`Couldn't send · <reason>`. The timestamps are mcode's own names here (`createdAt`), **not**
the `Ms`-suffixed ones the delegation members use, which is the kind of inconsistency a
guessed type gets wrong.

## The delegation-stop receipt — `chunks/chunk-E2AN54L4.js`

`stop()` ends with:

```js
return {schemaVersion:1, rootSessionId:e, rootStopped:n,
        stoppedSessionIds:[...r],
        activeSessionIds:d.members.filter(HP).filter(...).map(...),
        failedSessionIds:[...i]}
```

There is **no `stopped`**. The result sentence now reads `stoppedSessionIds` and
`failedSessionIds`, and reports a partial failure explicitly — *"stopped 1 delegated task;
2 could not be stopped"* — because reporting a clean stop when children refused is the same
lie in a quieter form. Five cases are pinned by test, including the no-receipt fallback.

## The plan — `chunks/run-acp-command-JPZMIXGP.js`

```js
e.client.notify(ee.client.session.update,{sessionId:i,update:{
  sessionUpdate:"plan_update", plan:{type:"markdown",planId:a,content:d}}})
```

The payload is `{type, planId, content}` — so `hasPlan`'s check for a string `content` is
**correct for mcode**, which is what I measured. But my rebuttal of the empty-box finding was
**too narrow, and the second audit was right to say so.**

`PlanUpdateContent` is a **three-way union**, not one shape:

```
PlanUpdateContent = (PlanItems    & {type:"items"})     -> {planId, entries: PlanEntry[]}
                  | (PlanFile     & {type:"file"})      -> {planId, uri}
                  | (PlanMarkdown & {type:"markdown"})  -> {planId, content}
```

`PlanItems` has **neither `content` nor `uri`** — so a conformant server sending that arm
produced a plan object the panel could not draw. And the mapper's `update.get("plan") or {}`
made a malformed `plan_update` produce `acpPlan = {}`, which is **non-null**.

That mattered because **two copies of the same rule disagreed**:

- `AcpBlock` (the child) gated on `typeof content === "string" || typeof uri === "string"`
- the outer `AgentState` panel gated on `acpPlan != null`

So the object passed the outer gate — the panel drew its rounded container — while the child
returned null. **A visible empty box.** I had dismissed this by checking only the one arm
mcode happens to send.

Three fixes, and the third is the one that matters:

1. The mapper no longer emits a plan object that says nothing (`or {}` removed; an empty
   `plan_update` produces **no event**, because saying nothing is the correct answer).
2. The `items` arm is **routed** rather than dropped: its `entries` go to the same todos
   channel the SDK's sibling `plan` variant already uses for the same data. Previously they
   were silently discarded — not drawn, not routed anywhere.
3. The rule is now **one function**, `planIsRenderable`, used by both the child and the
   parent. Two copies of a predicate is how they drifted apart; collapsing them is the fix,
   not updating both.

Note the naming: mcode sends `plan_update` as a **`sessionUpdate` variant** carrying an
object with `content`, while the ACP SDK's own `plan` variant carries `{entries:[…]}`. Rigma
handles both, and routes them to **different panels** — `plan_update` markdown/file → the
"plan review" block, `plan` → the todos channel. That is a deliberate choice, but it means
"the plan" is two different things depending on which wire produced it, and the `items` arm
of `plan_update` is a third case that belongs with the second.

## What is still not established

- **Nothing has been run against a real mcode.** All of the above is read from mcode's
  shipped code; no live `mcode acp` session was driven and no model was loaded.
- Whether mcode **replays** a pending permission request on a resumed session is unknown.
- **`members[].backgroundTaskId` is available and still unrendered.** `parentSessionId` is
  now used (the tree indents by depth), but a background-task delegation still shows no link
  to the task it names. Declared rather than fixed: the row is correct, just less than the
  data allows.
- No browser has rendered any of this.

## Coverage added, because the absence of it is why these survived

The second audit's most useful finding was not a bug — it was that **the fake never put a
member in `members`**, so `delegation/get` and `delegation_update` always sent an empty
list and the panel's `members.map` was never rendered by anything. `errorMessage` and
`failedReason` had **zero** occurrences across `tests/**` and every `*.test.ts`, and the
fake never emitted a mid-turn delegation update at all.

- `--delegation tree` seeds a three-deep tree from mcode's **measured** member shape: one
  `running` child, one `queued` grandchild, and one `failed` member carrying both
  `errorMessage` and `backgroundTaskId`. **Off by default**, because the existing tests
  assert an empty tree and turning members on everywhere would change what they mean
  without anyone deciding that.
- The fake emits the snapshot **mid-turn**, as mcode does, instead of only on
  `delegation/stop` — the omission that made the whole path unreachable.
- The root placeholder is resolved **where the snapshot is built**, not at `session/new`:
  a resumed process never runs `session/new`, and every control operation runs in its own
  process.
- The fake's stop receipt was corrected to mcode's real field names. It had answered
  `{"receipt": {"stopped": [...]}}` — **the same invented field the client read**, so the
  double and the client agreed with each other about a field the server does not have. A
  double that invents a shape validates the invention.

