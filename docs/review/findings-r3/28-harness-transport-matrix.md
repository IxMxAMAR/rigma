# Harness capability parity: the transport matrix

**Status: measured, not inferred.** Every claim below was verified on this machine on
2026-09-22 and the command or file that establishes it is named. Where something is
*advertised* rather than *observed*, it says so.

This is the consolidated answer to "which harness capabilities are unreachable from
Rigma, and why". The narrative record is in
[23-harness-capability-bridge.md](23-harness-capability-bridge.md) and the stage
checkpoints in [27-harness-parity-stage-checkpoint.md](27-harness-parity-stage-checkpoint.md).

---

## 1. The finding that explains everything else

**The DSH SDK wire is a one-way telemetry channel.** This is the root cause of every
"display-only" caveat in this project, and it is a property of the transport, not of
Rigma's UI.

`packages/sdk/protocol/src/types.ts:106-119` is the whole contract:

```ts
/** Server-to-client notifications by JSON-RPC method name. */
export interface HarnessSdkNotificationMap {
  'session.event': SessionEventNotification
  'session.status': SessionStatusNotification
  'subagent.started': SubagentStartedNotification
  'subagent.finished': SubagentFinishedNotification
}

/** Client-to-server request methods with their param and result shapes. */
export interface HarnessSdkRequestMap {
  'initialize': { params: InitializeParams; result: InitializeResult }
  'session/prompt': { params: SessionPromptParams; result: SessionPromptResult }
  'shutdown': { params: undefined; result: Record<string, never> }
}
```

The server half is **notifications only** — no `id`, no reply expected. The client half
is three methods, none of which answers anything. So there is **no way for Rigma to
respond to DSH**: not to an approval, not to a question, not to a permission prompt.

Everything Rigma bridges from that wire is therefore correctly display-only. A
clickable "Allow?" button would misrepresent the connection.

---

## 2. What DSH exposes, and what Rigma does with it

### 2.1 The event vocabulary: 51 types, 15 bridged as structured state

The authoritative list is the dispositions table, not the core types file — plugins add
the rest: `packages/session/session-format-v0-to-v1/src/dispositions.ts` (51 entries).

Rigma's `_STATE_EVENTS` (`src/rigma/_dsh_runner.py`) holds **15**:

| group | events | reachable? |
|---|---|---|
| goals | `goal/change` | yes |
| todos | `todo/write` | yes |
| plan mode | `plan/mode` | yes |
| subagents | `subagent/descriptor`, `subagent/catalog` | yes |
| governance | `approval/asked`, `approval/decided`, `approval/policy` | yes — **display-only** |
| confinement | `sandbox/mode`, `permission/preset` | yes — display-only |
| title | `session/title` | yes |
| compaction | `compaction/start`, `compaction/summary`, `compaction/prune`, `compaction/end` | yes |
| retry | `llm/retry`, `llm/retry-started` | yes |

Tools and usage travel their own paths (`tool/call`, `tool/result`,
`assistant/message`).

**The remaining 36 are deliberately not bridged**, and this is a decision rather than
an omission. They are bookkeeping (`step/start`, `step/end`, `request/header`,
`request/context`, `agent/inbox/spliced`), transport plumbing
(`session-log-deepseek/delivery-accepted`, `web/deepseek-search-llm-request`,
`session/title-llm-request`), or belong to plugins this profile does not mount
(`team/*` comes from `experimental/agent-team`; `tool-workflow/*`, `hook/*`,
`tool/code-dispatch`). Bridging them would add noise, not capability. The original
filter existed because a one-line turn produced fifteen notifications.

### 2.2 The plugin surface: 130 installed, ~24 in the profile, 24 patched in

`C:\AI\deepseek-harness\python\sdk-runtime\node_modules\@deepseek-ai` holds **130**
packages. `sdk-minimal` depends on 29 (mostly infrastructure: LLM, agent loop,
session, tools, bash, pwsh). Rigma's patch inserts 24 — goals, todos, skills,
subagents, plan mode, filesystem, search, str-replace, compaction.

**The two capabilities that cannot be mounted usefully**, because of §1:

| package | description | why not |
|---|---|---|
| `dsh-user-questions` | "Abstract user-questions seam (`ctx.userQuestions`) for asking the human during agent runs" | the seam is **empty in every profile**; it needs a client-side UI answerer |
| `dsh-tool-ask-user` | "Model-facing `ask_user_question` tool over the `ctx.userQuestions` seam" | mounting it alone means the tool has nothing to answer it |

Verified: the `web` profile mounts `@deepseek-ai/dsh-client-ui-user-questions` as the
answerer (`dsh --profile web --dump-config`). The `acp` profile mounts
`dsh-user-questions` and `dsh-user-approval` but **no answerer**. Rigma's profile
mounts neither. So the answerer is a **client-side UI plugin** — and Rigma is an SDK
client, not a UI plugin, on a wire with no reply method.

### 2.3 Why interactive questions cannot work on EITHER DSH transport

This closes the last "could Rigma do better here" question, and the answer is no — for a
structural reason worth writing down so nobody tries again.

`@deepseek-ai/dsh-client-ui-user-questions` states the rule outright in its README:

> The package is one ownership rule: **rendering a question is a host UI capability,
> having the tool is an agent capability**, so the `tool-ask-user` row belongs to the
> host.

And the seam's own README calls `dsh-user-questions` "the **Host-side** question seam
and its answerer waterfall".

So answering a question requires being the **Host with a UI**. Rigma is an SDK
*client* of a host process — it is on the wrong side of that boundary, on a wire with
no reply method. That is why:

- the `web` profile can do it: it mounts `dsh-client-ui-user-questions`, a
  composer-takeover question surface with option selection, custom answers and skips;
- the `acp` profile cannot: it mounts the seam with no answerer, and the ACP README
  lists **elicitation** among the explicitly unsupported surfaces;
- Rigma cannot: it mounts neither, and could not use them if it did.

Mounting `dsh-tool-ask-user` in Rigma's profile would therefore be worse than leaving
it out — the model would gain a tool that can only hang or abort. `dsh-tool-ask-user`
and `dsh-user-questions` are both installed in the runtime and both deliberately
absent from the patch.

### 2.4 Slash commands: unreachable by transport

DSH ships `dsh-command-goal`, `dsh-command-compact`, `dsh-command-feedback`.
`parseCommand` (`interaction/commands/src/index.ts:125`) recognises the syntax and its
only caller is `CommandRuntime.execute` (`:367`), decorated **`@Remote`** — a web-layer
remote service. There is no wire method to invoke a command with, which is why Rigma
has its own command surface instead (§4).

### 2.5 `session/fork` does not exist

Verified three ways: no match in the 51-entry dispositions list, no match in
`HarnessSdkRequestMap`, no match anywhere in `packages/`. DSH forks by seeding an event
log through a **host-side** API — `ctx.sessions.create(id, { seed })`
(`core/session/src/index.ts:443-529`). The `subagent_fork` *tool* forks a subagent,
not a chat, which is what made it look reachable.

---

## 3. DSH's ACP transport: the interactive path, verified working

**This is the significant finding of the round, and it was not in the original plan.**

DSH ships `@deepseek-ai/dsh-acp` — "Automation-only Agent Client Protocol server for
driving DeepSeek Harness agents over JSON-RPC stdio" — and an `acp` profile already
exists at `C:\Users\amren\.dsh\profiles\acp`. **Unlike mcode's ACP server, it requires
no authentication.**

Started and probed for real (`tools/dsh_acp_probe.py`). `initialize` answered:

```
protocolVersion: 1
agentInfo: {"name": "deepseek-harness-acp", "version": "0.0.1"}
sessionCapabilities: {"close": {}, "list": {}, "resume": {}}
mcpCapabilities: {"http": true}
promptCapabilities: {"image": false, "audio": false, "embeddedContext": false}
```

`session/new` returned a real session with **two configuration options**:

| option | values |
|---|---|
| `model` | a grouped catalog: `deepseek-official`, `agentrouter`, `agentrouter-openai`, `openrouter`, `openrouter-ox`, `openrouter-extra`, `tokenjuice`, `omniroute` — dozens of models |
| `reasoning_effort` | `off`, `low`, `high`, `max` |

**And it works.** `session/set_config_option` switched the model to
`["tokenjuice","deepseek-ai/DeepSeek-V4.1-Flash"]` and returned the complete resulting
state; a `session/prompt` then ran to `{"stopReason": "end_turn"}`.

### What ACP would give Rigma that the SDK wire cannot

1. **`session/request_permission`** — the README: "a permission prompt with one-shot
   allow/reject choices; your client can answer automatically". A real Allow button.
   **Not observed in my probes**: a workspace write and even a write *outside* the
   workspace both succeeded with no prompt, because DSH's default policy is permissive.
   So this is an **advertised** capability that is policy-gated, not something I
   demonstrated.
2. **`session/set_config_option`** — real per-session model and reasoning-effort
   selection, **verified working**. The SDK wire has no configuration method at all.
3. **`session/list` / `session/resume` / `session/close`** — real session enumeration
   and lifecycle, advertised and mounted.

### What ACP would COST

From the same README, and this matters: ACP "intentionally omits DSH-specific
presentation data and interactive UI features". Explicitly unsupported: **plans,
commands, terminals, modes, elicitation, fork, deletion, `session/load`**. Raw provider
deltas, **retry attempts**, and DSH presentation data stay off the wire.

So the two DSH transports are **complementary, not substitutes**:

| | SDK wire (Rigma today) | ACP |
|---|---|---|
| goals, todos, plans | **yes** | no (plans explicitly omitted) |
| compaction, retry | **yes** | no (retry explicitly off the wire) |
| subagent lifecycle | **yes** | no |
| answer a permission | **no** | yes (policy-gated) |
| select model / effort | **no** | **yes, verified** |
| list / resume sessions | **no** | yes |
| auth required | no | no |

---

## 4. What Rigma does instead, and why that is the right shape

Because DSH's commands and interactive seams are unreachable, Rigma supplies its own:

| Rigma feature | stands in for | backed by |
|---|---|---|
| `/compact`, `/permission`, `/new`, `/stop`, `/skills`, `/help` | DSH's slash commands | `chat/commands.ts`; `/compact` → the existing compact route |
| `GovernanceBlock` | approval visibility | `chat/governance.ts` — display-only, and says so |
| `CompactionBlock` | compaction progress | `chat/compaction.ts` |
| retry line | a stalled turn | `chat/retry.ts` |
| capability panel | what the harness can do | `chat/Sidecar.tsx:871-889`, both directions |

---

## 5. mcode: the other harness

Rigma drives mcode **only** through `exec --output-format stream-json`.

### 5.1 The stream-json surface is fully extracted

mcode's own event projector is the authority
(`@minimax-ai/code/chunks/run-exec-command-*.js`): its `project()` emits exactly
`exec.started`, `session.started|session.resumed`, `turn.started`,
`item.started|item.updated|item.completed`, then `turn.completed|turn.failed` and
`exec.completed` — **8 types**, and its internal `er()` returns `[]` for anything that
is not an assistant delta or message.

Rigma's adapter handles **all 8** (`harness_mcode.py:770-1046`). So mcode's
stream-json surface is fully covered; nothing is being missed there.

### 5.2 What mcode exposes that stream-json cannot reach

The bundle contains **53 dotted event types**, including a complete goal state machine
and a delegation control plane:

```
goal.created  goal.state_transitioned  goal.admission_decided
goal.breaker_decided  goal.budget_decided  goal.budget_updated
goal.queue_item_deferred  goal.reminder_injected  goal.turn_bound
goal.turn_settled  goal.verification_child_started
goal.verification_decided  goal.verification_dispatched
goal.worker_proposal_decided
thread_goal.updated  thread_goal.cleared  thread_goal.objective_updated_steering
permission.ask  permission.resolved
questionnaire.ask  questionnaire.dismiss  questionnaire.superseded
session.llm_retry  session.title_updated  compaction.completed  message.rewind
```

**None reach stream-json** — the projector drops them all.

### 5.3 mcode's ACP server: real, larger, and BLOCKED

`mcode acp` is a real subcommand. Probed (`tools/mcode_acp_probe.py` and companions):

```
agent: minimax-code 0.5.4      protocolVersion: 1
loadSession: true
sessionCapabilities: {list, fork, resume, close}
mcpCapabilities: {http: true, sse: true}

14 METHODS:
  session/activate, mcode/session/activate, mcode/session/steer,
  mcode/session/queue/{list,enqueue,update,delete,steer},
  mcode/session/goal/{create,get,patch,clear},
  mcode/session/delegation/{get,stop}

4 NOTIFICATIONS:
  current_session_update, delegation_update, goal_update, queue_update
```

That is a genuinely larger surface than stream-json. **And every session method
refuses:**

```json
{"error":{"code":-32000,"message":"Authentication required: Run `mcode login` and try again."}}
```

`~/.minimax/auth/prod/cn/mcode-public/` contains **only `auth.lock` files and no
credentials** — re-checked at the start of this round, still not signed in.
`initialize` and the method *advertisement* work unauthenticated; nothing else does.

**Conclusion: not built.** It is the largest remaining piece of work and it would be
written against an API that cannot be exercised even once.

---

## 6. What is genuinely left

| item | blocked on | evidence |
|---|---|---|
| mcode ACP client (goals, delegation, queue, fork, resume) | **`mcode login`** — a browser sign-in to a MiniMax account | `-32000 Authentication required` on every session method |
| DSH ACP client (answerable permission, model/effort selection, session list/resume) | **nothing technical** — verified working; it is a design decision, since it would add a second DSH transport and give up goals/todos/plans on that path | `tools/dsh_acp_probe.py` output |
| interactive approvals on the SDK wire | **impossible** — `HarnessSdkNotificationMap` is notifications-only | §1 |
| `ask_user_question` on **either** DSH transport | **impossible** — answering is a Host UI capability and Rigma is an SDK client; ACP omits elicitation | §2.3 |

### The honest recommendation on DSH ACP

Worth doing **only as an addition, never a replacement**, and the first thing to build
would be the two things the SDK wire provably cannot do:

1. `session/set_config_option` — a model and reasoning-effort picker per chat.
   Verified working end to end.
2. `session/request_permission` — a real Allow/Reject prompt. Advertised, and
   **policy-gated**, so it must be provoked before it can be tested; my probes did not
   provoke one.

Both would need an ACP client in Rigma (`src/rigma/`), a second subprocess transport,
and a session-id mapping between the two transports. That is a feature-sized piece of
work, not a bridge — which is why it is written down here with the evidence rather
than half-built.

---

## 7. Probe scripts

Committed so this work resumes from measured facts rather than a guess:

| script | what it establishes |
|---|---|
| `tools/mcode_acp_probe.py` | mcode ACP answers `initialize`; session methods refuse with `-32000` |
| `tools/mcode_acp_methods.py` | the 14 methods and 4 notifications |
| `tools/mcode_acp_notifications.py` | the notification list, and login state |
| `tools/dsh_acp_probe.py` | DSH ACP starts, needs no auth, and exposes model + reasoning_effort |
| `tools/dsh_acp_permission_probe.py` | `set_config_option` works; a turn runs to `end_turn` |

All are read-only apart from creating a scratch workspace, and none of them touch the
original repository.
