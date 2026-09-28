# Harness parity — what each backend can do, and what Rigma shows

*Checkpoint for the capability-bridge round, 2026-09-28. Companion to
`findings-r3/23-harness-capability-bridge.md`, which records the investigation;
this is the reference table.*

Rigma drives three agent backends through one seam. This is what each one can do,
what Rigma adds or withholds, and where the evidence lives. It exists because the
gap that prompted the round was not a broken feature — it was that **capability
was real and invisible**, and a table is what makes that checkable.

---

## 1. The three backends

| | native | dsh | mcode |
|---|---|---|---|
| what it is | Rigma's own loop | DeepSeek Harness | MiniMax Code |
| how a turn is handed over | in-process | `python -m rigma._dsh_runner`, NDJSON over stdio | `mcode exec --output-format stream-json` |
| verified against | itself (`built_in`) | `0.1.6-alpha.2` | `0.5.4` |
| streams text | **yes** | no — the SDK reports a turn's text when it ends | yes |
| cancels | **yes** | kills the subprocess | kills the subprocess |
| applies `permission` | **yes** | **no** (accepted and ignored) | only via `--permission full` |
| Rigma's system prompt | **used** | not used; arrives as a delimited preamble | not used |
| Rigma's tool repair / undo / artifact check | **yes** | no | no |
| sandbox | Rigma's own | pinned `danger-full-access` | needs `--permission full` |

`harness.py` declares `installed` and `runnable` separately on purpose: a backend
can be present on the machine and still not be wired up, and collapsing the two
is how a probe becomes a silent fallback. `resolve()` never falls back — a
session that asked for DSH and quietly got native would be a lie the user cannot
see from the output.

---

## 2. Tool surfaces

**DSH, unpatched, advertises exactly ONE tool to the model.** That was measured,
not inferred: a local server was put in place of the model and the request body
recorded.

| configuration | tools the model is offered |
|---|---|
| `sdk-minimal` | `pwsh` — 1 |
| `sdk-minimal` + `data/dsh/agent-capabilities.patch.yml` | 19 |

The 19, verbatim from the captured request:

```
create_goal  get_goal  update_goal  todo_write  exit_plan_mode
subagent  subagent_fork  send_message  interrupt_agent  list_agents
skill  workflow  read  write  edit  str_replace_editor  glob  grep  pwsh
```

mcode owns its own 18-tool roster, plus whatever MCP adds. Rigma can subtract
from that set, never replace it:

```
read  write  edit  bash  grep  glob  todowrite  skill  web_fetch  web_search
task  task_append  task_query  task_output  task_stop  website_deploy
update_goal  get_goal
```

The native loop has 42 tools in `tools.py`, gated by `needs=` (`workspace`,
`code`, `vision`, `method_builder`, `run`, `rag`) and by profile.

---

## 3. What Rigma ADDS to each backend

Declared by the adapter (`Harness.capabilities`), because only the adapter knows
what it mounts. Rendered as an open disclosure in the chat Sidecar, above the
"what it does not get" list — "what can this do" is the question a user is asking.

**dsh** — seven capabilities, every one of which `sdk-minimal` does not ship:

- goals, with a round driver and a tool to set, read and revise one
- subagents, spawned in-process or forked from this conversation
- a todo list the model maintains and the transcript renders
- plan mode, so it can propose before it changes anything
- skills, discoverable and loadable by the model itself
- filesystem tools: read, write, edit, glob and grep
- agent instructions from `AGENTS.md`, and context compaction

**mcode** — a different kind of list, because mcode owns its own goals and todos:

- Rigma's own tools as an MCP server, registered in `<dataDir>/mcp.json`
  (`~/.rigma/mcode/mcp.json`): the roster is `search_my_documents`, `remember`,
  `recall`, `undo_last_change`. `ensure_mcp` runs every turn and re-reads that
  file, and **repairs an entry that has drifted** rather than only writing one
  that is absent — see §8.
- the chat's workspace and its `AGENTS.md`, written in before a turn
- a session that survives the process, so its plan, subagents and goals continue
  across turns instead of restarting each one

---

## 4. What Rigma does NOT give them

`Harness.unsupported`, rendered in the Sidecar as "what X does not get from
Rigma". Shared by both external backends (`_LEAVES_BEHIND`):

- Rigma's tools
- the weak-model repair layer
- the run progress log, artifact verification and the one-action-per-turn guard

Plus, per backend:

| dsh | mcode |
|---|---|
| streaming — text arrives when the turn ends | driveable only as a subprocess; no library API |
| cancel — Stop ends the subprocess and the turn | its own tool roster: Rigma can subtract, never replace |
| the sandbox: `sdk-minimal` pins `danger-full-access` | its own system prompt; Rigma's is not passed through |
| | the sandbox: a headless turn needs `--permission full` |

---

## 5. Structured state, and where it comes from

The seam's `TurnEvent` has six kinds — `text`, `thinking`, `tool`, `tool_result`,
`notice`, `error` — plus `state`, which carries a backend's own event name and
payload for facts that have no home in a sentence. It was added this round,
because squashing a goal or a token count into `text` means re-parsing our own
prose in the UI.

| fact | dsh source | mcode source | SSE event |
|---|---|---|---|
| goal | `goal/change` (envelope: `{operation, goal, roundsStarted}`) | `update_goal`/`get_goal` result `details.goal` | `goal` |
| todos | `todo/write` | `todowrite` result `details.todos` | `todos` |
| plan mode | `plan/mode` | — (mcode uses its own `exit_plan_mode` tool) | `plan_mode` |
| subagents | `subagent.started` / `subagent.finished` | `task`/`task_*` result `details.task_id`, `sub_session_id` | `subagent` |
| usage | `assistant/message` → `usage` | `turn.completed` → `usage` | `usage` |

Two shape differences the UI normalises once, in `chat/goal.ts` and
`chat/subagents.ts`, rather than branching in a render path:

- DSH nests the goal under `goal` and calls the phase `phase` with a round count;
  mcode sends it flat and calls it `status` with tokens instead.
- DSH wraps subagent lifecycle in `{event, data}`; mcode reports a task's ids flat
  and its `status` **is** the lifecycle step.

An event name from a newer backend than this build knows is dropped rather than
guessed at: a wrong rendering is worse than none.

---

## 6. The five SSE events the UI used to drop

Found by diffing the two vocabularies — the server emits 15 names and the store
handled 10. All five missing ones fell into `case "message": default:`, which
returns the turn unchanged for a payload with no `delta`, so they vanished with
no error anywhere.

| event | what was lost |
|---|---|
| `meta` | live per-turn tok/s; **and server-side retitling never reached the rail** |
| `housekeeping` | a long compaction looked like a frozen screen |
| `masked` | observation masking never mentioned |
| `compacted` | auto-compaction silent; only its *failure* was re-emitted |
| `info` | a queued prompt produced an empty-looking stream, unexplained |

There was no case of the frontend handling an event the backend never emits.

---

## 7. Verification, and what is still unverified

| check | result |
|---|---|
| `dsh --dump-config` with the patch | exit 0; tree 130 → 239 lines; every row present |
| runtime **boots** with the patch (3-way isolation) | STARTED, including Rigma's exact config |
| tools **advertised** to the model | 1 → 19, listed above |
| `tests/test_dsh_capabilities.py` | 20 passed |
| `tests/test_mcode_state.py` | 17 passed |
| `tests/test_harness_mcode.py` | 71 passed |
| `tests/test_harness_seam.py` | 37 passed |
| frontend `vitest` | 21 files, 248 passed |
| `tsc --noEmit` | clean |
| `ruff check src tests tools` | All checks passed |

**Not verified:**

- No turn was driven far enough for the model to actually **call** `todo_write` or
  spawn a subagent and for the resulting event to travel to the UI. Every
  translation layer is unit-tested against recorded shapes and the composition is
  measured; one full round trip is not.
- `todowrite`, `get_goal` and all five `task_*` tools were **never invoked in any
  session on this machine** — zero occurrences across 108 message lines. Their
  schemas are real (`llm-call.json`) and the goal object's field list is real, but
  their success payloads come from mcode's shipped code, not observed data. Only
  `update_goal` was observed, and only on its error path.
- mcode's tool results still cannot always say `ok`: it is derived from the
  wire's `status`, and anything unrecognised is `None` = unknown.
- Rigma's own autonomous `objective` is visible now but is still a string field of
  a run's compiled spec — no id, no status, no mutator, and a chat cannot have one.

---

## 8. The MCP registration could be wrong forever

Found by checking a claim I had written in §3 rather than by testing the code,
which is worth recording as the reason it was found at all.

`ensure_mcp` points mcode at `python -m rigma.mcp_server`, which is how Rigma's
own tools — `remember`, `recall`, `undo_last_change` — reach an external agent
without Rigma editing a single message the agent sends to its model. It runs
every turn and re-reads the file each time.

On this machine the registration named an interpreter that **cannot import
Rigma's MCP server at all**:

```
$ <the registered interpreter> -m rigma.mcp_server
No module named rigma.mcp_server          (exit 1)
```

The registered command was the global `Python312\python.exe`, whose Rigma is a
stale **0.10.0** that ships `mcp_client.py` and no `mcp_server.py` — that module
is new in 0.11.0. So mcode's memory tools were dead, and nothing said so: the
file looks configured, and a missing MCP server is not an error the arm reports.

**The defect was not the stale path.** Rigma has used `sys.executable` for this
since the feature was first committed (`git log -S'sys.executable'` →
`83c1d1b`), and its docstring says why: *"the interpreter that has Rigma
installed is the one already running this."* What wrote the global path is **not
established** and is not needed: the global 0.10.0 has no `harness_*` module and
no `harness` support in its CLI at all, so it cannot have written this file, and
the entry's shape (`RIGMA_HOME`, `RIGMA_MCP_ALLOW_CODE`) matches what Rigma
writes. The path is the symptom. The defect was that the entry could **never be
corrected**:

```python
if wanted:
    servers["rigma"] = _mcp_spec(cwd)      # assigned ONLY when adding
else:
    servers.pop("rigma", None)
```

An entry that is **present but wrong** falls through both branches. It is not
added, because it is already there; it is not removed, because there is
something to offer. So it stayed exactly as it was, every turn, indefinitely. A
stale registration is not a rare state — an interpreter moves, a virtualenv is
rebuilt, Rigma is reinstalled, or the file is hand-edited — and each one was
permanent.

The fix compares instead of assuming, which costs nothing because the function
already reads the file:

- `_same_registration` checks the **command**, the **args**, and the two `env`
  keys that identify *which* Rigma and *whether* code tools were granted.
- `RIGMA_MCP_WORKSPACE` is deliberately excluded from the comparison: it is the
  chat's directory and moves from turn to turn, so counting it as drift would
  rewrite the file constantly. Verified: changing the workspace leaves the bytes
  and the mtime untouched.
- A hand-written non-object entry (`"rigma": "python -m rigma.mcp_server"`) is
  replaced rather than raising inside a turn.
- The fix is about the **class**, not this one file: an interpreter moves, a
  virtualenv is rebuilt, Rigma is reinstalled, or the file is edited by hand —
  and before this change every one of those was permanent.
- Other servers in the file are preserved. Rigma manages this file; it does not
  own it exclusively.

Verified end to end against a temporary home seeded with the exact stale entry:
repaired to `sys.executable`, foreign server intact, idempotent on a second run
(same bytes, same mtime), unchanged when the workspace moves. Five tests in
`tests/test_mcode_state.py` pin it.
