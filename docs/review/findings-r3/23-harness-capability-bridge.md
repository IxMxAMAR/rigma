# R3 capability bridge — what the harnesses could do, and what Rigma showed

*Owner request, 2026-09-27:* "Clearly some functionalities of the harnesses have
been limited? I need every functionality to be in Rigma, including anything that
may need a UI or visual representation at all… I want everything the harnesses
have to offer, just through my UI."

This documents what was actually limited, why, and what changed. It is a
capability round rather than an adversarial one, so the shape is different from
`22-adversarial-round-fixes.md`: the question is not "what was wrong" but "what
was reachable and what was not".

---

## 0. The finding, in one paragraph

Rigma drives three agent backends (its own loop, DeepSeek Harness, MiniMax Code).
All three were **already running with their full capability**; almost none of it
was **representable**. The seam between a backend and the chat had a fixed
six-word vocabulary — `text`, `thinking`, `tool`, `tool_result`, `notice`,
`error` — and no word for a goal, a todo list, a plan, a subagent, or a token
count. DSH additionally had a hard filter that reduced its entire internal
lifecycle to a six-keyword substring match against a two-field summary,
truncated to 200 characters. The result was that a DSH turn arrived as one
notice line and one final text block: **zero tool chips, zero reasoning, no
visible work**. And the capabilities themselves were partly absent from the
runtime, not merely unreported — see §1.

---

## 1. The capabilities were not only unreported, they were not mounted

This is the finding that changes the shape of the fix, and it was not obvious
from reading Rigma.

Rigma boots DSH with `profile="sdk-minimal"`. That profile is **not** a trimmed
`sdk`; it is a *standalone* Cordis tree whose only model-facing tools are two
persistent shells. It mounts no goal plugin, no subagent plugin, no todo tool, no
skill tool, no plan mode, no filesystem tools and no compaction. So forwarding
events could not have revealed goals or subagents — **there were none to forward.**

The obvious fix — switch to `profile="sdk"` — was rejected, and the reasons are
in `src/rigma/data/dsh/agent-capabilities.patch.yml`. That profile flips the
sandbox to `workspace-write`, mounts an approval gate that fails closed (Rigma is
headless, so every write would be denied), switches session logs to zstd, makes
`settings.yaml` and the credential store live inputs, and changes the system
prompt shape. It would trade a working integration for a broken one to gain a
tool list.

So the capabilities are **inserted** into `sdk-minimal` with a pure-`insert`
Cordis patch. The mechanism was verified rather than assumed: patch layers apply
in order and the last write wins, `insert` is unconditional at the root, and
Rigma's `patches=(...)` lands in `context.overlays` above every bundle layer.

**Measured, not assumed.** `dsh --profile sdk-minimal --patch <the file>
--dump-config` exits 0 and the composed tree grows from **130 to 239 lines**,
adding all 40+ rows. A typo in the patch would *warn and skip* rather than fail,
so `tests/test_dsh_capabilities.py` reads the file and asserts the rows by id
rather than trusting a successful boot.

---

## 2. The DSH notice choke point

`_dsh_runner._NOTICE_WORTHY` was six keywords, and `_notice_text` read exactly
two fields out of the notification — `payload["event"]["type"]` or
`payload["status"]` — then lowercased them, substring-matched, and truncated to
200 characters. Its own comment says why, and the reasoning was sound: a one-line
turn produced fifteen notifications and forwarding them all buried the reply.

The cost was not stated anywhere, and it was large:

| Fact | What reached Rigma |
|---|---|
| a tool call | the literal string `"session.event tool/call"` — no name, no arguments |
| a tool result | nothing distinguishable from a call |
| a subagent starting **and** finishing | nothing (`"subagent.finished"` contains "finish", not "fail") |
| a goal being set or changed | nothing |
| a todo list being written | nothing |
| a plan-mode switch | nothing |
| the step's token usage | nothing |
| reasoning | nothing |

`_run_turn` also emitted only three of `RunResult`'s fields — `final_response`,
`finish_reason`, `session_id` — so everything else the SDK computed was dropped
at the process boundary.

**The fix** replaces the substring match with a projection:
`_dsh_runner._project(notification)` returns a *list* of structured events, one
per fact. Tool calls carry the tool name, its arguments and the backend's call
id; tool results carry the error flag; goals, todos, plan mode, subagent
lifecycle and usage become `{"type": "state", "event": <the backend's own event
name>, "data": <its payload>}`. The old one-line behaviour is retained for
everything else, so the noise floor is unchanged.

Two design choices are worth naming:

- **The whole envelope is kept, not the interesting field.** `goal/change`'s
  payload is `{operation, goal: <snapshot>, roundsStarted, createdAt, updatedAt}`.
  An earlier version of this change unwrapped it to the snapshot, which threw away
  the round count and the operation — exactly the context a UI needs to say
  "round 4 of 60, paused". The test that caught this is
  `test_a_goal_change_is_structured_not_a_progress_line`.
- **The call id is forwarded.** Without it, two calls to the same tool in one
  turn collapse onto a single chip, because the frontend matches a result to the
  first still-running chip with that id. The server previously fell back to
  `ev.name`, so this was a live bug on the external path, not a theoretical one.

---

## 3. The seam had no word for these things

`TurnEvent` gained two fields — `event` and `data` — both defaulted, so every
existing adapter and call site kept working. They exist because a goal, a todo
list and a token count are **structured facts with no home in `text`**. Squashing
them into a sentence would mean re-parsing our own prose in the UI, which is how a
field quietly becomes a sentence and stops being data.

`harness_dsh._event_for` translates the new `state` kind; `serve.py`'s external
turn loop maps each backend event name onto its own SSE name:

| backend event | SSE event | rendered as |
|---|---|---|
| `goal/change` | `goal` | objective, phase, round N of M, blocked reason |
| `todo/write` | `todos` | the list, with ✓ ◌ ○ × per status |
| `plan/mode` | `plan_mode` | "proposing, not changing anything yet" |
| `subagent.started` / `.finished` | `subagent` | a row per child, with its closing message |
| `assistant/message` | `usage` | the step's token accounting |

An event name from a **newer** DSH than this build knows is dropped rather than
guessed at, the same way an unknown `TurnEvent` kind already was: a wrong
rendering is worse than none.

---

## 4. Five events the server had always sent and the UI had always dropped

Found by diffing the two vocabularies rather than by reading either one. The
server emits **15** SSE names; `chatStore.applyEvent` handled **10**. The five
missing ones all fell into `case "message": default:`, which returns the turn
unchanged when the payload carries no `delta` — so they vanished with no error
anywhere.

| event | was | now |
|---|---|---|
| `meta` | live per-turn tok/s unused; **server-side retitling never reached the rail**, so a renamed chat kept its old name until a reload | `title` on the turn |
| `housekeeping` | "compacting the context" invisible — a long compaction looked like a frozen screen | a status line |
| `masked` | observation masking never mentioned | "masked N earlier observations" |
| `compacted` | an auto-compaction was silent (only its *failure* was re-emitted as a notice) | "compacted N messages into the summary" |
| `info` | a queued prompt produced a 200 with an empty-looking stream and no explanation | "queued behind the running reply — N waiting" |

There was **no** case of the frontend handling an event the backend never emits.
One stale comment claimed otherwise for `citations`; `serve.py` does emit it and
the case was already re-added. Corrected in place.

---

## 5. What Rigma *adds* to a backend was undisclosed

The seam published `unsupported` — what a backend does not get from Rigma — and
`Sidecar.tsx` renders it, along with the drift warning. I initially recorded this
as a gap and **that was wrong**; both were already rendered at `Sidecar.tsx:864`
and `:885`.

The real gap was the other direction. Nothing said what Rigma *supplies*. For DSH
that is the entire §1 story: its goals, subagents, todos, skills, plan mode and
filesystem tools are Rigma's doing, and without a line saying so a user cannot
tell Rigma's additions from DSH's own — the capability simply appears.

`Harness` gained a `capabilities` tuple, declared by each adapter because only
the adapter knows what it mounts. DSH lists seven; mcode lists three, and they
are a different kind of thing — mcode owns its own goals and todos, so its list
is about Rigma's memory (an MCP server), the workspace and `AGENTS.md`, and a
session that survives the process. The UI renders it as an open disclosure
**above** the "does not get" list, because "what can this do" is the question a
user is asking.

---

## 6. `delegate` was a subagent with no identity

`delegate` hands a research question to a fresh-context helper whose tool calls
deliberately never re-enter context — the context firewall, and the single
biggest context-engineering win in the codebase. Its calls are recorded on the
assistant message as `delegate_trace`, a UI-only field, precisely so they do not
pollute `tool_trace` (which feeds context, run scoring and the turn signature).

Nothing rendered it. A delegated answer therefore arrived as one `delegate` chip
and then an answer, with no sign that twelve file reads had happened behind it.

It is now a collapsed disclosure under the tool chips, summarised as "a research
helper ran 12 tool calls using read_file, grep, list_directory — 1 failed —
2 refused". The refusal count is included deliberately: the helper's roster is
read-only by design, so a refusal is normal, and it explains a thin answer.

---

## 7. Verification

| check | result |
|---|---|
| `dsh --profile sdk-minimal --patch … --dump-config` | exit 0; tree grows 130 → 239 lines; every inserted row present |
| `tests/test_dsh_capabilities.py` | **20 passed** — patch rows, projection, adapter translation, envelope shape |
| `tests/test_harness_seam.py` | **37 passed**, including the new `capabilities` assertions |
| frontend `tsc --noEmit` | clean |
| frontend `vitest` | **20 files, 222 passed** (was 17 / 184) |
| `ruff check src tests` | All checks passed |
| full backend suite | see §8 |

Two bugs were found by the new tests rather than by review, and both were real:

1. **A double space at every content-block boundary.** `lastAssistantMessage` is
   a list of arbitrary slices, so `"found "` + `"three"` is one sentence. Joining
   on a space inserted a gap at every boundary. Caught by
   `carries the child's closing message, clipped`.
2. **The goal envelope was being unwrapped**, losing `roundsStarted` and
   `operation`. Caught by `test_a_goal_change_is_structured_not_a_progress_line`.

A third was caught by reading the code back before trusting it: `_project`'s
state-event table was written as a dict mapping event name → payload key, which
silently dropped everything in the envelope except that one key.

---

## 8. What this round did not do

Stated plainly, because a declared gap is worth more than a fake green.

- **mcode's own goal and todo state is still not read.** mcode keeps its plan,
  subagents and goals inside its session — the adapter's own docstring says so.
  Rigma stores the session id and can kill the tree, and nothing more. Reading
  them means either parsing mcode's private SQLite (`runtime-state.sqlite`,
  `local_runtime_thread_goals`) or its `messages.jsonl`/`llm-call.json`, both of
  which are undocumented internals that can change under a version bump. This
  round did **not** couple Rigma to them.
- **mcode's `usage` and `status` are still collected and discarded.**
  `harness_mcode` writes `state["usage"]` and `state["status"]`; `serve.py`
  forwards only `session_id`. The values are already in hand — wiring them is
  small and was not done.
- **Every mcode tool result still claims success.** The adapter's own comment
  says `ok` cannot be derived, so `TurnEvent.ok` stays `True` and the trace
  records `ok: True`. A tool that does not exist comes back as status 3 with the
  reason as text.
- **`harness_dsh._Run.hstate` is dead code and its docstring is false.** It is
  declared and never written; `drive_turn(state=...)` never reads or writes it,
  contradicting its own docstring. Continuity actually comes from the pooled
  runner process. `serve.py` therefore always persists `backend_session=""` for
  DSH. Either implement it or delete it — it is not implemented.
- **DSH still cannot stream or cancel.** The SDK reports a turn's text when it
  ends and has no wire-level cancel; Stop kills the subprocess. Unchanged, and
  unfixable from Rigma's side.
- **No live DSH turn was run with the patch applied.** The patch was verified to
  *compose* (dump-config), and every translation layer is unit-tested against
  recorded shapes, but a real end-to-end turn exercising a goal or a subagent was
  not executed. The first live turn is the remaining unknown.
- **Rigma's own autonomous `objective` is still not a first-class goal.** It
  exists as a string in a run's compiled spec, pinned into the system prompt and
  reachable only via `GET /api/runs/{rid}`. It has no id, no status, no mutator
  and no route of its own. `AutonomousSurface` still renders the plan list but
  not the objective above it.
