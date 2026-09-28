# Round 6 checkpoint: the harness control planes

Status as of this round: `main`-branch fork at `C:\ComfyUI\RD\rigma-review`, working
tree clean. Round 5 ended at `57b8cd1`; this round adds seven commits on top.

## What round 6 set out to do, and where it came from

Round 5 closed with three named next steps, in the order the user was given them:

1. **MCP parity for DSH** — small, concrete, closes a real asymmetry.
2. **The ACP client** — unblocked by the R5-ACP-CORRECTION, and the big one.
3. `_ok_of` — "still needs a live mcode turn to observe."

Item 3 turned out to be answerable without a live turn, and item 2 turned out to be
much larger than one commit. All three are done.

## The correction this round rests on

Round 5 recorded mcode's ACP transport as **blocked on `mcode login`**. That was
WRONG, and the user was right to challenge it. The probe drove `mcode acp` with stdin
attached to a FILE, so mcode read one request, hit EOF, and exited before answering —
and an unanswered request was read as a refusal. The binary does contain the string
`Authentication required`, which is why the false quote survived review.

Measured properly, with no credentials on this machine: `initialize`, `session/new`,
`mcode/session/goal/get` and `session/set_mode` all answer. The only failure was
`session/prompt`, because Rigma's engine was not running on 11601 — mcode reaches a
model through the BYOK provider `~/.minimax/config.yaml` already registers
(`custom_provider: rigma`), which the README says "does not require a MiniMax login".

**The lesson, and it is the reason several tests below exist: an unanswered probe is
not a negative result.** Two error TYPES are now used to keep that distinction:
`AcpError` is "the server answered with an error", `AcpUnavailable` is "the server did
not answer at all". They are separate classes so they cannot be caught as one thing by
accident.

## 1. R6-MCP — Rigma's own tools reach DSH, not only mcode

`harness_mcode.ensure_mcp` has given mcode an `mcp.json` naming four MCP tools since
R4. DSH got nothing: `patch_file` passed no MCP configuration and no shipped profile
mounts `dsh-mcp-client`. So a DSH turn could not search the user's indexed documents
or remember anything, and **no line anywhere said so** — the same class of defect as
the plan-mode claim R5-PLANMODE removed.

`dsh-mcp-client` was already installed in the packaged runtime with its dependencies,
and sdk-minimal mounts both of its optional peers. Nothing was missing but the row.

- `harness_dsh.mcp_patch_file(tmpdir, cwd)` generates a one-row patch over stdio with
  `serverName: rigma`. Generated rather than shipped because `command` must be
  `sys.executable` and the patch carries the chat's cwd.
- `_dsh_runner.py` composes it as `patches = (cap, skills, mcp, patch)`.
- `harness.py`: DSH's `capabilities` names the four tools as the MODEL sees them
  (`mcp__rigma__remember`), and `_LEAVES_BEHIND` no longer claims RAG and undo are
  left behind — that had become a FALSE NEGATIVE.

**Why the env is in the row.** `dsh-mcp-client` hands the child `scrubbedParentEnv()`
plus exactly the row's `env`, so `RIGMA_MCP_ALLOW_CODE` and `RIGMA_MCP_WORKSPACE`
cannot be inherited. Relying on the ambient environment would have worked on this
machine and failed on a clean one.

`failOnStartupError` is deliberately left `false`: a broken Rigma MCP server must cost
the four tools, not the whole turn. The cost of that choice is that a malformed row
fails SILENTLY — which is why the tests pin the row against the plugin's real schema
rather than against our own idea of it.

**Measured:** Rigma's MCP server handshaken over stdio exactly as the patch launches
it — `initialize` OK (`{"name": "rigma", "version": "0.11.0"}`), `tools/list` OK with
3 tools (`remember`, `recall`, `undo_last_change`). `search_my_documents` is correctly
ABSENT because nothing is indexed on this machine, which is the roster being live
rather than hardcoded.

## 2. R6-ACP — the client, and the three declarations

`harness_mcode_acp.py` is a bidirectional JSON-RPC client over the child's stdio.
`mcp_client.McpServer` could not be reused: ACP's server sends
`session/request_permission` and `elicitation/create` and WAITS, so the reader must
dispatch inbound REQUESTS. stdin must also stay open for the session's life.

### The declaration that is easy to miss and loses everything

mcode gates its interactive and extension surface on what the CLIENT declares:

| declaration | unlocks |
|---|---|
| `elicitation.form` | `elicitation/create` instead of the silent fallback |
| `plan` | the `plan_update` plan review |
| `_meta["minimax-code/extensions"] = {version: 1, notifications: true}` | ALL FOUR extension notifications |

The third is the dangerous one, because **it is not symmetric**: mcode ADVERTISES its
extension method list in the `initialize` RESPONSE. A client can read that list, call
every extension method SUCCESSFULLY, and never receive a single update —
`goal/create` answers correctly and `goal_update` is dropped. That reads as "mcode does
not notify" rather than "we never said we could hear it". Two tests pin both halves.

`supports()` deliberately has NO fallback to the module's own constant. An earlier
version fell back to `EXTENSION_METHODS`, which made it answer True for a server that
had advertised nothing — a claim about the server derived from our file, the same
shape of error as the false blocker.

### A drift guard caught five missing wrappers

`test_every_advertised_extension_method_has_a_client_call` found `session/activate`,
`mcode/session/activate`, `queue/update`, `queue/delete`, `queue/steer` — methods mcode
advertises that Rigma could not reach and would not have noticed.

### Two real bugs in the client, found by its own tests

- `stop()` filled every pending slot's error but never SIGNALLED its event, so a
  blocked caller waited out its full timeout (or forever) instead of being released.
  The thread stayed alive, which is how it was caught.
- The permission-mode error assertion had been written against a message my own fake
  had fabricated. The real text is `Unsupported permission mode: <value>`, from the
  runtime's own guard `nS`.

## 3. R6-OKOF — the tool-status enum, resolved from its definition

`_ok_of` returned None for every real call, so every mcode tool chip rendered UNKNOWN.
It refused to guess, and that refusal was correct: the captured fixture pairs status 3
with "Tool not found", but the projector contains no `status: 3` literal, so mapping 3
to a cross could have inverted every chip if 3 had meant success.

The enum's DEFINITION settles it. In the installed 0.5.4, `chunk-QWAB5G2D.js` byte
6132401:

    zu = {Start: 1, Finished: 2, Failed: 3, Preparing: 4, Prepared: 5}

is the value domain of the thrift field `tool_call_status`, and the same module pins
the polarity with `isError: e.tool_call_status === zu.Failed`. The projector does not
invent the number — it passes the runtime tool-call object through verbatim — so this
is authoritative. The earlier reasoning was sound about the evidence it had and wrong
about where to look.

2 -> True, 3 -> False, 1/4/5 -> None. The string arms stay, because `Dwn` in the same
module proves "preparing"/"prepared"/"completed"/"failed" are that ONE enum's other
spelling. A bool status is rejected explicitly, since `True == 1` in Python.

**The gap that remains:** NO capture in this repo contains status 2, so "2 means
success" rests on the enum's names plus the runtime's own `isError` predicate — strong
evidence, still not an observation.

## 4. R6-ACP-UI — the control plane in the panel

`AgentState.tsx` gains an `AcpBlock`: the live configOptions, the plan review, the
delegation tree, the queue, and the advertised commands. Each renders only when the
backend reported it, so a turn that never spoke ACP draws exactly as before. Nothing
is guessed — a missing field is omitted rather than defaulted.

The DURABLE panel deliberately does not restore any of it: a queue is messages waiting
for the NEXT turn and the configOptions are the selects in force right now, so
restoring them from a durable copy would show a queue that has already drained.

Five new SSE names, each REPLACING rather than merging, because mcode sends the whole
list every time and a merge would resurrect a deleted queue item.

### The cross-layer drift guard found a real bug

An ACP event crosses four hops — mapper names it, `serve.py` routes that name to an SSE
name, `chatStore.ts` has an arm for the SSE name — and a name that stops matching at
any hop is dropped SILENTLY. The guard reads all three name sets from source and
reported:

    the ACP mapper emits these and serve.py drops them: ['session_title']

The mapper was emitting `session_title`, the SSE name, one hop too early; `serve.py`
routes the BACKEND name `session/title`. Every ACP session retitle would have been
dropped with no error anywhere. Fixed, and the pair is now pinned.

## What is deliberately NOT done, and why

- **The chat turn still runs over `exec`.** The ACP client exists and works; swapping
  the live transport is a separate, riskier change than building the client. The
  `drives` line says so rather than implying the turn uses ACP.
- **The approval channel is built but not wired to a chat turn**, for the same reason:
  it is only reachable if the turn runs over ACP. What exists is the mechanism
  (`answer_permission`, tested) and the disclosure that mcode offers it.
- **`ask_user` over mcode ACP is not wired either.** The client declares
  `elicitation.form` and can answer; nothing in a chat turn asks yet.

## What was NOT verified, and must not be claimed

- **No real `mcode acp` session was driven.** `session/prompt` calls a model and the
  user's standing order forbids loading one. The handshake, the advertised surface and
  the rejection texts are measured facts; end-to-end turn behaviour is not.
- **No DSH process was booted**, so the composed MCP patch tree is unconfirmed.
- **The ACP panel has never been seen in a browser.** There is no `*.test.tsx` in this
  repo at all, so the render layer is covered by typecheck plus reducer and formatter
  tests. That is the honest ceiling.
- `_ok_of`'s status 2 has never been observed on the wire.

## Gates at this checkpoint

- Python: full suite, counts in the round-6 commit messages.
- Frontend: 375 tests across 27 files; tsc clean.
- Bundle: `index-vrblvNIh.js` + `index-B3TPff.css`, referenced from `index.html`.
- ruff clean.

## Next, in the order I would take it

1. **Wire the ACP turn** — make mcode's chat turn run over ACP instead of `exec`, which
   is what makes the approval channel, the queue and steering live rather than
   reachable. The client is the hard part and it is done.
2. **`/export` and `/feedback`** — `/export` has a route already
   (`GET /api/sessions/{sid}/export`); `/feedback` has no Rigma equivalent.
3. The unactioned DSH findings from round 5's audits: F5
   (`subagent/descriptor` dies at `subagents.ts:139`), F6
   (`subagent/model-selection-policy` dropped by `_notice_text`), F7 (workflow/PTC
   reduced to an opaque string), F8 (`web_fetch` omitted for a reason that applies only
   to search providers).
