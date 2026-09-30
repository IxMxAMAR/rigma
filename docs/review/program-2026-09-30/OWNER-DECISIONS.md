# OWNER-DECISIONS — Rigma improvement program, 2026-09-30

Decisions that are the owner's, not the Head Agent's (brief §0.8). Each has the options, the
evidence, a recommendation, and what happens if nothing is decided. Nothing here has been
implemented; nothing here changes a global default.

---

## OD-1 — What should the default run profile be? (finding 13-2, prompt injection → secret exfiltration)

**State today:** the run profile defaults to `"all"` (`serve.py:2630`, `:6262`), which grants the
full network + delete surface to an autonomous run. The finding's other three mitigations are
already in the tree and both the read-confinement and outbound-post grants default **OFF**
(`serve.py:2642-2643`), and an unknown profile is currently *silently coerced* to `"all"`
(`serve.py:6324`) — that coercion is being fixed as item A7 regardless.

**Options**
1. Keep `"all"` as the default. A run that silently cannot reach the network is a run that fails
   in a confusing way; the grants exist and the UI shows them.
2. Default to `"read"` (no network, no delete). Safer by construction; a run that needs to fetch
   something now stops and asks, which the approval UI supports.
3. Default to a new `"ask"` profile: every network/delete action raises an approval prompt.
   Safest, but it turns a long autonomous run into a click-through.
4. Leave the default alone and make the *first* launch of a new run show the profile chooser with
   `"all"` preselected — informed consent without changing behaviour.

**Evidence:** `docs/review/findings/13-adversarial-security.md` (finding 13-2); the grants are at
`serve.py:2642-2643`; the profile is read at `serve.py:2630,6262`.

**Recommendation: option 4.** It is the only one that does not change a global default, it costs one
dialog, and the finding's real risk is *uninformed* consent rather than the profile value. If you
want a behavioural change, option 2 is the next best.

**If nothing is decided:** nothing changes; the current default stands. This is not a blocker.

---

## OD-2 — `view_image` reads absolute paths outside the workspace; `copy/move_files` can write anywhere (R3-9, R3-10)

**State today:** `read_file` refuses an absolute path outside the workspace without a grant, but
`view_image` does not (`findings-r3/04-sandbox-r3.md:225`), and `copy_files`/`move_files` accept any
absolute destination — including the Startup folder (`:250`). Both are read/write confinement gaps,
not exploits by themselves, but together they are a file-exfiltration and a persistence primitive
for a model that has been talked into it.

**Options**
1. Route `view_image` through the same confinement check as `read_file` (needs a grant for outside
   paths), and restrict `copy/move_files` destinations to the workspace + a configured allowlist.
   Consistent, and the grant mechanism already exists.
2. Add `view_image` to the confinement check only, and leave `copy/move_files` as-is.
3. Add both to a new "write outside the workspace" grant that defaults OFF, and keep the current
   behaviour when the user turns it on.
4. Do nothing; document the gap.

**Recommendation: option 1**, with the destination allowlist seeded from the paths the owner already
works in. It reuses the existing grant machinery (`serve.py:2642-2643`) rather than inventing a
second one, and it is the smallest change that closes both.

**If nothing is decided:** nothing changes. This is a security-policy trade-off and therefore not
the Head Agent's call.

---

## OD-3 — A pre-fix RAG index still holds secrets until it is reindexed (R3-11 residual)

**State today:** the credential-`exclude` denylist fix (`R3-17`) applies to *new* indexing. An index
built before the fix still contains whatever was indexed. Reindexing is the fix, but reindexing
walks and re-reads the owner's documents — which the Head Agent must not trigger.

**Options**
1. Reindex on the next RAG use, automatically, once, and say so in the log.
2. Offer a "rebuild index" button in the RAG panel and leave the decision to the owner.
3. Do nothing.

**Recommendation: option 2.** Auto-reindexing touches the owner's files without being asked, which
§0.8 excludes.

**If nothing is decided:** the stale index keeps its old contents. Recorded here so it is not lost.

---

## OD-4 — LoRA attach (`--lora`/`--lora-scaled` + `/lora-adapters`)

**State today:** Rigma has no LoRA code at all. The engine accepts `--lora`/`--lora-scaled` and the
server exposes `/lora-adapters` for runtime scaling. The owner trains LoRAs, so this is the one
lever in the R4 survey that is a missing *feature* rather than a missing knob.

**Options**
1. Add a per-model LoRA list (path + scale) that is appended to the launch argv, plus a
   `/lora-adapters` passthrough so scale can change without a relaunch. A real feature: a model
   editor row, a store, a UI control, tests.
2. Add the argv fields only (CLI + `LaunchDefaults`), no UI.
3. Defer.

**Recommendation: option 1, but as its own program** — it is larger than any single item in this
run's backlog and deserves its own design note. Recorded so it is not rediscovered.

**If nothing is decided:** no LoRA support; nothing regresses.

---

## OD-5 — Two fit terms that can only be settled with the card (NEEDS-GPU)

The fit currently charges the whole GGUF file to VRAM, charges the recurrent state zero (being fixed
as A2), and uses a flat `COMPUTE_BUFFER_MB = 150` (`resolve.py:28`). A real load log
(`.scratch/prism-v.log`) shows `CPU_Mapped 322.07 MiB` + `ROCm0 6539.67 MiB` for a 6872 MiB file,
`compute buffer 410.28 MiB`, `RS 149.62 MiB`, `graph splits 2`.

- **`token_embd` host residency.** Whether the host-resident 322.07 MiB is a property of this
  fork/model or of hybrid models in general is UNVERIFIED. Subtracting it universally could
  over-commit a model that does put it on the card. Needs a Rigma-launched log at `--parallel 2`.
- **`COMPUTE_BUFFER_MB`.** The 150 was deliberately lowered from 400 because it was obtained by
  differencing measured VRAM and already contains the draft head's buffers (`resolve.py:14-28`).
  Raising it back would double-count. Needs a measurement, not a guess.

**Recommendation:** run the extended plan in `docs/review/findings-r3/37a-262k-verification-plan.md`
(the Head Agent must not; §0.1) and hand the two numbers back. Both are recorded as `NEEDS-GPU` in
`BACKLOG.md` (E1, E2).

**If nothing is decided:** the RS term (A2) is still correct and strictly improves the plan; the
other two stay as they are, with the divergence now *detectable* because A17/S2 parses the engine's
own log.

---

## OD-6 — DSH ACP client, and answering `ask_user`

**State today:** the DSH ACP client is not on the wire (`findings-r3/28-harness-transport-matrix.md`
§320), and mcode `ask_user`/plan are gated on the client declaring `elicitation.form`/`plan`, which
Rigma does not (`28 §6a`). Answering a question needs a live mcode turn, which §0.1 forbids in this
run.

**Recommendation:** adopt the ACP client — it is the only transport that gives Rigma a real
permission round-trip instead of display-only notifications — and declare the elicitation
capability at the same time. Both need one live mcode session to verify.

**If nothing is decided:** the seam keeps working as it does today; questions continue to be
auto-declined, and the drop is now *visible* (item B2 makes dropped events a notice rather than a
silent discard).

---

## OD-7 — How many method drafts are kept on disk, and how long is a draft "in use"? (10-R3-17 / A15)

**State today:** `~/.rigma/method_drafts/` had no budget at all — a draft was removed only by
`promote()`, so a creation chat the user abandons left its file there forever. Item A15 adds
`DELETE /api/methods/drafts/{id}` (plus the singular `/api/methods/draft/{id}` its siblings use) and
a cap enforced when a new draft is created. **This one is implemented**, unlike OD-1..6; the two
numbers are the product decision.

**The numbers A15 chose** (both in `src/rigma/method_drafts.py`):
- `MAX_DRAFTS = 20` — drafts kept.
- `_DRAFT_LIVE_SECONDS = 12 h` — a draft written inside this window counts as in use and is never
  evicted while any older candidate exists. The builder chat rewrites its draft on every tool call,
  so an open chat keeps its draft live.

**Options**
1. Keep 20 / 12 h. A draft is a single-digit-KB JSON, so 20 is disk hygiene rather than a resource
   bound; 12 h makes it very unlikely an open chat loses its draft, and abandoned ones age out
   within a day.
2. A smaller cap or shorter lease (e.g. 5 / 1 h). Reaps faster, but can drop a draft a chat is still
   pointed at — the builder tool then answers "the draft is gone".
3. Reap on a timer or at startup instead of on creation. More even, but adds a background task for
   a few KB.

**Recommendation: option 1.** The cap deliberately yields rather than drop a live draft: if every
draft is fresh the store stays over the cap until one ages. Dropping a live draft is the one outcome
the eviction rule must not produce.

**If nothing is decided:** the numbers stand; changing either is a one-line edit.

---

## OD-8 — The mid-turn prompt queue cap, and how long a question waits for an answer (A12 / B4)

**State today:** two bounds in `serve.py` are product decisions rather than engineering ones, and are
recorded here because the brief requires it.

- **The mid-turn prompt queue.** A prompt typed while a reply is streaming is held in memory
  (`_queued`) until the running turn drains it. It is capped at `_QUEUE_MAX = 32` per chat, enforced
  at the door: the 33rd prompt is refused with HTTP **429** and a body that says how many are already
  queued — the client is TOLD, rather than the prompt being silently discarded later. 32 is a
  typing-spree bound, not a memory bound: each entry is one message. (Implemented in R3-5/A12, already
  on the branch.)
- **The elicitation wait.** mcode's `ask_user` arrives as an ACP `elicitation/create`; `serve.py` now
  answers it on the same channel as a permission. `QUESTION_WAIT_SECS = 5.0` is how long the ACP
  reader thread waits for the user before returning `None` (a decline). It is deliberately shorter
  than `_APPROVAL_WAIT_SECS` (120 s) because **no client renders elicitation yet** — a two-minute
  hang per question would be worse than the auto-decline it replaces.

**Options**
1. Keep 32 and 5 s. Raise `QUESTION_WAIT_SECS` toward `_APPROVAL_WAIT_SECS` in the same change that
   ships the UI card; keep the queue cap.
2. A smaller or larger queue cap. 32 is already far past what a person types during one reply; a
   larger cap only lengthens the in-memory tail.
3. Wait the full 120 s for a question now. Rejected: with no UI to answer, it stalls the turn for two
   minutes to reach the same decline.

**Recommendation: option 1.** Both numbers are one-line edits. The important half is behavioural: the
queue refuses *visibly* (429) and a question returns a real decline instead of hanging.

**If nothing is decided:** the numbers stand. What the UI still needs for B4 is a card for the
`approval/asked` event when `data.kind == "question"` that renders `data.question` + `data.schema` and
POSTs `{"requestId": data.id, "answer": {...}}` to `/api/sessions/{sid}/approval`. Until it exists,
every question declines after `QUESTION_WAIT_SECS`.
---

## OD-9 — ACP session management (list / load / activate) was removed as dead code; wiring it to a route + UI is a feature of its own (B5)

**State today:** three `AcpClient` wrappers had no product consumer. Item B5 **deleted two of them**
from `src/rigma/harness_mcode_acp.py`: `session_list` (`session/list`) and `session_load`
(`session/load`, the alternate wire spelling of the `session_resume` that `drive_control` already
calls at `:1339`). **`session_activate` was kept, because it is not dead**:
`tests/test_harness_mcode_acp.py::test_both_spellings_of_activate_work_and_the_notice_is_gated`
(`:471`) calls it against a real child process (`tests/fake_acp_server.py`) over real pipes and
asserts both spellings emit `mcode/session/current_session_update` — so the backlog's "zero callers
anywhere" is wrong for that one name. `session/list` and `session/load` had no caller in `src/`,
`tests/`, `frontend-v2/` or `src/rigma/data/`, and nothing reaches them dynamically: `drive_control`
is an explicit `if/elif` chain, `_CONTROL_OPS` (`:1181`) lists neither, and the client is never
dispatched through `getattr`.

**Options**
1. Leave them deleted. The ACP control plane keeps exactly the session operations it has a caller for.
2. Re-add `session_list` and wire it to a route + a UI session picker, so a chat can continue an mcode
   session Rigma did not start. Needs the route, the UI, and a decision about which listed session
   becomes the chat's.
3. Re-add `session_list` + `session_load`/`session_activate` as a full session-management panel: list,
   reattach, mark active. The largest option, and `session_activate`'s only observable effect is the
   `current_session_update` notice, so it needs a reason to exist beyond the panel itself.

**Evidence:** the deletion is in this commit; `harness_mcode_acp.py:1339` (`session_resume` is the
reattach that is actually wired); `tests/test_harness_mcode_acp.py:471`; the `STANDARD_METHODS`
entries `session/load` (`:69`) and `session/list` (`:71`) were **removed in item B5c**, and the guard at
`tests/test_acp_control.py:514-529` was made non-vacuous at the same time. The old fallback accepted a
method whose literal appeared anywhere in the file — including the table itself — so it could never
fail for *any* of the seven entries (measured: pre-filter `['session/list','session/load']`,
post-filter `[]`). Keeping the two rows would have made the table a false record of what the client
can send.

**Recommendation: option 1**, with option 2 as its own item if the owner wants a chat to continue an
mcode session Rigma did not start. Deleting was right because `session_list`'s only proposed consumer
was a route with no UI caller — the anti-pattern backlog row D4 complains about — and a wrapper nothing
calls is a false signal of capability. One honest caveat: `session_load` is *not* a capability
`session_resume` lacks (both reattach; only the wire method name differs), so its deletion removes a
fallback spelling, not a feature.

**If nothing is decided:** nothing changes; Rigma reaches mcode sessions only through `session/new` and
`session/resume`. This is not a blocker.

---

## OD-10 — The DSH approval audit trail is mounted but inert; making it real changes the system prompt (B8 / O4)

**State today:** `src/rigma/data/dsh/agent-capabilities.patch.yml` mounts `dsh-user-approval`
(policy `never`) and `dsh-permission-presets` — the missing hop O4 identified, and the rows are correct
and match the base bundle. But an independent verifier measured that this **does not yet make the
governance panel fill**, for three independent reasons, all now stated in the file itself:

- `sdk-minimal` sets `includeRuntimeContext: false`, which suppresses **every** runtime-context
  contribution — including the `approval:policy` entry `dsh-user-approval` registers — so the model is
  never told the policy. (The file previously claimed the opposite; that claim was false and is fixed.)
- `sdk-minimal`'s `sandbox-policy` is hardcoded `danger-full-access` and only a strictly-wider target
  escalates, so nothing ever asks and `approval/asked`/`approval/decided` stay quiet.
- `dsh-permission-presets` also needs a confining `ctx.shell`; sdk-minimal mounts the PTY `terminal`
  seam instead, and the packaged runtime closure carries neither `dsh-pwsh-sandbox` nor
  `dsh-bash-sandbox`.

The rows' only observable effect today is creating `ctx.approval` — a different deny *reason* if an ask
ever fires. The tests pin row presence and config values, so they pass on inert config; they are not
evidence the trail fills.

**Options**
1. **Leave it.** The rows are a correct, documented prerequisite; the trail stays quiet until a
   confining shell row exists. Cost: the panel still looks live and shows nothing.
2. **Also set `includeRuntimeContext: true`** so the model is told the policy. This changes the system
   prompt on **every turn** — a global behaviour change that also perturbs prompt-cache stability,
   which Rigma deliberately protects (`--parallel 2 --kv-unified` exists partly for that).
3. **Hide the governance panel** until the trail can fill, so the UI stops promising a signal it cannot
   show.
4. **Compose a sandboxed shell row** (`dsh-pwsh-sandbox`) so `permission/preset` can log — a packaging
   change, not a config one.

**Recommendation: option 1 now, option 4 when a sandboxed shell is packaged.** Option 2 should not be
taken merely to make a *panel* look alive; option 3 is UI honesty but removes a diagnostic that becomes
correct the moment option 4 lands.

**If nothing is decided:** the rows stay mounted and inert, exactly as measured, and the file says so.
---

## OD-11 — Should a shipped registry model be allowed to store launch defaults? (D2)

**State today:** `hangar.set_launch_defaults` refuses any spec whose `custom` is False
(`hangar.py:1123-1125`): "its launch settings are hand-authored and are not overwritten here". The D2
backend work did **not** change that — it added the missing GET, accepted `backend`, and made `null`
clear a field. So the new first-load dialog can read a registry model's defaults and shows `custom:
false`, but its Save is refused with 400.

**Options**
1. Keep the refusal. A shipped spec's launch settings are the program's evidence-backed choice; the
   dialog disables Save for `custom:false` and says why. A user who wants different numbers edits the
   model's file or installs their own copy.
2. Allow defaults for registry models too, stored in the user's custom overlay rather than in the
   shipped file. Consistent ("my machine, my launch"), but it introduces a second place a spec can be
   overridden, and a shipped spec can then disagree with what actually launched — the class of bug
   `list_models`' "running" row and the fit math already have to be careful about.
3. Allow it only for the shipped models whose settings are NOT evidence-backed (none are marked as
   such today, so this needs new metadata).

**Recommendation: option 1.** It is the status quo, it is what the recon recommended, and the cost is
one disabled button with a reason. Option 2 is a real product change and should be its own item with
its own merge/precedence rules.

**If nothing is decided:** the refusal stands and the dialog explains it. Not a blocker.

---

## OD-12 — What happens to a question after its 5-second window closes? (B4b)

**State today:** item B4 puts mcode's `ask_user` on the existing permission channel with
`kind: "question"` and a 5-second bound (`QUESTION_WAIT_SECS = 5.0`, OD-8). When the window closes
the server returns a real **decline** — but it emits **no `approval/decided` event**. The frontend
half (B4b, merged) now draws a real question form; because no decided-event arrives, **the row stays
`awaiting` and the form stays clickable after the question has already been declined.** A click then
hits the route's 409 and shows up as `lastError`.

**Options**

1. **Emit a server-side expiry event** (`approval/decided` with `decision: "expired"`) when
   `QUESTION_WAIT_SECS` elapses, and have the UI fold the row to a disabled "expired" state. The
   client then never guesses the clock. **Cost:** a small server change plus the fold; the event is
   one more thing on the wire that the harness must tolerate.
2. **A client-side countdown** that disables the form after 5 s. No server change. **Cost:** the
   client guesses the server's clock and the request's start time, so it will sometimes disable a
   question that is still live and sometimes leave a dead one clickable — the exact class of
   wrong-state bug this program has been removing.
3. **Leave it.** The window is 5 s and a late click is harmless (a 409 surfaced as a notice).

**Recommendation: option 1.** The server knows when it declined; making it say so is the honest
version, and it is the only one that cannot drift. Option 3 is acceptable if the deadline is close,
but it should then be written down as a known rough edge rather than left implicit.

**If nothing is decided:** the form stays clickable for up to 5 s after the decline and a late click
reports the route's 409. Not a data-loss risk; a rough edge.

---

## OD-13 — Should a remotely-streaming chat have a stop affordance? (D3b)

**State today:** D3b (merged) makes a reloaded chat read the server's `streaming`/`partial` state,
suppress the false "interrupted" notice, show a rail dot, and poll. There is **no way to stop a turn
this tab did not start** — the client's stop control only exists for a turn the tab owns. A user who
reloads mid-generation can watch it but not cancel it.

**Options**

1. Add a stop control for a remotely-owned streaming turn (POST the existing stop route for that
   session id). **Cost:** small; needs care that two tabs do not both believe they own the stop.
2. Leave it: reloading is rare, and the turn finishes on its own.

**Recommendation: option 1** if a follow-up wave has room — it is a small, contained UI addition on
an existing route, and "I can see it but cannot stop it" is the kind of gap the owner asked to have
covered. Otherwise option 2 with the gap recorded.

**If nothing is decided:** the turn runs to completion and the tab only observes it.

---

## OD-14 — `taskkill /T` walks reused parent pids; the real fix is a Job Object per harness child (DR9)

**State today:** on Windows, killing a harness process tree uses `taskkill /T` (`tools.py:4321`), which
walks `ParentProcessId` links. Rigma's own server is spawned **detached** by a CLI process that then
exits (`cli._spawn_detached`), so the server's recorded parent pid is free for reuse. If a harness
descendant is assigned that pid, `taskkill /T` could treat the server (and `llama-server` under it) as
part of the tree. The **root** pid is safe — the `Popen` handle pins it.

**Options**

1. **A Windows Job Object per harness child** (`CreateJobObject` + `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE`,
   assign the child, terminate the job instead of walking pids). This is the correct fix: it is immune
   to pid reuse because the kernel tracks the members, not the links. **Cost:** real Windows API work
   (`ctypes`), a new failure surface on a path that currently works, and it cannot be exercised
   meaningfully on a non-Windows CI. It is a contained module plus tests against the API surface.
2. **Leave it.** The scenario needs a pid to be reused *and* assigned to a harness descendant inside a
   narrow window. It is a plausible-but-unwitnessed hazard, and the guard that refuses to signal
   Rigma's own process group (B1) already covers the POSIX half.

**Recommendation: option 2 for now, option 1 as a scoped item of its own.** It is design work, not a
one-liner, and the current behaviour is the status quo the owner has been running. Recording it here
is the point: the next session should not rediscover it as a mystery.

**If nothing is decided:** `taskkill /T` keeps walking parent links. No observed incident.
