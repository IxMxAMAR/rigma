# What changed in Rigma today — a summary for the owner

## Tonight, after you went to sleep

You said you would trust the recommendations, so I implemented the ones that fit in the evening and
wrote the rest down as accepted-and-scheduled. This is the short version of what changed for you.

**A late review found a real hole, and it is closed.** After you went to sleep, an independent review of
the backup/restore and file-tool work found that a new chat using the default home workspace could be
talked into writing a startup script, a `.gitconfig` or a PowerShell profile — a way to make something run
later — with no permission asked. That is fixed: the file tools now refuse those specific locations
whatever the chat is allowed to do. The first attempt was rejected by its own checker, which found two ways
around it (a trailing dot on a folder name, and a network-share spelling of the same path); both are closed
and re-checked, and no way around it survived. The review's three smaller findings are also fixed: moving a
file out of your workspace now needs the same permission as writing there; the list of folders a new chat
may write into can no longer contain your whole home folder or a drive root; and restoring a backup now
clears the "always allow" trust on any method it rewrites, so a backup from another machine cannot smuggle
in a trusted method that writes files. The restore card now says exactly what a restore replaces and lists
which methods it deleted. One more way around the file-tools fix was caught by a second look: the tools checked the folder a copied
file was going into but not the file's own name, so a file named `.bashrc` could still be dropped into your
home folder — that is closed as well. And the two crash-safety items I first listed as next-session work
are done: a restore interrupted by a crash is recovered from a small on-disk journal when Rigma next
starts, a save can no longer race a restore, and a crash right after a restore finished no longer undoes it.

**Restoring a backup now actually restores.** The Backup/Restore card promised that a restore
"replaces the whole store — anything not in the file is gone". It didn't: it replaced your memories
but quietly merged settings and methods, so a method you created after the backup survived and an old
setting stayed. It now really replaces all three. Because that can delete things you made after the
backup, it still rolls back completely if anything fails halfway — an independent check forced a
failure mid-restore and confirmed your settings, every method file and your memory came back
byte-for-byte identical.

**You can now clear a stale document index.** The credential filter we added earlier only applies when
a document is indexed, so an index built before it could still hold a `.env`, an SSH key or a browser
cookie database. There is now a **rebuild index** button in the Grounding card. It stops the indexer,
deletes the old index and rebuilds it from your folders with the filter in place. It never runs on its
own — you press it.

**An agent can no longer copy a file into your Startup folder just because it can write somewhere.**
Copying and moving files outside your workspace already needed an explicit grant; now there is also a
list of allowed destinations, seeded from the folders you already work in, so the grant is no longer
all-or-nothing. An independent reviewer tried 32 ways to escape that list — `..`, prefix lookalikes,
directory aliases, UNC paths, short names and case tricks — and none worked.

**The two failing tests are fixed, and so is what they were hiding.** The suite had two failures that
had been blamed on your locked `D:` drive. They were real: when Windows cannot read a path (locked
drive, offline share, permission error), Rigma's mission compiler was aborting and the run started
with no plan at all, silently. It now falls back the way it was designed to, and the tests build their
own missing paths so they no longer depend on one drive's state.

**A smaller one an independent reviewer caught while checking the above:** asking to view an image
whose "file" is actually a folder returned success and sent the model a picture it could never see. It
is refused properly now.

**What I did not do, and why.** The LoRA feature and the ACP client are accepted but are their own
programs — one needs a live agent session, which this run is not allowed to start. The two GPU fit
measurements wait for your card. Making the DSH approval panel real needs a system-prompt change you
should decide on, and the Job Object for Windows process trees is accepted as its own item. Nothing
changed about your default run profile: the chooser was already there, and it now has a test so it
cannot quietly disappear.

**How it was checked — and one thing you should know.** Every change above was made by one agent and
checked by a *different* one that did not write it, each required to name a state the test does not
cover and try the code there. The full suite now passes: **3513 passed, 3 skipped, 0 failed** in 8
minutes 41 seconds, and the two `D:` failures are gone. But I have to be straight about the runs before
that one: two of my four attempts stalled part-way (a rare stall this project has seen before, in a
known-flaky lifecycle test) and one crashed while I was instrumenting it. The stall is intermittent and
pre-dates tonight, but I could not prove it away, so it is written down as the first thing to re-test.
Nothing in the product is broken by it — but you should not be told the suite is rock-solid when it is
merely reliable.

---

*Plain language. Written 2026-09-30 by the review session working on
`review/deep-audit-2026-09-22`; this version covers the whole day, including the evening
re-verification wave. Every claim here is traceable to a commit and to the documents under
`docs/review/program-2026-09-30/`. **Nothing in this file was measured on your GPU tonight** — the
machine was never allowed to load a model, and every memory and speed figure below is either quoted
from a log that already existed or labelled as a prediction.*

---

## 1. What Rigma can do now that it could not do this morning

**You can set the launch knobs per model — and the fit still has the last word.** `-b` (logical
batch), `-ub` (physical batch) and `-ngl` (layers offloaded) are now real, first-class settings, in
the API, in the CLI and in the launch dialog. Before today they were constants in the resolver: the
only way to change the batch size was to edit Rigma. Setting an impossible pair is now refused with
the *server's own* sentence, and the dialog shows you the fit's answer next to your choice.

**`rigma up` restores your KV cache instead of always cold-starting.** The save/restore path existed
but was never wired into the launch — every start re-computed the prompt cache from scratch. It now
restores it, and if the restore is refused you get a warning and a sentence instead of silence.

**You can stop a chat that is streaming on another device.** The stop control used to be tied to the
turn the page had started; a chat streaming from a phone or another tab had no way to be stopped from
here. There is now a real server route for it, and a request that arrives late cannot stop the *next*
turn by accident.

**A question that times out says so.** When an agent asked you something and you did not answer within
five seconds, the question was declined but the UI was never told — the form stayed clickable and a
late click produced an error. The server now emits a "this expired" event and the form goes disabled;
when you do answer, the row shows an answered tick.

**The launch dialog tells you what the fit actually answered, and refuses to invent a number.** It
used to guess a context window when it could not read one, offer you steps that were impossible, store
them, and let the backend quietly clamp them. It now says "unknown window" and disables the steps
instead.

**A backup/restore card that tells the truth.** It used to say "restore" replaces your settings.
Measured: it replaces memory and *merges* settings and methods. The card now says so, and the deeper
question is on your desk (OD-15).

**The budget surface names the numbers it is guessing at.** The VRAM estimate for the hybrid model's
recurrent state used to be invisible — you could not tell a considered estimate from a placeholder.
The plan and the UI now name it, with its provenance. The KV-cache charge now carries an explicit
"this geometry is only partly known" flag rather than being presented as a measurement.

**The plan-versus-actual report is honest again.** See §2 — this one was broken *by its own fix*,
twice, and is now fixed properly.

---

## 2. The bugs that would have bitten you — in the order they would have hurt

These are the ones worth reading. Each was fixed with a test that fails without the fix, and — for
every item in waves 16 and 17 — checked by a **separate** agent that did not write it.

### Your chat history could be silently damaged
- **A reply that failed to save could erase someone else's messages.** When a save failed, the code
  fell through into compaction holding a stale snapshot of the conversation. Compaction then wrote
  that stale snapshot back — deleting anything written in between — and marked the unsaved reply as
  finished. Fixed, and the "not saved" path now stops before compaction.
- **A transient unreadable `run.json` wedged a run forever.** The run's driver died, the run stayed
  `running` with its slot claimed, and every control lied about why: pause said "no driver", restart
  said "running", a new run said "already active" — and only Stop cleared it. **This was found by the
  test suite, not by reading code.** Fixed.
- **Worse: a run whose file could not be *read* was overwritten with a 4-field stub.** That destroyed
  the session id, so "restart" answered *"the run's chat session was deleted"* — permanently. The
  test that was supposed to protect this pinned the destructive overwrite as the desired behaviour.
  Fixed, and the boot-time reaper now reconciles orphaned runs *without* touching an unreadable file.
- **A driver could be mistaken for a live one after its process id was recycled.** Each run records
  *which process* drives it, but the check that read that record did not actually run on the path that
  mattered, and an identity that could not be read at all could wedge the run. Both fixed.

### The plan could be wrong on your exact model
- **The fit did not charge the hybrid model's recurrent state.** Your 27B is 48 recurrent layers of 64;
  their state is 149.62 MiB per sequence, and you run two sequences. The plan under-reserved VRAM by
  ~300 MiB on the model you actually use. Fixed, with the arithmetic checked against your own load log.
- **"We could not tell" was stored as "zero".** A missing measurement was charged as 0 MiB, which is
  indistinguishable from a real 0 and is the unsafe direction. Fixed.
- **The compute-buffer charge was ~1 GiB too low at a large physical batch.** The constant `150 MiB`
  was a *differenced* quantity; your own log says the engine really uses **410.28 MiB** at ubatch 512.
  Scaling that difference up to ubatch 2048 under-reserved by **1041 MiB**. Fixed so the measured term
  is what scales.
- **A plan-versus-actual warning fired on every launch of a healthy load — twice, and then a third
  time.** First the split baseline was wrong; then its fix compared the engine's *device* VRAM against
  the plan's *whole-file* prediction, so a perfectly healthy 33-of-65-layer spill was reported as a
  divergence (6017 vs 9038 MiB, −33 %). That was fixed by refusing to compare a non-device-resident
  load — but the two sides of the comparison were then still on **different bases**: the engine's
  device figure includes its compute/reserve buffers and the plan's deliberately excludes them. A
  512 MiB tolerance hid that until the spill was deep, and then a healthy deep spill was reported as
  broken again. **Fixed properly in the evening** by charging the plan the same compute term the
  resolver already reserves against — the two sides now agree to within a rounding error — **not** by
  widening the tolerance, which would have hidden real divergences too. A genuinely wrong figure still
  fires, at every spill depth, and that is now pinned by a test.
- **The plan recorded the placement you asked for, not the one it used.** On a model that has to spill
  layers to system memory, the recorded figure was the request rather than the outcome. It now records
  what it actually did, so the comparison above is meaningful on your MoE model too.
- **Two settings that contradict each other could reach the engine.** The engine silently clamps the
  physical batch to the logical batch; Rigma checked that pair on most paths but not on the one that
  merges the command line into a plan whose logical batch came from a **stored calibration row**. The
  plan then charged the batch you *asked* for while the engine ran the smaller one. It is now refused
  with the same clear sentence the other paths use. (The verifier also caught that the fix's own
  commit message overstated how big a plan this could mislead — the tolerance is not a flat 512 MiB —
  and that correction is recorded rather than quietly dropped.)

### A sweep could quietly make every later launch worse
- **The calibration sweep could crown "disable attention rotation" on speed alone.** A trial that
  turned off a quality feature won on tokens/sec, was written into `~/.rigma/calibration.json`, and
  was then applied to **every subsequent launch** — silently trading quantized-KV quality for speed
  (measured in the repo's own notes: +2.42 % perplexity with it off versus +0.19 % on). Fixed on both
  the write path and the read path, and the Windows-specific case-sensitivity hole in that gate is
  closed too (`getenv` on Windows ignores case; the pinned source reads the variable directly).
- **…and the same guard was wrong one lever over.** It checked the *cache type* on the setting being
  trialled rather than on what the trial would actually run with. A calibration row can put `q4_0` on
  the plan, and then every trial of a tools-capable model ran the very cache the guard exists to
  avoid. Fixed, and a plan that already carries `q4_0` now **says so** instead of quietly shrinking
  the sweep.
- **A sweep trial could be launched with settings the sweep never trialled.** The trial's flags were
  built in two places and could drift apart. There is now one construction, and the test pins which
  function called it rather than just its name.
- **A tune that never happened was cached as if it had.** When there is nothing to sweep, Rigma now
  remembers that so it stops warning on every launch — but the first version of that memory keyed the
  decision without the settings that made it a no-op, so a plan that *later* became sweepable was
  skipped forever. **Caught by its verifier the same hour** and fixed: the cached decision now records
  *why* it was made and is re-checked on every launch.

### A question card could lose an answer, or send an empty one
- **Two questions at once, and one answer was lost.** The server kept one question slot per chat, so a
  second question overwrote the first — and the first one's cleanup then freed the *second* one's
  slot. The first answer was refused and the still-live second question became unanswerable. Fixed by
  keying each question on its own request id.
- **…and the first fix dropped a safety check.** With slots keyed by request id, an answer for one
  chat was accepted through another chat's route. **Caught by the verifier the same hour** and closed;
  the permission path (which is a different thing) was verified untouched.
- **The question form could submit a required field as absent.** The "is the form ready?" check
  counted the rows on screen while the code that builds the answer dropped blank rows — so a required
  list containing only blank rows passed validation and posted an answer with that key missing
  entirely. One function now decides whether a field is empty, so the two cannot disagree. Verified
  against 22 hand-built shapes plus a 4,000-case random fuzz: 198 bad submissions before, 0 after.

### Processes and files
- **On Windows, stopping an ACP agent left the agent and its subagents running.** The stop killed the
  `.cmd` shim and nothing else. **Your platform.** Fixed.
- **A boot-time cleanup could mark a *live* run as interrupted.** If you ever run two Rigma servers on
  one home directory — the CLI's own "pass a different `--port`" invites it — the second one's startup
  sweep would write `interrupted` into a run the first one was still driving. Each run now records
  *which process* is driving it, and the sweep leaves another live process's run alone.
- **A KV cache could be restored under the wrong engine build** when the engine's version could not be
  measured — it fell back to the *pin's* identity instead of saying "unknown". Fixed.
- **A grep pattern could wedge the server.** An exponential regex could not be interrupted even with
  Ctrl-C. Now bounded by wall clock and by pattern size.
- **A glob `?` matched a whole run instead of one character.**

### Things that made other people's work fail
- **A flaky test.** `test_a_pause_does_not_burn_the_clock` failed about 1 run in 10. It was a *test*
  race, not a product bug — the test set up the exact state the pause bound stalls on and released the
  engine turn too early. A flaky test makes everyone else's work look broken; it is fixed, and the
  full-suite run below is the proof.
- **Two full test-suite runs at once deadlock, and the suite leaks its fake servers.** Two overlapping
  runs both froze (not slow — *stopped*, 0 CPU over 25 s) with **72** orphaned `fake_acp_server.py`
  processes. Alone, the suite is green. **Diagnosed and fixed:** the shared resource was not one fixed
  TCP port but **six** of them (`11594`–`11599`) — test fixtures that bound a literal port instead of
  letting the kernel choose one, so a second run's child died on "address already in use" and the
  second run's readiness check was then answered by the *first* run's server, meaning its tests
  silently measured another process. All six now let the kernel choose. On top of that, a second
  **full** suite now refuses to start and exits with a clear message rather than competing for the
  machine; deliberately, running a *single test file* is not blocked, because that is what everyday
  work does. Verified end to end (a second full run exits 4, a single file runs fine, a run whose
  owner died is taken over).

---

## 3. How it was checked — and the one weakness, and how it was closed

- For **waves 0–15**, every change was verified by a **separate** agent that did not write it, on a
  test that was first observed **failing** on the unmodified code.
- **From ~16:20 UTC that stopped.** Your PC shut down mid-run at 15:55 UTC; when the session resumed,
  it had lost the ability to start subagents. Waves 14–15 were therefore written *and* checked by the
  same agent. That was written down here, in `HANDOFF.md` and in `verify-w15a.md` rather than glossed
  over — and **the first thing the evening session did was close it.** Every single-sourced verdict
  merged after the power cut was handed to a **fresh independent verifier**, each of which also had to
  answer the program's standing question (*"name one realistic state the test does not put the code
  in, and check the fix there"*). That re-verification wave found real defects: one fix (the KV
  "geometry unknown" flag) **failed** its re-verification outright — it flagged every pure-Mamba model
  as unknown — and had to be fixed and re-verified before it could stay. **There are no single-sourced
  verdicts left in the tree.**
- **Four independent deep reviews** were run over work that had already passed those checks. They
  found **23 real defects**, including several introduced *by a fix* and several that were the same
  mistake one lever over. That is why the reviews exist: the tests had been shaped to the fixes. The
  evening wave is the clearest proof — **two of the evening's own fixes were caught, by their
  verifiers, introducing a regression the same hour**, and both were fixed and re-verified before the
  night ended.
- `ruff` clean; the frontend builds reproducibly, type-checks, and its **709** tests pass.
- **The evening session ran its own verifiers, not just implementers.** Nineteen merges in the
  evening, each behind an independent verifier that had to make the new test fail on the unmodified
  code first. Three of those verifiers found something the implementer had got wrong: one caught a
  regression in the fix it was checking, one caught a *wrong claim in the commit message itself*, and
  one found that the route the Head Agent's brief named was not reachable — and found the real one
  instead.
- The full Python suite was re-run after the last merge, alone, with nothing else touching the tree:
  **3,477 passed, 3 skipped, 5 deselected, 2 failed** in 8 m 51 s. **Both failures are pre-existing
  and environmental, not regressions:** `test_view_image_missing_file` and
  `test_compiled_spec_seeds_the_plan` both use `D:/...` paths, and **`D:` is a BitLocker-locked drive
  on this machine** (it does not even appear in `Get-PSDrive`), so Windows answers with a BitLocker
  error instead of "no such file". Both were reproduced **identically at the base commit `7ea1827`**,
  and neither file was touched by this program. So the whole program — **59 merges and 202 other
  commits** on top of the base — has **zero regressions**.
- **One honest caveat about how that suite was run.** Twice, a full-suite run appeared to *freeze* —
  0 CPU for over a minute, with 36 of the suite's fake ACP servers still alive — which is exactly the
  symptom of the deadlock in §2. In both cases the run's output was being buffered rather than read;
  the identical command run with its output read live, or written straight to a file, **completed
  three times out of three** with the same result. So the most likely explanation is an artifact of
  *this session's* output capture (a full pipe buffer blocking the writing process), **not** a Rigma
  deadlock — but it is written down here rather than dismissed, because a future session should not
  mistake it for a new bug, and because the six-port fix's own guarantee is worth re-testing under
  load.

---

## 4. What is waiting on the GPU (nothing here was run tonight)

The single plan is `docs/review/findings-r3/37a-262k-verification-plan.md`; the consolidated list is
folded into it. The ones that matter most:

**Launch your 27B hybrid at 2 slots and capture the buffer lines.** That one run settles four separate
questions at once: where `token_embd` really lands, what the compute buffer really is at 2 slots,
whether the split count the code expects is right, and — if you also run it at ubatch 1024 or 2048 —
whether the new compute-buffer scaling is a measurement instead of a prediction.

**Run one launch with `--ubatch` set, above and below `-b`.** The plan-versus-actual comparison now
charges the compute buffer at the physical batch the launch actually used, read back from the run's
record — but Rigma cannot know whether the engine *clamped* that value, so one real load would confirm
the two agree.

Also waiting: the 262K-context verification itself; one live mcode session (to answer `ask_user`
end-to-end and to see the Windows ACP stop against a real agent); vLLM, which needs Linux/WSL; a
multi-GPU box; and a quality run before the fork's K-cache bias correction could ever be wired in.

---

## 5. What is waiting on you

Sixteen decisions, each written up with options, evidence and a recommendation, in
`OWNER-DECISIONS.md`. **Two were resolved today** (OD-12, the question-expiry event, and OD-16, the
concurrent-suite deadlock) because the recorded recommendation was clear and cheap. The ones that
would change behaviour if you chose differently:

- **OD-15** — should "restore a backup" really *replace* your settings and methods, or is the merge
  fine now that the card says so?
- **OD-14** — `taskkill /T` can hit a reused process id; a Job Object per child is the real fix.
- **OD-2** — how confined should `view_image` / `copy_files` be?
- **OD-11** — may a model shipped in the registry store your launch defaults?
- **OD-5** — the GPU measurements above.
- **A new one, recorded but not numbered:** the calibration sweep's quality-lever gate cannot see a
  quality setting exported in your shell's environment (the engine inherits it, but the sweep's gate
  does not). The obvious fix would make the sweep drop **every** candidate config, so it was
  deliberately **not** applied — it is a measurement caveat, and the right fix is an owner decision.

Nothing in this program rewrote history: the base commit `7ea1827` is untouched, every commit carries
your identity, and nothing was pushed.
