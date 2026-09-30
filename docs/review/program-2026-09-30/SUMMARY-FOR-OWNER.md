# What changed in Rigma today — a summary for the owner

*Plain language. Written 2026-09-30 by the review session that has been working on
`review/deep-audit-2026-09-22`. Every claim here is traceable to a commit and to the documents under
`docs/review/program-2026-09-30/`; nothing in this file was measured on your GPU tonight.*

---

## 1. What Rigma can do now that it could not do this morning

**You can set the launch knobs per model — and the fit still has the last word.** `-b` (logical
batch), `-ub` (physical batch) and `-ngl` (layers offloaded) are now real, first-class settings, in
the API, in the CLI and in the launch dialog. Before today they were constants in the resolver: the
only way to change the batch size was to edit Rigma. Setting an impossible pair is now refused with
the *server's own* sentence, and the dialog shows you the fit's answer next to your choice.

**You can stop a chat that is streaming on another device.** The stop control used to be tied to the
turn the page had started; a chat streaming from a phone or another tab had no way to be stopped from
here. There is now a real server route for it, and a request that arrives late cannot stop the *next*
turn by accident.

**The launch dialog tells you what the fit actually answered, and refuses to invent a number.** It
used to guess a context window when it could not read one, offer you steps that were impossible, store
them, and let the backend quietly clamp them. It now says "unknown window" and disables the steps
instead.

**A backup/restore card that tells the truth.** It used to say "restore" replaces your settings.
Measured: it replaces memory and *merges* settings and methods. The card now says so, and the deeper
question is on your desk (OD-15).

**The budget surface names the numbers it is guessing at.** The VRAM estimate for the hybrid model's
recurrent state used to be invisible — you could not tell a considered estimate from a placeholder.
The plan and the UI now name it, with its provenance.

**The plan-versus-actual report is honest again.** See §2 — this one was broken *by its own fix* and
is now fixed properly.

---

## 2. The bugs that would have bitten you — in the order they would have hurt

These are the ones worth reading. Each was fixed with a test that fails without the fix, and each fix
was checked by a **separate** agent that did not write it.

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
- **A plan-versus-actual warning fired on every launch of a healthy load — twice.** First the split
  baseline was wrong; then its fix compared the engine's *device* VRAM against the plan's *whole-file*
  prediction, so a perfectly healthy 33-of-65-layer spill was reported as a divergence (6017 vs 9038
  MiB, −33 %). Fixed: a load that is not device-resident is reported as *not comparable*, never as
  divergence — while a genuinely wrong figure still fires.

### A sweep could quietly make every later launch worse
- **The calibration sweep could crown "disable attention rotation" on speed alone.** A trial that
  turned off a quality feature won on tokens/sec, was written into `~/.rigma/calibration.json`, and
  was then applied to **every subsequent launch** — silently trading quantized-KV quality for speed
  (measured in the repo's own notes: +2.42 % perplexity with it off versus +0.19 % on). Fixed on both
  the write path and the read path, and the Windows-specific case-sensitivity hole in that gate is
  closed too (`getenv` on Windows ignores case; the pinned source reads the variable directly).

### Processes and files
- **On Windows, stopping an ACP agent left the agent and its subagents running.** The stop killed the
  `.cmd` shim and nothing else. **Your platform.** Fixed.
- **A KV cache could be restored under the wrong engine build** when the engine's version could not be
  measured — it fell back to the *pin's* identity instead of saying "unknown". Fixed.
- **A grep pattern could wedge the server.** An exponential regex could not be interrupted even with
  Ctrl-C. Now bounded by wall clock and by pattern size.
- **A glob `?` matched a whole run instead of one character.**

### Things that made other people's work fail
- **A flaky test.** `test_a_pause_does_not_burn_the_clock` failed about 1 run in 10. It was a *test*
  race, not a product bug — the test set up the exact state the pause bound stalls on and released the
  engine turn too early. A flaky test makes everyone else's work look broken; it is fixed.
- **Two full test-suite runs at once deadlock, and the suite leaks its fake servers.** Found tonight:
  two overlapping runs both froze (not slow — *stopped*, 0 CPU over 25 s) with **72** orphaned
  `fake_acp_server.py` processes. Alone, the suite is green. This is a defect in the *harness*, not in
  the product, and it is on your desk as **OD-16**.

---

## 3. How it was checked

- Every change was verified by a **separate** agent that did not write it, on a test that was first
  observed **failing** on the unmodified code.
- **Three independent deep reviews** were run over work that had already passed those checks. They
  found **19 real defects**, including one introduced *by a fix* and two that were the same mistake
  one lever over. That is why the reviews exist: the tests had been shaped to the fixes. The counter-
  measure is now a standing rule (*"name one realistic state the test does not put the code in, and
  check the fix there"*), written into the guidance file and every verifier brief.
- `ruff` clean; the frontend builds reproducibly, type-checks, and its 656 tests pass.
- The full Python suite was re-run after the last merge; the result is recorded in
  `.scratch/orchestrator/WAVES.log` and in `STATUS.md`.

---

## 4. What is waiting on the GPU (nothing here was run tonight)

The single plan is `docs/review/findings-r3/37a-262k-verification-plan.md`; the consolidated list is
now folded into it. The one that matters most:

**Launch your 27B hybrid at 2 slots and capture the buffer lines.** That one run settles four separate
questions at once: where `token_embd` really lands, what the compute buffer really is at 2 slots,
whether the split count the code expects is right, and — if you also run it at ubatch 1024 or 2048 —
whether the new compute-buffer scaling is a measurement instead of a prediction.

Also waiting: the 262K-context verification itself; one live mcode session (to answer `ask_user`
end-to-end and to see the Windows ACP stop against a real agent); vLLM, which needs Linux/WSL; a
multi-GPU box; and a quality run before the fork's K-cache bias correction could ever be wired in.

---

## 5. What is waiting on you

Fifteen decisions, each written up with options, evidence and a recommendation, in
`OWNER-DECISIONS.md`. The ones that would change behaviour if you chose differently:

- **OD-15** — should "restore a backup" really *replace* your settings and methods, or is the merge
  fine now that the card says so?
- **OD-16** — fix the concurrent-suite deadlock and the leaked processes, or just document it?
- **OD-12** — should a question that times out tell the client it expired, instead of just going quiet?
- **OD-14** — `taskkill /T` can hit a reused process id; a Job Object per child is the real fix.
- **OD-2** — how confined should `view_image` / `copy_files` be?
- **OD-5** — the GPU measurements above.
- **OD-11** — may a model shipped in the registry store your launch defaults?

Nothing in this program rewrote history: the base commit `7ea1827` is untouched, every commit carries
your identity, and nothing was pushed.
