# HANDOFF — Rigma improvement program, 2026-09-30

**Read this first if you are a new session with a different model.** It is written to be
self-sufficient: it names the commands, the paths, the rules and the state, and it does not assume
you share any context with the session that wrote it.

- Repo: `C:\ComfyUI\RD\rigma-review` — the review fork. **Never touch `C:\ComfyUI\RD\rigma`** (that
  is the owner's live checkout; only its `.venv\Scripts\python.exe` is used, as the interpreter).
- Integration branch: `review/deep-audit-2026-09-22`. No `git push`, ever.
- Program docs: `docs/review/program-2026-09-30/` — `BACKLOG.md`, `STATUS.md`, `OWNER-DECISIONS.md`,
  `HANDOFF.md` (this file). `docs/*` is git-ignored, so these need `git add -f`.
- **Steering channel: `.scratch/orchestrator/GUIDANCE.md` — RE-READ IT BEFORE EVERY WAVE.** Newer
  entries override the brief. This is not optional: on 2026-09-30 entries 4 and 5 sat unread for
  40 minutes and wave 6 was started without them, and the orchestrator had to inject a note into
  `STATUS.md` to get them actioned. Entries 4+5 were a **real defect in already-merged work** (A17
  would have flagged the owner's healthy log on every launch). If `.scratch/orchestrator/STOP`
  exists, finish the commit in hand, write this hand-off, end.
- Wave log: `.scratch/orchestrator/WAVES.log`. Recon reports: `.scratch/orchestrator/recon-*.md`.
  Implementer reports: `.scratch/orchestrator/impl-*.md`. Verifier reports:
  `.scratch/orchestrator/verify-*.md`. Verifier template: `VERIFIER-TEMPLATE.md`.

## How to run anything

```powershell
$env:PYTHONPATH='C:\ComfyUI\RD\rigma-review\src'      # or the worktree's src
& C:\ComfyUI\RD\rigma\.venv\Scripts\python.exe -m pytest -m "not hardware" -q
& C:\ComfyUI\RD\rigma\.venv\Scripts\python.exe -m ruff check src tests
```

At most **3 pytest processes across all agents**; use the shared semaphore so this is enforced
rather than remembered (run it from your worktree root — it sets `PYTHONPATH` to that worktree's
`src` and uses the owner's venv):

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File C:\ComfyUI\RD\rigma-review\.scratch\orchestrator\run-pytest.ps1 tests/test_foo.py -q -m "not hardware"
```

**Semaphore notes.** The script has **no `param()` block** and reads `$args`; it had one, and
`powershell.exe -File` then bound `-m` as a *parameter name*, so every run that passed
`-m "not hardware"` died with *"A parameter cannot be found that matches parameter name 'm'"* —
i.e. the §0.1 marker was silently unpassable. The script now also **appends `-m "not hardware"`
itself when the caller did not pass a marker**. `addopts` in `pyproject.toml` is only `-q` (so `-q`
on the command line makes it `-qq`, which **suppresses the pass count** — use `-o addopts=` when you
need a readable summary). If a lock is held by a **dead** PID (`Get-Process -Id`), remove only that
stale lock directory; a lock held by a live PID belongs to another agent — leave it.

**Environment notes (verified 2026-09-30).** `pwsh` (PowerShell 7) is **not installed** on this
host — only Windows PowerShell 5.1, and `powershell -File` needs `-NoProfile -ExecutionPolicy
Bypass`. Use the invocation above, not `pwsh -File`. (The Head Agent's own `pwsh` tool works because
the harness invokes it directly; a subagent shell does not have it on PATH.) `Get-Date -AsUTC` does
**not** exist here — use `[DateTime]::UtcNow`. `Get-ChildItem .scratch -Recurse` can time out;
scope it.

Worktrees: `git worktree add .scratch/wt-<name> -b impl/<name> review/deep-audit-2026-09-22`. Merge
serially into the integration branch from the main tree. No links or junctions; never
`git worktree remove --force`; never `git clean -d/-x`. **Frontend work happens in the MAIN tree**
(one agent at a time) because `node_modules` exists only there.

Baseline at program start (2026-09-30, base `7ea1827`): **2935 non-hardware tests, all pass, exit 0**;
`ruff check src tests` clean.

## The rules that cannot be broken (§0 of the brief — full text in `.scratch/orchestrator/STANDING-RULES.md`)

1. **No GPU work and no model loading, ever.** No `llama-server`/`llama-bench`/`llama-cli`/
   `llama-fit-params`, no `rigma up/serve/bench/sweep/calibrate/switch`, no vLLM/Ollama/`vulkaninfo`,
   no code that launches an engine, no live harness turn (no dsh/mcode model calls). Fakes, fixtures
   and protocol handshakes only. Every pytest run uses `-m "not hardware"`.
2. Work only in `rigma-review`; the owner's `.rigma` config is read-metadata-only; no `pip install`,
   no new dependencies, no `git push`, no links/junctions, no destructive host commands, no process
   killing from the shell.
3. Never read the owner's prose: chat/session bodies, `/slots` prompt text, chapter/story files, RAG
   document contents. Metadata, shapes and counts only.
4. Research via DuckDuckGo through `web_fetch` (`web_search` is unconfigured). Source files from
   `raw.githubusercontent.com` at a pinned commit — PrismML fork `PrismML-Eng/llama.cpp` @ `87268f77`,
   mainline `ggml-org/llama.cpp` @ `b9867`. Never clone.
5. Nothing can be measured in this run: a speed/memory claim cites an existing measurement or is
   labelled **PREDICTION** with its arithmetic.
6. One coherent change per commit, with a test that fails before and passes after; a reason, a cause
   and a foreseeable resolution. "If it works and the margin for improvement is not that high, leave
   it be" — write down what you skipped and why.
7. Targeted test files only; the FULL suite is run by the Head Agent alone, once per wave. `ruff`
   clean. Frontend changes need `npm.cmd run build` + `npx.cmd vitest run` + `npx.cmd tsc --noEmit`,
   with the rebuilt `src/rigma/data/ui_v2` in the same commit, in the MAIN tree (never link
   `node_modules` into a worktree). `renderToStaticMarkup` (react-dom/server) works with **zero new
   deps**; `@testing-library/react` is forbidden.
8. Owner decisions go to `OWNER-DECISIONS.md` with a recommendation — do not make them.
9. Subagents run DeepSeek-V4.1-Flash on tokenjuice.

**Additions made during the run (from `GUIDANCE.md` / incidents):**

10. **Never override the git author.** Plain `git commit` only — no `-c user.name`, no `-c user.email`,
    no `GIT_AUTHOR_*`/`GIT_COMMITTER_*`. The branch may be published, so every commit must carry
    `IxMxAMAR <officialamrendrasingh@gmail.com>`. Check with `git log -1 --format='%an <%ae>'`; audit
    the whole branch with
    `git log --branches --not 7ea1827 --format='%an <%ae>' | Select-String '@local|@rigma\.local'`.
    (The orchestrator broke this early on; the affected commits were re-authored in a metadata-only
    rewrite — see `STATUS.md`.)
11. **Never use `git stash`.** It is repository-wide, not per-worktree: one agent's `stash push` was
    consumed by another agent's `pop`, which then applied a foreign stash into a third worktree. Use
    `git show HEAD:path` or a copy for before/after evidence. Commit on your own branch instead.
12. **Re-read `.scratch/orchestrator/GUIDANCE.md` before every wave** (see the top of this file).

Also: prefer `git commit -F <file>` for messages. **Check the exit code of every command in a
sequence** — a merge was once committed with a conflict marker still in `models.py` because
PowerShell continued past a failed verification. After resolving any merge conflict, grep the
resolved files for `^(<<<<<<<|=======|>>>>>>>)` and `ast.parse` every changed `.py` **before**
committing.

## What has landed

_(see `STATUS.md` for the per-item table with outcomes and verifier corrections)_

| wave | commits | items |
|---|---|---|
| 0 | `8de0434` | recon + backlog + program docs only; no source change |
| 1 | `4eaec9d` `30e711e` `5df51a6` `f93ea8e` `e922c41` `6c83ab5` | A1, A17/S2, A4, A2, A3, A13, A5, A6 — each independently verified (new test observed failing on the unmodified base, then passing) |
| 2 | `b040751` `ca83bc0` | B2, A14, A15 merged; D1 sent back once (its "unchanged default path" was 11.1 % shorter than before and the test tolerance hid it) |
| 3 | `e6e8c3d` `f0b78b3` `e745820` `d53d65d` | A7, A11, B1, A8, B7 merged; B1 rejected once (the first version still signalled Rigma's own process group on POSIX) |
| 4 | `dbb2110` `e2a9ce0` `46b53f5` `e770ac0` `9c47346` `a78a82e` `7f8991c` `959bb46` | D1/S3, B5, A16, B3, A9, B4, B8, C1, C2, C3 merged; C3 rejected once (the fork gate was re-derived at argv-build time, so a fork-only flag reached the pinned mainline binary) |
| 5 | `b63ff92` `036a5d3` `335dffe` `6361ca2` | Frontend wave 1: D5 (the first `*.test.tsx`), A10, A8c, B6b |
| 6 | `ddd0dfc` `50f24e7` `622f6ef` `baa1178`…`4edea5e` | B5c/B6a, A13b/A13d/A2b/A2d, **A18**/D2-backend/D3a/B7d, frontend wave 2 (W5F1a/A16b/B6c/D4a) |
| 7 | `099e2c1` `30122c3` `773d39d` | **A17b/A17c (GUIDANCE 4+5)**, B1b/D4, A5c/A11c/A7b |
| 8 | `acc5c41` | A2d-gap/A11b/A16e |
| 9 | _in flight at hand-off_ | A18c/A18d/A17d/A11d (`w9a`), A17e (`w9b`), A2d-budget/A2e (`w9c`), frontend wave 3 |

Integration head: **`773d39d`** (code) — `d2212c0` adds the docs that record waves 6–8. `STATUS.md`
carries the per-item table and the note on the metadata-only author rewrite (with the old→new hash
mapping for every pre-rewrite commit).

**A18 is the one that mattered most** and it was found by the full suite, not by reading: the run
loop's *first* `_runs.load` was unguarded, so a transient unreadable `run.json` raised
`TypeError: 'NoneType' object is not subscriptable` at `serve.py:5602`, killed the driver, and left
the run `running` forever with its slot claimed — pause/inject answered 409 "no driver", restart
409 "running", a new run 409 "already active", and only Stop cleared it. It is fixed and the wedge
is confirmed closed by an independent verifier. Its sibling — a **readable but unusable** record
(`{}`, no `session_id`) — is the same wedge and is `A18c` in `BACKLOG.md`.

## What is in flight / what to do next

Wave 9 implementers are running at hand-off (see `STATUS.md` and `.scratch/orchestrator/impl-*.md`).
`BACKLOG.md`'s *"Follow-ups opened by the verifiers"* table is the work queue — read it before
starting anything. The highest-value open items at hand-off:

1. **A18c** — the same wedge on a readable-but-unusable `run.json`.
2. **A17d** — wire `expected_vram_mb` into `/api/server/findings` so the VRAM axis is live (it is
   correct but inert).
3. **B6c-mid** — the runnable-command control cannot deliver mid-turn; needs a product decision
   (queue it via `api.control(sid,"queue_enqueue",…)`, or hide the control).
4. **Frontend wave 3/4** — the W5F1a residual (four call sites are still deletable with the suite
   green), B6d (mode control), B4b (question card), D3b, then D2's dialog and D4b–D4d.
5. **OD-11** and the other owner decisions in `OWNER-DECISIONS.md` (now OD-1…OD-11).

## What is waiting on the GPU (`NEEDS-GPU`)

The single plan to extend is `docs/review/findings-r3/37a-262k-verification-plan.md` — **do not start
a second one.** The open fit questions it should now also answer:

1. Launch the 27B hybrid (`Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf`) at ctx 65536,
   ubatch 512, `--parallel 2`, and capture the `load_tensors`/`sched_reserve`/`llama_memory_recurrent`
   buffer lines. Then: (a) does `token_embd` still land in `CPU_Mapped` host memory at 2 slots, and is
   that a property of the fork or of the model? (b) what is `compute buffer size` at 2 slots? Feed
   both back into `resolve.py` (see `OWNER-DECISIONS.md` OD-5). **This is now also the check for
   A17b**: the code expects `graph splits = 2` on this exact load, and `.scratch/prism-v.log` is the
   only evidence that it does.
2. The 262K-context verification itself (already planned in 37a).
3. Answering mcode `ask_user` and the DSH ACP client (OD-6) need one live mcode session.
4. vLLM phases 1–5 need Linux/WSL + a GPU; on Windows `vllm_availability` refuses
   (`engines.py:288-291`). Only unit-level work with fakes is possible here.
5. A multi-GPU machine would settle A17e (the device-count baseline is derived from the load's own
   device labels and cannot be executed here).

## What is waiting on the owner

`OWNER-DECISIONS.md` — OD-1 run-profile default, OD-2 `view_image`/`copy_files` confinement, OD-3
stale RAG index, OD-4 LoRA feature, OD-5 the two GPU measurements, OD-6 the ACP client, OD-7 draft
retention, OD-8 the queue cap and question wait, OD-9 ACP session management, OD-10 the DSH approval
trail, OD-11 whether a shipped registry model may store launch defaults. Each has options, evidence
and a recommendation.

## How to continue the program in a new session

1. Read `.scratch/orchestrator/GUIDANCE.md`, `.scratch/orchestrator/STANDING-RULES.md`, then this
   file and `STATUS.md`.
2. `git log --oneline -25` on `review/deep-audit-2026-09-22` to see what landed.
3. `[DateTime]::UtcNow` — the tokenjuice model retires **2026-09-30 23:59 UTC**. Start no new item
   after **22:00 UTC**; the hand-off must be committed by **22:45 UTC**.
4. Pick the next `DO` item from `BACKLOG.md` (ranked), spawn one implementer per item on disjoint
   files (brief = the item's cause/resolution/evidence + the §0 text verbatim + the pytest
   semaphore + "read GUIDANCE.md"), then one INDEPENDENT verifier per diff (fresh context; reads only
   the diff and the brief; answers: does the test fail without the fix, does the fix do what the
   commit says, what does it break, is it this repo's style). Merge the ones that pass, serially
   (grep for conflict markers and `ast.parse` before each merge commit). Run the full suite once.
   Update `STATUS.md`, `WAVES.log`, `HANDOFF.md`.
5. A verifier rejection goes back to the implementer once; a second rejection is recorded and the
   item is dropped. PASS-WITH-NITS is mergeable — record the nits in `BACKLOG.md`.

## Standing caveats a new session must know

- `docs/review/FINDINGS.md`, `docs/RESUME-2026-09-21-arm.md` §11 and `docs/HANDOFF-speedups.md`
  contain **stale** claims; `BACKLOG.md` Tier F lists what is already done and must not be redone.
- The unmerged branch `r3/harness` is a stale-base rewrite of already-merged work — **do not merge
  it**; port only the `kill_tree` lines (items B1/B1b, both now done).
- `.scratch/prism-v.log` is a real load log and the ground truth for the fit arithmetic. It contains
  **two** loads; the second (~line 4485) is the real one; the first is a fitting pass that reports
  `0.00 MiB`. It begins with **two** `common_params_print_info:` markers (lines 1–2), so a parser
  that treats the first marker as the start of the run truncates the file to line 2.
- `subagent_fork` returns a job id that **cannot** be steered with `send_message` — a fork cannot be
  sent back to. A fix needs a fresh agent.
