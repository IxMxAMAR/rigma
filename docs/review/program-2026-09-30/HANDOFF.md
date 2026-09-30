# HANDOFF — Rigma improvement program, 2026-09-30

**Read this first if you are a new session with a different model.** It is written to be
self-sufficient: it names the commands, the paths, the rules and the state, and it does not assume
you share any context with the session that wrote it.

- Repo: `C:\ComfyUI\RD\rigma-review` — the review fork. **Never touch `C:\ComfyUI\RD\rigma`** (that
  is the owner's live checkout; only its `.venv\Scripts\python.exe` is used, as the interpreter).
- Integration branch: `review/deep-audit-2026-09-22`. No `git push`, ever.
- Program docs: `docs/review/program-2026-09-30/` — `BACKLOG.md`, `STATUS.md`, `OWNER-DECISIONS.md`,
  `HANDOFF.md` (this file). `docs/*` is git-ignored, so these need `git add -f`.
- Steering channel (written by the orchestrator on the owner's behalf):
  `.scratch/orchestrator/GUIDANCE.md` — **newer entries override the brief**. If
  `.scratch/orchestrator/STOP` exists, finish the commit in hand, write this hand-off, end.
- Wave log: `.scratch/orchestrator/WAVES.log`. Recon reports: `.scratch/orchestrator/recon-*.md`.
  Implementer reports: `.scratch/orchestrator/impl-*.md`. Verifier reports:
  `.scratch/orchestrator/verify-*.md`.

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

**Semaphore note (fixed 2026-09-30):** the script now has **no `param()` block** and reads `$args`.
It had one, and `powershell.exe -File` then bound `-m` as a *parameter name*, so every run that
passed `-m "not hardware"` died with *"A parameter cannot be found that matches parameter name 'm'"*
— i.e. the §0.1 marker was silently unpassable. The script now also **appends `-m "not hardware"`
itself when the caller did not pass a marker**, so a hardware test cannot be run by forgetting the
flag. `addopts` in `pyproject.toml` is only `-q`; it does **not** exclude hardware.

**Environment note (verified 2026-09-30):** `pwsh` (PowerShell 7) is **not installed** on this host —
only Windows PowerShell 5.1, and `powershell -File` needs `-NoProfile -ExecutionPolicy Bypass`.
Use the invocation above, not `pwsh -File`. (The Head Agent's own `pwsh` tool works because the
harness invokes it directly; a subagent shell does not have it on PATH.)

Worktrees: `git worktree add .scratch/wt-<name> -b impl/<name> review/deep-audit-2026-09-22`. Merge
serially into the integration branch from the main tree. No links or junctions; never
`git worktree remove --force`; never `git clean -d/-x`.

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
   `node_modules` into a worktree).
8. Owner decisions go to `OWNER-DECISIONS.md` with a recommendation — do not make them.
9. Subagents run DeepSeek-V4.1-Flash on tokenjuice.

**Two additions made during the run, both from `GUIDANCE.md` / an incident:**

10. **Never override the git author.** Plain `git commit` only — no `-c user.name`, no `-c user.email`,
    no `GIT_AUTHOR_*`/`GIT_COMMITTER_*`. The branch may be published, so every commit must carry
    `IxMxAMAR <officialamrendrasingh@gmail.com>`. Check with `git log -1 --format='%an <%ae>'`. (The
    orchestrator broke this early on; the affected commits were re-authored in a metadata-only
    rewrite — see `STATUS.md`.)
11. **Never use `git stash`.** It is repository-wide, not per-worktree: one agent's `stash push` was
    consumed by another agent's `pop`, which then applied a foreign stash into a third worktree. Use
    `git show HEAD:path` or a copy for before/after evidence. Commit on your own branch instead.

Also: prefer `git commit -F <file>` for messages (the PreToolUse guard no longer inspects heredoc text,
but a `-F` file is unambiguous).

## What has landed

_(updated at every wave boundary — see `STATUS.md` for the per-item table)_

| wave | commits | items |
|---|---|---|
| 0 | `8de0434` | recon + backlog + program docs only; no source change |
| 1 | `4eaec9d` `30e711e` `5df51a6` `f93ea8e` `e922c41` `6c83ab5` | A1, A17/S2, A4, A2, A3, A13, A5, A6 — all merged, each independently verified (new test observed failing on the unmodified base, then passing) |
| 2 | `b040751` `ca83bc0` | B2, A14, A15 merged; D1 sent back once (its "unchanged default path" is 11.1 % shorter than before and the test tolerance hides it) |
| 3 | `e6e8c3d` | A7, A11 merged; A8 and B7 implemented and in verification |
| 4 | _in flight_ | A16, A9, A12, B4, C1, C3, B8 |
| 1 | _sent back once, fixed_ | B1 — the verifier rejected the merge because delegating to `tools._kill_tree` would `killpg` Rigma's own process group on POSIX (the harness children are not detached); fixed in `8afc625` and re-verifying |

Integration head: **`e6e8c3d`**. `STATUS.md` carries the per-item table and a note on the
metadata-only author rewrite (the pre-rewrite hashes are listed there).

## What is in flight

Wave 3 implementers, one per item, each in its own worktree on `impl/w3a`…`impl/w3c` plus
`impl/w2d`. See `STATUS.md` for live state, `.scratch/orchestrator/impl-*.md` for implementer
reports and `.scratch/orchestrator/verify-*.md` for verifier verdicts.

**Verifier follow-ups that are NOT yet fixed** are listed in `BACKLOG.md` under *"Follow-ups opened
by the verifiers (wave 1+2)"* — read that table before starting new work; several are one-liners
(`A2c`, `A13c`, `S2b`, `S2c`) and are good first items for a new session.

## What is waiting on the GPU (`NEEDS-GPU`)

The single plan to extend is `docs/review/findings-r3/37a-262k-verification-plan.md` — **do not start
a second one.** The two open fit questions it should now also answer:

1. Launch the 27B hybrid (`Ternary-Bonsai-2-27B-Uncensored-Heretic-PQ2_0.gguf`) at ctx 65536,
   ubatch 512, `--parallel 2`, and capture the `load_tensors`/`sched_reserve`/`llama_memory_recurrent`
   buffer lines. Then: (a) does `token_embd` still land in `CPU_Mapped` host memory at 2 slots, and is
   that a property of the fork or of the model? (b) what is `compute buffer size` at 2 slots? Feed
   both back into `resolve.py` (see `OWNER-DECISIONS.md` OD-5).
2. The 262K-context verification itself (already planned in 37a).
3. Answering mcode `ask_user` and the DSH ACP client (OD-6) need one live mcode session.
4. vLLM phases 1–5 need Linux/WSL + a GPU; on Windows `vllm_availability` refuses
   (`engines.py:288-291`). Only unit-level work with fakes is possible here.

## What is waiting on the owner

`OWNER-DECISIONS.md` — OD-1 run-profile default, OD-2 `view_image`/`copy_files` confinement, OD-3
stale RAG index, OD-4 LoRA feature, OD-5 the two GPU measurements, OD-6 the ACP client. Each has
options, evidence and a recommendation.

## How to continue the program in a new session

1. Read `.scratch/orchestrator/GUIDANCE.md` and `.scratch/orchestrator/STANDING-RULES.md`, then this
   file and `STATUS.md`.
2. `git log --oneline -20` on `review/deep-audit-2026-09-22` to see what landed.
3. `Get-Date -AsUTC` — the tokenjuice model retires **2026-09-30 23:59 UTC**. Start no new item after
   **22:00 UTC**; the hand-off must be committed by **22:45 UTC**.
4. Pick the next `DO` item from `BACKLOG.md` (ranked), spawn one implementer per item on disjoint
   files (brief = the item's cause/resolution/evidence + the §0 text verbatim + the pytest
   semaphore), then one INDEPENDENT verifier per diff (fresh context; reads only the diff and the
   brief; answers: does the test fail without the fix, does the fix do what the commit says, what
   does it break, is it this repo's style). Merge the ones that pass, serially. Run the full suite
   once. Update `STATUS.md`, `WAVES.log`, `HANDOFF.md`.
5. A verifier rejection goes back to the implementer once; a second rejection is recorded and the
   item is dropped.

## Standing caveats a new session must know

- `docs/review/FINDINGS.md`, `docs/RESUME-2026-09-21-arm.md` §11 and `docs/HANDOFF-speedups.md`
  contain **stale** claims; `BACKLOG.md` Tier F lists what is already done and must not be redone.
- The unmerged branch `r3/harness` is a stale-base rewrite of already-merged work — **do not merge
  it**; port only the `kill_tree` lines (item B1).
- `.scratch/prism-v.log` is a real load log and the ground truth for the fit arithmetic. It contains
  **two** loads; the second (~line 4485) is the real one; the first is a fitting pass that reports
  `0.00 MiB`.
