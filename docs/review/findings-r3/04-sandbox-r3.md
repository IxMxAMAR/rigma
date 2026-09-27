# 04-sandbox-r3 — breaking the run-2 confinement

worktree: `C:\ComfyUI\RD\rigma-review\.wt-r3-b` (branch `r3/sandbox`, base `5b5790b`)
fork import verified: `python -c "import rigma; print(rigma.__file__)"` ->
`C:\ComfyUI\RD\rigma-review\.wt-r3-b\src\rigma\__init__.py`

Threat model: a page the model fetches, a file it reads, a downloaded repo, or a
third-party MCP server is the attacker; the service is unauthenticated on
127.0.0.1:11500; the run profiles (`confined`, `no-delete`, `no-network`) are
promises the tool layer is supposed to keep.

Nothing dangerous was handed to anything that executes. Every execution-tool
finding was proved by calling `exec_decision`/`_text_refusal` on TEXT, or by
stubbing the launcher. File-system proofs ran inside
`.scratch\r3-b\tmp*` or a pytest `tmp_path`.

## Findings

### R3-1 [HIGH] A junction inside the workspace escaped every walker (find_files, grep, watch, undo, pack_folder)
- **Where:** `src/rigma/tools.py:1843-1880` (`_iter_workspace_files`, shared by
  `find_files` and `grep`), `src/rigma/watch.py:78-113` (`_iter_files`),
  `src/rigma/tools.py:1462-1474` (`_glob_under`),
  `src/rigma/tools.py:2071-2090` (`_newest_undo_key`),
  `src/rigma/workspace.py:41-46` (`pack_folder`)
- **Trigger:** any Windows JUNCTION inside the workspace that points outside it
  (`cmd /c mklink /J ws\link C:\Users\<owner>` — no privilege needed), then
  `grep(pattern="<secret>")`, `find_files`, `sample_files`, an arm edit through
  the link, or `POST /api/workspace/pack`.
- **Consequence:** the walkers read files OUTSIDE the workspace while `read_file`
  on the same path is correctly refused — including under the `confined`
  profile, whose whole promise is that everything stays workspace-relative.
  Measured through the real tools:
  ```
  os.path.islink(<junction dir>) = False        # not seen as a link
  read_file('link/secret.txt')  -> error running read_file: path is outside the workspace
  grep('TOPSECRET')             -> 'link/secret.txt:1: TOPSECRET-APIKEY=abcd1234'
  find_files('**/*.txt')        -> 'inside.txt\nlink/secret.txt'
  watch.Watcher(ws).poll_once() -> recorded [<ws>\link\secret.txt]
  undo_last_change()  no-arg    -> 'restored <ws>\link\secret.txt ...'
  outside/secret.txt before: 'VERSION-TWO'   after: 'VERSION-ONE'   # WRITTEN outside
  pack_folder(ws)               -> <file path="link/secret.txt"> TOPSECRET-KEY </file>
  ```
  `os.walk(followlinks=False)` and `Path.is_symlink()` only recognise a
  NAME-SURROGATE reparse point. A junction is a MOUNT-POINT reparse point, so it
  reads as an ordinary directory and is descended into. The undo case is the
  worst of the four: the recorded key is lexically inside the workspace, so
  `_newest_undo_key`'s containment test accepts it and the restore writes through
  the link to a file outside.
- **Cause:** the link test is `is_symlink()`/`followlinks=False`, which is the
  symlink case only, and `_newest_undo_key` tested containment on the
  un-resolved key.
- **Fix:** implemented. `watch.is_reparse_dir` is the one predicate (symlink, or
  a directory carrying `FILE_ATTRIBUTE_REPARSE_POINT`; a non-symlink reparse
  FILE is a cloud placeholder and stays ordinary content) and it is applied to
  `dirnames` in `watch._iter_files`, `tools._iter_workspace_files` and
  `workspace.pack_folder`. `_glob_under` tests containment on the resolved path;
  `_newest_undo_key` resolves each key before the containment test.
- **Verified:** `tests/test_r3_junction_confinement.py` — 5 tests, all 5 failed
  before the fix (`grep` returned `TOPSECRET`, the watcher recorded the link,
  the undo wrote outside), all pass after. Commit `b4012a1`.

### R3-2 [HIGH] The credential denylist was one dropped character deep (`_fuzzy_file` re-checked nothing)
- **Where:** `src/rigma/tools.py:1336-1372` (`_fuzzy_file`), called from
  `_read_file` (`:2559`), `_transfer_sources` (`:3011`), `_resolve_image` (`:3159`)
- **Trigger:** a near-miss spelling of a denied file:
  `read_file(".en")`, `read_file("id_rs")`, `read_file("credential")`,
  `move_files(paths=["id_rs"], dest="public")`.
- **Consequence:** the 13-2 credential denylist is bypassed by one character, and
  the recovery note names the file it used — so it is also a reliable way to
  DISCOVER the real credential name. Measured:
  ```
  read_file('.env')       -> error ... that looks like a credential file
  read_file('.en')        -> " (you asked for '.en' — used '.env')SECRET=sk-live-1234"
  read_file('id_rs')      -> " (you asked for 'id_rs' — used 'id_rsa')-----BEGIN OPENSSH PRIVATE KEY-----"
  read_file('credential') -> " (you asked for 'credential' — used 'credentials')aws_secret=AKIA"
  ```
- **Cause:** `_read_path`/`_write_path` decide on the name the MODEL gave;
  `_fuzzy_file` then substitutes the closest real sibling and nothing re-checks
  the substitute. Check-then-substitute.
- **Fix:** implemented. `_fuzzy_file(p, ctx=None)` drops any candidate the
  credential denylist refuses; all three call sites pass `ctx`. A near miss on an
  ordinary file still resolves.
- **Verified:** `tests/test_r3_credential_fuzzy.py` — 6 tests, all failed before,
  all pass after. `tests/test_audit_tools.py`'s `_fuzzy_file` stub was updated to
  mirror the new parameter (assertions unchanged). Commit `7721b51`.

### R3-3 [MEDIUM] A sentinel tool was the one way to deliver a raw NUL run
- **Where:** `src/rigma/tools.py:736-747` (`run_tool`'s `t.sentinel` branch),
  reached from `_resolve_image`'s error path (`:3135-3148`) and `_view_images`
- **Trigger:** `view_image(path="C:\\nope\\" + "\u0000"*300 + "x.png")`, or
  `view_images(folder="C:\\x\\" + "\u0000"*300)` under `confined`.
- **Consequence:** 300 raw NULs reach the model context, un-defused, through a
  `sentinel=True` tool. Measured: `view_image` NUL -> len 334, 300 NULs;
  `view_images` -> 352/300; confined `view_images` -> 395/300; the same payload
  through `read_file` (not a sentinel tool) -> 0 NULs. A NUL flood is the
  strongest end-of-document signal a model knows: it emits EOS as its first
  token and the turn dies at "generating" — the 2026-07-21 failure
  `_defuse_control_bytes` was written for.
- **Cause:** 04-7 correctly moved sentinel detection from "sniff the result text"
  to "a property of the tool", but the property then exempted the tool's
  ORDINARY results too, and a sentinel tool's error text embeds the model's own
  argument. The fix traded one hole for another.
- **Fix:** implemented. `_defuse_sentinel_result` passes a result around the
  guard only when it really starts with one of the three sentinels, and even then
  defuses the trailing note (after the NUL separator serve.py splits on). A real
  `IMAGE_SENTINEL` payload and the `use_tools`/`delegate` payloads still
  round-trip byte-for-byte.
- **Verified:** `tests/test_r3_sentinel_defuse.py` — 4 of 5 failed before, all
  pass after. Commit `7bf6f8b`.

### R3-4 [MEDIUM] `run_shell` carried the payload `run_python` refuses
- **Where:** `src/rigma/tools.py:3745-3800` (`exec_decision`), `:3900-3920`
  (`_run_subprocess`)
- **Trigger:** `run_shell(command='python -c "import shutil; shutil.rmtree(\'/\')"')`,
  or under `no-delete` `run_shell(command='python -c "import os; os.remove(\'x\')"')`.
- **Consequence:** the Python-specific blocklist (`_BLOCKED_PY`/`_DELETE_PY`,
  which exists because the shell wordlist false-positives on Python) was bypassed
  by wrapping the same payload in a shell call, and the `no-delete` profile was
  defeated for ordinary deletion. Measured (TEXT only, nothing executed):
  ```
  shell 'python -c "import shutil; shutil.rmtree(\'/\')"'  _BLOCKED_CMD=False  _DELETE_CMD=False  exec_decision(all)=(True,'')
  shell 'python -c "import os; os.remove(\'x\')"'          _DELETE_CMD=False  exec_decision(no-delete)=(True,'')
  py    'import shutil; shutil.rmtree(\'/\')'              _BLOCKED_PY=True   exec_decision(all)=(False,'blocked ...')
  ```
- **Cause:** `exec_decision` chose its wordlist from which tool called it, so the
  shell branch never applied the Python rules; the two rule sets also existed in
  two copies (`exec_decision` and `_run_subprocess`) that could drift.
- **Fix:** implemented. One shared `_text_refusal(cmd, python_src, prof)` used by
  both; the shell branch tests the shell list AND the Python list. The reverse is
  deliberately not done — the shell wordlist matches ordinary Python
  (`{}`.format, the `del` keyword). Verified `echo hi`, `python -m pytest`,
  `ls -la`, `git status --short`, `python -c "print('hello')"` and ordinary
  Python still pass.
- **Verified:** `tests/test_r3_shell_python_blocklist.py` — 7 of 8 failed before,
  all pass after. Commit `503055d`.

### R3-5 [MEDIUM] The ReAct rescue's whole-reply test was vacuous
- **Where:** `src/rigma/tools.py:299-301` (`_REACT_PREFIX`), used at `:448-457`
- **Trigger:** a reply whose first line is `Thought: …` followed by explanatory
  prose, ending with an Action block:
  ```
  Thought: The user is asking HOW to delete a folder. Here is the syntax you
  would use — do not run it, it is only an example.
  It takes a -Recurse flag and a path.
  Action: run_shell
  Action Input: {"command": "echo pwned"}
  ```
- **Consequence:** `rescue_tool_call` returned
  `('run_shell', {'command': 'echo pwned'})` and serve.py's loop EXECUTES a
  rescued pair — so prose that merely discusses a call became a call. This is the
  05-2 shape the whole-reply test was added to stop.
- **Cause:** `^(?:\s*(?:Thought|Thinking|Reasoning)\s*:.*)?\s*$` compiled with
  `re.S`: the `.*` swallowed newlines, so any reply merely STARTING with
  "Thought:" matched however much prose followed.
- **Fix:** implemented. `[^\n]*` in place of `.*`, and no `re.S` — only a single
  Thought LINE may precede the block, which is what the docstring always claimed.
  A bare Action block, a single Thought line and blank lines after the Thought
  all still rescue.
- **Verified:** `tests/test_r3_react_rescue.py` — 2 of 7 failed before, all pass
  after. Commit `0ec7168`.
- **Residual (reported, not fixed):** a reply whose FIRST line is
  `Thought: <prose quoting the call>` and which ends with the Action block still
  executes. The marker is a deliberate reasoning signal and the quoted-call case
  is introduced by ordinary prose, so this is the narrowest defensible line.

### R3-6 [MEDIUM] The watcher never forgot a file that was gone
- **Where:** `src/rigma/watch.py:78-113` (`_iter_files`), `:150-200`
  (`poll_once`), `_remember`
- **Trigger:** ordinary churn — a workspace where files are created and deleted
  (build output, temp files, an arm that rewrites and removes files).
- **Consequence:** `_known`/`_stats` were written every pass and never pruned,
  and `_bytes` was only decremented when the SAME key was re-remembered. Measured
  300 create+delete cycles -> `remembered_files 300, remembered_bytes 0`. The
  64 MB budget therefore fills with garbage, after which `_remember` refuses
  every NEW file: a watcher that looks alive and records nothing, i.e.
  `undo_last_change` silently stops working for the arm. The key count is
  unbounded too, because a 0-byte file costs nothing against the budget.
- **Cause:** no eviction pass; only re-remembering the same key adjusted `_bytes`.
- **Fix:** implemented. `poll_once` collects the keys the walk visited and
  `_forget_unseen` drops the rest from both dicts with an exact `_bytes`
  adjustment. `_iter_files` reports `state["truncated"]` when the MAX_FILES cap
  ended the walk early, and eviction is skipped on such a pass so a huge tree
  cannot lose the baseline for files the pass never reached.
- **Verified:** `tests/test_r3_watch_eviction.py` — 2 of 3 failed before, all
  pass after; `tests/test_watch.py` (26 tests) still passes. Commit `2452518`.

### R3-7 [MEDIUM] The image tools skipped the credential denylist
- **Where:** `src/rigma/tools.py:3155-3200` (`_resolve_image`)
- **Trigger:** `view_image(path=".ssh/a.png")`,
  `view_images(paths=[".ssh/a.png"])`, or an absolute image inside a browser
  profile / Rigma's state dir.
- **Consequence:** the image tools were the one read the 13-2 fix did not cover:
  they take `_resolve_image`, not `_read_path`. Measured: `read_file(".ssh/a.png")`
  is refused ("that is a credential directory") while
  `view_image(".ssh/a.png")` returned `IMAGE_SENTINEL + <abs path>` and
  `view_images(paths=[...])` the same — and `view_images(folder=".ssh")` refused,
  so the two modes of the SAME tool disagreed.
- **Cause:** `_resolve_image` applied neither `_credential_path_reason` nor the
  grant check.
- **Fix:** implemented for the denylist: `_resolve_image` applies
  `_credential_path_reason` (the fuzzy-recovered path included) before the
  extension check.
- **Verified:** `tests/test_r3_image_credentials.py` — 3 of 5 failed before, all
  pass after; `tests/test_run_tools.py` and `tests/test_phase0_contracts.py`
  still pass. Commit `218ce28`.

### R3-8 [LOW] The run-progress exemption outranked every credential rule
- **Where:** `src/rigma/tools.py:1609-1639` (`_credential_path_reason`)
- **Trigger:** `read_file(".ssh/progress.md")` with `allow_absolute_reads`, or the
  same name in `.aws`, `.gnupg`, or a browser profile.
- **Consequence:** the exemption exists so the run loop can hand the model its own
  progress log, which lives under Rigma's state dir — it needs to punch through
  the STATE-DIR rule only. Tested first, it punched through the credential-file,
  credential-directory and browser-profile rules too. Measured:
  `read_file('.ssh/progress.md')` -> the file body ("PRIVATE KEY MATERIAL").
- **Cause:** an early `return ""` before any other rule.
- **Fix:** implemented. The exemption moved inside the state-dir branch. Verified
  the reason it exists still holds (a workspace that IS the state dir reads its
  `progress.md`) and that an ordinary workspace progress log is unaffected.
- **Verified:** `tests/test_r3_progress_exemption.py` — 3 of 5 failed before, all
  pass after. Commit `1fe52f8`.

## Open findings (not fixed)

### R3-9 [MEDIUM] `_resolve_image` accepts an absolute image outside the workspace with no grant
- **Where:** `src/rigma/tools.py:3135-3148`
- **Trigger:** `view_image(path="C:\\Users\\<owner>\\Pictures\\anything.png")` on
  a session whose `allow_absolute_reads` is OFF.
- **Consequence:** `read_file` on an absolute path outside the workspace is
  refused ("... enable 'allow absolute reads' ...") while `view_image` on a file
  in the same directory succeeds and base64s the bytes into the conversation.
  Measured:
  ```
  _read_path(abs outside)      -> DENIED
  _resolve_image(abs outside)  -> (WindowsPath('.../outside/pic.png'), '', '')
  view_image(abs outside)      -> __RIGMA_IMAGE__...\outside\pic.png
  view_images(folder=abs out)  -> error: reading an absolute path outside the workspace is disabled
  ```
  The blast radius is bounded (image extensions, ≤20 MB) but it is a read the
  13-2 policy says is disabled, and the two image modes disagree.
- **Cause:** the 04-6 fix only consulted the profile; the grant was never wired in.
- **Fix:** NOT implemented. Making `_resolve_image` require the grant is the
  consistent change, but it is pinned as intended behaviour by
  `tests/test_run_tools.py` (`test_view_images_recovers_a_batch_of_mangled_absolute_names`
  and the two `_resolve_image(<abs>, {})` tests) and is the product's headline
  photo mission ("go through D:\Good Stuff"), so it is a product decision. The
  smallest safe fix, if the lead wants it: require `_absolute_reads_allowed(ctx)`
  for an absolute path outside the workspace and surface the grant in the UI.

### R3-10 [MEDIUM] A `copy_files`/`move_files` destination can be any absolute path (Startup persistence)
- **Where:** `src/rigma/tools.py:1642-1665` (`_write_path`), `:3046-3090`
  (`_do_transfer`)
- **Trigger:** `copy_files(paths=["payload.bat"], dest="C:\\Users\\<owner>\\AppData\\Roaming\\Microsoft\\Windows\\Start Menu\\Programs\\Startup")`
- **Consequence:** the write half of the confinement is a no-op. Measured:
  `_write_path(ctx, <an absolute dir outside the workspace>)` returns the path
  unchanged under the default profile; the credential denylist catches `.ssh`/
  `.aws`/browser profiles but not the Startup folder, `%TEMP%`, or a PowerShell
  profile directory. `copy_files` is `safe=False, needs="code"`, so it needs
  `allow_code` — but nothing in the path layer bounds where the bytes land.
- **Cause:** 13-2 confined READS and deliberately preserved the pre-fix write
  behaviour, which is pinned by
  `tests/test_audit_sec13.py::test_move_files_to_an_absolute_destination_is_still_a_write`.
- **Fix:** NOT implemented (pinned as intended). If it should change: add the
  autostart/profile locations to the denylist, or require
  `allow_absolute_reads` (a "session may touch paths outside the workspace"
  grant) for an absolute destination.

### R3-11 [MEDIUM] `rag.add_source` accepts anything, and the index is model-readable — the credential denylist has a second door
- **Where:** `src/rigma/rag.py:87-95` (`add_source`), `:54-59`
  (`_source_globs`), `src/rigma/tools.py:1193-1216` (`search_my_documents`)
- **Trigger:** `rigma rag add C:\Users\<owner>` (or `POST /api/rag/sources`), then
  any `search_my_documents` call.
- **Consequence:** `.env`, `.ssh/id_rsa`, `.git-credentials`, `.aws/credentials`
  and browser cookie DBs under the added folder are embedded into a local vector
  index with no denylist and no confirmation, and the model retrieves them with a
  `safe=True` (auto-run) tool. This defeats the premise the 13-2 denylist is built
  on ("the model has no legitimate reason to put a key into the conversation")
  through a door the fix did not close. Measured:
  `rag.add_source(<tmp>)` -> `['<tmp>']`, `_source_globs()` -> `['<tmp>/**/*']`
  with no path validation at all.
- **Cause:** the denylist was added to the tool read path only; the RAG ingest
  path was not considered part of the same surface.
- **Fix:** FIXED as `0c03523` by the orchestrator, by the second half of the
  suggestion above — and by the second half *only*, deliberately. The generated
  `raggity.toml` now carries an `exclude` list DERIVED from the same
  `_CREDENTIAL_FILES`/`_CREDENTIAL_DIRS` tuples the read path uses, so a pattern
  added there is excluded from indexing by construction; a test fails if a denied
  pattern is not excluded. Refusing the source was rejected: the index is what the
  user asked to build, and with credentials excluded, adding a home directory is a
  legitimate thing to want.
- **Verified against REAL raggity 0.13.0, with a control.** A canary corpus was
  ingested twice — once with `exclude = []`, once with the derived list — and each
  index was then ASKED whether the canary was retrievable. The control is what makes
  this evidence: my first two attempts at the measurement were **vacuous** and I
  nearly shipped their conclusion, because the canaries had non-indexable extensions
  (`.json`, `.pem`, none) or dotfile names raggity prunes on its own, so "clean"
  proved nothing. With indexable extensions: control = 8 of 10 files indexed and
  canaries A/B/C/G/H all retrievable; guarded = 2 (the two real documents), none.
- **The measurement found a gap the derived list alone did NOT close.** `**/credentials`
  and `**/*.pem` are exact basenames, so an appended extension defeats them:
  `credentials.md`, `my.api_key.md`, `credentials.json.md`, `token_api_key.txt` and
  `server.pem.txt` were ALL still indexed and ALL still retrievable. So every pattern
  also gets a `.*` variant. The reachable case is a README about key rotation, or a
  `.env.md` note: its NAME carries the secret's identity into an index an auto-run
  tool reads. Final measured state: 8 canaries, 0 retrievable, both real documents
  indexed and answerable.
- **Not fixed, and named rather than implied:** an index built BEFORE this fix still
  holds whatever was already embedded. Regenerating the config changes what a future
  ingest includes; it does not remove existing rows. A user who added a folder
  containing credentials should reindex (`rigma rag reindex`). Doing it automatically
  would mean writing to the index from a config write.
- **Asymmetry, deliberate:** the read path keeps its `fnmatch` behaviour
  (`credentials.md` is still readable there). The index is the stricter side, because
  it is consulted by an auto-run tool with no human in the loop.
- **Verified:** 11 tests in `tests/test_r3_rag_credential_excludes.py`, two of them
  mutation-checked (short-circuiting the glob list fails the appended-extension test;
  removing the `exclude` key fails the config test). The generated TOML is parsed
  with `tomllib` in the test, so a malformed list fails there rather than at
  raggity's first ingest.

### R3-12 [LOW] Alternate data streams are accepted as ordinary files
- **Where:** `src/rigma/tools.py:1416-1441` (`_bad_write_char`), `_ws_path`
- **Trigger:** `write_file(path="notes.txt:ads", content="…")`, then
  `read_file("notes.txt:ads")`.
- **Consequence:** measured — the write reports success and the read returns the
  stream; the plain file is untouched, the stream is invisible to Explorer, to
  `find_files` (size is the base file's) and to the user, so it is a place to park
  content that a later `grep`/`read_file` will surface. `_bad_write_char` rejects
  `*?"<>|` but not `:`; `_ws_path` passes the colon through. Not a confinement
  escape (measured: `inside.txt:..\..\evil` resolves to `ws\evil`, i.e. `..`
  collapses inside the workspace, and four levels up are refused).
- **Fix:** NOT implemented. Smallest change: reject `:` in a relative path
  component unless it is a drive prefix (i.e. add `:` to the write guard when the
  component is not the first).

### R3-13 [LOW] TOCTOU between `_ws_path`'s `resolve()` and the filesystem call
- **Where:** `src/rigma/tools.py:1490-1540` (`_ws_path`) and every caller
- **Trigger:** a directory component inside the workspace replaced by a junction
  between the resolve and the `open`/`write_bytes`.
- **Consequence:** the containment decision is made on a resolved path, then the
  path is used unresolved (`p.parent.mkdir`, `p.write_bytes`). serve.py runs up to
  8 tool calls concurrently under one semaphore, and `run_shell` can create a
  junction (`mklink /J`) without privilege, so two parallel calls make the window
  reachable in principle: one call swaps `ws\sub` for a junction to `C:\Windows`,
  the other writes `ws\sub\x`.
- **Cause:** check-then-use with no re-check and no handle-based containment.
- **Fix:** NOT implemented. Proving it needs a symlink/junction swap raced against
  a live write across two concurrent tool calls; not attempted (and the R3-1 fix
  removes the walker half of the same class). Mitigation: re-resolve immediately
  before the open, or open the parent directory and use
  `os.open(..., O_NOFOLLOW)`-style semantics (not available for directories on
  Windows), or serialise writes to the workspace.

### R3-14 [LOW] The undo journal is never pruned
- **Where:** `src/rigma/tools.py:1996-2030` (`_snapshot_before_write`),
  `_undo_dir`
- **Trigger:** ordinary use over a long-lived install — every distinct path ever
  written gets a snapshot file.
- **Consequence:** `~/.rigma/undo/` grows without bound (one file per distinct
  path per change, plus an ever-growing `index.json`), and nothing removes the
  snapshots of files deleted long ago. Measured on this box after a few sessions
  of review work: 50 files in the undo dir, no pruning code anywhere (the only
  `unlink` in the module is `_atomic_bytes`' temp-file cleanup).
- **Fix:** NOT implemented. Smallest change: a size/age cap applied inside
  `_snapshot_before_write` (drop the oldest entries and their snapshot files past
  N bytes or M days).

### R3-15 [LOW] A background job outlives the confirmation that authorised it, and `job_output` is not profile-gated
- **Where:** `src/rigma/tools.py:3544-3585` (`start_job`), `:3590-3600`
  (`job_output`), `:3633-3643` (`kill_jobs_for_run`), `:3664-3692` (`kill_all_jobs`)
- **Trigger:** `start_job` in a chat (not a run), then the session's
  `confirm_exec` is turned off (or the run/profile changes), then `job_output`.
- **Consequence:** the process is detached (`CREATE_NEW_PROCESS_GROUP`) and
  `_JOBS` is in-process, so nothing reaps a chat-started job: `stop_run` calls
  `kill_jobs_for_run(run_id)` and an empty `run_id` matches nothing, so the job
  survives the turn, the chat, and the revocation of the confirmation that
  authorised it; only process shutdown calls `kill_all_jobs`. `job_output` and
  `kill_job` are `needs="code"` with no `kind`, so they are NOT exec-gated:
  measured, `tool_specs(profile="confined")` offers `job_output` and `kill_job`,
  so a confined run can read (and kill) a job another session started, and
  `job_output` returns its output.
- **Fix:** NOT implemented. Smallest change: have `job_output`/`kill_job` require
  `confirm_exec` (or record the session id in the job and check it), and give
  `start_job` a bounded lifetime.

### R3-16 [MEDIUM] `find_files`/`grep` glob patterns can be a measured ReDoS
- **Where:** `src/rigma/tools.py:1802-1841` (`_glob_re`), called per file from
  `_iter_workspace_files`
- **Trigger:** `find_files(pattern="**/**/**/**/**/**/**/**/**/**/x")` in a repo
  whose paths are ~24 components deep.
- **Consequence:** each `**/` becomes an optional `(?:.*/)?` group, and
  `rx_glob.match(rel)` is called for EVERY file. Measured, per single path:
  ```
  **/ x4  -> 0.000s     x6 -> 0.016s     x8 -> 0.172s     x10 -> 2.187s
  ```
  (×12.7 for every two extra groups, so x12 is ~30 s and x14 ~6 min PER FILE).
  `find_files` and `grep` are `safe=True` (auto-run), so a model — or a
  prompt-injected page that talks the model into "searching" — holds a server
  worker thread for minutes. A second bug in the same function:
  `_glob_re("[z-a]")` raises `re.error` (the character class is copied through
  unescaped), which `run_tool` turns into
  `error running find_files: bad character range z-a at position 5`.
- **Fix:** NOT implemented. Two smallest changes: collapse a run of `**/` groups
  into one (the regex is equivalent), and wrap the class in a `try: compile`
  that escapes a malformed class instead of raising.

### R3-17 [LOW] `pack_folder` has no credential denylist
- **Where:** `src/rigma/workspace.py:34-70`, route `src/rigma/serve.py:3589-3598`
- **Trigger:** `POST /api/workspace/pack {"folder": "C:\\Users\\<owner>"}`.
- **Consequence:** every text file under the folder is packed into a prompt block
  and returned to the caller, including `.env`, `.ssh/id_rsa`, `.aws/credentials`
  and `.git-credentials` (none of `.ssh`/`.aws`/`.gnupg` is in `_SKIP_DIRS`). The
  13-2 denylist is not consulted, so the same bytes `read_file` refuses are handed
  out here. (The junction half of this walk is fixed by R3-1.)
- **Fix:** NOT implemented. Smallest change: skip `_CREDENTIAL_DIRS` names and
  `_CREDENTIAL_FILES` patterns while walking, reusing
  `tools._credential_path_reason`.

### R3-18 [LOW] `search_my_documents` is offered under `no-network` and `confined`
- **Where:** `src/rigma/tools.py:83` (`_NETWORK_TOOLS`), `:1193-1216`
- **Trigger:** a `no-network` run calling `search_my_documents`.
- **Consequence:** measured, `tool_specs(profile="no-network")` and
  `tool_specs(profile="confined")` both advertise `search_my_documents`, which
  performs an HTTP POST to the local RAG sidecar (`rag.ask`). It is not in
  `_NETWORK_TOOLS` and its `kind` is empty, so neither profile filter sees it. A
  local call is not the same as egress, but `no-network` reads as "this run cannot
  reach the network" and the tool is the one outbound HTTP call left.
- **Fix:** NOT implemented. Smallest change: add `search_my_documents` to
  `_NETWORK_TOOLS` (or give it `kind="network"` and filter that).

## Checked and NOT exploitable (so nobody re-runs it)

- `..`, `....//`, mixed `/` and `\`, `sub/../x`, `./x`, trailing dots and spaces,
  `C:foo` (drive-relative), `\\?\C:\…`, `\\.\PhysicalDrive0`,
  `\\localhost\C$\…`, `/Windows/System32/...` — all either collapse inside the
  workspace or are refused by `_ws_path`/`_read_path`. Notably `Path.resolve()`
  normalises `notes.txt.` and `.env ` to the real name, so the credential
  denylist catches the trailing-dot/space spelling of `.env` (measured: DENIED).
- 8.3 short names: `resolve()` expands them, so containment and the denylist are
  both tested on the long name.
- Case-insensitivity: both sides of the containment test go through `resolve()`.
- The MCP spill filename: `serve._spill_big_result` builds
  `rigma_home()/results/<tool name>-<hex>.txt` from the raw (server-chosen) tool
  name, but MCP names are always prefixed `mcp__<server>__`, so the first path
  component is a directory that cannot be made to exist (`write_file` is
  workspace-confined). Measured with a crafted name: the write failed and the
  function fell back to `_clip`. Not a traversal.
- `rescue_tool_call` prose shapes: a fenced JSON block inside prose, an XML call
  inside prose, an XML call with trailing text, a fenced XML call, prose followed
  by a bare Action block, and a trailing-text Action block are all refused
  (measured); a bare call, two adjacent XML calls, and a bare JSON argument
  object are still rescued.
- The `sentinel` flag is a property of the tool, and only `use_tools`,
  `delegate`, `view_image`, `view_images`, `view_sample` set it; their payloads
  are `json.dumps`-escaped or real resolved paths, so R3-3's fix cannot be used
  to forge a sentinel.

## Could not execute

- The TOCTOU in R3-13 (needs a junction swap raced against a live write across
  two concurrent tool calls; the R3-1 fix removes the walker half).
- Windows symlinks (as opposed to junctions): creating one needs a privilege this
  session does not have (`WinError 1314`, reproduced). Junctions cover the same
  code path and need no privilege, so the finding stands on the junction.
- The larger R3-16 ReDoS cases (x12, x14) — deliberately not run, since they are
  the hang the finding describes; the growth is measured up to x10.
- Live reachability of R3-10/R3-11 end-to-end (writing to a real Startup folder,
  or indexing a real home directory) — both were verified at the guard/route
  layer only, which is where the missing check is.
- `serve.py`'s run-loop budget/profile plumbing beyond the `tctx` construction at
  `serve.py:2049-2071` and the sentinel/spill call sites; whether a given profile
  is reachable from the UI is out of this area.

<!-- coverage: src/rigma/tools.py L36-155, L185-270, L286-470, L563-605, L660-790, L1131-1220, L1328-1540, L1545-1680, L1752-1935, L1942-2200, L2507-2760, L2762-2960, L2956-3100, L3098-3310, L3339-3700, L3700-3930 (read); src/rigma/workspace.py (read fully); src/rigma/watch.py (read fully); src/rigma/mcp_server.py (read fully); src/rigma/mcp_client.py (read fully); src/rigma/rag.py L1-110, L305-365 (read); src/rigma/serve.py L2030-2145, L2290-2320, L3576-3600, L5480-5505 (read); src/rigma/runs.py (progress-log sites, grepped) -->
