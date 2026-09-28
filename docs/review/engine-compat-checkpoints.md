# Engine compatibility — stage checkpoints

Resumable work log for the engine/quant compatibility objective. **Read this file
first** in a new session; it is written so that no prior conversation is needed.

* Goal id: `goal-94877b0c-81bb-457b-b76b-d3564863bf4c`
* Fork: `C:\ComfyUI\RD\rigma-review` (branch `review/deep-audit-2026-09-22`)
* Original repo `C:\ComfyUI\RD\rigma` is **READ-ONLY** and must stay untouched
* Venv to use: `C:\ComfyUI\RD\rigma\.venv\Scripts\python.exe` (the fork has none)

---

## Standing directives from the user

These are the user's own words, carried forward so a new session inherits them
without the user having to repeat anything.

1. **Fork only.** "You can Modify the Whole Code-Base if you deem it to be, Just
   make sure that The work is being done in a Fork/Branch and not the Original repo
   itself."
2. **Reason for everything.** "Everything that you do has to have a Reason, A cause
   and A Fore-seeable Solution/Resolution."
3. **Judge UI/UX too**, and improve it where warranted.
4. **Delegate the bulk.** "Your Context is 262K, So pretty small, Do More than 50%
   of your Work or More, Through Sub-Agents and Workflows." Research subagents must
   be `provider: tokenjuice`, `model: deepseek-ai/DeepSeek-V4.1-Flash` — the same
   route the main agent runs on.
5. **Don't force it.** "You don't have to Force anything, It it Works and the Margin
   for Improvement is not that high, Just leave it be."
6. **Research online, then implement.** "Research every point online. Find a Solution
   that they Deem Right, Then Implement."
7. **This objective.** Solve the engine/quant compatibility gap "as many Model
   Architectures as you can, and As Many Engines as you can. Including but not
   Limited to Voice Models. Go ahead on full throughput, Leave Documented Checkpoints
   in Docs on each Stage that you pass and complete."
8. **Method for web research.** DuckDuckGo's `lite.` and POST endpoints serve an
   anti-bot CAPTCHA to this machine. **Use `web_fetch` on
   `https://html.duckduckgo.com/html/?q=...`** — it egresses from a different address
   and works. Prefer `raw.githubusercontent.com` source over blog posts.

### House rules that have bitten before

* **Never blanket-kill processes.** Track PIDs/job ids; stop only what you started.
  A blanket `Get-Process python | Stop-Process` previously destroyed a live ComfyUI
  session.
* **`pytest -q` suppresses the final summary line on this build.** Read totals from
  `--junit-xml`, not the console tail.
* **`pytest-timeout` is NOT installed** — never pass `--timeout`.
* Set `PYTHONPATH` **inline in every `pwsh` call** (it does not persist):
  `"C:\ComfyUI\RD\rigma-review\src;C:\ComfyUI\RD\rigma-review\.scratch\00-recon\plug"`
* Run pytest with `-p no:cacheprovider -p dsh_tmpfix --basetemp=<unique dir>`.
* `git commit -F <file>` — PowerShell has no heredoc. Use
  `git -c core.hooksPath= commit` to bypass the pre-commit hook when needed.
* `docs/*` is **gitignored**; force-add (`git add -f`) anything that must persist.

---

## The defect

Rigma planned `ternary-bonsai-2-27b` (quant `PQ2_0`, ctx 262144) and its own pinned
engine could not load the model at all:

```
gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 42)
```

## Root cause (VERIFIED, and not the obvious one)

The obvious reading — "the pin is too old, bump the pin" — is **wrong**, and the
truth is worse in a more interesting way. Three separate facts, each measured:

### Fact 1 — Rigma never verifies which engine build it is running

Four directories, all under `~/.rigma/engines/b9867/`, i.e. all claiming to be the
manifest's pinned version:

| directory | `--version` says | loads type 142? |
|---|---|---|
| `b9867/cpu` | `9867 (152d337fa)` | **NO** |
| `b9867/vulkan` | `9867 (152d337fa)` | **NO** |
| `b9867/rocm-mainline-b9867` | `9867 (152d337fa)` | **NO** |
| `b9867/rocm` | `0.2.0-dev (build 10709, commit 9a9394a89)` | **YES** |

`runtime.ensure_engine()` returns the binary if `exe.exists()` and a `.ready`
sentinel is present — **it never runs `--version`**. And
`server_ops.engine_version()` returns **the manifest's version string**, not the
binary's. So calibration could be labelled `b9867` while measured on build 10709,
and `bench.calibration_stale`'s engine check compares the manifest against itself
and can never fire.

### Fact 2 — the newer build is a THIRD-PARTY FORK, not a newer mainline

`~/.rigma/engines/b9867/rocm/ENGINE-PROVENANCE.txt`, written by the owner by hand:

```
This is NOT the mainline llama.cpp ROCm build rigma pins.

  PrismML-Eng/llama.cpp, branch 'prism', release prism-b10709-9a9394a
  (tracks upstream b10709; adds PQ2_0/PTQ1_0 ternary types plus the runtime
   Walsh-Hadamard activation transform those weights require)

Installed by hand because Ternary-Bonsai-2-27B cannot run on any stock build:
mainline rejects PQ2_0 as an unknown type, and the Vulkan backend silently
falls back to CPU for ternary tensors (~0.06 t/s measured by others).

rigma's ENGINE_URL_ALLOWLIST is untouched; this rides the .ready short-circuit
in runtime.ensure_engine rather than the download path.
```

So the user had already worked out the correct fix and applied it by hand, and
Rigma had no way to represent it.

### Fact 3 — **bumping the pin cannot fix this**, confirmed against both headers

Mainline `ggml.h` (fetched from `master`) *does* now have these types:

```
GGML_TYPE_Q1_0    = 41,
GGML_TYPE_Q2_0    = 42,
GGML_TYPE_COUNT   = 43,      <-- so b9867's [0, 42) is "before Q1_0/Q2_0"
```

But the model does not use those. The fork's `ggml.h`:

```
GGML_TYPE_Q1_0    = 41,
GGML_TYPE_Q2_0    = 42,
GGML_TYPE_PQ2_0 = 142,       // Prism-private
GGML_TYPE_PTQ1_0 = 143,      // Prism-private ternary, group 128
GGML_TYPE_COUNT   = 144,
```

**`PQ2_0` is 142 and `PTQ1_0` is 143 — private numbers far above mainline's
maximum of 42/43.** No mainline build, however new, will ever load this model. The
fork also needs a runtime Walsh-Hadamard activation transform, which is a *behaviour*
difference a version number cannot express.

### The actual defect, stated once

Rigma has exactly one engine: whatever the manifest pinned. It cannot:
1. **verify** the engine it is running is the one it thinks it is;
2. **detect** that a model needs a different engine before failing at load with an
   opaque `ggml type` error;
3. **represent** a non-stock engine at all — `ENGINE_URL_ALLOWLIST` is
   `("https://github.com/ggml-org/llama.cpp/releases/download/",)`, and there is no
   custom-engine mechanism, which is why the owner's fork had to be smuggled in via
   the `.ready` short-circuit.

### Consequence chain

1. Engine build identity is fiction (a manifest string, not a measurement).
2. Nothing can detect a model/engine incompatibility — it appears only at load.
3. Calibration records the wrong engine and cannot go stale on an engine change.
4. Behaviour differs per backend directory for reasons nothing explains.
5. Users with a model that needs a fork are silently unsupported and must hand-hack
   the install directory.

## What is already verified about the fix's building blocks

* **`--version` works and is precise**, on both `llama-server` and
  `llama-fit-params`, for every build tested. Two formats exist:
  * `version: 9867 (152d337fa)` — the `bNNNN` scheme
  * `version: 0.2.0-dev (build 10709, commit 9a9394a89)` — the newer scheme
  * both are followed by `built with Clang X for Windows <arch>`
* **`llama-fit-params` parses a GGUF without starting a server** and reports the
  `ggml type` rejection in ~0.1 s, so a model's loadability can be probed cheaply
  *before* committing to a long download or a launch.
* `--log-jsonl` is **absent** from b9867 (answers `invalid argument`).

---

## Stage log

### Stage 0 — scope and reproduce · **COMPLETE**

Reproduced the defect; established the three facts above; confirmed `--version` is a
usable identity source and `llama-fit-params` a usable capability probe; enumerated
the four builds and which load the model; read the owner's `ENGINE-PROVENANCE.txt`
and both `ggml.h` headers to prove a pin bump cannot help.

**Artifact shipped:** `src/rigma/engine_build.py` + `tests/test_r3_engine_build.py`
(16 tests, all green). Parses both version schemes, compares builds by NUMBER rather
than by string, and degrades to `ok=False` rather than inventing a version. Verified
against all four real binaries:

```
cpu                  -> id=b9867+152d337fa    build=9867  pin_match=True
rocm                 -> id=b10709+9a9394a89   build=10709 pin_match=False
rocm-mainline-b9867  -> id=b9867+152d337fa    build=9867  pin_match=True
vulkan               -> id=b9867+152d337fa    build=9867  pin_match=True
```

### Stage 1 — ggml type registry · **COMPLETE**

Output: [`_research/reports/ggml-type-registry.md`](../../_research/reports/ggml-type-registry.md)
(521 lines, every claim URL-cited, READ vs INFERRED marked, 9 declared gaps).

**It refuted the task's premise, which is the most valuable thing it could have done.**
The brief assumed type 142 was added to llama.cpp between b9867 and build 10709 — a
version lag. It is not. A commit search over `ggml-org/llama.cpp` for `PQ2_0` returns
`total_count: 0`; type 142 is `GGML_TYPE_PQ2_0`, private to the fork, and mainline's enum
stops at `Q2_0 = 42` / `COUNT = 43`. **No mainline build, however new, will ever load
this model.**

Other findings that changed the design:

* **`bNNNN` and `build NNNNN` are the SAME counter** (`git rev-list --count HEAD`), not
  two schemes. `b9867` → commit `152d337fa`, dated 2026-07-03.
* **The `[0, N)` bound is `GGML_TYPE_COUNT`** — a raw enum count including seven
  permanently-unused holes. Unreliable as a capability signal, but a precise
  **provenance fingerprint**: 42 = mainline pre-Q2_0, 43 = mainline now, 144 = PrismML.
* **No runtime type query exists.** No flag or API enumerates supported types; the only
  reliable probe is attempting a load and reading the error.
* **The load failure is backend-independent.** `gguf.cpp` validates against
  `GGML_TYPE_COUNT` before any backend is chosen, so CPU/Vulkan/ROCm-mainline all fail
  identically — the backend split is a red herring for this defect.
* **PQ2_0 has no Vulkan kernel even on the fork that defines it**; its README points
  Vulkan users at the group-64 `Q2_0` file.
* **A false negative in my own Stage 1 check** — see Stage 4, design decision 2.

### Stage 2 — engine versioning landscape · **COMPLETE**

Output: [`_research/reports/engine-versioning-landscape.md`](../../_research/reports/engine-versioning-landscape.md)
(391 lines, 18 declared gaps).

* **The GitHub Releases API already publishes `assets[].digest = "sha256:<hex>"`**, and
  recent builds carry **sigstore attestations** binding artifact → commit. b11146 has
  one; **b9867 returns 404** (it predates the `actions/attest@v4` step). So Rigma gets
  strong verification on new builds and digest-only on b9867-era ones.
* **GGUF has NO minimum-engine-version key**, and `general.quantization_version` is a
  dead signal (`GGML_QNT_VERSION` unchanged at 2). The tensor type enum is the only
  capability signal — which is what Stages 1 and 5 act on.
* **`general.file_type` is a trap**: the spec enum and `ggml.h`'s `ggml_ftype` use the
  same integers with different meanings. Do not build a table from the spec doc.
* **Asset names DRIFT between builds** (`hip-radeon` → `rocm-10.0`), and the Windows
  cudart asset has no build number and is byte-identical across releases. A pinned
  downloader must not template asset names.
* **None of the six tools surveyed auto-upgrades on a too-new model** — all fail loudly
  — and **no tool parses `--version` to confirm engine identity.** Rigma doing so is
  ahead of the field, not behind it.
* **Disk**: ≈373 MB per Windows build, so ten retained builds ≈ 3.7 GB. Retention is the
  decision, not download.

### Stage 3 — model architecture coverage · **COMPLETE**

Output: [`_research/reports/model-architecture-coverage.md`](../../_research/reports/model-architecture-coverage.md)
(570 lines).

* **b9867 is dated 2026-07-03**, not mid-2025 as the brief assumed.
* **The real coverage gap is ROLE, not the arch list.** llama.cpp master has 153 LLM
  arch strings (b9867: 135), but `ModelSpec` has no embedding / reranker / ASR / TTS /
  diffusion role. All four non-chat families ARE reachable from a pinned build:
  * ASR → `/v1/audio/transcriptions`, requires an audio `mmproj`
  * embeddings → `--embedding` + `/v1/embeddings`
  * rerankers → `--embedding --pooling rank` + `/v1/rerank`
  * diffusion LMs → `dream` / `llada` / `llada-moe` / `rnd1`
* **TTS exists in llama.cpp now but is CLI-only** (`qwen3tts`, `pockettts`; `llama-tts`
  needs two projectors). **There is no `/v1/audio/speech` route anywhere in
  llama-server**, so a chat-shaped UI cannot serve TTS.
* **Vision/audio are a SECOND namespace**: `clip-impl.h` has 62 projector type strings
  independent of the LLM arch. SmolVLM / Pixtral / Moondream run on `llama` / `mistral` /
  `phi2` base archs, so "is this multimodal?" is not answerable from
  `general.architecture` alone — the projector type string distinguishes vision vs audio
  vs TTS before launch.
* Best "one OpenAI client for text + voice": llama-server for text/ASR plus a sidecar
  (`speaches` or LocalAI) for TTS.

### Stage 4 — design decision · **COMPLETE**

The evidence forced a design that is **not** "bump the pin". Rigma had three distinct
gaps, each needing its own mechanism:

| gap | mechanism | why the obvious fix fails |
|---|---|---|
| cannot **verify** the engine it runs | measure `--version` | there was nothing to compare against |
| cannot **detect** an incompatible model | read the GGUF's own tensor types | the failure is opaque and arrives at load |
| cannot **represent** a non-stock engine | an explicit registry | `ENGINE_URL_ALLOWLIST` is mainline-only |

Design decisions, each traceable to a specific finding:

1. **Capability beats version.** Selection asks "can this build load *this* model's
   tensor types" — a fact we can check — rather than comparing version strings, which
   is a guess. Stage 1 proved a version number cannot express a fork.
2. **The bound belongs to the ENGINE, not a constant.** Stage 3 found that
   `MAINLINE_MAX_TYPE = 42` describes master while the pin has `COUNT = 42` and rejects
   id 42. A module-level bound passed Q2_0 files that then died on the pin.
3. **Provenance, not a minimum build number.** The counter is shared between mainline
   and the fork, so build number cannot distinguish them. The `[0, N)` fingerprint can.
4. **Unknown is never a refusal.** Every check returns "cannot say" rather than "cannot
   load" when evidence is missing, because a false block refuses models that work.
5. **`ENGINE_URL_ALLOWLIST` untouched.** Auto-downloading third-party binaries is a
   trust decision for the user, not a side effect of wanting to run a model. The pin
   stays the default and registration is purely additive.

### Stage 5 — implementation · **COMPLETE**

Six commits, each with tests, `ruff` clean throughout:

| commit | what |
|---|---|
| `d97a373` | `engine_build.py` — measure the build, parse both version schemes |
| `3249a8e` | `engine_compat.py` + `gguf_meta` type histogram — predict from the file |
| `eb408e0` | `engine_registry.py` + `rigma engines` / `engine-register` / `engine-forget` |
| `392e856` | the engine-specific bound (a false negative in the above) |
| `d6f5d94` | wire into `--verify`; do not cry wolf about a model that loads |
| `f0c664c` | calibration records the MEASURED engine, so it can go stale |

**Verified live on the owner's real machine**, not only in tests:

```
$ rigma engines
pinned     vulkan                 b9867+152d337fa        the pin (b9867)
pinned     rocm                   b10709+9a9394a89       NOT the pin (b9867)
pinned     cpu                    b9867+152d337fa        the pin (b9867)
note: a directory claiming to hold the pin holds a different build. That is
allowed — some models need a fork — but any calibration measured on it is not a
measurement of the pin.
```

`_engine_compat_note`, on three real combinations:

| engine | model | result |
|---|---|---|
| pinned `vulkan` (COUNT 42) | Ternary-Bonsai PQ2_0 | explains: names PrismML, says a pin bump will not help, adds the Vulkan kernel note |
| PrismML fork (COUNT 144) | the same model | **silent** — it loads, so there is nothing to warn about |
| pinned `vulkan` | SmolLM2 Q2_K | silent |

`server_ops.engine_version()` before → after:

```
backend=''        -> 'b9867+152d337fa'
backend='vulkan'  -> 'b9867+152d337fa'
backend='rocm'    -> 'b10709+9a9394a89'    <- the fork, no longer disguised
backend='cpu'     -> 'b9867+152d337fa'
```

The engine fingerprint, read from each build's own `[0, N)` refusal:

```
cpu / vulkan / rocm-mainline-b9867  -> 42    (refuse the ternary model)
rocm (the PrismML fork)             -> None  (LOADS it, so it never refuses)
```

**Two bugs I found in my own work by testing rather than by reasoning:**

1. **Cried wolf.** The first wiring ran the model-side check unconditionally, so pointed
   at the fork *installed to load this model* it still said "THIS ENGINE CANNOT LOAD
   THIS MODEL". Fixed by letting the engine override the model-side guess: a load that
   succeeds is authoritative.
2. **The bound was a constant** (design decision 2) — found by the Stage 3 research
   reviewing my Stage 1 code. It is the more dangerous of the two because it fails
   silently in the *permissive* direction: it says "fine" about a file the engine
   cannot open.

### Stage 6 — verification and handoff · **COMPLETE**

* **Full suite: 2502 tests, 0 failures, 0 errors, 3 skipped** (JUnit XML, `-m "not
  hardware"`), up from 2410 before this objective. `ruff check src tests tools` clean.
* Working tree clean; **original repo `C:\ComfyUI\RD\rigma` untouched**; upstream push
  still disabled (`DISABLED-no-push-from-fork`).
* **The intermittent suite hang documented in `RUN3.md` recurred once** (73 s CPU then
  stalled, no XML). It is not a regression — the same selection passed cleanly when
  re-run — but `pytest-timeout` is still absent, so a hang must be capped from outside.
  **This remains the one piece of test infrastructure worth fixing.**

### Crash recovery, 2026-09-27

A power loss during this objective left `.git/refs/heads/review/deep-audit-2026-09-22`
**empty**, so `git` reported "your current branch appears to be broken" and every ref
looked like `invalid sha1 pointer 0000000000000000000000000000000000000000`.

No work was lost. The reflog (`git/logs/HEAD` and `git/logs/refs/heads/...`) retained
the full commit chain, the commit object `f0c664cc` was intact, and the working tree was
untouched. `git update-ref` **refuses** to repair a broken ref ("unable to resolve
reference … reference broken"), so the fix was to write the SHA into the ref file
directly. `git fsck` is clean afterwards.

**Worth knowing if this recurs:** the reflog is what saved this, so do not run
`git reflog expire` or `git gc --prune` on this repo until the work is pushed
somewhere. There is no remote backup — `upstream` is a local path with push disabled.

---

## What is NOT done, and what a follow-up should pick up

Ordered by value. Each is a real, evidenced gap rather than a nice-to-have.

1. **`ModelSpec` has no role for embedding / reranker / ASR / TTS / diffusion.** Stage 3
   showed all four non-chat families are reachable from a pinned build and none can be
   expressed. This is the largest remaining coverage gap and it is a *data model* change
   (`family` / `kind` / `capabilities` would gain a role axis), not a flag.
2. **TTS cannot be served by this UI.** There is no `/v1/audio/speech` route anywhere in
   llama-server, so a sidecar (`speaches`, LocalAI) is the only route. Whether Rigma
   should manage a sidecar is a product decision, not a technical one.
3. **Download verification.** Stage 2 found the Releases API publishes
   `assets[].digest` (SHA256) and recent builds are sigstore-attested. Rigma records a
   first-use digest but never compares it against the published one. Comparing would be
   a genuine improvement on new builds, and is impossible on b9867-era ones (404).
4. **Asset-name drift.** `windows/rocm` maps to `llama-b9867-bin-win-hip-radeon-x64.zip`,
   but newer releases call it `rocm-10.0`. Rigma's manifest is a single pinned version,
   so this is latent rather than broken — it breaks the moment the pin moves.
5. **Retention.** ≈373 MB per Windows build. Registering engines is additive with no
   pruning, so a user who registers many will accumulate disk.
6. **`pytest-timeout`.** Not installed, so the documented hang cannot be capped from
   inside. This has now cost time twice.
7. **The `general.quantization_version` question.** Stage 2 could not verify whether it
   has ever been bumped above 2, and the spec's own "the quantization version is 4"
   example is unexplained. If it has moved, it becomes a usable signal.

## Open questions — final status

1. ~~What is ggml type 142?~~ **ANSWERED**: `GGML_TYPE_PQ2_0`, fork-private. Never in
   mainline; a commit search returns zero results.
2. ~~Is it backend-specific?~~ **ANSWERED**: the *load* failure is backend-independent
   (`gguf.cpp` checks `GGML_TYPE_COUNT` before any backend is chosen). Separately, PQ2_0
   has no Vulkan *kernel*, so it loads nowhere useful on Vulkan. "Can it load" and "is
   it usable" are two different questions, and the code now says both.
3. ~~Can a minimum build be derived?~~ **ANSWERED**: yes for mainline types (integer
   compare of `bNNNN`), but **not** for fork types, where the counter is shared and
   therefore meaningless. Hence provenance.
4. ~~Does llama.cpp publish checksums?~~ **ANSWERED**: yes — `assets[].digest` in the
   Releases API, plus sigstore attestations on recent builds.
5. ~~Can a GGUF declare a minimum engine version?~~ **ANSWERED**: no such key exists.
   The tensor type enum is the only signal.
6. ~~Which voice/audio stacks are in scope?~~ **ANSWERED** by Stage 3, and the answer is
   "not via this UI": ASR yes (`/v1/audio/transcriptions`), TTS no (no route).
7. ~~Did Rigma produce the mislabelled directory?~~ **ANSWERED**: no. The owner installed
   it by hand and documented why.
8. **Should Rigma ever auto-download a fork?** **DECIDED, pending the owner's
   confirmation**: no. Registration is opt-in and the allowlist is untouched. If the
   owner wants Rigma to fetch forks, that is a policy change with a real trust cost and
   should be an explicit decision rather than a default.
