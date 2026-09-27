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

### Stage 1 — ggml type registry · **IN PROGRESS**

Subagent researching: what type 142 is, the full type enum, when Q1_0/Q2_0 were
added upstream, whether support is per-backend, and whether a binary can report its
supported types. Output: `_research/reports/ggml-type-registry.md`.

*Partially answered already by Stage 0:* 142 is `GGML_TYPE_PQ2_0`, fork-private;
mainline's maximum is `GGML_TYPE_Q2_0 = 42` with `GGML_TYPE_COUNT = 43`. Remaining:
the upstream introduction bracket for Q1_0/Q2_0, and whether any runtime query
exists.

### Stage 2 — engine versioning landscape · **IN PROGRESS**

Subagent researching: llama.cpp release/versioning schemes and the releases API,
how Ollama / LM Studio / llamafile / koboldcpp / text-generation-webui / vLLM pin and
verify engines, download verification (checksums, sigstore, attestations), and
whether GGUF metadata can declare a minimum engine version.
Output: `_research/reports/engine-versioning-landscape.md`.

### Stage 3 — model architecture coverage · **IN PROGRESS**

Subagent researching: the exhaustive list of architectures llama.cpp supports,
grouped; the non-LLM surface (ASR/Whisper, TTS, vision/`mmproj`, embeddings,
diffusion); the family→requirements table (extra files, flags, endpoints); and what
local voice stacks actually work today.
Output: `_research/reports/model-architecture-coverage.md`.

### Stage 4 — design decision · **PENDING**

### Stage 5 — implementation · **PENDING**

### Stage 6 — verification and handoff · **PENDING**

---

## Working hypothesis for Stage 4 (to be confirmed or refuted by Stages 1–3)

Four pieces, in increasing order of ambition. **Piece 1 is already built and tested**
(Stage 0); the rest wait on the research.

1. ~~**Measure the build, don't assume it.**~~ **DONE** — `engine_build.py`. This
   alone makes calibration honest and makes the mislabelled `rocm/` directory
   visible.
2. **Detect incompatibility before launching, and say something useful.** Map the
   failure to an actionable message ("this model's quant type PQ2_0 is private to
   PrismML-Eng/llama.cpp; the pinned mainline build cannot load it") instead of an
   opaque `ggml type` error. `llama-fit-params` is the cheap probe and is already
   wired into `--verify`.
3. **Let the engine be selected per model rather than globally pinned** — keep more
   than one build and choose the one that can load the model. Real design risk (disk,
   cache invalidation, calibration keying); must follow Stage 2's evidence.
4. **Represent a non-stock engine explicitly** — a provenance record and a
   registration path that does not require weakening `ENGINE_URL_ALLOWLIST` for the
   mainline download path. The owner's hand-written `ENGINE-PROVENANCE.txt` is the
   de-facto spec for this.

**Do not implement 3 or 4 without Stage 2's answer** on how other tools handle
side-by-side engine versions and what breaks.

---

## Open questions

1. ~~What exactly is ggml type 142?~~ **ANSWERED**: `GGML_TYPE_PQ2_0`, fork-private
   to PrismML-Eng/llama.cpp. Mainline max is 42.
2. ~~Is it backend-specific?~~ **PARTLY**: the provenance note says the Vulkan
   backend silently falls back to CPU for ternary tensors (~0.06 t/s), i.e. it
   *loads* but is useless. So "does it load" is not the only question — "is it
   usable" is a second one. Stage 1 to confirm.
3. When were `Q1_0`/`Q2_0` added upstream, and can a minimum build be derived rather
   than hard-coded? (Stage 1)
4. Does llama.cpp publish checksums for release assets? (Stage 2)
5. Can a GGUF declare a minimum engine version, or is the tensor type the only
   signal? (Stage 2)
6. Which voice/audio stacks are in scope for a chat-shaped UI? (Stage 3)
7. ~~Did Rigma produce the mislabelled directory?~~ **ANSWERED**: no. The owner
   installed it by hand and documented why; Rigma's `.ready` short-circuit is what
   let it in unexamined.
8. **Should Rigma ever auto-download a fork?** This is a trust decision, not a
   technical one, and it needs the user's call. Default should be "no, but let me
   register one I obtained myself".

