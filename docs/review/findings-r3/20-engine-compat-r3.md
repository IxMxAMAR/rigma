# R3-ENG-1..6 — the engine/quant compatibility gap

**Status:** fixed, tested, verified live. Six commits, 2502 tests passing.
**Branch:** `review/deep-audit-2026-09-22` in the fork `C:\ComfyUI\RD\rigma-review`.
**Full handoff:** [`docs/review/engine-compat-checkpoints.md`](../engine-compat-checkpoints.md).

---

## The symptom

Rigma planned `ternary-bonsai-2-27b` (quant `PQ2_0`, ctx 262144) and its own pinned
engine could not load the model at all:

```
gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 42)
```

## The obvious reading, and why it is wrong

"Rigma's engine pin is too old; bump it." That is wrong, and acting on it would have
shipped a fix that changed nothing.

`GGML_TYPE_PQ2_0 = 142` is **private to a third-party fork**,
`PrismML-Eng/llama.cpp` branch `prism`. A commit search over `ggml-org/llama.cpp` for
`PQ2_0` returns `total_count: 0`. Mainline's enum stops at:

```c
GGML_TYPE_Q1_0    = 41,
GGML_TYPE_Q2_0    = 42,
GGML_TYPE_COUNT   = 43,      // so b9867's [0, 42) is "before Q1_0/Q2_0"
```

while the fork allocates:

```c
GGML_TYPE_PQ2_0 = 142,       // Prism-private
GGML_TYPE_PTQ1_0 = 143,      // Prism-private ternary, group 128
GGML_TYPE_COUNT   = 144,
```

142 is not a newer mainline type. It is **different numbering**, and the fork also needs
a runtime Walsh-Hadamard activation transform that is not upstream — a *behaviour*
difference no version number can express. **No mainline build, however new, will ever
load this model.**

## The real defect: three gaps, not one

### Gap 1 — Rigma never verified which engine it was running

Four directories on the owner's machine, **all under `~/.rigma/engines/b9867/`**, i.e.
all claiming to be the pin:

| directory | `--version` says | loads type 142? |
|---|---|---|
| `b9867/cpu` | `9867 (152d337fa)` | **NO** |
| `b9867/vulkan` | `9867 (152d337fa)` | **NO** |
| `b9867/rocm-mainline-b9867` | `9867 (152d337fa)` | **NO** |
| `b9867/rocm` | `0.2.0-dev (build 10709, commit 9a9394a89)` | **YES** |

`runtime.ensure_engine()` returned whichever binary existed, checking only for the file
and a `.ready` sentinel. It never ran `--version`. And `server_ops.engine_version()`
returned **the manifest's version string** — so the same model loaded or failed depending
on which directory happened to hold a newer binary, for reasons nothing surfaced.

### Gap 2 — a model/engine incompatibility could not be detected before load

The information needed was **in the model's own tensor table the whole time**.
`gguf_meta._read_tensors` already walked every tensor and already read the ggml type id
(it sits between the dims and the offset, so it cannot be skipped) — and discarded it.

### Gap 3 — a non-stock engine could not be represented

The owner had already worked out the correct fix and applied it **by hand**, leaving
`ENGINE-PROVENANCE.txt` beside the fork:

```
This is NOT the mainline llama.cpp ROCm build rigma pins.
  PrismML-Eng/llama.cpp, branch 'prism', release prism-b10709-9a9394a
  ...
rigma's ENGINE_URL_ALLOWLIST is untouched; this rides the .ready short-circuit
in runtime.ensure_engine rather than the download path.
```

It worked, and that was the problem: Rigma could not name it, verify it, select it, or
restore the build it displaced. `ENGINE_URL_ALLOWLIST` is mainline-only and there is no
custom-engine path, so hand-editing the install directory was the only option.

### The consequence chain

1. Engine identity was fiction (a manifest string, not a measurement).
2. Nothing could detect an incompatibility — it appeared only at load, opaquely.
3. Calibration recorded the wrong engine and **could never go stale on an engine
   change**, because `calibration_stale` compared the manifest string against itself.
4. A user with a model needing a fork was silently unsupported.

---

## The fix

| module | what it does |
|---|---|
| `engine_build.py` | runs `--version`, parses both schemes, compares builds by NUMBER, degrades to `ok=False` rather than inventing a version |
| `engine_compat.py` | judges a GGUF's tensor types against an engine's own `GGML_TYPE_COUNT`, and fingerprints the engine from its `[0, N)` refusal |
| `engine_registry.py` | registered non-stock engines, with capability-based selection |
| `gguf_meta.py` | captures the tensor-type histogram that was already being read and thrown away |
| `cli.py` | `rigma engines`, `rigma engine-register`, `rigma engine-forget`, and the `--verify` pre-flight check |
| `server_ops.py` / `bench.py` / `hwid.py` / `resolve.py` | calibration records and compares the MEASURED build |

The opaque error becomes an actionable diagnosis:

```
before: invalid ggml type 142. should be in [0, 42)
after:  this engine's type table ends at 42 (mainline llama.cpp: before Q2_0 was
        appended (this is b9867)); this model uses PQ2_0=142, which is private to
        PrismML-Eng/llama.cpp — mainline llama.cpp numbers its types only up to 42
        (Q2_0), so NO mainline build can load this file
        fix: install a build of PrismML-Eng/llama.cpp and point Rigma at it;
        upgrading the pinned mainline engine will not help
```

### Design decisions, each traceable to evidence

1. **Capability beats version.** Selection asks "can this build load *this* model's
   tensor types" — a fact we can check — rather than comparing version strings, which is
   a guess.
2. **The bound belongs to the ENGINE, not a constant.** See the self-caught bug below.
3. **Provenance, not a minimum build number.** The counter is shared between mainline and
   the fork, so build number cannot distinguish them. The `[0, N)` fingerprint can.
4. **Unknown is never a refusal.** Every check returns "cannot say" rather than "cannot
   load" when evidence is missing, because a false block refuses models that work.
5. **`ENGINE_URL_ALLOWLIST` untouched.** Auto-downloading third-party binaries is a trust
   decision for the user, not a side effect of wanting to run a model.

---

## Two bugs I found in my own work

Both were found by testing rather than by reasoning, and both are recorded because the
second is the more instructive.

### 1. It cried wolf

The first `--verify` wiring ran the model-side check unconditionally. Pointed at the
**PrismML fork installed specifically to load this model**, it still printed "THIS ENGINE
CANNOT LOAD THIS MODEL". A warning about a model that loads is worse than no warning.

Fixed by letting the engine override the model-side guess: `_engine_type_count` returns
`None` precisely when the engine did *not* refuse the file, and that is authoritative.

### 2. The bound was a module constant (a false negative)

`engine_compat.MAINLINE_MAX_TYPE = 42` describes **master**. The **pinned b9867** reports
`should be in [0, 42)`, i.e. `GGML_TYPE_COUNT = 42`, i.e. it accepts `0..41` —
`GGML_TYPE_Q2_0 = 42` was appended on master only (2026-07-07, after b9867's
2026-07-03). So a Q2_0 file **passed Rigma's check and then died on the pinned engine**.

This is the more dangerous direction: it fails silently in the *permissive* way. A check
that says "fine" about a file the engine cannot open is worse than no check at all. The
bound is now a property of the build, and both directions are tested — b9867 must reject
Q2_0 (or a file fails at load) *and* must accept Q1_0 = 41 (or a file that loads fine is
refused).

The two failure modes now also get **opposite advice**, which is the point of separating
them:

* too-new mainline type → "upgrade the engine build; no fork is needed"
* fork-private type → "install PrismML-Eng/llama.cpp; upgrading will not help"

Telling a Q2_0 user to install a fork would send them somewhere useless, and vice versa.

---

## Verified live on the real machine

Not only in tests. `rigma engines`:

```
pinned     vulkan                 b9867+152d337fa        the pin (b9867)
pinned     rocm                   b10709+9a9394a89       NOT the pin (b9867)
pinned     cpu                    b9867+152d337fa        the pin (b9867)
note: a directory claiming to hold the pin holds a different build. That is
allowed — some models need a fork — but any calibration measured on it is not a
measurement of the pin.
```

`server_ops.engine_version()` before → after:

```
backend=''        -> 'b9867+152d337fa'
backend='vulkan'  -> 'b9867+152d337fa'
backend='rocm'    -> 'b10709+9a9394a89'    <- the fork, no longer disguised
backend='cpu'     -> 'b9867+152d337fa'
```

The engine fingerprint, read from each build's own refusal to load the ternary model:

```
cpu / vulkan / rocm-mainline-b9867  -> 42    (refuse it)
rocm (the PrismML fork)             -> None  (LOADS it, so it never refuses)
```

And the pre-flight verdict on three real combinations:

| engine | model | result |
|---|---|---|
| pinned `vulkan` (COUNT 42) | Ternary-Bonsai PQ2_0 | explains, names PrismML, adds a Vulkan note |
| PrismML fork (COUNT 144) | the same model | **silent** — it loads |
| pinned `vulkan` | SmolLM2 Q2_K | silent |

### A regression this would have caused, and the guard against it

Engine strings now come from two producers: older calibration entries recorded the bare
`b9867`, current code records `b9867+152d337fa`. `hwid.soft_reasons` compared them with
`!=`, which would have called them different and **silently discarded every valid
calibration on upgrade**. `engine_build.same_build` compares by build number and, when
both sides carry one, by commit:

```
same_build("b9867", "b9867+152d337fa")            -> True
same_build("b10709+9a9394a89", "b9867+152d337fa") -> False
same_build("b9867+152d337fa", "b9867+deadbeef0")  -> False
```

That last case is why identity carries the commit at all: a re-tagged release and a fork
can share the counter.

---

## What this does NOT fix

Named so a follow-up does not have to rediscover them.

1. **`ModelSpec` has no role for embedding / reranker / ASR / TTS / diffusion.** All four
   non-chat families are reachable from a pinned build and none can be expressed. This is
   the largest remaining coverage gap, and it is a *data model* change, not a flag.
2. **TTS cannot be served by this UI.** There is no `/v1/audio/speech` route anywhere in
   `llama-server`. A sidecar (`speaches`, LocalAI) is the only route.
3. **Download verification is not compared against the published digest.** The Releases
   API publishes `assets[].digest` (SHA256) and recent builds are sigstore-attested;
   Rigma records a first-use digest but never compares. b9867-era releases have no
   attestation (404), so this only helps on newer builds.
4. **Asset-name drift** (`hip-radeon` → `rocm-10.0`) is latent: Rigma's manifest is a
   single pinned version, so it breaks the moment the pin moves.
5. **No retention policy.** ≈373 MB per Windows build; registration is additive with no
   pruning.
6. **`pytest-timeout` is still not installed**, so the documented intermittent suite hang
   cannot be capped from inside. This has now cost time twice.

---

## Corrections to this document, made later

Two claims here were wrong and are corrected in
[`35-prismml-install-and-corrections.md`](35-prismml-install-and-corrections.md):

* The **"adds the Vulkan kernel note"** behaviour described above asserted that PQ2_0 has
  no Vulkan kernel. It does — the fork ships `bin-win-vulkan-x64` and measured ~52 t/s on
  it — and the note recommended the deprecated `Q2_0` format, which will not load at all.
  The note now says that instead.
* The verdict table's **"PrismML fork (COUNT 144) → silent"** row was the *correct*
  behaviour but was not what the code did: `check_types` rejected fork types without
  consulting the engine's bound, so a bound-144 engine was told a PrismML model could not
  load. A test asserted that wrong behaviour, so the table and the test contradicted each
  other and the test was wrong. Both are fixed.

Also from that round: `engine_registry.select` had no callers, so a registered engine was
describable but never selectable — `rigma up` still chose the pin for a PQ2_0 model. The
launch path now consults the registry first.
