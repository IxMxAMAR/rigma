# ggml type registry — what type 142 is, and why `Ternary-Bonsai-2-27B-PQ2_0.gguf` fails on b9867

Research date: 2026-09-27. Method: `web_fetch` on `raw.githubusercontent.com` and the GitHub REST API
(per the brief, DuckDuckGo only via `https://html.duckduckgo.com/html/?q=...`).

Every factual claim below carries a URL. Claims I read directly in a source file are marked **[READ]**;
claims I derived by combining sources are marked **[INFERRED]**. Unresolvable items are collected in
**Could NOT verify**.

---

## Verdict

**The framing "ggml type 142 was added to llama.cpp between b9867 and build 10709" is wrong in one
important way, and the correction changes what Rigma should do.**

1. **Type 142 is `GGML_TYPE_PQ2_0`, and it is PrismML-private. Upstream llama.cpp has never contained
   it.** A GitHub commit search over `ggml-org/llama.cpp` for `PQ2_0` returns `total_count: 0`
   ([search](https://api.github.com/search/commits?q=repo:ggml-org/llama.cpp+PQ2_0)), and `ggml-org`
   `master`'s enum ends at `GGML_TYPE_Q2_0 = 42` / `GGML_TYPE_COUNT = 43` with no 142/143
   ([ggml.h @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)). **[READ]**

2. **The working binary is a PrismML *fork* build, not a newer mainline build.** The reported
   `version: 0.2.0-dev (build 10709, commit 9a9394a89)` matches the fork's release tag
   `prism-b10709-9a9394a` → commit `9a9394a895b96003ca842a6041cb28ac49a108f7` in **both** the build
   number and the commit hash
   ([fork tag list](https://api.github.com/repos/PrismML-Eng/llama.cpp/tags?per_page=40)). The tree at
   that commit contains `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143`, `GGML_TYPE_COUNT = 144`
   ([ggml.h @ 9a9394a89](https://raw.githubusercontent.com/ggml-org/llama.cpp/9a9394a895b96003ca842a6041cb28ac49a108f7/ggml/include/ggml.h)). **[READ]**

3. **Therefore the b9867 failure is not a version-lag problem.** Moving the engine pin from b9867 to any
   newer *mainline* build will not make this file load. The bound in the error message moves `42 → 43`
   when `Q2_0` merged, but 142 remains far outside it in mainline forever. **[INFERRED, from (1)+(2)]**

4. **Rigma's compatibility matrix must key on engine *provenance*, not on build number.** The load
   gate is "is this a PrismML-derived binary", not "is this build ≥ N". Build number cannot express
   this: build 10709 is a fork build, and mainline builds exist in the same numeric range. **[INFERRED]**

5. **A second, independent rule applies even on PrismML builds: PQ2_0 is not a Vulkan format.**
   PrismML's own README says `*-PQ2_0.gguf` (id 142) is "preferred on Metal, CUDA, HIP and CPU", while
   the group-64 `*-Q2_0_g64.gguf` (id 42) "runs on every backend here AND on mainline llama.cpp"
   ([fork README @ 4725def](https://github.com/PrismML-Eng/llama.cpp/commit/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d)). **[READ]**

6. **The `[0, N)` bound is `GGML_TYPE_COUNT` — a raw enum count that includes permanently-unused holes
   (4, 5, 31–33, 36–38).** It tells you which enum a binary was compiled from; it is *not* a capability
   signal. **[READ]** (see Q7)

---

## Q1 — What is ggml type 142?

| | `PQ2_0` | `PTQ1_0` |
|---|---|---|
| Enum name | `GGML_TYPE_PQ2_0` | `GGML_TYPE_PTQ1_0` |
| Value | **142** | **143** |
| Block / group size | **128** | **128** |
| Packing | one trit in a **2-bit** slot | **dense trit** packing |
| Family | ternary `{-1, 0, +1}` + FP16 group scale | same weights, denser trit packing |
| Bits per weight | **~2.13** (docs also say 2.16 — see note) | **~1.75–1.76** |
| `general.file_type` | `GGML_FTYPE_MOSTLY_PQ2_0 = 128` | `GGML_FTYPE_MOSTLY_PTQ1_0 = 129` |

Enum, group size and the `ftype` values are **[READ]** from the fork's header
([ggml.h, `prism` branch](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/include/ggml.h)):

```c
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
        // Prism-private Q2_0 at group size 128 (upstream Q2_0 is group 64). High id so it
        // slots above upstream types; type_traits is sized to COUNT (143) with 43..141 unused.
        GGML_TYPE_PQ2_0 = 142,
        GGML_TYPE_PTQ1_0 = 143, // Prism-private ternary, group 128
        GGML_TYPE_COUNT   = 144,
```
```c
        GGML_FTYPE_MOSTLY_PQ2_0 = 128, // except 1d tensors (Prism-private group-128 Q2_0)
        GGML_FTYPE_MOSTLY_PTQ1_0 = 129, // except 1d tensors (Prism-private group-128 ternary)
```

Note the comment says `type_traits is sized to COUNT (143)` while `COUNT` is actually 144 — an
off-by-one in the comment, not in the code. **[READ]**

**Publisher: PrismML** (HF org `prism-ml`, GitHub org `PrismML-Eng`, docs at `docs.prismml.com`).
Your suspicion of "a specific vendor's ternary/1.58-bit line" is correct. PrismML's formats page names
the families:

| Family | GGUF type | Notes |
| --- | --- | --- |
| Bonsai (1-bit) | `Q1_0` | Weights in `{−1, +1}`, packed one bit each. |
| Ternary-Bonsai | `Q2_0` | Weights in `{−1, 0, +1}` packed in 2 bits. |
| Ternary Bonsai 2 | `PTQ1_0` | Ternary g128 at 1.76 bpw — the smaller footprint (5.93 GB for the 27B language model). |
| Ternary Bonsai 2 | `PQ2_0` | Ternary g128 at 2.16 bpw (7.25 GB) — a simpler 2-bit representation that is cheaper to unpack, so prompt processing is faster. |

([docs.prismml.com/download/formats](https://docs.prismml.com/download/formats)) **[READ]**

The same page's "three ternary GGUF files" table gives PQ2_0 as "Group size 128 (**2.13 bpw**)" — so
the page itself carries both 2.16 and 2.13 for PQ2_0. The published file size settles it:
7,206,168,928 bytes × 8 ÷ 27×10⁹ ≈ **2.135 bpw**, matching 2.13. **[INFERRED]**
The corresponding PTQ1_0 file is 5,946,648,928 bytes → ≈ **1.762 bpw**, matching 1.76.
(File sizes are **[READ]** from the issue body, [issue #29058](https://github.com/ggml-org/llama.cpp/issues/29058).)

**Do not confuse these with the lookalikes.** PrismML publishes three different ternary files, and the
distinction is a live footgun ([docs.prismml.com/download/formats](https://docs.prismml.com/download/formats), [fork README](https://github.com/PrismML-Eng/llama.cpp/commit/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d)): **[READ]**

| File | Format | Runs on |
|---|---|---|
| `*-PQ2_0.gguf` | group 128, its own id **142** | PrismML fork binaries `prism-b10658+` |
| `*-Q2_0_g64.gguf` / 27B `*-Q2_g64.gguf` | official group 64, id **42** | mainline llama.cpp **and** fork binaries |
| `*-Q2_0.gguf` (older repos, no `g64`) | **deprecated** group-128 stored under id 42 | only old `prism-v5` releases; newer builds refuse it |

The deprecation matters: the docs warn that on stock llama.cpp this legacy id-42 group-128 file
"loads without a warning and outputs gibberish"
([docs.prismml.com/download/formats](https://docs.prismml.com/download/formats), surfaced via
[DuckDuckGo](https://html.duckduckgo.com/html/?q=GGML_TYPE_PQ2_0+llama.cpp)). **[READ]** — that is a
*silent wrong-answer* path, worse than the loud failure you hit.

`block_pq2_0` / `block_ptq1_0` C struct layouts were **not** read — see **Could NOT verify**.

---

## Q2 — The complete current ggml type enum

### Upstream `ggml-org/llama.cpp` @ master (authoritative)

**[READ]** from [ggml.h @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h):

| # | Name | # | Name | # | Name |
|---:|---|---:|---|---:|---|
| 0 | `GGML_TYPE_F32` | 15 | `GGML_TYPE_Q8_K` | 29 | `GGML_TYPE_IQ1_M` |
| 1 | `GGML_TYPE_F16` | 16 | `GGML_TYPE_IQ2_XXS` | 30 | `GGML_TYPE_BF16` |
| 2 | `GGML_TYPE_Q4_0` | 17 | `GGML_TYPE_IQ2_XS` | 34 | `GGML_TYPE_TQ1_0` |
| 3 | `GGML_TYPE_Q4_1` | 18 | `GGML_TYPE_IQ3_XXS` | 35 | `GGML_TYPE_TQ2_0` |
| 6 | `GGML_TYPE_Q5_0` | 19 | `GGML_TYPE_IQ1_S` | 39 | `GGML_TYPE_MXFP4` |
| 7 | `GGML_TYPE_Q5_1` | 20 | `GGML_TYPE_IQ4_NL` | 40 | `GGML_TYPE_NVFP4` |
| 8 | `GGML_TYPE_Q8_0` | 21 | `GGML_TYPE_IQ3_S` | 41 | `GGML_TYPE_Q1_0` |
| 9 | `GGML_TYPE_Q8_1` | 22 | `GGML_TYPE_IQ2_S` | 42 | `GGML_TYPE_Q2_0` |
| 10 | `GGML_TYPE_Q2_K` | 23 | `GGML_TYPE_IQ4_XS` | **43** | **`GGML_TYPE_COUNT`** (sentinel, not a type) |
| 11 | `GGML_TYPE_Q3_K` | 24 | `GGML_TYPE_I8` | | |
| 12 | `GGML_TYPE_Q4_K` | 25 | `GGML_TYPE_I16` | | |
| 13 | `GGML_TYPE_Q5_K` | 26 | `GGML_TYPE_I32` | | |
| 14 | `GGML_TYPE_Q6_K` | 27 | `GGML_TYPE_I64` | | |
| | | 28 | `GGML_TYPE_F64` | | |

**Removed / skipped numbers — present as commented-out gaps, not as live values [READ]:**

- **4, 5** — `Q4_2`, `Q4_3`: `// support has been removed`
- **31, 32, 33** — `Q4_0_4_4`, `Q4_0_4_8`, `Q4_0_8_8`: `// support has been removed from gguf files`
- **36, 37, 38** — `IQ4_NL_4_4`, `IQ4_NL_4_8`, `IQ4_NL_8_8`: commented out
- **43…∞** — `GGML_TYPE_COUNT = 43` is the exclusive upper bound; there is no type 43.

**There are no aliases** in `enum ggml_type`. (`GGML_PREC_DEFAULT` / `GGML_PREC_UNDEFINED` are aliases
of each other, but that is a different enum.) **[READ]**

The header's own rule for extension: `// NOTE: always add types at the end of the enum to keep backward
compatibility` **[READ]** — which is exactly why the Prism types were placed at 142/143 rather than
appended at 43.

### PrismML fork additions

On the fork's `prism` branch: `GGML_TYPE_PQ2_0 = 142`, `GGML_TYPE_PTQ1_0 = 143`,
`GGML_TYPE_COUNT = 144`, with **43–141 deliberately unused**
([ggml.h, `prism` branch](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/include/ggml.h)). **[READ]**

The fork's `master` branch is *not* the fork's main line — its default branch is `prism`
([PR #24448 head repo metadata](https://api.github.com/repos/ggml-org/llama.cpp/pulls/24448), and
`ggml.h` on fork `master` still has `GGML_TYPE_COUNT = 43` with no 142/143). **[READ]**

---

## Q3 — When was type 142 added?

**I could not find the introducing commit or PR.** Declared as a gap rather than guessed.

What the evidence does bracket:

- **Upstream: never.** Commit search over `ggml-org/llama.cpp` for `PQ2_0` → `total_count: 0`
  ([search API](https://api.github.com/search/commits?q=repo:ggml-org/llama.cpp+PQ2_0)). **[READ]**
- **The fork: by 2026-08-27 (build 10658) at the latest.** PrismML's docs state Ternary Bonsai 2's
  "GGUF packings need the PrismML llama.cpp fork (**prism-b10658 or newer**)"
  ([docs.prismml.com/bonsai-2-27b](https://docs.prismml.com/bonsai-2-27b), via
  [DuckDuckGo](https://html.duckduckgo.com/html/?q=GGML_TYPE_PQ2_0+llama.cpp)). Tag `prism-b10658-4725def`
  points at commit `4725def0acca71f2a62d64f6a5fcd0d5ed73e64d`, dated **2026-08-27**
  ([tag list](https://api.github.com/repos/PrismML-Eng/llama.cpp/tags?per_page=40),
  [commit](https://api.github.com/repos/PrismML-Eng/llama.cpp/commits/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d)). **[READ]**
- **Lower bound: after 2026-07-07.** The upstream `Q2_0` PR (#24448, "ggml: add Q2_0 quantization
  support (CPU)") was merged 2026-07-07T19:05:47Z as `bec4772f6a2527d371557b5d2032641e5ff7619c`. That PR
  says: "We plan to also maintain a sibling **group-128** variant (`PQ2_0`) in our fork since the 0.125
  extra bpw becomes significant on larger models." So `PQ2_0` existed as a *plan/format* by
  2026-06-11 (PR opened) and the fork had to move it to a new id once upstream claimed id 42 for
  group-64 ([PR #24448](https://api.github.com/repos/ggml-org/llama.cpp/pulls/24448)). **[READ]**

**[INFERRED]** Type 142 was introduced in the fork's `prism-v7` rebase onto a mainline that already
owned id 42 for group-64, i.e. between 2026-07-07 and 2026-08-27. The fork's own README states the
mechanism explicitly:

> This branch is rebased onto current mainline llama.cpp, where **`Q2_0` (ggml type id 42) is the
> official group-64 format**. The fork's original group-128 format now lives under its own name and id:
> **`PQ2_0` (ggml type id 142)**.

([README diff @ 4725def](https://github.com/PrismML-Eng/llama.cpp/commit/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d)) **[READ]**

### Gating

- **No compile-time gate.** In the fork's `ggml.h`, 142/143 are plain enum entries — no `#ifdef`, no
  CMake option ([ggml.h, `prism` branch](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/include/ggml.h)). **[READ]**
- **A runtime/model-level gate exists: a Walsh–Hadamard transform.** PrismML: "the rotated weight basis
  requires a runtime Walsh-Hadamard transform that is not upstream yet"
  ([docs.prismml.com/bonsai-2-27b](https://docs.prismml.com/bonsai-2-27b)). The fork's header carries a
  dedicated op hint for it: `GGML_HINT_SRC0_IS_HADAMARD`
  ([ggml.h, `prism` branch](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/include/ggml.h)). **[READ]**
  The model card says weights are stored after a blockwise Hadamard (block 1024) and the runtime must
  apply the matching transform to activations or **refuse to load**
  ([issue #29058](https://github.com/ggml-org/llama.cpp/issues/29058)). **[READ]** This is why "just
  register the block size" is not sufficient — and why the legacy id-42 file produces gibberish.
- **A backend restriction exists** — see Q4. This is the real gate, and it is per-backend, not global.

---

## Q4 — Does support differ per backend? (Yes — but not in the way the failure suggests)

This is the most important structural answer, and it splits cleanly in two.

### 4a. Type *registration* and *parsing* is compile-time global — not per-backend

`enum ggml_type` is a single enum in the core header, and the GGUF loader's validation is a global
compile-time check with no backend involvement. **[READ]** from
[ggml/src/gguf.cpp @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/gguf.cpp):

```c
        // tensor type
        {
            ok = ok && gr.read(info.t.type);

            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = false;
                break;
            }
```

**Consequence for the reproduced defect:** the failure you see is a *binary-identity* failure, not a
backend failure. If a binary was compiled without type 142 in its enum, it refuses the file on **every**
backend — CPU, Vulkan and ROCm alike — before any backend is selected. Your observation that "the
failing binary is CPU and Vulkan and ROCm-mainline" is consistent with all three being **stock mainline
builds**, and the working ROCm one being the PrismML fork. **[INFERRED]**

### 4b. Type *execution* is per-backend, via an explicit capability query

Backends advertise what they can run, and the scheduler routes around them. **[READ]** from
[ggml/include/ggml-backend.h @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml-backend.h):

```c
    GGML_API bool ggml_backend_supports_op(ggml_backend_t backend, const struct ggml_tensor * op);
    GGML_API bool ggml_backend_dev_supports_op(ggml_backend_dev_t device, const struct ggml_tensor * op);
```
```c
        // The backends are selected based on:
        // - the backend that supports the operation
        // - the location of the pre-allocated tensors (e.g. the weights)
```

Crucially, **neither `ggml_backend_dev_caps` nor `ggml_backend_dev_props` contains a list of supported
tensor types** — the only per-backend capability surface is per-*op* plus generic feature flags:

```c
    struct ggml_backend_dev_caps {
        bool async; bool host_buffer; bool buffer_from_host_ptr; bool events; bool mmap_support;
    };
```
```c
    typedef struct ggml_backend_feature * (*ggml_backend_get_features_t)(ggml_backend_reg_t reg);
```
**[READ]**, same file.

### 4c. Concrete per-backend differences for this exact format

- **PrismML's own matrix:** `*-PQ2_0.gguf` (id 142) is "preferred on **Metal, CUDA, HIP and CPU**";
  `*-Q2_0_g64.gguf` (id 42) "runs on **every** backend here AND on mainline llama.cpp"
  ([fork README @ 4725def](https://github.com/PrismML-Eng/llama.cpp/commit/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d)). **[READ]**
  Vulkan is conspicuously absent from the PQ2_0 list.
- **PrismML's formats page** states the group-64 file has "widest backend coverage (**adds Vulkan and
  SYCL**)" relative to the PQ2_0 file ([docs.prismml.com/download/formats](https://docs.prismml.com/download/formats)). **[READ]**
- **Upstream `Q2_0` landed CPU-only, with other backends explicitly deferred.** PR #24448: "This PR is
  **CPU only** (ARM NEON + generic scalar fallback). … We have the x86, Metal, CUDA, and Vulkan backends
  ready to submit later." ([PR #24448](https://api.github.com/repos/ggml-org/llama.cpp/pulls/24448)). **[READ]**
  So a quant type can be *fully present in the enum and loadable* while individual backends lack kernels
  for it. That is the general shape of the answer to your question.
- **A third-party fork exists solely to add the missing Vulkan kernel:**
  `drishal/llama.cpp-pq2-vulkan`, described as "Fork of PrismML-Eng/llama.cpp", whose file table says
  `*-PQ2_0.gguf (fork group-128, ggml id 142): preferred on Metal, CUDA, HIP and CPU"
  ([github.com/drishal/llama.cpp-pq2-vulkan](https://github.com/drishal/llama.cpp-pq2-vulkan), via
  [DuckDuckGo](https://html.duckduckgo.com/html/?q=GGML_TYPE_PQ2_0+llama.cpp)). **[READ]**

**Explicit answer:** ggml type support is **not** purely compile-time-global and **not** purely
per-backend — it is both, at different layers. Enum membership + GGUF parsing + CPU dequantisation are
global to the binary; kernel availability for a type is per-backend and queried per-op. **PQ2_0 (142)
needs a PrismML-derived binary to load at all, and on that binary it still has no Vulkan kernel.**

**[INFERRED / not verified]** What exactly happens at runtime when a *weight* type has no kernel on the
chosen backend (silent CPU fallback for those ops vs. hard error vs. gibberish) — I read that the
scheduler picks "the backend that supports the operation", but I did not trace the weights-placement
path. Do not encode a specific behaviour for this case without testing it.

---

## Q5 — Is there a way to query a binary's supported types at runtime?

**No. There is no flag, subcommand, or API that enumerates supported ggml tensor types.** That is a
genuine negative result, and it is the most actionable gap for Rigma.

What exists, and why it does not answer the question:

- **`--list-devices` exists, but lists *devices*, not types.** It is a real llama-server flag
  ([issue #16659, "Misc. bug: 'llama-server --list-devices' doesn't show the CPU device"](https://github.com/ggml-org/llama.cpp/issues/16659), via
  [DuckDuckGo](https://html.duckduckgo.com/html/?q=llama.cpp+%22--list-devices%22+flag+llama-server+version+output)). **[READ]**
- **The C API has no type-list function.** The complete device-properties surface is
  `name / description / memory_free / memory_total / type / device_id / caps`, and `caps` holds only
  `async, host_buffer, buffer_from_host_ptr, events, mmap_support`
  ([ggml-backend.h @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml-backend.h)). **[READ]**
  The nearest capability query is `ggml_backend_supports_op()`, which is per-**op** and requires you to
  already hold a tensor of the type in question — i.e. you can only ask "can you run this?" after
  successfully loading a model that uses it. That is circular for a pre-flight compatibility check.
- **`ggml_type_name()` / `ggml_type_size()` / `ggml_blck_size()` exist**, but they are lookups into the
  compiled-in `type_traits` table — they *describe* a type you already named, they do not enumerate.
  **[READ]** — `ggml_type_size()` / `ggml_blck_size()` are called immediately after the bound check in
  [gguf.cpp](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/gguf.cpp).

**Practical consequence for Rigma:** the only reliable pre-flight capability probe is to attempt the
load and parse the error — which is exactly what your reproduced run did. The error message is
informative by accident: `should be in [0, N)` leaks `GGML_TYPE_COUNT` and therefore identifies the
enum the binary was built from. Parsing `N` is a usable build-identity signal (see Q7).

**[Scope caveat]** I did not read `common/arg.cpp` in full, so I cannot claim exhaustive coverage of
every flag. The negative is established for the public C API surface in `ggml-backend.h`, which is the
authoritative capability interface.

---

## Q6 — Version identification

### What `--version` prints

**[READ]** from [common/build-info.cpp.in @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/build-info.cpp.in):

```cpp
int LLAMA_BUILD_NUMBER = @LLAMA_BUILD_NUMBER@;
char const * LLAMA_COMMIT = "@LLAMA_BUILD_COMMIT@";
...
const char * llama_build_info(void) {
    static std::string s = "b" + std::to_string(LLAMA_BUILD_NUMBER) + "-" + LLAMA_COMMIT;
    return s.c_str();
}

void llama_print_build_info(const char * llama_version, FILE * stream) {
    fprintf(stream, "version: %s (build %d, commit %s)\n", llama_version, llama_build_number(), llama_commit());
    fprintf(stream, "built with %s for %s\n", llama_compiler(), llama_build_target());
}
```

So `llama-server --version` emits exactly two lines:

```
version: <LLAMA_VERSION> (build <LLAMA_BUILD_NUMBER>, commit <LLAMA_COMMIT>)
built with <compiler> for <target>
```

Your working binary's `version: 0.2.0-dev (build 10709, commit 9a9394a89)` is this format with
`llama_version = "0.2.0-dev"`, build `10709`, short commit `9a9394a89`. **[READ]**

### Can a build be identified precisely from it?

**Yes — the triple (version string, build number, commit hash) is precise, and the commit hash is the
part that matters.** Note `llama_build_info()` also exposes a `b<number>-<commit>` compact form. **[READ]**

### `bNNNN` vs `0.2.0-dev (build NNNNN, commit HASH)`

- They are **the same counter**, not two schemes. The `bNNNN` form is the *release-tag* naming; the
  `build NNNNN` form is the same number printed by the binary. Evidence: mainline release tag `b9867`
  resolves to commit `152d337fadb93c2a099653c4072d5512c92c5bfd`
  ([tag ref](https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/b9867)), and the fork's tags
  follow `prism-b<number>-<shorthash>` (e.g. `prism-b10709-9a9394a`, `prism-b10658-4725def`,
  `prism-b9601-68faa14`) ([tag list](https://api.github.com/repos/PrismML-Eng/llama.cpp/tags?per_page=40)). **[READ]**
- The `0.2.0-dev` prefix is a semantic-version label; `llama_version` is passed into
  `llama_print_build_info` rather than being generated there. **[READ]** I did **not** find the commit
  that introduced the `0.2.0-dev` label — see **Could NOT verify**.
- **[INFERRED]** Because the counter is shared between mainline and the fork, a build number alone does
  **not** identify provenance. `10709` is a PrismML fork build; mainline almost certainly has a
  different commit at build 10709. Rigma must not treat build number as a proxy for capability.

### Does `b9867` map to a build number?

**Yes, directly.** **[READ]** [tag ref b9867](https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/b9867):

- tag `b9867` → commit `152d337fadb93c2a099653c4072d5512c92c5bfd`
- that commit: **2026-07-03**, `spec: support spec-draft-p-min in DFlash (#25246)`
  ([commit](https://api.github.com/repos/ggml-org/llama.cpp/commits/152d337fa))

So `b9867` is a real mainline release tag from 2026-07-03, four days *before* `Q2_0` merged
(2026-07-07). That date ordering is exactly what produces the `[0, 42)` bound — see Q7.

---

## Q7 — The `should be in [0, 42)` bound

**The upper bound is `GGML_TYPE_COUNT`, verbatim. [READ]** —
[ggml/src/gguf.cpp @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/gguf.cpp):

```c
            // check that tensor type is within defined range
            if (info.t.type < 0 || info.t.type >= GGML_TYPE_COUNT) {
                GGML_LOG_ERROR("%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n",
                    __func__, info.t.name, info.t.type, GGML_TYPE_COUNT);
                ok = false;
                break;
            }
```

The `%d` in `[0, %d)` is `GGML_TYPE_COUNT`. **[READ]**

**It is a count of enum entries, not "the last non-deprecated type".** Proof: `GGML_TYPE_COUNT` includes
the permanently-dead holes (4, 5, 31–33, 36–38) inside its range. In master, `COUNT = 43` while the
highest *live* type is `Q2_0 = 42` and there are seven numbers inside the range that are not types at
all ([ggml.h @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)). **[READ]**

**Cross-checked against three independent real error strings — this is the strongest confirmation:**

| Observed error | Implied `COUNT` | Implies enum contains | Source |
|---|---|---|---|
| type `42` → `[0, 42)` | 42 | `Q1_0 = 41` yes, `Q2_0 = 42` no | [HF discussion #39](https://huggingface.co/prism-ml/Ternary-Bonsai-27B-gguf/discussions/39) |
| type `142` → `[0, 42)` | 42 | same era as above | your b9867 reproduction |
| type `143` → `[0, 43)` | 43 | `Q2_0 = 42` present | [kvmem/llama.cpp issue #31](https://github.com/kvmem/kvmem-llama.cpp/issues/31) |

All three are consistent with `COUNT` = the enum's own sentinel, and with b9867 (2026-07-03) predating
the `Q2_0` merge (2026-07-07). **[INFERRED from READ sources]**

**Is the bound a reliable capability signal? Partly — and only as a *build-identity* signal.**

- **As a capability signal: no.** A type can be *inside* `[0, COUNT)` and still be a removed hole
  (4/5/31–33/36–38) with no implementation. Conversely, a type can be *outside* the bound and still be
  perfectly supported by that binary if the binary is a fork that added it — which is precisely your
  case, and the fork's own comment confirms the design intent ("High id so it slots above upstream
  types").
- **As a build-identity signal: yes, and it is useful.** The printed `N` is a direct readout of the
  binary's compiled-in `GGML_TYPE_COUNT`: `42` = mainline before `Q2_0` merged (≤ 2026-07-07),
  `43` = mainline after, `144` = PrismML fork. Rigma can use `N` as a cheap, high-confidence
  *fingerprint* to distinguish mainline from PrismML-derived binaries, which is exactly the
  discrimination this defect needs. **[INFERRED]**

---

## Could NOT verify

Listed explicitly. These are gaps, not conclusions.

1. **The introducing commit/PR for `GGML_TYPE_PQ2_0 = 142` in the PrismML fork.** Commit search over
   both repos for `PQ2_0` returned `total_count: 0` ([ggml-org](https://api.github.com/search/commits?q=repo:ggml-org/llama.cpp+PQ2_0),
   [fork](https://api.github.com/search/commits?q=repo:PrismML-Eng/llama.cpp+PQ2_0)), so I could not
   retrieve it by that route. Bracketed by other evidence to 2026-07-07 … 2026-08-27, but the exact
   commit/PR is unknown.
2. **`block_pq2_0` / `block_ptq1_0` C struct layouts and exact bytes-per-block.** Not read from
   `ggml-common.h` or the fork's quant kernels. The issue author states the same gap ("I do not have a
   public spec for the exact `block_pq2_0` / `block_ptq1_0` C layouts")
   ([issue #29058](https://github.com/ggml-org/llama.cpp/issues/29058)). Group size 128 and bpw figures
   are read; the byte layout is not.
3. **The commit that introduced the `0.2.0-dev (build N, commit H)` version format.** I read the
   *current* format string and confirmed it matches the observed output, but did not locate the change
   that added the semantic-version prefix or the date the scheme changed.
4. **The PR that actually merged `Q1_0 = 41` upstream.** The issue cites
   [#20048](https://api.github.com/repos/ggml-org/llama.cpp/pulls/20048) "Q1_0 / follow-ups", but that PR
   shows `"state":"closed"`, `"merged":false`, `merged_at:null` (closed 2026-03-02T23:27:50Z). `Q1_0 = 41`
   *is* present in master, so it landed via some follow-up PR I did not identify.
5. **Exhaustive coverage of CLI flags.** I did not read `common/arg.cpp` in full. The negative answer in
   Q5 is established for the public C API in `ggml-backend.h` (no type enumeration) and for
   `--list-devices` (devices, not types), but I cannot claim to have checked every flag string.
6. **Runtime behaviour when a weight type has no kernel on the selected backend** (silent CPU fallback
   for those ops vs. hard error). The scheduler's selection rule is read; the weights-placement and
   failure path is not traced.
7. **Whether the PrismML fork's Vulkan backend has any PQ2_0 kernel.** The absence is strongly implied
   by the fork README (PQ2_0 "preferred on Metal, CUDA, HIP and CPU"; group-64 "runs on every backend"),
   by PrismML's formats page (group-64 "adds Vulkan and SYCL"), and by the existence of a dedicated
   Vulkan PQ2_0 fork — but I did not read the fork's Vulkan source.
8. **The real binaries on this machine.** I searched for `llama-server.exe` / `llama-cli` under
   `C:\ComfyUI`, `C:\Users\<user>`, `C:\Program Files`, `C:\tools`, `C:\llama.cpp`, `C:\AI` and found
   only **test fixtures** — 1-byte and 96-byte stubs under
   `C:\ComfyUI\RD\rigma-review\.scratch\00-recon\` (e.g. `bt-es8\...\home\engines\b9867\vulkan\llama-server.exe`,
   96 bytes). No real llama.cpp build was inspected, so the binary-identity conclusion rests entirely on
   the version string ↔ release-tag match, not on running the binaries.
9. **An anomaly I could not explain.** `https://raw.githubusercontent.com/ggml-org/llama.cpp/<9a9394a89>/ggml/include/ggml.h`
   returns a header containing `PQ2_0 = 142`, and `api.github.com/repos/ggml-org/llama.cpp/commits/9a9394a89`
   resolves — yet `ggml-org` `master` has `COUNT = 43` with no 142, `ggml-org` `master`'s
   `.github/workflows` listing contains no `release-prism.yml`, and a commit search over `ggml-org` for
   `PQ2_0` returns 0. That commit's own diff touches `.github/workflows/release-prism.yml` and references
   fork PR `#198`, and the fork's default branch is `prism`
   ([PR #24448 metadata](https://api.github.com/repos/ggml-org/llama.cpp/pulls/24448)). **[INFERRED]** it
   is a fork/`prism`-branch commit that the `ggml-org` endpoints also resolve; I could not confirm why.
   This does **not** weaken the operative conclusion — the *stock mainline* tree demonstrably lacks 142 —
   but I am flagging it rather than papering over it.

---

## Source index

Primary sources read in full or in the relevant part:

- [`ggml/include/ggml.h` @ ggml-org master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h) — authoritative upstream enum, `GGML_TYPE_COUNT = 43`
- [`ggml/include/ggml.h` @ PrismML-Eng `prism`](https://raw.githubusercontent.com/PrismML-Eng/llama.cpp/prism/ggml/include/ggml.h) — `PQ2_0 = 142`, `PTQ1_0 = 143`, `COUNT = 144`, `GGML_HINT_SRC0_IS_HADAMARD`
- [`ggml/include/ggml.h` @ 9a9394a89](https://raw.githubusercontent.com/ggml-org/llama.cpp/9a9394a895b96003ca842a6041cb28ac49a108f7/ggml/include/ggml.h) — the working build's tree
- [`ggml/src/gguf.cpp` @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/src/gguf.cpp) — the `[0, GGML_TYPE_COUNT)` check
- [`ggml/include/ggml-backend.h` @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml-backend.h) — per-backend `supports_op`, device caps/props, scheduler rule
- [`common/build-info.cpp.in` @ master](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/common/build-info.cpp.in) — the `--version` format string
- [llama.cpp issue #29058](https://github.com/ggml-org/llama.cpp/issues/29058) — the PQ2_0/PTQ1_0 feature request, file sizes, reproduction
- [PR #24448 — add Q2_0 (CPU)](https://github.com/ggml-org/llama.cpp/pull/24448) — merged 2026-07-07, `PQ2_0` plan, CPU-only scope
- [PR #20048 — add Q1_0](https://github.com/ggml-org/llama.cpp/pull/20048) — closed unmerged
- [PrismML formats & runtime support](https://docs.prismml.com/download/formats) and [Ternary Bonsai 2 27B](https://docs.prismml.com/bonsai-2-27b) — vendor matrix, bpw, `prism-b10658+`, Hadamard requirement
- [PrismML-Eng/llama.cpp tags](https://api.github.com/repos/PrismML-Eng/llama.cpp/tags?per_page=40) — `prism-b10709-9a9394a` and the `prism-b<number>-<hash>` scheme
- [Fork README commit 4725def](https://github.com/PrismML-Eng/llama.cpp/commit/4725def0acca71f2a62d64f6a5fcd0d5ed73e64d) — id-42 → id-142 migration, per-backend file guidance
- [ggml-org tag b9867](https://api.github.com/repos/ggml-org/llama.cpp/git/ref/tags/b9867) and [commit 152d337fa](https://api.github.com/repos/ggml-org/llama.cpp/commits/152d337fa) — b9867 ↔ 2026-07-03
- [drishal/llama.cpp-pq2-vulkan](https://github.com/drishal/llama.cpp-pq2-vulkan) — third-party Vulkan PQ2_0 fork
- [kvmem/llama.cpp issue #31](https://github.com/kvmem/kvmem-llama.cpp/issues/31) — independent `[0, 43)` bound observation for type 143
- [llama.cpp issue #16659](https://github.com/ggml-org/llama.cpp/issues/16659) — `--list-devices` exists, lists devices
