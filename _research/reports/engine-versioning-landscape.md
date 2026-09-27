# Engine Versioning Landscape — pinning, identifying, selecting and verifying llama.cpp builds

Research for the Rigma defect: `~/.rigma/engines/b9867/rocm/` contained `0.2.0-dev (build 10709, commit 9a9394a89)` while `b9867/cpu`, `b9867/vulkan` and `b9867/rocm-mainline-b9867` genuinely were `9867 (152d337fa)`. Rigma checked only that a file existed.

All claims below carry a source URL. Anything I derived rather than read is marked **inference**. Everything I could not source is in **Could NOT verify** at the end.

---

## Verdict

**The defect is fully fixable, and Rigma is currently leaving four independent, free identity signals on the table.**

1. **A build's identity is exactly and cheaply knowable.** `bNNNN` is *not* an opaque CI counter — it is `git rev-list --count HEAD`, i.e. the total commit count on master at that commit ([get-tag-name/action.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/actions/get-tag-name/action.yml)). So `b9867 ↔ 152d337fa` is a bijection, and the GitHub Releases API hands you the commit directly in `target_commitish`. I confirmed b9867's `target_commitish` is `152d337fadb93c2a099653c4072d5512c92c5bfd` ([releases/tags/b9867](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867)) — byte-for-byte the commit the user's *correct* directories reported.

2. **`bNNNN` is monotonic, so "is this build new enough?" is a total order, not a guess.** Because the number is a commit count on `master`, larger `bNNNN` always means later. **Inference:** this is the single most useful primitive for Rigma — it can express a model requirement as `min_build: 10709` and compare integers, rather than comparing version strings.

3. **A SHA256 per asset is already published, for free, by the API.** Every release asset carries `"digest": "sha256:<hex>"` ([b9867 assets](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867), [b11146 assets](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11146)). llama.cpp publishes **no separate checksum file** — but it does not need to.

4. **Recent builds are sigstore-attested, and the attestation binds the artifact to the exact git commit.** The current release workflow runs `actions/attest@v4` over `release/*` with `id-token: write` + `attestations: write` ([release.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml)), and b11146's release body publishes an attestation URL. I verified the attestation is programmatically retrievable and contains all 33 asset digests plus `resolvedDependencies[0].digest.gitCommit = 7fe450e19305b828c199d602c23a8337aaa1f03b` ([attestations API](https://api.github.com/repos/ggml-org/llama.cpp/attestations/sha256:55a378aa095b466979d85075234f66d7655c7a7483222af0c006c0e55b4d7bd6)).

**But there is no free lunch on the model side.** GGUF has **no minimum-engine-version key**. `general.quantization_version` is a dead signal — it has been `2` since forever. The only real capability signal is the set of tensor types actually present in the file, and Rigma will have to maintain its own `type/architecture → min bNNNN` table. **Inference:** that table is small, changes slowly, and is worth building; a declared minimum beats trusting a directory name.

**And the one thing Rigma must not rely on is `--version` parsing alone.** The output format itself changed between the two builds in this very defect, and the version-printing surface is still moving (`llama-bench --version` was only added in PR #28971, per the [v0.5.0 release body](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest)).

---

## 1. llama.cpp release + versioning scheme

### Two schemes run in parallel

**Scheme A — `bNNNN` nightly/build tags.** The tag name is computed by a composite action, quoted verbatim:

```bash
BUILD_NUMBER="$(git rev-list --count HEAD)"
SHORT_HASH="$(git rev-parse --short=7 HEAD)"
if [[ "${{ env.BRANCH_NAME }}" == "master" || "${{ env.BRANCH_NAME }}" == "b${BUILD_NUMBER}" ]]; then
  echo "name=b${BUILD_NUMBER}" >> $GITHUB_OUTPUT
else
  SAFE_NAME=$(echo "${{ env.BRANCH_NAME }}" | tr '/' '-')
  echo "name=${SAFE_NAME}-b${BUILD_NUMBER}-${SHORT_HASH}" >> $GITHUB_OUTPUT
fi
```
— [.github/actions/get-tag-name/action.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/actions/get-tag-name/action.yml)

Consequences, all verified against the API:

| Fact | Evidence |
|---|---|
| `bNNNN` = commit count on `master` | the action above |
| `bNNNN` → commit is exact | b9867 `target_commitish = 152d337fadb93c2a099653c4072d5512c92c5bfd` ([source](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867)) |
| `bNNNN` is monotonic | **inference** from "count of commits on a linear-ish branch"; corroborated by b9867 (2026-07-03) < b11146 (2026-09-23) |
| nightlies are `prerelease: true` | b11146 `"prerelease":true` ([source](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11146)) |
| non-master branches get `NAME-bNNNN-hash` | the `else` branch above |

Note the mismatch in hash width: the tag action uses `--short=7`, while the build embeds a longer short hash. **Inference:** do not assume the `--version` commit token length matches the tag action's.

**Scheme B — semver releases (`vMAJOR.MINOR.PATCH`).** `docs/release.md`:

> llama.cpp uses [semantic versioning](https://semver.org) (`MAJOR.MINOR.PATCH`).

with the bump rules (MAJOR = breaking change to `include/llama.h`, MINOR = backward-compatible features/model support/API addition, PATCH = bug fix), the version living in three `CMakeLists.txt` variables (`LLAMA_VERSION_MAJOR/MINOR/PATCH`), and:

> By default, `LLAMA_BUILD_IS_DEV=ON` which appends a `-dev` suffix to `LLAMA_VERSION`, marking the build as a nightly/development build. Distributors building from a release tag must pass `-DLLAMA_BUILD_IS_DEV=OFF` to produce a clean version string (e.g. `0.1.0` instead of `0.1.0-dev`).
>
> Currently releases are not published to github releases, only nightly/development builds are available there.
— [docs/release.md](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/docs/release.md)

**This is the origin of the `0.2.0-dev` string in the defect report.** A build with `LLAMA_BUILD_IS_DEV=ON` prints `0.2.0-dev`; the same code built from a release tag with `-DLLAMA_BUILD_IS_DEV=OFF` prints `0.2.0`. **Inference:** the `-dev` suffix is therefore a reliable "this is not a tagged release" marker, but it carries no build identity — which is exactly why the build number and commit must be parsed alongside it.

### Can a `bNNNN` be mapped to a build number? Is it a build number?

**`bNNNN` *is* the build number** — there is no separate mapping to find. The action computes it as `git rev-list --count HEAD` and uses it verbatim as the tag. **Inference:** the `build 10709` token in the defect's version string is the same integer that the tag would have been.

**How semver and `bNNNN` relate:** the semver release names its matching nightly in the body. The current `v0.5.0` release object (published 2026-09-23T20:50) contains:

> ## Assets
> **Nightly build:** [b11146](https://github.com/ggml-org/llama.cpp/releases/tag/b11146)
— [releases/latest](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest)

Caveat worth recording: `v0.5.0`'s own `target_commitish` is `d2e54583c7452353eb35d40431281f6ee984332f`, whereas b11146's is `7fe450e19305b828c199d602c23a8337aaa1f03b` (the commit whose message is `llama.cpp : bump version to 0.5.0 (#29333)`, the first line of v0.5.0's own changelog). **Inference:** the semver release points at a *later* commit than the nightly it advertises, so "semver → bNNNN" is approximate, not a bijection. Do not use a semver release's `target_commitish` as the build identity of the binaries.

### When and why did it change?

**Partially answered — see Could NOT verify #1.** Verified: the semver scheme exists now and is documented; `v0.1.x` releases existed before the current `v0.5.0` (the discussion snippet "take the newest v0.1.x, then read its body for the matching **bNNNN**" is the indexed abstract of [discussions/27340](https://github.com/ggml-org/llama.cpp/discussions/27340)); and `0.2.0` was being written about by 2026-08-21 ([freedom.tech](https://freedom.tech/posts/2026-08-21-llama-cpp-0-2-0/)). The *reason* is stated in `docs/release.md`: consumers needed a stable, human-meaningful version to pin against, while nightlies continued unchanged. I could not source the exact PR or date, and the discussion body itself was unreadable (GitHub nav chrome truncated it, and DuckDuckGo later served a CAPTCHA to this IP).

### Machine-readable releases API

**Yes.** `https://api.github.com/repos/ggml-org/llama.cpp/releases`, `/releases/latest`, `/releases/tags/{tag}`, `/releases/{id}/assets`.

Fields observed on a release object ([b9867](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867), [v0.5.0](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest)):

`url, assets_url, upload_url, html_url, id, author, node_id, tag_name, target_commitish, name, draft, immutable, prerelease, created_at, published_at, updated_at, assets[], tarball_url, zipball_url, body, reactions`

Fields observed on each asset object:

`url, id, node_id, name, label, uploader, content_type, state, size, digest, download_count, created_at, updated_at, browser_download_url`

**`digest` is the important one**: `"digest":"sha256:843333f1aa7d0c52d2e8ba346e635d2ef9832f0d6d667723351c60fd26d71454"` for b9867's Windows Vulkan zip. This is a first-class, machine-readable checksum — see §3.

### Asset naming convention

Format: `llama-b<NNNNN>-bin-<os>-<backend-or-arch>[-<toolchain-version>]-<arch>.<ext>`. Verified names from the b9867 and b11146 asset lists:

**Windows** (`.zip`)
- `llama-b9867-bin-win-cpu-x64.zip`, `-bin-win-cpu-arm64.zip`
- `llama-b9867-bin-win-cuda-12.4-x64.zip`, `-bin-win-cuda-13.3-x64.zip`
- `llama-b11146-bin-win-cuda-13.4-x64.zip`, `-bin-win-cuda-13.4-arm64.zip`
- `llama-b9867-bin-win-vulkan-x64.zip`
- `llama-b9867-bin-win-sycl-x64.zip`
- `llama-b9867-bin-win-openvino-2026.2.1-x64.zip` / `llama-b11146-bin-win-openvino-2026.4-x64.zip`
- `llama-b9867-bin-win-opencl-adreno-arm64.zip`
- **ROCm on Windows — name changed:** b9867 `llama-b9867-bin-win-hip-radeon-x64.zip` → b11146 `llama-b11146-bin-win-rocm-10.0-x64.zip`

**Linux** (`.tar.gz`)
- `llama-b9867-bin-ubuntu-x64.tar.gz`, `-bin-ubuntu-arm64.tar.gz`, `-bin-ubuntu-s390x.tar.gz`
- `llama-b9867-bin-ubuntu-vulkan-x64.tar.gz`, `-bin-ubuntu-vulkan-arm64.tar.gz`
- `llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz`, `-bin-ubuntu-cuda-13.4-x64.tar.gz`, `-bin-ubuntu-cuda-13.4-arm64.tar.gz`
- **ROCm on Linux — name changed:** b9867 `llama-b9867-bin-ubuntu-rocm-7.2-x64.tar.gz` → b11146 `llama-b11146-bin-ubuntu-rocm-10.0-x64.tar.gz`
- `llama-b9867-bin-ubuntu-openvino-2026.2.1-x64.tar.gz`, `-bin-ubuntu-sycl-fp32-x64.tar.gz`, `-bin-ubuntu-sycl-fp16-x64.tar.gz`
- `llama-b11146-bin-linux-arm64-snapdragon.tar.gz`

**macOS / Apple:** `llama-b9867-bin-macos-arm64.tar.gz`, `-bin-macos-x64.tar.gz`, `llama-b9867-xcframework.zip`

**Android:** `llama-b9867-bin-android-arm64.tar.gz`, `llama-b11146-bin-android-arm64-snapdragon.tar.gz`

**CUDA runtime DLLs (separate asset, paired with the CUDA zip):**
- `cudart-llama-b11146-bin-ubuntu-cuda-12.8-x64.tar.gz`
- `cudart-llama-bin-win-cuda-12.4-x64.zip` ← **no build number in the name**

**UI:** `llama-b9867-ui.tar.gz`

Three traps for a pinned downloader, all verified:

1. **Asset names are not stable across builds.** `hip-radeon` became `rocm-10.0`; `rocm-7.2` became `rocm-10.0`; OpenVINO `2026.2.1` became `2026.4`. The backend token and embedded toolchain version both drift. The ROCm version comes from a workflow matrix variable (`ROCM_VERSION: "10.0.0"` → `ROCM_VERSION_SHORT` = `10.0`) — [release.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml).
2. **The Windows cudart asset has no build number and is byte-identical across releases.** `cudart-llama-bin-win-cuda-12.4-x64.zip` has digest `sha256:8c79a9b226de4b3cacfd1f83d24f962d0773be79f1e7b75c6af4ded7e32ae1d6` in **both** b9867 and b11146. A downloader keyed on "asset name contains the tag" will silently reuse a stale cached file.
3. **There is no asset named `rocm`.** Rigma's directory token `rocm` maps to `hip-radeon` (b9867) or `rocm-10.0` (b11146) on Windows, and `ubuntu-rocm-<ver>` on Linux. Rigma's backend vocabulary is its own and must be mapped explicitly.

---

## 2. How other projects pin and verify engine builds

Answers keyed to (a) multiple versions side by side, (b) binary verification, (c) which build for which model, (d) mismatch behaviour.

### Ollama

(a) **Multiple *backends* yes; multiple engine *versions* not evidenced.** The docs state:

> Ollama includes multiple LLM libraries compiled for different GPUs and CPU vector features. Ollama tries to pick the best one based on the capabilities of your system. … In the server log, you will see a message that looks something like this (varies from release to release):
> `Dynamic LLM libraries [rocm_v6 cpu cpu_avx cpu_avx2 cuda_v11 rocm_v5]`
— [docs.ollama.com/troubleshooting](https://docs.ollama.com/troubleshooting)

Note `rocm_v5` and `rocm_v6` coexisting — these are different GPU-runtime builds of the *same* llama.cpp, not two llama.cpp versions. **Inference:** Ollama pins one llama.cpp per Ollama release and cross-compiles it for several backends.

(b) **No verification described.** Not verified either way — see Could NOT verify #4.

(c) Hardware autodetection, with a documented manual override:

> **Experimental LLM Library Override** You can set `OLLAMA_LLM_LIBRARY` to any of the available LLM libraries to bypass autodetection, so for example, if you have a CUDA card, but want to force the CPU LLM library with AVX2 vector support, use:
> `OLLAMA_LLM_LIBRARY="cpu_avx2" ollama serve`
— [same page](https://docs.ollama.com/troubleshooting)

(d) Not applicable — the user cannot select a version, only a backend.

### LM Studio

(a) **Yes — multiple runtimes side by side.** `lms runtime` provides `ls` ("list installed runtimes"), `get` ("download a runtime"), `select` ("set the active runtime"), `remove` ("uninstall a runtime"), `update` ("update an installed runtime") — [lmstudio.ai/docs/cli/runtime/runtime](https://lmstudio.ai/docs/cli/runtime/runtime). Runtimes are described as "inference engines (such as llama.cpp, MLX, etc.)" — [deepwiki.com/lmstudio-ai/lms/5.2-runtime-management](https://deepwiki.com/lmstudio-ai/lms/5.2-runtime-management).

(c) **Selection is machine-wide, not per-model — and this is an open feature request, not a shipped feature.** Issue #599 is titled "Feature request: allow per-model runtime selection and pinning":

> I use LM Studio with custom llama.cpp backends on NVIDIA Blackwell hardware. Some models perform best with a specialised build, while others are more stable or compatible with the stock LM Studio runtime. A per-model runtime binding would let both coexist without repeatedly changing a machine-wide runtime **or risking an unintended fallback**.
— [lmstudio-ai/lms#599](https://github.com/lmstudio-ai/lms/issues/599)

(d) The requester explicitly names "unintended fallback" as a risk of the current machine-wide model. **Inference:** LM Studio can silently fall back to another runtime; Rigma must not copy this design.

**How LM Studio labels a runtime:** not with a `bNNNN`. Issue #1729 says "The bundled **llama.cpp backend (v2.10.0)** apparently doesn't support whatever quantization type 41 is" ([lmstudio-bug-tracker#1729](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1729)), and #1742 shows the CLI flow `lms runtime select --latest -> observe llama.cpp-linux-arm64` ([lmstudio-bug-tracker#1742](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1742)). **Inference:** LM Studio interposes its own runtime version over the llama.cpp build, which makes the underlying build number opaque to users — and plausibly contributed to #1729 being hard to triage.

### llamafile

(a) **Effectively yes, trivially** — each `.llamafile` is a standalone file.

(c) **There is no selection at all: the engine is welded into the file.**

> We're doing that by combining llama.cpp with Cosmopolitan Libc into one framework that collapses all the complexity of LLMs down to a single-file executable (called a "llamafile") that runs locally on most operating systems and CPU architectures, with no installation.
— [mozilla-ai/llamafile](https://github.com/mozilla-ai/llamafile)

> A llamafile bundles the llamafile executable, model weights, and a set of default arguments into a single self-contained file using the APE (Actually Portable Executable) format, which supports ZIP as a container for extra data.
— [docs.mozilla.ai — creating llamafiles](https://docs.mozilla.ai/llamafile/using-llamafile/creating_llamafiles)

(b) No verification mechanism found — see Could NOT verify #6. (d) Not applicable.

### koboldcpp

(a) **Side-by-side by convention only** — it is one self-contained distributable per version, so users keep multiple `.exe` files.

> KoboldCpp is an easy-to-use AI text-generation software for GGML and GGUF models… It's a single self-contained distributable that builds off llama.cpp and adds many additional powerful features.
— [LostRuins/koboldcpp](https://github.com/LostRuins/koboldcpp)

(c) **Explicit flag or GUI picker, not inference.** Release notes:

> This can be overridden by picking a specific backend (eg. `--usecpu`, `--usevulkan`, `--usecublas`).
— [koboldcpp v1.75.1 release notes](https://newreleases.io/project/github/LostRuins/koboldcpp/release/v1.75.1)

Backends documented as `--usecpu`, `--usevulkan`, `--usecublas`, `--useclblast`, with Vulkan recommended for Intel and for AMD "when ROCM isnt an option" — [koboldai.com — KoboldCpp Quicklaunch](https://koboldai.com/Guides/Kobold_CPP/KoboldCpp_Quicklaunch/).

**Notable for Rigma's ROCm lane:** ROCm support is not in the official build.

> Concedo's KoboldCPP Official. Does not support RoCM. YellowRoseCx's KoboldCPP With RoCM support (for AMD GPUs only)
— [wiki.pygmalion.chat/backend/kobold-cpp](https://wiki.pygmalion.chat/backend/kobold-cpp)

(b) No verification found. (d) Not applicable — the engine is the application.

### text-generation-webui

**Not adequately sourced.** It consumes `llama-cpp-python` (a Python binding, not llama.cpp binaries), and a DeepWiki overview page exists at [deepwiki.com/oobabooga/text-generation-webui/3.1-llama.cpp-integration](https://deepwiki.com/oobabooga/text-generation-webui/3.1-llama.cpp-integration), but I did not reach the wheel-selection or pinning logic. See Could NOT verify #8. **Inference from the ecosystem:** because the engine arrives as a prebuilt wheel, its llama.cpp build is whatever the wheel was compiled against — a layer of indirection Rigma deliberately avoids.

### vLLM

**The analogue is real but I could not source its exact check.** What is verified:

> **ModuleNotFoundError: No module named 'vllm._C'** … The `vllm._C` module is built when you install vLLM.
— [vllm-project/vllm#1814](https://github.com/vllm-project/vllm/issues/1814)

**Inference:** vLLM separates a Python package from a compiled extension (`vllm._C`) that must match it; a mismatch surfaces as an import failure rather than a version-diff message. A secondary Chinese Q&A page quotes a distinct warning, `WARNING: Failed to import vllm._C due to CUDA version mismatch`, and claims it "does not block startup but significantly degrades performance, forcing fallback" ([ask.csdn.net](https://ask.csdn.net/questions/9213121)) — **treat that as secondary and unverified**; it is the closest thing I found to a "warn and silently degrade" behaviour, and it is exactly the anti-pattern Rigma must avoid. See Could NOT verify #9.

### 7. Does any tool run the engine's `--version` and parse it to confirm identity?

**No example found.** See Could NOT verify #10. The evidence points the other way: llama.cpp's own version-printing surface is still moving (`llama-bench --version` to "print build info" was added in PR #28971, listed in the [v0.5.0 changelog](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest)), and the format demonstrably differs between the two builds in this defect. **Inference:** `--version` parsing is a useful *secondary* confirmation but is too unstable to be the primary identity mechanism.

---

## 3. Verifying that a downloaded binary is what it claims

**llama.cpp publishes no checksum manifest file.** Neither the b9867 nor the b11146 asset list contains a `SHA256SUMS`, `.sha256` or equivalent asset — only archives, `ui.tar.gz`, `xcframework.zip`, and (on semver releases) a 7-byte `nightly-tag.txt` ([b9867 assets](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867), [b11146 assets](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11146)).

**But GitHub's Releases API publishes a SHA256 per asset anyway.** Every asset object carries `digest`:

```json
{"name":"llama-b9867-bin-win-vulkan-x64.zip","size":32164547,
 "digest":"sha256:843333f1aa7d0c52d2e8ba346e635d2ef9832f0d6d667723351c60fd26d71454"}
```
— [releases/tags/b9867](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867)

**And recent builds carry sigstore-backed GitHub artifact attestations.** The current workflow grants the needed permissions and runs the step:

```yaml
    permissions:
        contents: write # for creating release
        id-token: write
        attestations: write
...
      - name: Attest release artifacts
        id: attest
        uses: actions/attest@v4
        with:
          subject-path: 'release/*'
```
— [.github/workflows/release.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml)

and the generated release body embeds the resulting URL:

```
**Attestations:**
- <${{ steps.attest.outputs.attestation-url }}>
```
— [same file](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml)

**b11146 has one, and it is machine-verifiable.** Its body reads `**Attestations:** - <https://github.com/ggml-org/llama.cpp/attestations/49623059>` ([releases/tags/b11146](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b11146)). Querying the API by the Windows Vulkan zip's digest returns HTTP 200 with a full sigstore bundle:

`GET https://api.github.com/repos/ggml-org/llama.cpp/attestations/sha256:55a378aa095b466979d85075234f66d7655c7a7483222af0c006c0e55b4d7bd6`

Contents verified in the response:
- `mediaType: application/vnd.dev.sigstore.bundle.v0.3+json`
- a Rekor transparency-log entry, `logIndex 2924240238`, with inclusion proof and checkpoint signed by `rekor.sigstore.dev`
- a Fulcio certificate whose SAN encodes the workflow identity `https://github.com/ggml-org/llama.cpp/.github/workflows/release.yml@refs/heads/master`
- an in-toto statement with `predicateType: https://slsa.dev/provenance/v1`
- `buildDefinition.buildType: https://actions.github.io/buildtypes/workflow/v1`, `externalParameters.workflow.path: .github/workflows/release.yml`
- **all 33 asset names with their sha256 digests** in the subject list
- **`resolvedDependencies[0].digest.gitCommit = 7fe450e19305b828c199d602c23a8337aaa1f03b`** — the exact commit

**b9867 does NOT have one.** `GET .../attestations/sha256:843333f1...` (its Windows Vulkan zip digest) returns **HTTP 404**. Likewise v0.5.0's `nightly-tag.txt` digest returns 404 — expected, since that release is produced by the separate `make-release.yml` workflow, not `release.yml`. b9867 was published 2026-07-03 and b11146 on 2026-09-23, so the attestation step was added somewhere in between (Could NOT verify #14).

**Inference — the verification chain available to Rigma today:**

1. Pin the tag (`bNNNN`) and resolve it via `/releases/tags/{tag}`, which yields `target_commitish` (the commit) and per-asset `digest`.
2. Download, hash locally, compare against `digest`. This alone closes the "wrong bytes" hole.
3. For builds with an attestation, additionally run `gh attestation verify <file> --repo ggml-org/llama.cpp`. This is strictly stronger: it proves the artifact was produced by `release.yml` from a specific commit, which `digest` alone does not (a `digest` fetched from a compromised API would match a compromised file).
4. **Treat `--version` output as a third, weaker signal** — useful to catch the exact defect at hand (a mislabelled directory), but never as the sole check, because its format is unstable.

For the specific defect, note that step 1 alone would have caught it: the API says b9867 *is* commit `152d337fa`, so a directory named `b9867` holding a `10709`/`9a9394a89` binary is provably wrong before any bytes are hashed.

---

## 4. GGUF metadata as a capability signal

### The standardized `general.*` key set

From the GGUF specification ([ggml/docs/gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)):

**Required:** `general.architecture: string`, `general.quantization_version: uint32`, `general.alignment: uint32`

**General metadata:** `general.name`, `general.author`, `general.version`, `general.organization`, `general.basename`, `general.finetune`, `general.description`, `general.quantized_by`, `general.size_label`, `general.license`, `general.license.name`, `general.license.link`, `general.url`, `general.doi`, `general.uuid`, `general.repo_url`, `general.tags`, `general.languages`, `general.datasets`, `general.file_type`

**Source metadata:** `general.source.url|doi|uuid|repo_url`, `general.base_model.count`, `general.base_model.{id}.name|author|version|organization|url|doi|uuid|repo_url`

### Is there a `general.min_llama_version` key? **No.**

There is no such key in the spec's standardized list, and no key with that role under any name. A DuckDuckGo search for the exact term returned literally:

> # No results found for **llama.cpp "min\_llama\_version" OR "general.min\_llama\_version" gguf minimum engine version**

Note the spec's `general.version` is **not** an engine version:

> `general.version: string`: The version of the model.
— [ggml/docs/gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)

and the GGUF *naming* convention confirms it is the model's `v<Major>.<Minor>`.

### `general.quantization_version` — what it is, and why it is a dead signal

Spec definition:

> **`general.quantization_version: uint32`**: The version of the quantization format. Not required if the model is not quantized (i.e. no tensors are quantized). If any tensors are quantized, this _must_ be present. This is separate to the quantization scheme of the tensors itself; the quantization version may change without changing the scheme's name (e.g. the quantization scheme is Q5_K, and the quantization version is 4).
— [ggml/docs/gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)

**The current value is 2, and it has not moved.** Verbatim from the current source:

```c
#define GGML_QNT_VERSION        2    // bump this on quantization format changes
#define GGML_QNT_VERSION_FACTOR 1000 // do not change this
```
— [ggml/include/ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)

The spec's "the quantization version is 4" is an illustrative example, not the current value — the code comment says `2` and directs a bump only "on quantization format changes". Real GGUFs in the wild carry `'general.quantization_version': '2'` ([HF city96/t5-v1_1-xxl-encoder-gguf discussion 8](https://huggingface.co/city96/t5-v1_1-xxl-encoder-gguf/discussions/8)).

**When absent, llama.cpp assumes it.** The converter writes `gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION)`, and llama.cpp "assumes 2" for files that omit it, so such models work in llama.cpp but break other readers:

> Turns out it didn't include the `"general.quantization_version"` metadata. In the case that llama.cpp reads a file without a version, it assumes 2 (grep for the line `gguf_set_val_u32(ctx_out, "general.quantization_version", GGML_QNT_VERSION);`), so this model works with llama.cpp but fails with rusformers/llm.
— [rustformers/llm#447](https://github.com/rustformers/llm/issues/447)

**Conclusion:** `quantization_version` is a single monotonic integer that has been `2` throughout the period in question. It cannot distinguish a build that supports `MXFP4`/`Q1_0` from one that does not. **Inference: Rigma should read it for completeness and then ignore it for capability decisions.** (Open question: whether it was ever bumped above 2 — see Could NOT verify #15.)

### The real capability signal: the tensor type enum

`ggml_type` is an append-only enum with deliberate gaps, and it currently extends well past what the published spec documents:

```c
    // NOTE: always add types at the end of the enum to keep backward compatibility
    enum ggml_type {
        ...
        GGML_TYPE_BF16    = 30,
        // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
        // GGML_TYPE_Q4_0_4_8 = 32,
        // GGML_TYPE_Q4_0_8_8 = 33,
        GGML_TYPE_TQ1_0   = 34,
        GGML_TYPE_TQ2_0   = 35,
        // GGML_TYPE_IQ4_NL_4_4 = 36,
        // GGML_TYPE_IQ4_NL_4_8 = 37,
        // GGML_TYPE_IQ4_NL_8_8 = 38,
        GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
        GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
        GGML_TYPE_Q1_0    = 41,
        GGML_TYPE_Q2_0    = 42,
        GGML_TYPE_COUNT   = 43,
    };
```
— [ggml/include/ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)

Compare the published spec, which stops at `GGML_TYPE_MXFP4 = 39` / `GGML_TYPE_COUNT = 40` and does not mention `NVFP4`, `Q1_0` or `Q2_0` at all ([ggml/docs/gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)). **The spec is stale relative to the code.**

**This is the mechanism behind the LM Studio bug — cross-verified.** Issue #1729 reports `invalid ggml type 41` for Bonsai models and infers:

> The bundled llama.cpp backend (v2.10.0) apparently doesn't support whatever quantization type 41 is. This is likely related to the 1-bit quantization used by Bonsai models — the GGUF was probably produced with a newer llama.cpp that added support for this type.
— [lmstudio-bug-tracker#1729](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1729)

Type `41` is `GGML_TYPE_Q1_0` — the 1-bit quantization type. The user's guess and the enum agree exactly.

### `general.file_type` is also unreliable — the published enum contradicts the code

The spec documents one `general.file_type` enum, and `ggml.h` defines a different one with the same integers:

| Value | [gguf.md spec](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md) | [ggml.h code](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h) |
|---|---|---|
| 11 | `MOSTLY_Q3_K_S` | `GGML_FTYPE_MOSTLY_Q3_K` |
| 12 | `MOSTLY_Q3_K_M` | `GGML_FTYPE_MOSTLY_Q4_K` |
| 13 | `MOSTLY_Q3_K_L` | `GGML_FTYPE_MOSTLY_Q5_K` |
| 14 | `MOSTLY_Q4_K_S` | `GGML_FTYPE_MOSTLY_Q6_K` |
| 15 | `MOSTLY_Q4_K_M` | `GGML_FTYPE_MOSTLY_IQ2_XXS` |
| 18 | `MOSTLY_Q6_K` | `GGML_FTYPE_MOSTLY_IQ1_S` |
| 25 | *(absent)* | `GGML_FTYPE_MOSTLY_MXFP4` |
| 28 | *(absent)* | `GGML_FTYPE_MOSTLY_Q2_0` |

Same integers, different meanings. `ggml_ftype` tops out at 28 while `ggml_type` reaches 42. **Inference: anyone building a `file_type → capability` table from the spec document will be wrong.** This is a live trap for Rigma.

### Unknown architecture — the other hard-fail signal

`general.architecture` is a free-form lowercase string; an engine that predates an architecture fails loudly. Real reports in §5.

### GGUF container version

> `uint32_t version;` — The version of the format implemented. Must be `3` for version described in this spec, which introduces big-endian support. This version should only be increased for structural changes to the format.
— [ggml/docs/gguf.md](https://raw.githubusercontent.com/ggml-org/ggml/master/docs/gguf.md)

History from the same document: **v3** adds big-endian support; **v2** changed most countable values from `uint32` to `uint64`; **v1** initial.

Caution: `ggml.h` also defines `#define GGML_FILE_MAGIC 0x67676d6c // "ggml"` and `#define GGML_FILE_VERSION 2` ([ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)) — these belong to the legacy **GGML** format, *not* GGUF, which uses magic `GGUF` and version `3`. Do not conflate them. The exact error text for a too-new GGUF container version is unverified — see Could NOT verify #11.

### Can GGUF metadata infer a minimum engine version?

**Partially — and only via a table Rigma maintains.** Ranked by usefulness:

| Signal | Usable? | Why |
|---|---|---|
| set of distinct tensor `ggml_type` values present | **Yes** | append-only enum with documented gaps; the newest type present bounds the minimum build. This is what actually fails at load (`invalid ggml type 41`). |
| `general.architecture` | **Yes** | new architectures are hard failures on older builds. |
| `general.file_type` | **No** | spec enum contradicts the code enum (§ above). |
| `general.quantization_version` | **No** | constant `2`. |
| `general.version` | **No** | it is the *model's* version. |
| declared minimum | **Does not exist** | no key. |

**Inference:** Rigma should build a small, versioned `capability → min_bNNNN` table keyed on architecture names and tensor-type IDs, and treat a GGUF's own metadata as the *query* against it. Because `bNNNN` is a monotonic commit count, the table's values are plain integers and comparison is trivial. And because there is no declared minimum, Rigma should fail *open* on unknown-but-plausible metadata (warn) and fail *closed* only on a type/architecture it positively knows postdates the selected build.

---

## 5. What tools do when a quant needs a newer engine

**Short answer: they all fail loudly, and none auto-upgrades.** The resolution offered is always "update llama.cpp".

### Failure mode A — unknown quantization type

**`invalid ggml type 41`**, LM Studio, Bonsai-8B GGUF:

> The bundled llama.cpp backend (v2.10.0) apparently doesn't support whatever quantization type 41 is. This is likely related to the 1-bit quantization used by Bonsai models — the GGUF was probably produced with a newer llama.cpp that added support for this type. GPU and memory are not the issue (RTX A4500 20 GB VRAM, 62 GB RAM, only ~1.5 GB needed for the model).
— [lmstudio-bug-tracker#1729](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1729) (filed 2026-04-02)

Behaviour: hard load failure; model unusable. No fallback, no auto-upgrade. Cross-verified against `GGML_TYPE_Q1_0 = 41` in [ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h).

### Failure mode B — unknown model architecture

Three independent real reports:

1. `llama_model_load: unknown model architecture` — [ggml-org/llama.cpp#14951](https://github.com/ggml-org/llama.cpp/issues/14951) (2025-07-29, closed as stale, i.e. never fixed by a code change).
2. Gemma 4 on LM Studio, Linux arm64 + CUDA — and note the user had *already* selected the newest runtime:

> Metadata loads correctly (general.architecture str = gemma4, GGUF V3, tensors listed) before failure. … Steps to reproduce: Install / update lms and run lms daemon up … `lms runtime select --latest` -> observe `llama.cpp-linux-arm64` …
— [lmstudio-bug-tracker#1742](https://github.com/lmstudio-ai/lmstudio-bug-tracker/issues/1742) (2026-04-03)

3. Gemma 7B, HF discussion: "How to get the gguf version to load with llama_cpp? I get the error 'unknown model architecture gemma' and it aborts." — [google/gemma-7b-it discussion 34](https://huggingface.co/google/gemma-7b-it/discussions/34)

Behaviour: hard abort. `#1742` is the most instructive — it shows that **"select the latest runtime" is not a reliable remedy**, because the failure can be in the model/engine pairing rather than in staleness of the engine relative to the *app*.

### The standard remedy, and the diagnosis rule

A widely-mirrored troubleshooting pattern:

> **unknown model architecture** — Cause: Your llama.cpp build is too old to support this architecture. Fix: Update llama.cpp
— [notes.itsvasugrover.com — llama.cpp troubleshooting](https://notes.itsvasugrover.com/kb/ai/llama-cpp/troubleshooting/)

And the useful disambiguation rule, which distinguishes "engine too old" from "you converted it wrong":

> The architecture is genuinely new and your llama.cpp build predates support for it — this produces a named unknown architecture (e.g. 'qwen35moe') rather than a blank one, and needs a llama.cpp update, not a reconversion.
— [markaicode.com — llama.cpp quantization fix](https://markaicode.com/errors/llamacpp-quantization-fix/)

**Inference:** a *named* unknown architecture = engine too old; a *blank* one = malformed conversion. That is a cheap, actionable distinction Rigma could encode in its error message.

### Auto-upgrade: no example found

None of the six tools in §2 auto-upgrades the engine on encountering a too-new model. LM Studio has `lms runtime update` but it is user-initiated, and #1742 shows `select --latest` still failing afterwards. See Could NOT verify #17.

### Silent wrong output: not found

No confirmed case of an old engine producing *silently wrong* results on a too-new model. Every case above is a loud failure. **Inference:** this is reassuring for Rigma's risk model — the realistic failure is a confusing load error, not corrupted output. The one adjacent risk I found is a *performance/quality* fallback rather than a correctness one, and its source is secondary: vLLM's reported `WARNING: Failed to import vllm._C due to CUDA version mismatch` allegedly "does not block startup but significantly degrades performance, forcing fallback" ([ask.csdn.net](https://ask.csdn.net/questions/9213121)) — unverified, flagged.

---

## 6. Side-by-side engine versioning pitfalls

### Disk usage per build

Sizes are real, from the API's `size` field on [b9867](https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/b9867):

| Asset | Bytes | Human |
|---|---:|---:|
| `cudart-llama-bin-win-cuda-12.4-x64.zip` | 391,443,627 | 373 MiB |
| `cudart-llama-bin-win-cuda-13.3-x64.zip` | 390,970,417 | 373 MiB |
| `llama-b9867-bin-win-hip-radeon-x64.zip` | 323,204,347 | 308 MiB |
| `llama-b9867-bin-win-cuda-12.4-x64.zip` | 266,142,585 | 254 MiB |
| `llama-b9867-xcframework.zip` | 254,366,440 | 243 MiB |
| `llama-b9867-bin-win-cuda-13.3-x64.zip` | 161,400,603 | 154 MiB |
| `llama-b9867-bin-ubuntu-rocm-7.2-x64.tar.gz` | 133,202,526 | 127 MiB |
| `llama-b9867-bin-win-sycl-x64.zip` | 114,639,056 | 109 MiB |
| `llama-b9867-bin-ubuntu-openvino-2026.2.1-x64.tar.gz` | 101,263,346 | 97 MiB |
| `llama-b9867-bin-win-openvino-2026.2.1-x64.zip` | 79,957,332 | 76 MiB |
| `llama-b9867-bin-android-arm64.tar.gz` | 78,716,148 | 75 MiB |
| `llama-b9867-bin-ubuntu-sycl-fp16-x64.tar.gz` | 47,510,499 | 45 MiB |
| `llama-b9867-bin-ubuntu-sycl-fp32-x64.tar.gz` | 47,319,531 | 45 MiB |
| `llama-b9867-bin-win-vulkan-x64.zip` | 32,164,547 | 31 MiB |
| `llama-b9867-bin-ubuntu-vulkan-x64.tar.gz` | 31,212,832 | 30 MiB |
| `llama-b9867-bin-ubuntu-vulkan-arm64.tar.gz` | 25,511,976 | 24 MiB |
| `llama-b9867-bin-win-cpu-x64.zip` | 17,486,019 | 17 MiB |
| `llama-b9867-bin-ubuntu-x64.tar.gz` | 15,862,965 | 15 MiB |
| `llama-b9867-bin-ubuntu-s390x.tar.gz` | 14,792,151 | 14 MiB |
| `llama-b9867-bin-ubuntu-arm64.tar.gz` | 12,864,189 | 12 MiB |
| `llama-b9867-bin-win-opencl-adreno-arm64.zip` | 11,970,460 | 11 MiB |
| `llama-b9867-bin-win-cpu-arm64.zip` | 11,378,885 | 11 MiB |
| `llama-b9867-bin-macos-x64.tar.gz` | 11,450,643 | 11 MiB |
| `llama-b9867-bin-macos-arm64.tar.gz` | 11,134,835 | 11 MiB |
| `llama-b9867-ui.tar.gz` | 2,735,058 | 2.6 MiB |
| **total (all 25 assets)** | **2,588,701,017** | **≈2.41 GiB** |

**Inference — the cost model that matters for Rigma:** a Windows build set of cpu + vulkan + rocm costs 17.5 + 32.2 + 323.2 ≈ **373 MB per build**, so ten retained builds ≈ **3.7 GB**. The ROCm/HIP zip alone is 87% of that. **Retention policy, not download, is the disk decision.**

**A non-obvious multiplier, verified in the workflow:** the Windows backend zips are not lean deltas — the release job injects the full CPU toolset into every one of them:

> the windows-cpu zip contains the full toolset (llama-server with the embedded UI, ggml-cpu) - inject it into the other windows zips so that every archive ships the same binaries, only with a different backend library on top
> …
> echo "Injecting windows-cpu binaries (llama-server + CPU backend) into the backend zips..."
— [.github/workflows/release.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml)

**Inference:** `llama-server.exe` is byte-identical across all Windows backend zips of one build. Rigma cannot dedupe by treating backend zips as disjoint; and it *can* rely on the fact that within one build, the only differing files are the `ggml-*` backend libraries.

### Cache invalidation (prompt / KV caches keyed on build)

**Not verified — see Could NOT verify #12.** What I did establish:

- llama.cpp state save/load is abstracted behind `llama_io_write_i` / `llama_io_read_i`, and the read interface carries an explicit failure hook: `// drop tensor data that has been read but not yet applied (e.g. when a restore fails)` — [src/llama-io.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/src/llama-io.h). So restores *can* fail, and callers are expected to handle it.
- `src/llama-io.h` contains **no** state-file magic or version constant; those live further down `src/llama-context.cpp`, which my fetch truncated before the `// state save/load` section.

**Inference (design guidance, not a sourced fact):** given that a restore can fail, Rigma should key any prompt/KV cache on the full build identity (`bNNNN` + commit, or better the asset digest) and treat a key mismatch as a cache *miss* — never attempt a load and hope. Given the §3 finding that asset digests are published, the digest is the natural cache key.

### Calibration measured on one build, applied to another

**Directly verified: not valid. Inferred from the changelog's own contents.** Two lines of evidence:

1. **Benchmarks historically did not record the build.** `llama-bench --version` to "print build info" is listed as a *new* feature in the v0.5.0 changelog (PR #28971) — [releases/latest](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest). Before that, a benchmark number carried no build identity.
2. **Performance changes materially between builds, by design.** v0.5.0's own highlights are backend performance work — "Accelerate CUDA `conv2d` with implicit GEMM (#29135)", "Add Metal MoE and SSM_CONV fusion optimizations (#28948)" — and its changelog is dense with tuning commits, e.g. "cuda : tune MMVQ to MMQ crossover for SM70 (Volta) (#28912)", "CUDA: tune FA for Gemma 4 on Ampere or newer (#29152)", "HIP: broaden MoE ncols_opt tile heuristic on RDNA3.5 architecture (#28935)", "Performance tune for gemma4-26b-a4b flash attention shape. (#28450)" — [releases/latest](https://api.github.com/repos/ggml-org/llama.cpp/releases/latest).

**Inference:** a tok/s calibration is only meaningful as a triple (build, backend, model). Because one build ships the same commit compiled for many backends, the *backend* is a first-class calibration dimension even within a single build — and the build dimension cannot be collapsed. This directly threatens the Rigma design where "calibration measured on one build [is] applied to another".

### Mixing builds / stale shared libraries

**Verified mechanism — the binaries are built to load sibling backend libraries:**

```
-DGGML_BACKEND_DL=ON \
-DCMAKE_INSTALL_RPATH='$ORIGIN' \
-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
```
on Linux, and `-DCMAKE_INSTALL_RPATH='@loader_path'` on macOS — [.github/workflows/release.yml](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/.github/workflows/release.yml).

`$ORIGIN` / `@loader_path` mean "load `ggml-*.so` from the directory containing the executable". **Inference:** this is exactly the property Rigma depends on for per-backend directories to be self-contained — and exactly the property that turns a mislabelled or partially-overwritten directory into an ABI mismatch rather than a clean error. Windows has no `$ORIGIN`; its search order (application directory, then `PATH`) is the classic route to loading a *different* `ggml.dll`. **I found no issue thread quoting such a Windows failure** — see Could NOT verify #13.

### Abandoned / stale build directories

**No real bug report found** of a tool picking up an old engine build because of a wrong version-directory lookup or a stale path. See Could NOT verify #13.

**However, the defect under investigation *is* an instance of this class** — a version-named directory containing a different version's binary. The verified facts that make it detectable without trusting the directory name are exactly the four signals in the Verdict.

### Two Rigma-specific hazards implied by the above (inference)

1. **The `--version` format changed between the two builds in this very defect.** The user reports the correct directories printing `9867 (152d337fa)` and the mislabelled one printing `0.2.0-dev (build 10709, commit 9a9394a89)`. *(This contrast comes from the defect report; I did not independently source the older format or date the change — see Could NOT verify #1.)* A parser keyed on `build (\d+), commit` will fail on `9867 (152d337fa)` and vice versa. **Inference:** parse leniently — look for `b?(\d{3,6})` and a `[0-9a-f]{7,40}` token independently, rather than matching a format.
2. **Rigma's backend token is not llama.cpp's.** `rocm` maps to `hip-radeon` (b9867) or `rocm-10.0` (b11146) on Windows, `ubuntu-rocm-<ver>` on Linux. Combined with the finding that these names *drift between builds*, Rigma should resolve the asset name from the release's asset list (match on a parsed backend/OS/arch tuple) rather than construct it from a template.

---

## Could NOT verify

Declared gaps, in rough order of how much they matter.

1. **The exact date, PR and reason for the `bNNNN` → semver transition, and the exact `--version` output format change.** `docs/release.md` documents the *current* scheme; the discussion body at [discussions/27340](https://github.com/ggml-org/llama.cpp/discussions/27340) was truncated by GitHub's navigation chrome and could not be re-fetched (DuckDuckGo subsequently served an anti-bot CAPTCHA to this IP). The claim that older builds print `9867 (152d337fa)` comes from the defect report, not from a source I read.
2. **The content of `nightly-tag.txt`** (7 bytes). **Inference:** it holds the matching `bNNNN`, since v0.5.0's body names b11146 as its nightly.
3. **Whether the `LLAMA_BUILD_NUMBER` compiled into a binary always equals its release tag.** The tag action makes this true by construction, but I did not read the CMake wiring or test a binary.
4. **Ollama:** whether it verifies its runner binaries at all, and whether it ever ships more than one llama.cpp *version* (only multiple backends were verified, from the documented `Dynamic LLM libraries [...]` log line).
5. **LM Studio:** whether it verifies runtime binaries; the exact error text on a runtime mismatch; and the precise semantics of the "unintended fallback" named in issue #599.
6. **llamafile:** any verification mechanism for the embedded engine.
7. **koboldcpp:** whether it verifies anything, and how (or whether) it exposes the embedded llama.cpp build number. I saw only its own `v1.75.x` versioning.
8. **text-generation-webui:** its llama.cpp build pinning and wheel-selection logic — not sourced at all. Only a DeepWiki overview page was located.
9. **vLLM:** the in-source version-check code path and exact error/warning text. Only `ModuleNotFoundError: No module named 'vllm._C'` ([#1814](https://github.com/vllm-project/vllm/issues/1814)) was verified; the "WARNING: Failed to import vllm._C due to CUDA version mismatch … forcing fallback" text is from a secondary Chinese Q&A page ([ask.csdn.net](https://ask.csdn.net/questions/9213121)) and is unverified.
10. **Whether any tool runs the engine binary's `--version` and parses it to confirm identity before use.** No example found — a negative result, not a confirmed absence.
11. **llama.cpp's exact error strings for:** an unknown `ggml_type` at load; a `general.quantization_version` newer than supported; an unknown `general.file_type`; and a newer GGUF container version. I have real *user-quoted* errors (`invalid ggml type 41`, `unknown model architecture`) but not the source-level messages.
12. **llama.cpp state/session file magic and version constants** (e.g. any `LLAMA_STATE_SEQ_MAGIC` / `LLAMA_STATE_SEQ_VERSION`) and the mismatch error text. `src/llama-io.h` has no such constant; the `src/llama-context.cpp` fetch truncated before its `// state save/load` section, and the spill file cut off at ~100 KB. `grep.app` returned HTTP 429 (Vercel checkpoint) and DuckDuckGo CAPTCHA'd, so no alternative route was available.
13. **Any real bug report of** a Windows `ggml.dll` load-order/version-mixing failure, or of a tool picking up a stale engine build directory because of a wrong version-directory lookup. The *mechanism* is verified (`$ORIGIN`/`@loader_path`/`GGML_BACKEND_DL`); a user-reported instance is not.
14. **When the `actions/attest@v4` step was added.** b11146 has an attestation and b9867 (≈11 weeks earlier) does not; I did not find the introducing PR or date.
15. **Whether `general.quantization_version` has ever been bumped above 2** in llama.cpp history. The current value is 2 ([ggml.h](https://raw.githubusercontent.com/ggml-org/llama.cpp/master/ggml/include/ggml.h)); the spec's "the quantization version is 4" example is unexplained and I could not account for it. This is a genuine open question and worth a follow-up, because if it *has* moved, it is a better signal than §4 concludes.
16. **Which `file_type` enum is authoritative for readers.** I verified that the published spec and `ggml.h` disagree on the meaning of integers 11–18, but not which one the loader actually consults, nor whether readers use `file_type` at all rather than inferring from tensor types.
17. **Any auto-upgrade-on-too-new-model behaviour** in any of the six tools surveyed. No example found — negative result.
18. **Any confirmed case of an old engine producing silently wrong output** (as opposed to a loud load failure) on a too-new model. No example found — negative result.

---

## Method note

Sources were fetched with `web_fetch`. Two retrieval constraints are worth recording for whoever continues this work:

- **DuckDuckGo rate-limited this machine mid-research.** `https://html.duckduckgo.com/html/?q=...` worked initially (as briefed) and then began returning an anti-bot CAPTCHA (`anomaly.js`, `cc=botnet`) after roughly a dozen queries. `grep.app` returned HTTP 429 (Vercel security checkpoint). **Plan for this: front-load the `raw.githubusercontent.com` and `api.github.com` fetches, which stayed reliable throughout, and spend DuckDuckGo queries only on things the raw sources cannot answer.**
- **GitHub HTML pages (issues, discussions) are unreliable through this tool** — the navigation chrome consumes the response and the body is truncated. The GitHub *API* worked perfectly for releases and attestations. For issue bodies, the DuckDuckGo result snippet was often more useful than the page itself. Where I quote an issue, the quote is from the search snippet and is attributed to the issue URL.
