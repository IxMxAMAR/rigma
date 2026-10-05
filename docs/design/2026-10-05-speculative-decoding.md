# Speculative decoding: MTP, DFlash, DFlash2, DSpark, EAGLE-3, P-EAGLE

Status: implemented on `review/deep-audit-2026-09-22`.
Scope: Rigma's CLI/plan/launch path, the GGUF probe, the fit, the KV fingerprint,
and the two engine runtimes (`llamacpp`, `vllm`). The PrismML fork is covered as
a *build* of the llama.cpp runtime, not as a fourth engine.

## 1. The problem this replaces

`--spec` used to be a string Rigma forwarded verbatim to `--spec-type`, with one
hand-written exception for `draft-mtp`. That was fine while llama.cpp had one
draft family. It stopped being fine because:

* the engine now has **five** draft families with **different artefacts**, and
  asking for a head a file does not carry is a **Vulkan driver reset**, not an
  error message (`hangar.file_has_mtp` documents the measured case);
* two of them (**DFlash2**, **DSpark**) are *variants of an existing
  `--spec-type`*, selected by the file's tensors rather than by the flag, so the
  flag alone cannot express the request and the parser cannot check it;
* one of them (**P-EAGLE**) does not exist in llama.cpp at all;
* vLLM spells every one of them differently, and two of vLLM's spellings are not
  what a user would guess (`peagle`, `dflash2` are **not** method ids).

So the mode vocabulary, the per-engine translation and the refusal conditions now
live in one module, `src/rigma/spec_decode.py`, and every entry in it cites the
upstream line it comes from.

## 2. Vocabulary

Canonical mode ids keep llama.cpp's spelling, because stored specs, registry
combos, launch defaults and calibration rows already contain them
(`spec_decode.LEGACY_MODES` is pinned by a test). Two are new:

| Rigma mode | llama.cpp `--spec-type` | artefact | vLLM `method` |
|---|---|---|---|
| `none` | `none` | — | — |
| `draft-mtp` | `draft-mtp` | **the target file** (`blk.N.nextn.*`) | `mtp` (or `deepseek_mtp`) |
| `draft-eagle3` | `draft-eagle3` | separate gguf | `eagle3` |
| `draft-peagle` | **none — refused** | separate gguf | `eagle3` + `parallel_drafting: true` |
| `draft-dflash` | `draft-dflash` | separate gguf | `dflash` |
| `draft-dflash2` | `draft-dflash` (alias) | separate gguf, must be DFlash2 | `dflash` |
| `draft-dspark` | `draft-dspark` | separate gguf, must be DSpark | `dspark` |
| `draft-simple` | `draft-simple` | separate gguf, any head | `draft_model` |
| `ngram-*` (5) | `ngram-*` | — | `ngram` |

## 3. llama.cpp: what is measured, and on what

Everything below was read from a binary installed on this machine
(`llama-server --help`, plus a string scan of the engine install), not from
memory. `rigma` can reproduce it: `spec_decode.probe_llamacpp(<exe>)`.

**The two builds, measured end to end (0.3 s per probe):**

| mode | pin b9867 | PrismML prism-b10743 |
|---|---|---|
| `draft-mtp`, `draft-eagle3`, `draft-dflash`, `ngram-*` | supported | supported |
| `draft-dflash2` | **refused** — the type is accepted, the loader is v1 | supported |
| `draft-dspark` | **refused** — no such `--spec-type` | supported (both lineages) |
| `draft-peagle` | refused — llama.cpp has no P-EAGLE | same |

Also measured, and each one changes what Rigma may send:

* The help prints the type list **comma-separated and unbracketed**
  (`--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,...`),
  while upstream's `docs/speculative.md` shows a bracketed pipe list. A parser
  that knew only the doc form reported every mode as *unverified* on this
  machine; `parse_help` now reads both.
* The Windows engine is a **9 KB `llama-server.exe` shim plus DLLs**
  (`llama-server-impl.dll`, `llama-common.dll`, `llama.dll`). The loader markers
  are in the DLLs — prism-b10743's `selector_hidden` is in `llama.dll`, its
  `markov_w1` is in `llama-common.dll` and `llama.dll`, and its `markov_head_a`
  is in `llama.dll` — so scanning the executable alone would have reported
  "scanned and absent" for a build that supports the mode, i.e. a **false
  refusal**. `scan_markers` scans the install, not the file.
* `--spec-draft-p-min`, `--spec-draft-n-max`/`-min`, `--spec-draft-model`/`-md`,
  `--spec-draft-hf`/`-hfd`/`--hf-repo-draft`, `--spec-draft-p-split`,
  `--spec-draft-ngl`: present on both builds.
* `--spec-draft-conf-min`: present on **neither**. It is documented upstream for
  DSpark and Rigma knows how to emit it, but `spec_decode.unadvertised_flags`
  **drops** it (with a warning) on a build that does not advertise it, because
  an unknown argument is an argparse exit after the download. `--spec-draft-p-min`
  — the DFlash-family probability floor added by PR #25246 — is the lever both
  builds actually have, and Rigma exposes it as `--spec-p-min` rather than
  pretending the two knobs are the same.
* The whole DFlash family has **one** arch string (`dflash`) and **one**
  `--spec-type`; the variant is decided by the tensors. Master's loader throws
  `"DFlash2 model is missing conv/selector metadata"` when a DFlash2 request
  meets a v1 file, which is why the tensor names — not the version — are the
  capability Rigma probes for.

### Provenance (READ from the GitHub API)

| PR | what it added | merged | first build that *contains* it |
|---|---|---|---|
| #22105 | DFlash | 2026-06-28 | b9831 |
| #25246 | `spec-draft-p-min` in DFlash | 2026-07-03 | b9867 (the pin) |
| #25173 | DSpark | 2026-07-28 | b10164 |
| #27342 | DFlash2 (local convolutions + candidate selector) | 2026-08-27 | b10658 (the PR merged into `xsn/dflash2`; it reached master as `b10f9ca`, which *is* b10658) |

PrismML-Eng/llama.cpp `87268f77` is dated 2026-09-28, ~31.6 days after DFlash2
landed on master, so the fork contains it.

None of this is used for gating. Gating asks the binary (below); the table is
here so a reader can tell which claim is provenance and which is measurement.

### DSpark has two lineages

Mainline folds DSpark into `LLM_ARCH_DFLASH` and names the head
`markov_w1`/`markov_w2`/`conf_proj`. The PrismML fork declares its own
`LLM_ARCH_DSPARK` with `dspark_fc`, `markov_head_a`/`markov_head_b`,
`confidence_head`, `log_snr_fc1/2`, `corr_*`. A gguf converted for one is **not
known** to load on the other, so:

* `gguf_meta` reports the lineage as a distinct draft head
  (`dspark` vs `dspark-prism`);
* a *provable* mismatch (the file says prism, the build's binary contains only
  mainline names) is a refusal with the reason and the remedy;
* a build that was never scanned refuses nothing.

## 4. Capability gating: ask the binary, not a version table

`spec_decode.probe_llamacpp(exe)` runs `--help` and scans the executable in 4 MiB
chunks for the tensor-name markers (`selector_hidden`, `markov_w1`,
`markov_head_a`). The result is tri-state and that is the point:

| verdict | when | what launch does |
|---|---|---|
| `True` | the build advertises the type *and* carries the loader's markers | runs |
| `False` | the build is provably missing the type or the loader | **refuses**, naming the build and what it does accept |
| `None` | no binary to ask, no help text, a wrapper that will not answer | **warns** ("unverified") and runs |

`None` exists because "we could not ask" must never be reported as "not
supported": a model can be planned before the pin is downloaded, and refusing
there would make every mode unusable on a fresh install.

This is what makes the pin's gap honest: `--spec draft-dspark` on b9867 says
*"this engine build does not accept --spec-type draft-dspark (it accepts …)"*,
and `--spec draft-dflash2` on b9867 says the build accepts the type but cannot
load the file, citing PR #27342.

## 5. The artefact gate

A mode's required artefact is checked **before** the launch, because the failure
mode is a driver reset:

* `draft-mtp` → the **target** file's tensor table must show `blk.N.nextn.*`
  (`hangar.file_has_mtp`, which reads the file when the spec has no cached
  answer). A stored launch default that no longer matches is **dropped with a
  word**; an explicit `--spec` is **refused**.
* the draft-file modes → the draft gguf's own head, read with
  `gguf_meta.inspect_gguf`, must be one the mode accepts
  (`spec_decode.DRAFT_HEAD_ACCEPTS`). A near miss gets the sentence that names
  the mode which *would* fit (`--spec draft-dflash` for a plain DFlash file).
* `draft-dflash` on a DFlash2 file is **allowed** — upstream loads it under the
  same type — and `variant_note` says which family member is actually running.
* a draft whose tensors cannot be read is **refused**, not assumed: the unsafe
  direction is launching it anyway.
* a `--spec-draft` given to a mode that drafts **without** one (`draft-mtp` uses
  the target's own heads, the n-gram modes use the context) is not passed to the
  engine as `-md` — that would load a model nothing uses — and the user is told
  it was ignored rather than watching the value vanish.

Both launch paths ask the same question, and both are tested end to end, not
just at the helper: `cli._spec_verdict` runs inside `up` before the download and
again for a stored launch default, and `server_ops.switch_model` — the UI's
relaunch — runs it where the chosen binary is known, so a mode the build cannot
serve is refused with its reason instead of becoming an argparse exit minutes
after calibration.

GGUF-side detection lives in `gguf_meta`: `_DRAFT_HEAD_TENSORS` (substring,
because the names carry a `blk.N` prefix), `_DRAFT_HEAD_EXACT` (`fc`/`d2t` are
compared with the `.weight` suffix stripped, so `fc1` is not EAGLE-3), and
`TensorIndex.draft_head`, which returns `None` on a **truncated** table rather
than claiming absence. The answer is stored per file
(`GgufFile.draft_head`) and healed on load alongside `mtp`.

## 6. vLLM

`--speculative-config` (alias `-sc`) takes ONE JSON object; `method` is the
algorithm. Read from `vllm/config/speculative.py`'s `SpeculativeMethod` literal:
`ngram, medusa, mlp_speculator, draft_model, suffix, custom_class, eagle, eagle3,
extract_hidden_states, ngram_gpu, dflash, dspark` plus 27 `MTPModelTypes`
(`mtp`, `deepseek_mtp`, `qwen3_next_mtp`, …). **There is no `peagle` and no
`dflash2`**, so:

* `draft-peagle` → `eagle3` + `parallel_drafting: true`. Two independent
  sources: vLLM's own max-slots table row is `P-EAGLE | eagle3 | Yes`, and
  `transformers_utils/configs/speculators/base.py` literally does
  `result.update({"method": "eagle3", "parallel_drafting": True})` when a
  checkpoint's `speculators_model_type` is `peagle`. Rigma passes both keys so a
  *bare* draft directory behaves like a speculators checkpoint (which derives
  them itself).
* `draft-dflash2` → `dflash`, because DFlash2 is an *architecture*
  (`DFlash2DraftModel` → `qwen3_dflash2` in the model registry) selected by the
  draft checkpoint, not a method — the same file remaps
  `elif result["method"] == "dflash2": result["method"] = "dflash"`, and
  `algos.py` says DFlash2 "reuses the same DFlash runtime (method=`dflash`)".
* `draft-dspark` → `dspark`, which **is** a method (`DSparkModelTypes`), and the
  speculators docs and a published checkpoint both show it verbatim:
  `"method": "dspark"`, with `"num_speculative_tokens": 7`.
* `draft-mtp` → `mtp`, and the weights are in the target, so no draft checkpoint
  is needed. `deepseek_mtp` is **not** the DeepSeek spelling to prefer: vLLM
  accepts it, warns `"method `deepseek_mtp` is deprecated and replaced with
  mtp."`, and rewrites it. Rigma emits `mtp` for every MTP target.
* the n-gram methods need no draft; everything else needs `model` = a HuggingFace
  repo id or a directory, which is **not** a Rigma gguf.

`engines.vllm_argv(..., spec_mode=..., spec_draft=...)` appends the JSON and
refuses a model-based method with no draft before vLLM starts, because vLLM
profiles the GPU for minutes before it reads its config. Two smaller honesty
notes ride along: an unset `--spec-n-max` is reported for a method whose
checkpoint may not carry its own `num_speculative_tokens`, and a llama.cpp-only
threshold (`--spec-conf-min`/`--spec-p-min`) on a vLLM launch is reported as not
applied rather than silently dropped.

## 7. VRAM

`resolve.draft_cache_mb` gained `draft_bytes`: a **separate** draft gguf is
loaded *beside* the target, so its size is charged into the same non-offloadable
overhead slot the projector uses (`with_launch_overheads`). For `draft-mtp` the
head is inside the target and the measured fixed term stands unchanged; for an
external draft the file's own weights dominate. `0` for an unmeasurable draft,
which is the safe direction — the launch re-fits once the download lands.

The KV fingerprint (`kvcache.FINGERPRINT_FIELDS`) gained `spec_draft`,
`spec_conf_min` and `spec_p_min`: a snapshot restored across a draft swap
describes a run that never happened, even when the KV layout is identical.

## 8. What is NOT verified

* **No vLLM speculative launch has ever run here.** vLLM cannot run on this
  machine (Windows), so the method table, the JSON shape and the flag are read
  from vLLM's source and docs only. The code says so
  (`spec_decode.VLLM_SPEC_UNVERIFIED`).
* **No DFlash/DFlash2/DSpark draft gguf was loaded**, because no draft gguf for
  these families exists on this machine. The tensor names come from upstream
  `gguf-py/gguf/constants.py` and the fork's `prism-arch.h`; the detection is
  tested against synthetic GGUF files built in the test suite, not against real
  converted artefacts.
* **DSpark's `--spec-draft-conf-min` is documented but unmeasured**: absent from
  both binaries here, so its emission path is tested against a synthetic flag set
  rather than a real build.
* Cross-lineage DSpark interchangeability is **unknown**, not disproved — hence a
  warning on an unscanned build rather than a blanket refusal.
* **The web UI has no spec-mode picker yet.** The launch-defaults API accepts
  `spec_type`, `spec_draft`, `spec_n_max`, `spec_conf_min` and `spec_p_min` and
  validates them, but the shipped frontend bundle (a prebuilt `data/ui_v2`
  asset) only renders the per-file MTP badge; it would need a rebuild to offer
  the new modes. The CLI and the API are complete.

## 9. Files

| file | change |
|---|---|
| `src/rigma/spec_decode.py` | new: vocabulary, translations, probe, verdicts, argv/JSON builders |
| `src/rigma/gguf_meta.py` | draft-head tensor families + `TensorIndex.draft_head` |
| `src/rigma/models.py` | `spec_draft`, `spec_conf_min`, `spec_p_min` on `ComboFlags` and `LaunchDefaults`; validation; `GgufFile.draft_head`; argv via `spec_decode` |
| `src/rigma/engines.py` | `vllm_argv(spec_mode=, spec_n_max=, spec_draft=)` |
| `src/rigma/cli.py` | `--spec` from the vocabulary, `--spec-draft`, `--spec-n-max`, `--spec-p-min`, `--spec-conf-min`, and `_spec_verdict` (the one gate) |
| `src/rigma/server_ops.py` | the UI/stored-default path uses the same gate |
| `src/rigma/hangar.py` | `resolve_draft_file`, `draft_head_of_file`, `draft_file_bytes`, heal of `draft_head` |
| `src/rigma/resolve.py` | `draft_bytes` in the fit; `draft_bytes_for` |
| `src/rigma/kvcache.py` | three new fingerprint fields |
| `src/rigma/serve.py` | the launch-defaults API accepts the new keys |
| `tests/test_spec_decode.py` | new: vocabulary, translations, probe, verdicts, gate |
| `tests/test_gguf_meta.py` | draft-head detection over synthetic GGUFs |
| `tests/test_kvcache.py` | the new fields invalidate a cache |
