"""Speculative decoding: ONE vocabulary, translated per engine runtime.

WHY THIS MODULE EXISTS. Speculative decoding used to be a single string Rigma
forwarded verbatim (`--spec-type draft-mtp`), which was fine while llama.cpp was
the only runtime and every draft head was the model's own. Three things broke
that:

  * the draft heads are now a FAMILY — MTP, DFlash, DFlash2, DSpark, EAGLE-3 —
    and they are not interchangeable: DSpark is a DFlash backbone plus a Markov
    head, DFlash2 is a DFlash backbone plus convolutions and a candidate
    selector, and each needs the draft gguf that matches it;
  * the engines spell the same idea differently. llama.cpp has one
    `--spec-type` value for the whole DFlash family and decides DFlash2 from the
    FILE, while vLLM names the algorithm in `--speculative-config`;
  * a wrong choice is not a clean error. Asking llama.cpp for a draft head the
    file does not carry is a documented Vulkan driver-reset loop, which is why
    `file_has_mtp` exists at all.

So a mode here is Rigma's own name for an intent, and every engine gets a
translation plus a capability check. Sources, all read 2026-10-01:

  * llama.cpp `docs/speculative.md` @ master — the flag grammar, the DFlash and
    DSpark usage lines (`--spec-type draft-dflash --spec-draft-n-max 15 -fa on
    --jinja`; `draft-dspark ... --spec-draft-conf-min P`), and the statement
    that `--spec-draft-n-max` "is clamped to the draft model's trained block
    size".
  * llama.cpp master `common/common.h` — `COMMON_SPECULATIVE_TYPE_DRAFT_MTP`,
    `_EAGLE3`, `_DFLASH`, `_DSPARK` ("DSpark speculative decoding (DFlash +
    Markov head)").
  * llama.cpp master `gguf-py/gguf/constants.py` — `MODEL_ARCH.DFLASH` with the
    optional DSpark (`markov_w1`, `markov_w2`, `conf_proj`) and DFlash2
    (`blk.{bid}.attn_conv_*`, `blk.{bid}.ffn_conv_*`, `selector_*`) tensors.
  * vllm-project/speculators `docs/user_guide/algorithms/decision_guide.md` —
    "Speculators currently supports six speculative decoding algorithms:
    Eagle-3, P-EAGLE, DFlash, DFlash2, DSpark, and MTP"; and `dspark.md` —
    'Serving uses vLLM's own `dspark` method (`"method": "dspark"` in
    `--speculative-config`)'.

WHAT THIS MODULE DOES NOT DO: it never decides that a draft head FITS. That is
`resolve.draft_cache_mb` and the fit's business. It decides what to ask the
engine for, and whether asking is honest.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

# --- the vocabulary ---------------------------------------------------------

NONE = "none"
DRAFT_SIMPLE = "draft-simple"
DRAFT_MTP = "draft-mtp"
DRAFT_EAGLE3 = "draft-eagle3"
DRAFT_PEAGLE = "draft-peagle"
DRAFT_DFLASH = "draft-dflash"
DRAFT_DFLASH2 = "draft-dflash2"
DRAFT_DSPARK = "draft-dspark"
NGRAM_CACHE = "ngram-cache"
NGRAM_SIMPLE = "ngram-simple"
NGRAM_MAP_K = "ngram-map-k"
NGRAM_MAP_K4V = "ngram-map-k4v"
NGRAM_MOD = "ngram-mod"

# The order is the order the CLI help and the UI list them in: the modes that
# need nothing on disk first, then the draft-model family, then the n-grams.
MODES: tuple[str, ...] = (
    NONE,
    DRAFT_MTP,
    DRAFT_EAGLE3,
    DRAFT_DFLASH,
    DRAFT_DFLASH2,
    DRAFT_DSPARK,
    DRAFT_SIMPLE,
    DRAFT_PEAGLE,
    NGRAM_SIMPLE,
    NGRAM_MAP_K,
    NGRAM_MAP_K4V,
    NGRAM_MOD,
    NGRAM_CACHE,
)

# Modes that were already spelled this way in stored specs, registry combos,
# launch defaults and calibration rows. Kept as a named set so a future rename
# has one place to look and a test can pin it: a mode in here MUST NOT be
# dropped without a migration.
LEGACY_MODES = frozenset({
    NONE, DRAFT_SIMPLE, DRAFT_EAGLE3, DRAFT_MTP, DRAFT_DFLASH,
    NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE,
})

# --- llama.cpp --------------------------------------------------------------

# `docs/speculative.md` @ master, "General Speculative Parameters", verbatim:
#   --spec-type [none|draft-simple|draft-eagle3|draft-dflash|draft-dspark|
#                draft-mtp|ngram-cache|ngram-simple|ngram-map-k|
#                ngram-map-k4v|ngram-mod]
LLAMACPP_SPEC_TYPES = frozenset({
    "none", "draft-simple", "draft-eagle3", "draft-dflash", "draft-dspark",
    "draft-mtp", "ngram-cache", "ngram-simple", "ngram-map-k",
    "ngram-map-k4v", "ngram-mod",
})

# Rigma mode -> the `--spec-type` value llama.cpp is actually given.
#
# THE ONE NON-IDENTITY: `draft-dflash2` is a Rigma name, not an engine value.
# Upstream has ONE arch string for the whole family (`dflash`) and no
# `draft-dflash2` type — the DFlash2 path is selected by the draft file's own
# tensors (master src/llama-context.cpp: "DFlash2's convolutions and selector").
# So the alias exists to let a user SAY dflash2 and be checked against the file,
# and it translates to the one type the engine has.
LLAMACPP_TYPE: dict[str, str] = {
    NONE: "none",
    DRAFT_SIMPLE: "draft-simple",
    DRAFT_MTP: "draft-mtp",
    DRAFT_EAGLE3: "draft-eagle3",
    DRAFT_DFLASH: "draft-dflash",
    DRAFT_DFLASH2: "draft-dflash",
    DRAFT_DSPARK: "draft-dspark",
    NGRAM_CACHE: "ngram-cache",
    NGRAM_SIMPLE: "ngram-simple",
    NGRAM_MAP_K: "ngram-map-k",
    NGRAM_MAP_K4V: "ngram-map-k4v",
    NGRAM_MOD: "ngram-mod",
    # DRAFT_PEAGLE is deliberately absent: llama.cpp has no P-EAGLE type, and
    # mapping it onto draft-eagle3 would be a silent lie about what runs.
}

# Flags a mode needs BESIDE --spec-type, with the upstream line that shows them.
# A flag here is REQUIRED only when the option it carries was asked for; the
# probe still checks the binary advertises it.
LLAMACPP_FLAG_FOR = {
    "spec_draft": "--spec-draft-model",     # -md / --model-draft / --spec-draft-model
    "spec_draft_hf": "--spec-draft-hf",     # -hfd / --hf-repo-draft
    "spec_n_max": "--spec-draft-n-max",
    # DSpark's confidence cut, as `docs/speculative.md` @ master documents it.
    # MEASURED ABSENT from both builds installed on this machine (mainline
    # b9867 and PrismML prism-b10743, 2026-09-18) — see `unadvertised_flags`,
    # which drops it rather than sending an unknown flag into argparse.
    "spec_conf_min": "--spec-draft-conf-min",
    # The draft's per-position PROBABILITY floor, added for the DFlash family by
    # PR #25246 ("spec: support spec-draft-p-min in DFlash") and MEASURED
    # present in both builds above. It is a different lever from the confidence
    # head's, which is why Rigma keeps them apart instead of mapping one onto
    # the other.
    "spec_p_min": "--spec-draft-p-min",
    "spec_draft_ngl": "--spec-draft-ngl",
}

# Modes whose draft head lives in a SEPARATE gguf, passed with -md.
NEEDS_DRAFT_FILE = frozenset({
    DRAFT_SIMPLE, DRAFT_EAGLE3, DRAFT_PEAGLE,
    DRAFT_DFLASH, DRAFT_DFLASH2, DRAFT_DSPARK,
})

# Modes whose draft head is INSIDE the target file, so the target is what has
# to carry it. `draft-mtp` is the only one today.
TARGET_HEAD = {DRAFT_MTP: "mtp"}

# Which draft-head family (gguf_meta.TensorIndex.draft_head) each mode accepts.
# `draft-dflash` accepts dflash2 as well because upstream loads a DFlash2 file
# under the same `draft-dflash` type — the note in `head_mismatch` says so
# rather than pretending the two files are the same thing.
DRAFT_HEAD_ACCEPTS: dict[str, tuple[str, ...]] = {
    DRAFT_EAGLE3: ("eagle3",),
    DRAFT_PEAGLE: ("eagle3",),
    DRAFT_DFLASH: ("dflash", "dflash2"),
    DRAFT_DFLASH2: ("dflash2",),
    DRAFT_DSPARK: ("dspark", "dspark-prism"),
}

# Modes where the draft file is the ONLY thing that decides the variant, so a
# plain DFlash file under `draft-dflash2` must be refused rather than run.
STRICT_HEAD = frozenset({DRAFT_DFLASH2, DRAFT_DSPARK, DRAFT_EAGLE3, DRAFT_PEAGLE})

# --- vLLM -------------------------------------------------------------------

# Rigma mode -> the `method` field of `vllm serve --speculative-config`.
#
# READ: vLLM's `SpeculativeMethod` literal (vllm/config/speculative.py) is
# exactly ngram, medusa, mlp_speculator, draft_model, suffix, custom_class,
# eagle, eagle3, extract_hidden_states, ngram_gpu, dflash, dspark — plus every
# value of `MTPModelTypes` (27 of them: mtp, deepseek_mtp, qwen3_next_mtp,
# qwen3_5_mtp, gemma4_mtp, ...). There is NO `peagle` and NO `dflash2` method
# id, and mapping Rigma's modes onto those two names would be a pydantic
# failure inside vLLM at startup:
#
#   * P-EAGLE is method `eagle3` with `parallel_drafting: true` — READ from
#     speculative.py's own table row "P-EAGLE | eagle3 | Yes".
#   * DFlash2 is an ARCHITECTURE, not a method: the model registry maps
#     `DFlash2DraftModel` -> `qwen3_dflash2`
#     (vllm/model_executor/models/registry.py) and vLLM picks `dflash` from the
#     draft checkpoint's name/arch (speculative.py). Speculators' dflash2.md
#     agrees: it "can be served in vLLM using `vllm serve ./checkpoint`".
#
# So two Rigma modes share a method with a sibling, exactly as `draft-dflash2`
# shares `--spec-type draft-dflash` on the llama.cpp side. The extra keys that
# tell them apart live in VLLM_METHOD_EXTRA.
VLLM_METHOD: dict[str, str] = {
    DRAFT_MTP: "mtp",
    DRAFT_EAGLE3: "eagle3",
    DRAFT_PEAGLE: "eagle3",
    DRAFT_DFLASH: "dflash",
    DRAFT_DFLASH2: "dflash",
    DRAFT_DSPARK: "dspark",
    DRAFT_SIMPLE: "draft_model",
    NGRAM_SIMPLE: "ngram",
    NGRAM_MAP_K: "ngram",
    NGRAM_MAP_K4V: "ngram",
    NGRAM_MOD: "ngram",
    NGRAM_CACHE: "ngram",
}

# Method ids vLLM's own Literal admits, for validating a caller-supplied one.
# READ from `SpeculativeMethod` in vllm/config/speculative.py, whose nested
# literals are `EagleModelTypes` ("eagle", "eagle3", "extract_hidden_states",
# MTPModelTypes, DFlashModelTypes), `NgramGPUTypes`, and `DSparkModelTypes`
# ("dspark"). `MTPModelTypes` is 28 entries long — `qwen3_next_mtp`,
# `glm4_moe_mtp`, `gemma4_mtp`, ... — and vLLM REWRITES every one of them to
# `mtp` with a deprecation warning:
#
#   if self.method in get_args(MTPModelTypes) and self.method != "mtp":
#       logger.warning("method `%s` is deprecated and replaced with mtp.", ...)
#       self.method = "mtp"
#
# so `deepseek_mtp` is NOT "the DeepSeek spelling" — it is a legacy name that
# vLLM warns about. Rigma emits `mtp` for every MTP target, which is the current
# spelling and the one vLLM's own table documents.
VLLM_METHOD_VALUES = frozenset({
    "ngram", "ngram_gpu", "medusa", "mlp_speculator", "draft_model", "suffix",
    "custom_class", "eagle", "eagle3", "extract_hidden_states", "dflash",
    "dspark", "mtp", "deepseek_mtp",
})

# Extra JSON keys that make a mode's method the RIGHT one.
#
# READ from vllm/transformers_utils/configs/speculators/base.py, which is the
# code that serves a speculators checkpoint:
#
#   if result["method"] == "peagle":
#       result.update({"method": "eagle3", "parallel_drafting": True})
#   elif result["method"] == "dflash2":
#       result["method"] = "dflash"
#
# vLLM derives both of these from the checkpoint's own `speculators_model_type`,
# so passing them here is what makes a NON-speculators checkpoint (a bare draft
# directory) behave the same way. `speculative.py`'s own table agrees:
# "P-EAGLE  eagle3  Yes  K - 1".
VLLM_METHOD_EXTRA: dict[str, dict] = {
    DRAFT_PEAGLE: {"parallel_drafting": True},
}

# Modes whose draft weights may be omitted because they live in the target
# (MTP layers) or need no model at all (n-gram). READ: vLLM's speculative
# docs — "For `ngram`, `ngram_gpu`, `suffix`, and `mtp`, this can often be
# omitted". Everything else needs `model`.
VLLM_DRAFT_OPTIONAL = frozenset({
    DRAFT_MTP, NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE,
})

# What a user must know before believing a vLLM mode works here.
VLLM_CAVEAT: dict[str, str] = {
    DRAFT_DFLASH2: ("vLLM serves DFlash2 as an ARCHITECTURE under method "
                    "'dflash', chosen by the draft checkpoint; it is "
                    "experimental and needs the verifier's full, unquantized "
                    "language-model head, because its selector operates in the "
                    "complete verifier vocabulary"),
    DRAFT_MTP: ("vLLM's MTP method needs a model with native MTP layers "
                "(e.g. Qwen3-Next, Qwen3.5); the weights are in the target, so "
                "no draft checkpoint is needed"),
    DRAFT_PEAGLE: ("vLLM serves P-EAGLE as method 'eagle3' with "
                   "parallel_drafting: true — a speculators checkpoint declares "
                   "that mapping itself, and Rigma passes it so a bare draft "
                   "directory behaves the same way"),
    DRAFT_DSPARK: ("vLLM serves DSpark as method 'dspark'; a separate draft "
                   "checkpoint is needed for Qwen3/Gemma4/Kimi targets, while "
                   "DeepSeek V4 ships the DSpark stages inside the target"),
    DRAFT_DFLASH: "DFlash in vLLM is newer and support is improving rapidly",
}

# `num_speculative_tokens` is what vLLM drafts per step. A speculators
# checkpoint carries its own K and vLLM prefers it, but a bare draft directory
# does not, and the docs list only `ngram`/`ngram_gpu`/`suffix`/`mtp` as methods
# where the model key "can often be omitted" — the count is a separate field
# that this project's own DSpark example sets explicitly
# (`"num_speculative_tokens": 7`). So an unset count is worth a word.
VLLM_COUNT_OPTIONAL = frozenset({
    DRAFT_MTP, NGRAM_SIMPLE, NGRAM_MAP_K, NGRAM_MAP_K4V, NGRAM_MOD, NGRAM_CACHE,
})


def vllm_count_caveat(mode: str, n_max: int) -> str:
    """Why an unset `--spec-n-max` matters on vLLM, or "" when it does not."""
    if n_max > 0 or mode in VLLM_COUNT_OPTIONAL or mode not in VLLM_METHOD:
        return ""
    return (f"vLLM is being given no num_speculative_tokens for {mode}: a "
            "speculators checkpoint carries its own, but a bare draft directory "
            "may need one — pass --spec-n-max")

# vLLM cannot run on Windows (see engines.py), so nothing in this table has been
# executed here. The flag and the method ids are read from vLLM's own source;
# no `--speculative-config` launch has ever run on this machine.
VLLM_SPEC_UNVERIFIED = ("vLLM cannot run on this machine (Windows), so Rigma "
                        "has never executed a --speculative-config launch")


def vllm_needs_draft(mode: str) -> bool:
    """Whether vLLM needs a separate draft checkpoint for this mode."""
    return mode in VLLM_METHOD and mode not in VLLM_DRAFT_OPTIONAL


def vllm_speculative_config(mode: str, *, n_max: int,
                            draft: str = "") -> dict:
    """The `--speculative-config` JSON body for a mode, as a dict.

    Keys are `SpeculativeConfig` fields: `method`, `num_speculative_tokens`,
    `model` for a separate draft, plus whatever the mode needs to be the method
    it claims (`parallel_drafting` for P-EAGLE). `draft` is a HuggingFace repo
    id or a local directory — NOT a gguf path, because vLLM does not serve
    Rigma's gguf library (see `engines.vllm_argv`).
    """
    if mode not in VLLM_METHOD:
        raise ValueError(
            f"{mode!r} has no vLLM speculative method: vLLM serves "
            f"{', '.join(sorted(VLLM_METHOD_VALUES))}")
    body: dict = {"method": VLLM_METHOD[mode]}
    body.update(VLLM_METHOD_EXTRA.get(mode, {}))
    if n_max > 0:
        body["num_speculative_tokens"] = int(n_max)
    if draft:
        body["model"] = draft
    return body


def vllm_speculative_flag(mode: str, *, n_max: int, draft: str = "") -> list[str]:
    """`["--speculative-config", "<json>"]`, or [] for `none`."""
    if mode == NONE or not mode:
        return []
    return ["--speculative-config",
            json.dumps(vllm_speculative_config(mode, n_max=n_max, draft=draft),
                       separators=(",", ":"))]


# --- what the artefact has to be --------------------------------------------

def head_mismatch(mode: str, head: str | None, *,
                  target: bool = False) -> str | None:
    """Why this draft head does not satisfy this mode, or None when it does.

    `head` is `gguf_meta`'s verdict on the file: "mtp", "dflash", "dflash2",
    "dspark", "eagle3", or None when the file carries no draft head at all.
    `target=True` means the file under test is the TARGET model (the only place
    `draft-mtp` looks), False means it is the draft gguf.

    Returns a sentence a user can act on, in Rigma's voice: it names what was
    found, what was needed, and — where the two are close — the mode that would
    have matched.
    """
    if mode in TARGET_HEAD:
        want = TARGET_HEAD[mode]
        if target is False:
            return None                      # not this file's question
        if head is None:
            return (f"{mode} needs the model file itself to carry the MTP "
                    "tensors (blk.N.nextn.*), and this one has none")
        if head != want:
            return (f"{mode} needs the model's own {want} tensors, but this "
                    f"file carries a {head} draft head instead")
        return None
    accepts = DRAFT_HEAD_ACCEPTS.get(mode)
    if accepts is None:
        return None                          # n-gram and none need no artefact
    if head is None:
        return (f"{mode} needs a draft gguf, and this file carries no draft "
                "head Rigma can see — llama.cpp would fail on it, or reset the "
                "GPU driver on Vulkan")
    if head in accepts:
        return None
    # Close calls get the useful sentence: say which mode WOULD fit.
    for other, others in DRAFT_HEAD_ACCEPTS.items():
        if head in others and other != mode:
            return (f"{mode} needs a {accepts[0]} draft head, but this file is "
                    f"a {head} draft — use --spec {other} for it")
    return (f"{mode} needs a {accepts[0]} draft head, but this file carries "
            f"{head}")


def variant_note(mode: str, head: str | None) -> str:
    """A one-line remark when the artefact is not exactly what the mode names.

    Empty when there is nothing to say. The case this exists for: `--spec
    draft-dflash` on a DFlash2 file runs (upstream loads it under the same
    type), but the user asked for the older family member and should be told
    which one they actually got.
    """
    if mode == DRAFT_DFLASH and head == "dflash2":
        return ("note: this draft file is DFlash2 (it carries the convolutions "
                "and selector), running under llama.cpp's draft-dflash type; "
                "--spec draft-dflash2 states that explicitly")
    return ""


# --- binary markers: what the LOADER knows, not just the arg parser ---------
#
# `--help` proves which `--spec-type` values a build's argument parser accepts,
# and for most modes that is the whole question. It is NOT the whole question
# for two of them:
#
#   * DFlash2 runs under llama.cpp's `draft-dflash` type, so a build that
#     predates PR #27342 advertises the type and still cannot load the file.
#     Master's loader decides by looking up `selector_hidden.weight` and throws
#     "DFlash2 model is missing conv/selector metadata" — so the tensor NAME is
#     the capability, and that name is a string literal in the binary.
#   * DSpark exists in two lineages with different tensor names (mainline
#     `markov_w1/w2` + `conf_proj`; PrismML fork `markov_head_a/b` +
#     `confidence_head`), and a file converted for one is not known to load on
#     the other.
#
# So the probe also scans the engine INSTALL (the executable and the binaries
# beside it) for these names. A marker is evidence of ABSENCE when the scan ran
# and did not find it; a scan that could not run reports nothing at all, which
# stays "unknown".
MARKERS: dict[str, str] = {
    "dflash2": "selector_hidden",
    "dspark": "markov_w1",
    "dspark-prism": "markov_head_a",
}
_MARKER_CHUNK = 4 << 20      # 4 MiB
_MARKER_OVERLAP = 64         # a name cannot straddle a chunk boundary
# The files a marker can live in. MEASURED on this machine: the Windows engine
# is a 9 KB `llama-server.exe` SHIM plus the real code in DLLs, so scanning the
# executable alone finds nothing — and, far worse, would report "scanned and
# absent", turning a supported mode into a false refusal. PrismML b10743's
# DFlash2 marker lives in `llama.dll`; its DSpark markers live in `llama.dll`
# and `llama-common.dll`; the pin b9867 has none of the three anywhere.
_MARKER_SUFFIXES = (".exe", ".dll", ".so", ".dylib")


def _scan_file(path) -> frozenset | None:
    """Which markers one file contains, or None when it cannot be read."""
    needles = {name: mark.encode() for name, mark in MARKERS.items()}
    found: set[str] = set()
    tail = b""
    try:
        with open(path, "rb") as f:
            while True:
                block = f.read(_MARKER_CHUNK)
                if not block:
                    break
                buf = tail + block
                for name, needle in needles.items():
                    if name not in found and needle in buf:
                        found.add(name)
                if len(found) == len(needles):
                    break
                tail = buf[-_MARKER_OVERLAP:]
    except OSError:
        return None
    return frozenset(found)


def marker_files(path: str | Path) -> list:
    """The binaries a capability marker can hide in, best candidate first.

    The executable itself, then the binaries BESIDE it — the engine is an
    install, not one file, and on Windows it is a shim plus several DLLs.
    """
    exe = Path(path)
    cands = [exe]
    try:
        if exe.parent.is_dir():
            cands += [p for p in sorted(exe.parent.iterdir())
                      if p != exe and p.is_file()
                      and p.suffix.lower() in _MARKER_SUFFIXES]
    except OSError:
        pass
    return cands


def scan_markers(path: str | Path) -> frozenset | None:
    """Which known capability markers this ENGINE contains, or None if unreadable.

    Scans the executable and every binary beside it, because the loader is not
    always in the executable (see `_MARKER_SUFFIXES`). Chunked, so a 300 MB
    install never lands in memory whole, and bounded by the number of markers:
    the scan stops as soon as all of them are found.
    """
    found: set[str] = set()
    read_any = False
    for candidate in marker_files(path):
        got = _scan_file(candidate)
        if got is None:
            continue
        read_any = True
        found |= got
        if len(found) == len(MARKERS):
            break
    return frozenset(found) if read_any else None


# --- engine capability, measured --------------------------------------------

@dataclass(frozen=True)
class EngineSpecCaps:
    """What a llama.cpp-family BINARY says it supports, or why we cannot tell.

    `spec_types` and `flags` are None when the probe could not answer. That is
    NOT an empty set: "this build has no such flag" and "we did not manage to
    ask" are different answers, and only one of them justifies a refusal.
    """
    server_exe: str = ""
    spec_types: frozenset | None = None
    flags: frozenset | None = None
    error: str = ""
    # Capability markers found in the binary (see MARKERS). `markers_probed`
    # separates "scanned and absent" from "never scanned", which is the same
    # distinction `spec_types is None` draws for the help text.
    markers: frozenset = frozenset()
    markers_probed: bool = False

    @property
    def known(self) -> bool:
        return self.spec_types is not None


def parse_help(text: str) -> tuple[frozenset, frozenset]:
    """(accepted --spec-type values, long flags) from a `llama-server --help`.

    TWO renderings of the type list exist and both are handled, because guessing
    one is how a probe silently reports "no support":

      * `--spec-type [none|draft-simple|draft-dflash|...]` — the bracketed pipe
        list upstream's `docs/speculative.md` shows;
      * `--spec-type none,draft-simple,draft-eagle3,...` — the COMMA-separated
        list the shipped builds actually print (MEASURED on b9867 and
        prism-b10743). Reading only the bracketed form made every mode
        "unverified" on this machine.

    The flag inventory is every `--long-flag` the help mentions, which is how a
    build that predates `--spec-draft-conf-min` is caught without a version
    guess.
    """
    types: set[str] = set()
    for line in text.splitlines():
        if "--spec-type" not in line:
            continue
        rest = line.split("--spec-type", 1)[1]
        if "[" in rest and "]" in rest:
            inside = rest[rest.index("[") + 1:rest.rindex("]")]
            types |= {t.strip() for t in inside.split("|") if t.strip()}
            continue
        for token in rest.split():
            if token.startswith("-"):
                break                    # the next flag: nothing here to take
            candidates = {t.strip() for t in token.split(",") if t.strip()}
            if candidates & LLAMACPP_SPEC_TYPES:
                types |= candidates
                break
    flags = set(re.findall(r"--[a-z0-9][a-z0-9-]+", text))
    return frozenset(types), frozenset(flags)


def probe_llamacpp(server_exe: str | Path, *,
                   popen=subprocess.run, timeout: float = 20.0) -> EngineSpecCaps:
    """Ask a llama-server binary what speculative decoding it accepts.

    `--help` is the cheap, side-effect-free question: it allocates nothing and
    loads no weight. A build that cannot be asked (missing DLL, timeout, a
    wrapper that will not answer) returns `error` set and both sets None, which
    every caller must treat as "no evidence" rather than "no support".
    """
    exe = Path(server_exe)
    if not exe.exists():
        return EngineSpecCaps(server_exe=str(exe),
                              error=f"no engine at {exe}")
    try:
        cp = popen([str(exe), "--help"], capture_output=True, text=True,
                   timeout=timeout)
    except Exception as e:                      # missing dll, timeout, ENOENT
        return EngineSpecCaps(server_exe=str(exe),
                              error=f"the engine could not be asked: {e}")
    text = (cp.stdout or "") + (cp.stderr or "")
    if not text.strip():
        return EngineSpecCaps(
            server_exe=str(exe),
            error=f"the engine printed no help (exit {cp.returncode})")
    types, flags = parse_help(text)
    # The marker scan runs even when the type list could not be read: the two
    # are independent evidence, and `markers_probed` must mean "we looked"
    # rather than "we got that far".
    markers = scan_markers(exe)
    if not types:
        return EngineSpecCaps(
            server_exe=str(exe),
            error="the engine's help names no --spec-type values, so Rigma "
                  "cannot tell which speculative modes this build has",
            flags=flags, markers=markers or frozenset(),
            markers_probed=markers is not None)
    return EngineSpecCaps(
        server_exe=str(exe), spec_types=types, flags=flags,
        markers=markers or frozenset(), markers_probed=markers is not None)


@dataclass(frozen=True)
class SpecSupport:
    """Whether a mode can be asked of an engine, and the sentence that says so.

    `supported` is tri-state on purpose. None means the question was not
    answered (no engine binary to ask), which must read as "unverified" rather
    than as a refusal — Rigma refuses a launch on False and warns on None.
    """
    mode: str
    runtime: str
    supported: bool | None
    reason: str = ""
    caveat: str = ""

    def __bool__(self) -> bool:
        return self.supported is True


def support(mode: str, runtime: str, caps: EngineSpecCaps | None = None) -> SpecSupport:
    """Can this mode be asked of this engine runtime, and why not?

    `runtime` is `engines.LLAMACPP` or `engines.VLLM`; `caps` is the probe
    result for a llama.cpp-family binary (the pin, a registered fork, anything)
    and is ignored for vLLM, whose capability is a documented method table
    rather than a binary Rigma can interrogate on this machine.
    """
    if mode == NONE or not mode:
        return SpecSupport(mode, runtime, True, "speculative decoding is off")
    if mode not in MODES:
        return SpecSupport(mode, runtime, False,
                           f"{mode!r} is not a speculative mode Rigma knows")
    if runtime == "vllm":
        if mode not in VLLM_METHOD:
            return SpecSupport(
                mode, runtime, False,
                f"vLLM has no speculative method for {mode}; it serves "
                f"{', '.join(sorted(set(VLLM_METHOD.values())))}")
        return SpecSupport(mode, runtime, True,
                           f"vLLM serves this as method {VLLM_METHOD[mode]!r}",
                           caveat=VLLM_CAVEAT.get(mode, ""))
    # llama.cpp and its forks.
    want = LLAMACPP_TYPE.get(mode)
    if want is None:
        return SpecSupport(
            mode, runtime, False,
            f"llama.cpp has no --spec-type for {mode} "
            f"(it has {', '.join(sorted(LLAMACPP_SPEC_TYPES))})")
    if caps is None or not caps.known:
        why = (caps.error if caps is not None and caps.error
               else "Rigma has not asked this engine what it accepts")
        return SpecSupport(
            mode, runtime, None,
            f"this engine's speculative types are unverified: {why}")
    if want not in caps.spec_types:
        return SpecSupport(
            mode, runtime, False,
            f"this engine build does not accept --spec-type {want} "
            f"(it accepts {', '.join(sorted(caps.spec_types))})")
    # DFlash2 and DSpark share a `--spec-type` with their predecessors, so the
    # argument parser is not evidence that the LOADER can read the file.
    needed = {DRAFT_DFLASH2: ("dflash2",),
              DRAFT_DSPARK: ("dspark", "dspark-prism")}.get(mode)
    if needed and caps.markers_probed and not (set(needed) & set(caps.markers)):
        extra = (" — DFlash2 landed in llama.cpp PR #27342, and this build's "
                 "loader has no selector/convolutions"
                 if mode == DRAFT_DFLASH2 else
                 " — this build predates DSpark in both lineages")
        return SpecSupport(
            mode, runtime, False,
            f"this engine build accepts --spec-type {want} but cannot load a "
            f"{' / '.join(needed)} draft file{extra}")
    caveat = ""
    if mode == DRAFT_DSPARK and caps.markers_probed:
        lineage = dspark_lineage(caps)
        if lineage:
            caveat = (f"this build's DSpark loader uses the {lineage} tensor "
                      "names, so the draft gguf must come from that lineage")
    return SpecSupport(mode, runtime, True,
                       f"the engine accepts --spec-type {want}",
                       caveat=caveat)


def capability_refusal(mode: str, caps: EngineSpecCaps | None) -> str:
    """Why this engine build cannot serve this mode, or "" when it can or cannot tell.

    The one-line form of `support(...)` for a caller that has a binary in hand and
    only needs to know whether to refuse — the UI's relaunch path, which builds
    its argv far from the CLI and must not be the one place a mode gets through
    unchecked.
    """
    verdict = support(mode, "llamacpp", caps)
    return verdict.reason if verdict.supported is False else ""


def dspark_lineage(caps: EngineSpecCaps) -> str:
    """Which DSpark tensor naming a build carries: mainline, prism, both, or "".

    Mainline llama.cpp folds DSpark into the DFlash arch and names the head
    `markov_w1`/`markov_w2`/`conf_proj`; PrismML-Eng/llama.cpp declares its own
    `LLM_ARCH_DSPARK` with `markov_head_a/b`, `confidence_head` and friends.
    A gguf converted for one is not known to load on the other, so which one a
    build carries is a fact worth carrying forward to the launch.
    """
    if not caps.markers_probed:
        return ""
    main = "dspark" in caps.markers
    prism = "dspark-prism" in caps.markers
    if main and prism:
        return "both"
    return "mainline" if main else ("prism" if prism else "")


def artifact_lineage_ok(mode: str, head: str | None,
                        caps: EngineSpecCaps | None) -> str | None:
    """Why this BUILD cannot load this draft FILE's lineage, or None.

    Only DSpark has two lineages today. `None` covers both "they match" and
    "the build was never scanned" — the second is not a refusal, and the caller
    can say so from `caps.markers_probed`.
    """
    if mode != DRAFT_DSPARK or head not in ("dspark", "dspark-prism"):
        return None
    if caps is None or not caps.markers_probed:
        return None
    build_main = "dspark" in caps.markers
    build_prism = "dspark-prism" in caps.markers
    file_prism = head == "dspark-prism"
    if file_prism and not build_prism:
        return ("this draft is a PrismML-lineage DSpark file (markov_head_a/b), "
                "and this engine build's DSpark loader is the mainline one "
                "(markov_w1/w2) — convert the draft for the engine that will "
                "load it")
    if not file_prism and not build_main:
        return ("this draft is a mainline-lineage DSpark file (markov_w1/w2), "
                "and this engine build's DSpark loader is the PrismML one "
                "(markov_head_a/b) — convert the draft for the engine that will "
                "load it")
    return None


# --- the argv a llama.cpp-family launch gets --------------------------------

def draft_shape(value: str) -> str:
    """"path", "hf", or "" — which KIND of draft artefact a value is.

    The two kinds select two different engine flags (`-md` vs
    `--spec-draft-hf`), so the rule lives in ONE place and every caller —
    `ComboFlags`, `LaunchDefaults`, `draft_model_flag`, the launch-time file
    resolver — asks it rather than re-implementing the same string test. Raises
    ValueError for a value that is neither shape, which can only be a typo.
    """
    s = (value or "").strip()
    if not s:
        return ""
    if s.lower().endswith(".gguf") or s[0] in "./\\" or "\\" in s or ":" in s:
        return "path"
    if "/" in s:
        return "hf"
    raise ValueError(
        f"draft {value!r} is neither a .gguf path nor a HuggingFace "
        "user/model id — the first is passed with -md, the second with "
        "--spec-draft-hf, and Rigma cannot tell which you meant")


def draft_model_flag(value: str) -> list[str]:
    """The flag pair that hands llama.cpp a draft artefact, or [] for none.

    `-md` takes a file, and `--spec-draft-hf` takes a `user/model[:quant]` id
    the engine may download.
    """
    kind = draft_shape(value)
    if not kind:
        return []
    return (["-md", value.strip()] if kind == "path"
            else ["--spec-draft-hf", value.strip()])


def llamacpp_spec_args(mode: str, *, n_max: int, draft: str = "",
                       conf_min: float = 0.0,
                       p_min: float = 0.0) -> list[str]:
    """The speculative fragment of a llama.cpp argv, or [] when speculation is off.

    Translation ONLY: no capability check and no file read, because this runs
    where the argv is built and both of those belong at launch (where the engine
    binary and the model are known). A mode with no llama.cpp spelling raises
    rather than emitting a flag the engine would reject in argparse.
    """
    if not mode or mode == NONE:
        return []
    want = LLAMACPP_TYPE.get(mode)
    if want is None:
        raise ValueError(
            f"llama.cpp has no --spec-type for {mode!r}; it accepts "
            f"{', '.join(sorted(LLAMACPP_SPEC_TYPES))}")
    args = ["--spec-type", want]
    if int(n_max) > 0:
        args += ["--spec-draft-n-max", str(int(n_max))]
    # n_max < 1 is this codebase's "no opinion" sentinel (batch, ubatch and ngl
    # all use 0 that way), and the engine's own default is 3. Emitting a literal
    # `--spec-draft-n-max 0` would ask it to draft NOTHING, which is the opposite
    # of "unset" — so the flag is omitted and the engine's default stands.
    #
    # The draft artefact is only passed for a mode that HAS one: `draft-mtp`
    # drafts from the target's own heads and the n-gram modes draft from the
    # context, so `-md` there would load a model nothing uses. `_spec_verdict`
    # warns about the ignored value rather than dropping it silently.
    if mode in NEEDS_DRAFT_FILE:
        args += draft_model_flag(draft)
    if conf_min and float(conf_min) > 0:
        # `--spec-draft-conf-min P` — upstream documents it for DSpark
        # ("truncates each drafted block at the first position whose predicted
        # acceptance falls below P"). `unadvertised_flags` is what keeps this
        # off a build that does not have it.
        args += ["--spec-draft-conf-min", f"{float(conf_min):g}"]
    if p_min and float(p_min) > 0:
        # The draft's own probability floor. Same story: the flag is checked
        # against the binary before it is ever sent.
        args += ["--spec-draft-p-min", f"{float(p_min):g}"]
    return args


def unadvertised_flags(caps: EngineSpecCaps | None, *, draft: str = "",
                       conf_min: float = 0.0,
                       p_min: float = 0.0) -> list[tuple[str, str]]:
    """[(the setting asked for, the flag this build does not advertise)].

    Empty when the binary was never asked — silence is not evidence, so an
    unprobed engine must not cause a setting to be dropped. This exists because
    a flag Rigma invented or mis-remembered would otherwise reach argparse and
    kill the launch AFTER a download: `--spec-draft-conf-min` is documented
    upstream and is absent from both builds measured here, while
    `--spec-draft-p-min` is present in both. The probe decides, not a table.
    """
    if caps is None or not caps.flags:
        return []
    out: list[tuple[str, str]] = []
    if draft and draft_shape(draft) == "hf":
        out.append(("spec_draft_hf", LLAMACPP_FLAG_FOR["spec_draft_hf"]))
    if conf_min and float(conf_min) > 0:
        out.append(("spec_conf_min", LLAMACPP_FLAG_FOR["spec_conf_min"]))
    if p_min and float(p_min) > 0:
        out.append(("spec_p_min", LLAMACPP_FLAG_FOR["spec_p_min"]))
    return [(what, flag) for what, flag in out if flag not in caps.flags]
