"""Things llama-server says once, quietly, that change what you are running.

The engine reports several decisions as a single WARN line at load and then
never mentions them again. They sit in a log nobody opens while the behaviour
they describe persists for the whole session.

The one that prompted this module, found in this machine's own logs:

    W srv load_model: cache_reuse is not supported by this context,
                      it will be disabled

Rigma passes `--cache-reuse 256`. On a hybrid / DeltaNet architecture the
engine cannot KV-shift, so it turns the flag off. Everything still works; edits
that are not a clean prefix just reprocess from scratch, silently, forever. The
flag being *set* and the flag being *in effect* are different facts, and only
the log knew which one was true.

Pure function over log text, so the patterns are testable without an engine.
"""
from __future__ import annotations

import re

# (id, compiled pattern, severity, what it means for the user)
#
# `confirmed` marks patterns actually observed in this machine's logs. The
# others come from llama.cpp's source and upstream reports and are matched
# best-effort — a pattern that never fires is invisible, which is the right
# failure mode for a diagnostic.
_PATTERNS: list[tuple[str, re.Pattern, str, str, bool]] = [
    ("cache_reuse_disabled",
     re.compile(r"cache[_ ]reuse is not supported", re.I),
     "warn",
     "KV-shift reuse is OFF — this model's architecture does not support it, "
     "so the engine disabled --cache-reuse. Edits that are not a clean prefix "
     "reprocess from scratch. Nothing to fix: it is an architecture limit, not "
     "a setting.",
     True),
    ("swa_disabled",
     # AUDIT 02-6: this used to be `n_swa\s*=\s*0|swa.*will be disabled`. The
     # first alternative is a bare `key = value` parameter dump, not a warning,
     # so a non-SWA model's hparams line reported a no-op for a flag Rigma never
     # passes (grep: `--swa-full` appears nowhere in the launch path). Require
     # the warning shape — `swa` AND `disabl` on the same line.
     re.compile(r"swa.*disabl", re.I),
     "info",
     "The engine reports no sliding window, so it disabled SWA. Nothing to fix: "
     "it is a property of this architecture, not a setting. On hybrid models the "
     "reprocessing trigger is the recurrent component, which SWA does not touch.",
     False),
    ("checkpoint_cascade",
     re.compile(r"erasing.*checkpoint|checkpoint.*eras|no checkpoint found",
                re.I),
     "warn",
     "Context checkpoints were discarded, which forces a full prompt "
     "reprocess. Hybrid and recurrent layers cannot be rewound, so when no "
     "checkpoint matches the resume position the engine drops them all. "
     "Lowering --checkpoint-min-step toward your turn length makes this rarer.",
     False),
    ("kv_cache_full",
     re.compile(r"KV cache is full|context (?:is )?full|slot context shift",
                re.I),
     "warn",
     "The context filled and the engine shifted or dropped tokens. The oldest "
     "part of the conversation is gone from the model's view.",
     False),
    ("fallback_template",
     re.compile(r"chat template.*not found|using default chat template", re.I),
     "warn",
     "No chat template in the gguf, so the engine is using a generic one. The "
     "model is being prompted in a format it was not trained on.",
     False),
]


# One engine PROCESS starts at its parameter dump. `runtime.launch_server` opens
# `server-<port>.log` with mode "w", so the file is normally truncated per
# launch; a log that spans launches (an appended or rotated file) holds several
# of these, and the last one is the process that is up now.
_RUN_START = re.compile(r"common_params_print_info:")


def _current_run(log_text: str) -> str:
    """The part of `log_text` written by the engine process that is up now.

    AUDIT 06R3-10: `findings` used to scan the whole file and quote the last
    matching line, with a docstring asserting that line described the running
    engine. It does not follow: on a log spanning several launches the last
    occurrence can belong to a previous one, so the Engine page showed a
    diagnostic for a flag state the running engine may not be in.

    The engine's own parameter dump is the start marker — `common_params_print_info:`
    is the first line of every one of this machine's 16 `server-*.log` files
    (llama.cpp prints it once per process). When the text carries at least one
    marker, everything before the LAST one is a previous process and is dropped.
    When it carries none (a tail that begins mid-launch) the whole text is
    scanned, because there is no evidence to segment on; in that case the
    caller gets no current-run guarantee, which the docstring now says instead
    of claiming one.
    """
    text = log_text or ""
    starts = [m.start() for m in _RUN_START.finditer(text)]
    return text[starts[-1]:] if starts else text


def findings(log_text: str, *, expected_vram_mb: float | None = None
             ) -> list[dict]:
    """Significant one-off engine statements found in a log.

    Deduplicated: these fire once per load, and a log spanning several restarts
    would otherwise report the same fact many times. Ordered as listed, so the
    confirmed and consequential ones come first.

    Only the current engine process's lines are considered (see `_current_run`),
    so `count` and `example` describe the engine that is up now whenever the log
    carries a launch marker. A log with no marker at all is scanned whole and no
    current-run claim is made.

    A17/S2 is wired in here rather than behind a second endpoint: the engine's
    own load accounting is one more thing the engine said once and never
    repeated. The split verdict is derived from the load the engine itself
    reported (ngl, layer count, expert placement, backend, and the device count
    from the load's own device labels), so this needs no plan at the call site —
    and a HEALTHY load adds no finding at all.

    `expected_vram_mb` is optional and is the plan's own prediction
    (`memtruth.planned_mb(plan)`, i.e. weights + KV for the same ctx / cache /
    slots). Without it the VRAM axis is reported as NOT COMPARABLE and never
    raises a finding, because the engine's actual includes the KV cache and a
    bare file size would read as a divergence that is only the cache. A caller
    that has the running plan can pass it to make that axis comparable.
    """
    text = _current_run(log_text)
    out = []
    for key, pat, severity, message, confirmed in _PATTERNS:
        hits = [ln.strip() for ln in text.splitlines()
                if pat.search(ln)]
        if not hits:
            continue
        out.append({
            "id": key,
            "severity": severity,
            "message": message,
            "count": len(hits),
            "confirmed_here": confirmed,
            # the last occurrence in the CURRENT run (see _current_run)
            "example": hits[-1][:300],
        })

    # A17/S2: the load accounting. Only a real DIVERGENCE is a finding; a
    # healthy load and a "not comparable" one both add nothing.
    r = compare_plan(parse_load(text), expected_vram_mb)
    if r["known"] and r["diverges"]:
        # Quote the engine's own line, as the pattern findings do — and quote
        # the line that belongs to the axis that actually diverged.
        example = r["detail"][:300]
        if r["split_verdict"] == "diverges":
            for line in text.splitlines():
                if _GRAPH_SPLITS.search(line):
                    example = line.strip()[:300]
        out.append({
            "id": "plan_divergence",
            "severity": "warn",
            "message": _PLAN_DIVERGENCE_MESSAGE,
            "count": 1,
            "confirmed_here": False,
            "example": example,
        })
    return out


# ---------------------------------------------------------------------------
# A17/S2 — the engine's own load accounting
# ---------------------------------------------------------------------------
#
# `findings()` answers "did the engine quietly change a setting". It does not
# answer the other half of the same question: "what did the engine actually
# allocate". That matters because on Windows/WDDM an allocation that does not
# fit in VRAM SUCCEEDS out of system RAM at a fraction of the speed and prints
# no error — the plan and the reality diverge silently, and the engine's own
# load log is the only place the real numbers appear.
#
# The formats below are copied from this machine's real load log of the owner's
# 27B hybrid, `.scratch/prism-v.log` (second load, lines 4485-4672), whitespace
# preserved:
#
#   0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU
#   0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB
#   0.01.522.676 I load_tensors:        ROCm0 model buffer size =  6539.67 MiB
#   0.04.285.857 I llama_context: n_seq_max             = 1
#   0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB
#   0.04.369.729 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB
#   0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB
#   0.04.417.938 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB
#   0.04.417.939 I sched_reserve: graph splits = 2
#
# The same file's first load (lines 2159-2340) is a FITTING pass with
# `ROCm0 model buffer size = 0.00 MiB`. Two loads in one file is normal, so the
# parser segments on the `offloaded N/M layers` marker and callers report the
# LAST segment (the engine that is actually up), never the first.
#
# Pure text in, plain dicts out: no engine call, no file read, no I/O.
#
# ---------------------------------------------------------------------------
# Why `graph splits = 2` is the HEALTHY baseline, verified from the engine
# ---------------------------------------------------------------------------
#
# The first version of this comparison took `expected_splits = 1` as a constant,
# which flagged the load above — the healthy one — on every launch. The count
# comes from the scheduler, not from the number of backends:
#
#   * `sched_reserve` prints `n_splits` from `ggml_backend_sched_get_n_splits`:
#     `LLAMA_LOG_INFO("%s: graph splits = %d\n", __func__, n_splits_pp);`
#     — ggml-org/llama.cpp b9867 src/llama-context.cpp:626,637,672-676;
#       PrismML-Eng/llama.cpp 87268f77 src/llama-context.cpp:765,776,811-815.
#     `n_splits` is `sched->n_splits` (ggml/src/ggml-backend.cpp:1923-1925 at
#     b9867; :1995-1997 at 87268f77).
#   * `ggml_backend_sched_split_graph` starts a NEW split whenever the current
#     node's assigned backend differs from the current split's:
#       `if (node_backend_id != cur_backend_id || need_new_split) { ... }`
#     — b9867 ggml/src/ggml-backend.cpp:1303-1317;
#       87268f77 ggml/src/ggml-backend.cpp:1344-1358.
#     So the count is the number of maximal SAME-BACKEND RUNS, not the number of
#     backends. One backend is one split, not zero.
#   * the INPUT layer is always placed on the CPU —
#     "there is very little benefit to offloading the input layer, so always
#     keep it on the CPU", `pimpl->dev_input = { cpu_dev, &pimpl->cpu_buft_list };`
#     — b9867 src/llama-model.cpp:1296-1298; 87268f77 src/llama-model.cpp:1632-1634.
#     prism-v.log:4482 shows it for this very model:
#     `tensor 'token_embd.weight' (pq2_0) ... cannot be used with preferred
#     buffer type ROCm_Host, using CPU instead`.
#
# Therefore a dense model with ANY GPU layers on ONE device is 2: one CPU run
# (the token embedding, contiguous with the first CPU layers under a partial
# offload) and one GPU run. That is why an all-GPU load and a contiguous dense
# partial offload share the same baseline. `ngl = 0` is 1.
#
# The DEVICE COUNT widens that baseline (A17e). The scheduler starts a new run
# whenever the node's backend differs from the current run's, and the layers a
# `--tensor-split` assigns to a device are a contiguous range, so a dense load
# placed across D devices is 1 + D runs: the CPU embedding run, then one run per
# device (CPU -> ROCm0 -> ROCm1 = 3). D is taken from the load's own distinct
# DEVICE labels (`ROCm0`, `ROCm1`, `CUDA0`, ...); `CPU_Mapped` and every
# `<device>_Host` staging buffer are host RAM, not devices. A17e nit: the labels
# come from BOTH the buffer lines and the engine's direct device statements
# (`load_tensors: layer N assigned to device ROCm0`,
# `llama_prepare_model_devices: using device ROCm0`), so a truncated buffer
# section no longer loses the count; the direct per-layer line is the precise
# source and the `using device` list is a fallback (see `device_labels`). A load
# that reports GPU layers but names no device has an UNKNOWN device count, and
# the verdict is "not comparable" — the same rule as the MoE case below, never a
# single-device guess that would flag a two-device load on every launch.
#
# CIRCULARITY, stated precisely. Taking D from the log means the expectation
# moves with the log being judged. The failure this detector exists for is still
# caught: a silent per-op CPU fallback (the fused attention node -> CPU) does
# not remove or rename a device label, so D is unchanged and the split count
# rises above 1 + D. What is NOT visible is a device that silently DROPS OUT of
# the load — a plan for two devices where one fails to initialise and its layers
# are re-placed on the survivor: the surviving label set is smaller, the
# expectation shrinks with it, and the load reads "ok". Seeing that needs the
# plan's device list, which neither `ComboFlags` nor `RunPlan` carries
# (`--tensor-split` is only a known-flag entry, `engines.py:109`, never emitted)
# and which `/api/server/findings` does not have. It is recorded as a residual,
# not guessed.
#
# `offloaded N/M` is `std::min(n_gpu_layers, max_offloadable_layers)` over
# `max_backend_supported_layers = n_layer_all + 1` — the denominator COUNTS THE
# OUTPUT LAYER, so the model's layer count is M - 1
# (b9867 src/llama-model.cpp:1600-1603; 87268f77 src/llama-model.cpp:1941-1944).
#
# MoE expert offload is the case that must NOT be judged by this baseline:
# `--n-cpu-moe N` keeps the expert weights of the first N layers on the CPU
# (`llm_ffn_exps_block_regex(i)` -> `ggml_backend_cpu_buffer_type()`,
# 87268f77 common/arg.cpp:2755-2769), and a WEIGHT tensor on a different,
# incompatible backend forces a new split even when the op itself is on the GPU
# ("check if a weight is on a different and incompatible backend / by starting a
# new split, the memory of the previously offloaded weights can be reused",
# b9867 ggml/src/ggml-backend.cpp:1280-1287; 87268f77 :1321-1328). The expert
# weights are marked as weights for exactly this purpose
# (87268f77 src/llama-model.cpp:1919-1923). So the count depends on the model's
# expert geometry, which the plan's `ngl` does not carry — hence
# "not comparable", never a finding.

# A new load starts at this marker. The neighbouring lines "offloading output
# layer to GPU" and "offloading 63 repeating layers to GPU" also contain
# "offload" but not the `offloaded N/M` shape, so they are not boundaries.
_LOAD_MARKER = re.compile(r"offloaded\s+(\d+)/(\d+)\s+layers to GPU")

# `print_info: n_expert = 0` is printed once per process in the hparams dump
# that PRECEDES `offloaded N/M` (prism-v.log:141 for the fitting pass, :2466 for
# the load that is up). It is what separates a dense model from a MoE, and the
# expert count is what decides whether a split verdict is even possible. The
# regex cannot match the neighbouring `n_expert_used = N` / `n_expert_groups`
# lines: `n_expert` there is followed by `_`, not by whitespace and `=`.
_N_EXPERT = re.compile(r"n_expert\s*=\s*(\d+)")

# Device families the engine names in a buffer label (`ROCm0`, `CUDA0`,
# `Vulkan0`, `Metal`, `SYCL0`). Used to name the plan's backend from the log.
_DEVICE_FAMILIES = (
    ("ROCm", "rocm"), ("CUDA", "cuda"), ("Vulkan", "vulkan"),
    ("Metal", "metal"), ("SYCL", "sycl"),
)

# One buffer line. The label is a single token — `CPU`, `CPU_Mapped`, `ROCm0`,
# `ROCm_Host`, `CUDA0`, `Vulkan0`, `Metal`, ... The kind is one of the four the
# engine prints; `output buffer size` is deliberately not one of them, so the
# `llama_context: ... output buffer size` line is ignored.
_BUFFER = re.compile(
    r"(?P<label>[A-Za-z][\w.]*)\s+"
    r"(?P<kind>model|KV|RS|compute)\s+buffer size\s*=\s*"
    r"(?P<mb>\d+(?:\.\d+)?)\s*MiB")

_N_SEQ_MAX = re.compile(r"n_seq_max\s*=\s*(\d+)")
_GRAPH_SPLITS = re.compile(r"graph splits\s*=\s*(\d+)")

# The engine names the device DIRECTLY as well, without a buffer label:
#
#   D load_tensors: layer  33 assigned to device ROCm0, is_swa = 0
#   I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 9070 XT) ...
#
# (prism-v.log:2510-2574 — 65 lines per load — and :81; both pins print them,
# `llama-model.cpp` `load_tensors`: `ggml_backend_dev_name(dev)` at b9867:1287
# /:1292 and 87268f77:1623/:1628). They are the second source for the device
# count (A17e nit) and the only source when the buffer section is truncated.
#
# The two are kept apart on purpose. `assigned to device` names the device that
# HOLDS a layer, so each label is provably a backend run; `using device` names
# the model's prepared device list, which can include a device that received no
# layers. The precise source therefore wins and the prepared one is only a
# fallback, so this cannot raise the expectation above the layer evidence and
# mask a divergence.
_ASSIGNED_DEVICE = re.compile(r"assigned to device\s+(?P<dev>[A-Za-z][\w.]*)")
_USING_DEVICE = re.compile(
    r"llama_prepare_model_devices: using device\s+(?P<dev>[A-Za-z][\w.]*)")

_KIND_KEY = {
    "model": "model_buffers",
    "KV": "kv_buffers",
    "RS": "rs_buffers",
    "compute": "compute_buffers",
}
_BUFFER_KEYS = ("model_buffers", "kv_buffers", "rs_buffers", "compute_buffers")


def _is_host_buffer(label: str) -> bool:
    """True when a buffer label names host RAM rather than device VRAM.

    `CPU` and `CPU_Mapped` are system RAM (`CPU_Mapped` is the mmap'd weight
    copy); the `<device>_Host` labels (`ROCm_Host`, `Vulkan_Host`, `CUDA_Host`)
    are the pinned host-side staging buffers `sched_reserve` sets aside. Every
    other label (`ROCm0`, `CUDA0`, `Vulkan0`, `Metal`) is a device buffer.

    A17e nit: the `_Mapped` and `_pinned` suffixes are host spellings too. No
    label the engine actually prints takes the form `CUDA_Mapped` /
    `ROCm0_pinned` — the mmap'd copy is always `CPU_Mapped` and the staging
    buffers always end `_Host` — but if one ever did, counting it as a DEVICE
    would raise the expectation and so could MASK a real divergence (the
    over-count direction the verifier recorded). Classifying them as host is
    strictly safer and cannot touch a real device label, none of which ends in
    either suffix.

    Confirmed for `CPU_Mapped` / `ROCm0` / `ROCm_Host` in prism-v.log:4485-4670.
    The CUDA/Vulkan/Metal spellings follow the same llama.cpp buffer-type naming
    but were not observed in this machine's log; a synthetic test pins the
    classification (UNVERIFIED against llama.cpp source in this run).
    """
    return (label in ("CPU", "CPU_Mapped") or label.endswith("_Host")
            or label.endswith("_Mapped") or label.endswith("_pinned"))


def _empty_load() -> dict:
    """A load with nothing known about it. `found` is False, never a fake 0."""
    return {
        "found": False,
        "offloaded": None,          # "65/65" as printed
        "offloaded_layers": None,   # (65, 65) for arithmetic
        "n_expert": None,           # None = the hparams line was not in the text
        "backend": None,            # "rocm" / "cuda" / ... / "cpu" / None
        "model_buffers": [],        # [{"label", "mb", "host"}, ...]
        "kv_buffers": [],
        "rs_buffers": [],
        "compute_buffers": [],
        # A17e nit: the devices the engine named directly, not via a buffer.
        # `devices` is the precise per-layer evidence; `prepared_devices` is the
        # `using device` list, used only when nothing more precise exists.
        "devices": [],
        "prepared_devices": [],
        "n_seq_max": None,
        "graph_splits": None,
    }


def _backend_of(load: dict) -> str:
    """The backend the load's device labels name, or "" when none does.

    The device labels the engine prints are the plan's backend in the log:
    `ROCm0 model buffer size` means the weights were placed on a ROCm device.
    A17e nit: a device named directly (`assigned to device ROCm0`, `using
    device ROCm0`) counts too, so a truncated buffer section does not turn a
    GPU load into a "cpu" one. A load whose only evidence is host labels
    (`CPU` / `CPU_Mapped` / `*_Host`) ran on the CPU; a load with no evidence at
    all has no answer (""), which is not "cpu".
    """
    labels = [b["label"] for key in _BUFFER_KEYS for b in load.get(key, [])]
    labels += list(load.get("devices") or ())
    labels += list(load.get("prepared_devices") or ())
    for label in labels:
        for token, name in _DEVICE_FAMILIES:
            if label.startswith(token):
                return name
    if labels and all(_is_host_buffer(label) for label in labels):
        return "cpu"
    return ""


def device_labels(load: dict) -> list[str]:
    """The distinct DEVICE labels the load's own log names, sorted.

    A17e: the scheduler cuts one run per device, so the device count has to
    come from somewhere. The plan does not carry it — `ComboFlags`/`RunPlan`
    have no tensor split and `--tensor-split` is only a known-flag entry
    (`engines.py:109`), never emitted — and the `/api/server/findings` surface
    is given log text alone. The load's own lines are therefore the only source
    available here.

    A17e nit: TWO sources are unioned. The buffer labels (`ROCm0 model buffer
    size`) are the original one; the engine's direct statements (`load_tensors:
    layer N assigned to device ROCm0`, `llama_prepare_model_devices: using
    device ROCm0`) are the second, and they are the only evidence when the
    buffer section is truncated away. The union can only add a device the
    engine really named, so a single-device load reads exactly as before. The
    `using device` list is a FALLBACK only: it names the prepared device list,
    which can include a device with no layers, and using it while precise
    evidence exists could over-count and mask a divergence.

    Host buffers (`CPU`, `CPU_Mapped`, any `*_Host`, `*_Mapped`, `*_pinned`)
    are not devices. A label counts once however many lines carry it, and it
    counts even when `_DEVICE_FAMILIES` does not recognize its family: the
    question is how many separate device segments the engine reported, and an
    unknown-family label is still a device.

    Circularity is inherent to reading this from the log (see the module
    comment): the expectation moves with the labels. A per-op CPU fallback is
    still caught — it leaves the labels alone and raises the split count — but
    a whole device silently dropping out of the load is not.
    """
    labels = {b["label"] for key in _BUFFER_KEYS for b in load.get(key, [])}
    labels |= set(load.get("devices") or ())
    labels = {label for label in labels if not _is_host_buffer(label)}
    if not labels:
        labels = {label for label in (load.get("prepared_devices") or ())
                  if not _is_host_buffer(label)}
    return sorted(labels)


def parse_loads(log_text: str) -> list[dict]:
    """Every model load in the log, in order.

    One entry per `offloaded N/M layers to GPU` line; buffer, `n_seq_max` and
    `graph splits` lines attach to the load marker that precedes them. A log
    covering a fitting pass and the real load yields two entries — the last is
    the engine that is up now.

    The model's `n_expert` is printed BEFORE the `offloaded` marker (the hparams
    dump comes first), so it is carried forward as a pending value and attached
    to the next load, then cleared — a load whose own hparams dump is missing
    from the text gets `n_expert is None` (unknown), never the previous model's
    count.

    The direct device statements (`assigned to device`, `using device`) are
    carried forward the same way and for the same reason: the engine prints
    them, and the weight tensors, BEFORE the `offloaded N/M` summary line
    (prism-v.log:2510-2574 then :4485), so they belong to the load whose marker
    FOLLOWS them. Attaching them to the load already open would charge the
    previous load with the next one's devices.

    A log tail that starts after the marker (no `offloaded` line but buffer
    lines present) still yields one entry, so the data is not dropped; the
    pending devices attach to it too.
    """
    loads: list[dict] = []
    cur: dict | None = None
    pending_experts: int | None = None
    pending_devices: list[str] = []
    pending_prepared: list[str] = []

    def _attach(load: dict) -> None:
        load["devices"] = pending_devices[:]
        load["prepared_devices"] = pending_prepared[:]

    for line in (log_text or "").splitlines():
        m = _N_EXPERT.search(line)
        if m:
            pending_experts = int(m.group(1))
            continue

        m = _ASSIGNED_DEVICE.search(line)
        if m:
            if m.group("dev") not in pending_devices:
                pending_devices.append(m.group("dev"))
            continue
        m = _USING_DEVICE.search(line)
        if m:
            if m.group("dev") not in pending_prepared:
                pending_prepared.append(m.group("dev"))
            continue

        marker = _LOAD_MARKER.search(line)
        if marker:
            cur = _empty_load()
            cur["found"] = True
            cur["offloaded"] = f"{marker.group(1)}/{marker.group(2)}"
            cur["offloaded_layers"] = (int(marker.group(1)),
                                       int(marker.group(2)))
            cur["n_expert"] = pending_experts
            pending_experts = None
            _attach(cur)
            pending_devices.clear()
            pending_prepared.clear()
            loads.append(cur)
            continue

        buf = _BUFFER.search(line)
        if buf:
            if cur is None:
                cur = _empty_load()
                cur["found"] = True
                _attach(cur)
                pending_devices.clear()
                pending_prepared.clear()
                loads.append(cur)
            label = buf.group("label")
            cur[_KIND_KEY[buf.group("kind")]].append({
                "label": label,
                "mb": float(buf.group("mb")),
                "host": _is_host_buffer(label),
            })
            continue

        if cur is None:
            continue
        m = _N_SEQ_MAX.search(line)
        if m:
            cur["n_seq_max"] = int(m.group(1))
            continue
        m = _GRAPH_SPLITS.search(line)
        if m:
            cur["graph_splits"] = int(m.group(1))

    for load in loads:
        load["backend"] = _backend_of(load)
    return loads


def parse_load(log_text: str) -> dict:
    """The load the engine is running now: the last one in the log.

    Returns an explicit not-found load (`found is False`) when the log has no
    load lines, so a caller cannot mistake "unknown" for "allocated 0 MiB".
    """
    loads = parse_loads(log_text)
    return loads[-1] if loads else _empty_load()


def _sum_buffers(load: dict, *, host: bool) -> float:
    """Total MiB of the buffers whose `host` flag equals `host`."""
    return sum(b["mb"] for key in _BUFFER_KEYS
               for b in load.get(key, []) if b["host"] is host)


# ---------------------------------------------------------------------------
# The split expectation comes from the PLAN, never from a constant
# ---------------------------------------------------------------------------

# The split count for a DENSE plan is `_SPLITS_CPU_ONLY + n_devices` (see the
# provenance block at the top of this section for the engine lines behind it):
# one CPU run for the token embedding, then one run per device. A single device
# is the 2 this machine's own log prints; two devices are 3.
_SPLITS_CPU_ONLY = 1        # one CPU run: embedding and every layer


def plan_fields_from_load(load: dict) -> dict:
    """The plan fields the LOAD LOG itself reports.

    `ngl` and the layer count come from `offloaded N/M`: M counts the output
    layer (llama-model.cpp: `max_backend_supported_layers = n_layer_all + 1`),
    so the model has M - 1 layers and the engine placed N of them.

    `n_cpu_moe` is the one plan field the log does NOT report. `n_expert == 0`
    proves there are no experts to place, so it is 0; anything else leaves the
    expert placement UNKNOWN and is reported as None — a MoE whose experts are
    all resident cannot be told from one that offloaded them, and guessing 0
    is exactly the cry-wolf the guidance forbids.
    """
    layers = load.get("offloaded_layers")
    n_expert = load.get("n_expert")
    return {
        "ngl": layers[0] if layers else None,
        "n_layers": (layers[1] - 1) if layers else None,
        "n_cpu_moe": 0 if n_expert == 0 else None,
        "backend": load.get("backend") or "",
    }


def expected_splits(*, ngl, n_layers, n_cpu_moe=None,
                    backend: str = "", n_devices: int | None = None
                    ) -> int | None:
    """The `graph splits` count a PLAN should produce, or None.

    None means **NOT COMPARABLE**: the plan's own fields do not determine the
    count, so no verdict may be given and no finding may be raised.

      * `ngl == 0`, or a CPU-only backend: everything — the token embedding
        included — runs on the CPU, so the graph is one CPU run => 1 split.
      * a DENSE model with any GPU layers: the input embedding is always on the
        CPU and the offloaded layers form one contiguous GPU run PER DEVICE
        after it => 1 + `n_devices` splits. One device is the same 2 as before
        (the healthy baseline this machine's log prints), and the SAME 2 for an
        all-GPU load and a contiguous partial offload (`ngl` below the layer
        count), which is why a constant 1 flagged this machine's own healthy
        load on every launch.
      * `n_devices is None` or `< 1`: the number of devices the load actually
        used is unknown, and `ngl` alone does not determine the number of runs
        => not comparable (A17e). This is the same rule as the MoE case: a plan
        field that does not determine the count yields no verdict, never a
        single-device guess.
      * `n_cpu_moe > 0`: expert weights on the CPU inside otherwise-GPU layers
        force a new split per affected layer, so the count is a function of the
        model's expert geometry rather than of `ngl` => not comparable.
      * `n_cpu_moe is None`: the expert placement is unknown (a MoE whose
        `--n-cpu-moe` the log does not report) => not comparable.
      * no layer count (a log tail, or a header without `block_count`):
        not comparable.

    A count ABOVE the expectation is the signal worth surfacing: a silent CPU
    attention fallback moves the fused attention node to the CPU while the rest
    of the layer stays on the GPU, i.e. GPU -> CPU -> GPU, roughly two extra
    splits per attention layer, and the engine prints no error.
    """
    if n_layers is None or int(n_layers) <= 0:
        return None
    if ngl is None:
        return None
    if int(ngl) <= 0 or (backend or "").lower() == "cpu":
        return _SPLITS_CPU_ONLY
    if n_cpu_moe is None or int(n_cpu_moe) > 0:
        return None
    if n_devices is None or int(n_devices) < 1:
        return None
    return _SPLITS_CPU_ONLY + int(n_devices)


def expected_splits_for_load(load: dict) -> int | None:
    """`expected_splits` for a parsed load; None when it cannot be derived.

    The device count is the load's own distinct device labels (`device_labels`,
    the union of the buffer labels and the engine's direct `assigned to device`
    statements, with `using device` as a fallback). A load that reports GPU
    layers but names no device anywhere — not even on a direct statement — has
    an UNKNOWN device count, so it is NOT COMPARABLE rather than the
    single-device 2 that would flag its `graph splits` line on every launch.
    """
    fields = plan_fields_from_load(load)
    devices = device_labels(load)
    ngl = fields["ngl"]
    if ngl is not None and int(ngl) > 0 and not devices:
        return None
    return expected_splits(**fields, n_devices=len(devices))


def _split_why_not_comparable(load: dict) -> str:
    """The precise reason a split verdict cannot be given, for the detail text."""
    fields = plan_fields_from_load(load)
    if fields["n_layers"] is None:
        return ("the log does not report how many layers the model has, so the "
                "plan's layer count is unknown")
    if fields["n_cpu_moe"] is None:
        return ("the log reports n_expert = %s, a mixture-of-experts model, and "
                "never reports how many expert layers were kept on the CPU; "
                "expert weights on the CPU inside GPU layers force a new split "
                "per affected layer, so the count depends on n_cpu_moe"
                % load.get("n_expert"))
    if fields["ngl"] is None:
        return "the log does not report the offloaded layer count"
    if int(fields["ngl"]) > 0 and not device_labels(load):
        return ("the log reports %d layers offloaded to a GPU but names no "
                "device (no ROCm0 / CUDA0 / ... buffer line and no "
                "`assigned to device` line), so the number of devices — and "
                "therefore of GPU runs — is unknown"
                % fields["ngl"])
    return "the plan's placement does not determine a split count"


def _split_sentence(load: dict, splits, expected, verdict: str) -> str:
    """One clause naming the split outcome, with "not comparable" kept distinct."""
    if splits is None:
        return ("the log did not report graph splits, so the backend assignment "
                "is NOT COMPARABLE to the plan.")
    if verdict == "not_comparable":
        return ("graph splits = %d, but this is NOT COMPARABLE to the plan: "
                "%s." % (splits, _split_why_not_comparable(load)))
    if verdict == "diverges":
        return ("graph splits = %d, ABOVE the %d the plan expects (%s): some "
                "operations ran on a backend the plan did not assume. A silent "
                "CPU attention fallback adds about two splits per attention "
                "layer (GPU -> CPU -> GPU), so a count this far above the "
                "baseline is the signature to look for." % (
                    splits, expected, _expected_splits_words(load, expected)))
    return "graph splits = %d, matching the %d the plan expects." % (splits,
                                                                    expected)


def _expected_splits_words(load: dict, expected) -> str:
    """How to describe the expected count in a sentence.

    The words follow `expected` (the caller's number), not a fresh derivation
    from the load: an explicit `expected_splits` override must not produce a
    parenthetical that contradicts the number beside it (verifier nit on
    `_split_sentence`, where `expected_splits=1` on an all-GPU load read
    "ABOVE the 1 ... (an all-GPU dense load)"). When the load's own derivation
    agrees with `expected`, its placement is named for context; when it does
    not, the number is named on its own.
    """
    if expected == _SPLITS_CPU_ONLY:
        return "a CPU-only plan"
    n_devices = len(device_labels(load))
    if n_devices >= 1 and expected == _SPLITS_CPU_ONLY + n_devices:
        fields = plan_fields_from_load(load)
        ngl, n = fields["ngl"], fields["n_layers"]
        if n_devices > 1:
            return ("a dense load across %d devices (CPU embedding + %d GPU "
                    "runs)" % (n_devices, n_devices))
        if n and ngl is not None and ngl >= n + 1:
            return "an all-GPU dense load"
        return "a dense partial offload (%s of %s layers on the GPU)" % (ngl, n)
    return "a plan expecting %d backend runs" % expected


# The same slack `memtruth.compare` uses for the same two quantities: below it
# the difference is terms Rigma deliberately does not model (the compute buffer,
# page alignment), so flagging it would be noise. 15% or 512 MiB, whichever is
# larger.
_VRAM_SLACK_MB = 512.0
_VRAM_SLACK_PCT = 0.15

_PLAN_DIVERGENCE_MESSAGE = (
    "The engine's own load accounting does not match the plan. `graph splits` "
    "is the number of backend runs the scheduler cut the graph into; a dense "
    "load with any GPU layers is one CPU run (the token embedding always stays "
    "on the CPU, llama.cpp llama-model.cpp) plus one run per device, so a "
    "count above that means operations ran on a backend the plan did not "
    "assume — most often a silent CPU attention fallback, which adds about two "
    "splits per attention layer and costs speed on every token without "
    "printing an error."
)


def _unknown_plan(expected_vram_mb, expected_splits) -> dict:
    return {
        "known": False,
        "actual_vram_mb": None,
        "host_ram_mb": None,
        "expected_vram_mb": expected_vram_mb,
        "divergence_mb": None,
        "divergence_pct": None,
        "graph_splits": None,
        "expected_splits": expected_splits,
        # S2b: None, not False. This is an UNKNOWN load, so "are the splits
        # unexpected" has no answer; False would read as "no split problem" to a
        # caller looking at this key alone (the `known` flag is the intended
        # guard, but a key must not lie on its own).
        "unexpected_splits": None,
        "split_verdict": "not_comparable",
        "vram_verdict": "not_comparable",
        "diverges": None,       # no verdict, not "no problem"
        "detail": ("no buffer lines in the engine log: what the engine "
                   "allocated is UNKNOWN, not zero — a silent 0 would read as "
                   "'it fits'."),
    }


def compare_plan(parsed, expected_vram_mb: float | None = None,
                 expected_splits: int | None = None) -> dict:
    """What the engine allocated vs what the plan charged it for.

    `actual_vram_mb` sums the DEVICE buffers only:

        device model buffers + KV + RS + device compute

    That is the set that lives in VRAM and is therefore the set the plan's
    charge is supposed to cover. `CPU_Mapped`, `CPU` and every `<device>_Host`
    buffer are host RAM (weights mapped from disk, pinned staging buffers) and
    are reported separately as `host_ram_mb`; counting them as VRAM would
    overstate the overrun and could turn a real one into a false alarm.

    `expected_vram_mb` must be the PLAN'S OWN PREDICTION for the same ctx,
    cache type and slot count — `memtruth.planned_mb(plan)` (the weights plus
    the KV cache the resolver computed) — NEVER a bare file size. The engine's
    actual includes the KV cache, so a file size alone reports the cache as a
    divergence: on this machine's real log that is a +35% "overrun" that is
    entirely the 2,176 MiB KV at ctx 65536. Pass None when no plan-side
    prediction is available at this point; the VRAM axis is then
    `vram_verdict == "not_comparable"`, never a wrong number.

    `expected_splits` is derived from the load's own plan fields
    (`expected_splits_for_load`: ngl, layer count, expert placement, backend,
    and the DEVICE COUNT from the load's distinct device labels) when not given,
    so a healthy dense load on one device expects 2 and on two devices expects
    3 — never a single-device constant. A caller may override it; None means
    "not comparable" and is what the MoE, unknown-layer-count and
    unknown-device-count cases produce.

    Three outcomes per axis, kept distinct:
      * "ok"              — within the plan's expectation (VRAM within the same
                            15% / 512 MiB slack `memtruth.compare` uses; splits
                            at or below the derived count);
      * "diverges"        — splits ABOVE the expectation, or VRAM outside the
                            slack;
      * "not_comparable"  — no basis for a verdict, so none is given.

    `unexpected_splits` stays for callers that only want the split verdict:
    True for "diverges", False for "ok", None for "not_comparable". `diverges`
    is the overall flag (either axis diverges) and is None, not False, on an
    unknown load.

    `parsed` may be one load from `parse_loads` or the whole list (the last
    entry is used). When nothing was parsed the result is explicitly
    `known is False` with `actual_vram_mb is None`, never a 0 that reads as
    "it fits".
    """
    if isinstance(parsed, list):
        parsed = parsed[-1] if parsed else None
    if not parsed or not parsed.get("found"):
        return _unknown_plan(expected_vram_mb, expected_splits)
    if not any(parsed.get(key) for key in _BUFFER_KEYS):
        return _unknown_plan(expected_vram_mb, expected_splits)

    actual = _sum_buffers(parsed, host=False)
    host_ram = _sum_buffers(parsed, host=True)

    # --- VRAM axis ---------------------------------------------------------
    if not expected_vram_mb or expected_vram_mb <= 0:
        vram_verdict = "not_comparable"
        divergence = None
        pct = None
    else:
        divergence = actual - expected_vram_mb
        pct = divergence / expected_vram_mb * 100.0
        slack = max(_VRAM_SLACK_MB, expected_vram_mb * _VRAM_SLACK_PCT)
        vram_verdict = "ok" if abs(divergence) <= slack else "diverges"

    # --- split axis --------------------------------------------------------
    if expected_splits is None:
        expected_splits = expected_splits_for_load(parsed)
    splits = parsed.get("graph_splits")
    if expected_splits is None or splits is None:
        split_verdict = "not_comparable"
        unexpected = None
    else:
        unexpected = splits > expected_splits
        split_verdict = "diverges" if unexpected else "ok"

    detail = ("engine allocated %.2f MiB of device VRAM (device model + KV + RS "
              "+ device compute); host RAM %.2f MiB excluded. " % (actual,
                                                                   host_ram))
    if vram_verdict == "not_comparable":
        detail += ("The VRAM comparison is NOT COMPARABLE: no plan-side "
                   "prediction was supplied for this ctx / cache / slot count, "
                   "and the engine's figure includes the KV cache, so a bare "
                   "file size would read as a divergence that is only the "
                   "cache. ")
    else:
        detail += ("The plan's own prediction for the same ctx / cache / slots "
                   "was %.2f MiB; divergence %+.2f MiB (%+.1f%%), %s. " % (
                       expected_vram_mb, divergence, pct,
                       "matching the plan" if vram_verdict == "ok"
                       else "DIVERGES from the plan"))
    detail += _split_sentence(parsed, splits, expected_splits, split_verdict)

    return {
        "known": True,
        "actual_vram_mb": actual,
        "host_ram_mb": host_ram,
        "expected_vram_mb": expected_vram_mb,
        "divergence_mb": divergence,
        "divergence_pct": pct,
        "graph_splits": splits,
        "expected_splits": expected_splits,
        "unexpected_splits": unexpected,
        "split_verdict": split_verdict,
        "vram_verdict": vram_verdict,
        "diverges": (vram_verdict == "diverges"
                     or split_verdict == "diverges"),
        "detail": detail,
    }
