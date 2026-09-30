"""A17/S2: the engine's own load log is ground truth Rigma does not read.

Every number below is copied from this machine's real load log of the owner's
27B hybrid, `.scratch/prism-v.log` (second model load, lines 4485-4672). The
point of reading it is the silent-failure detector: on Windows/WDDM an
over-budget plan SUCCEEDS out of system RAM at a fraction of the speed and
prints no error, so the only evidence that plan and reality diverged is the
engine's own buffer accounting.

The fitting pass at lines 2159-2340 is a *different* load in the same file,
with `ROCm0 model buffer size = 0.00 MiB`; a parser that takes the first (or
any) occurrence reports zeros and reads as "it fits".

These are trimmed copies embedded here on purpose: the real file is 357,442
bytes (~350 KiB) and reading it at test time would couple a unit test to a
scratch artifact.

A17's FIRST version asserted that this real load was "unexpected" because
`compare_plan` took `expected_splits = 1` as a constant. That fixture IS the
healthy case — `offloaded 65/65 layers to GPU`, flash attention fused,
`graph splits = 2` — so the baseline was wrong and every launch would have been
flagged. The tests below pin the corrected reading: the real fixture is HEALTHY
(no finding), a synthetic CPU attention fallback is a finding, and a MoE that
offloads experts reads "not comparable" rather than a finding.

A17e closes the residual version of the same defect: the derived baseline knew
`ngl` but not the DEVICE COUNT, so a healthy dense load split across two devices
(CPU embedding + ROCm0 + ROCm1 = 3 splits) was flagged. The device count now
comes from the load's own distinct device labels; a load whose device count
cannot be derived reads "not comparable", never a single-device guess.
"""
import pytest

from rigma import engine_log

# ---------------------------------------------------------------------------
# Fixtures: verbatim lines from .scratch/prism-v.log (whitespace preserved).
# ---------------------------------------------------------------------------

# The fitting pass: prism-v.log:2159-2340. Model buffers are ZERO because this
# pass only reserves; the real weights land in the second load. `n_expert = 0`
# is prism-v.log:141 — the hparams dump of the same process, printed before the
# `offloaded` marker.
FITTING_PASS = (
    "0.00.521.409 I print_info: n_expert              = 0\n"
    "0.00.529.481 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.00.529.483 I load_tensors:        ROCm0 model buffer size =     0.00 MiB\n"
    "0.00.529.484 I load_tensors:    ROCm_Host model buffer size =     0.00 MiB\n"
    "0.00.534.448 I llama_context: n_seq_max             = 1\n"
    "0.00.547.064 I llama_kv_cache:      ROCm0 KV buffer size =     0.00 MiB\n"
    "0.00.559.976 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB\n"
    "0.00.578.703 I sched_reserve:      ROCm0 compute buffer size =   400.28 MiB\n"
    "0.00.578.709 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.00.578.710 I sched_reserve: graph splits = 2\n"
)

# The real load: prism-v.log:4485-4672. `n_expert = 0` is prism-v.log:2466, the
# hparams dump of the SAME process (the second one in the file), which is what
# makes this a dense model and the split count derivable.
REAL_LOAD = (
    "0.00.987.636 I print_info: n_expert              = 0\n"
    "0.01.522.664 I load_tensors: offloading output layer to GPU\n"
    "0.01.522.670 I load_tensors: offloading 63 repeating layers to GPU\n"
    "0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    "0.01.522.676 I load_tensors:        ROCm0 model buffer size =  6539.67 MiB\n"
    "0.04.285.857 I llama_context: n_seq_max             = 1\n"
    "0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB\n"
    "0.04.369.729 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB\n"
    "0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB\n"
    "0.04.417.938 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.04.417.939 I sched_reserve: graph splits = 2\n"
)

# The plan's OWN prediction for the real load, i.e. `memtruth.planned_mb` on the
# real plan (ternary-bonsai-2-27b-uncensored-heretic-pq2-0, ctx 65536, q8_0/q8_0):
# the WHOLE GGUF file (7,206,168,928 bytes = 6872.34 MiB) plus the KV cache the
# resolver computed for the same ctx, cache type and slot count (2176.00 MiB)
# = 9048.34 MiB. The weight term is the FILE, not the sum of the engine's weight
# buffer labels below (CPU_Mapped 322.07 + ROCm0 6539.67 = 6861.74 MiB): those
# buffers omit ~10.60 MiB of GGUF metadata, so using the label sum understates
# the plan (9037.74 instead of 9048.34). NOT the file size on its own either —
# the engine's actual includes the KV cache, and comparing against the weights
# alone reports a +35% "overrun" that is only the cache (see
# `test_a_bare_file_size_would_report_the_kv_cache_as_a_divergence`).
GGUF_BYTES = 7206168928
PLAN_PREDICTION_MB = GGUF_BYTES / 2**20 + 2176.00      # 9048.34

# A log that never loaded anything: the panel must say UNKNOWN, never 0.
NO_BUFFERS = (
    "0.00.001 I srv    init: initializing slots, n_slots = 2\n"
    "0.00.002 I srv    main: server is listening on 127.0.0.1:11499\n"
)

# Synthetic, to pin the label classification for the other backends the engine
# names in the same position: CUDA0 / Metal / Vulkan0 are devices; a bare
# `CPU` and any `*_Host` are host RAM.
OTHER_BACKENDS = (
    "0.00.1 I load_tensors: offloaded 20/33 layers to GPU\n"
    "0.00.2 I load_tensors:        CUDA0 model buffer size =  1000.00 MiB\n"
    "0.00.3 I load_tensors:         Metal model buffer size =  2000.00 MiB\n"
    "0.00.4 I load_tensors:           CPU model buffer size =   500.00 MiB\n"
    "0.00.5 I llama_kv_cache:     Vulkan0 KV buffer size =   100.00 MiB\n"
    "0.00.6 I sched_reserve:    Vulkan0 compute buffer size =    50.00 MiB\n"
    "0.00.7 I sched_reserve: Vulkan_Host compute buffer size =    10.00 MiB\n"
    "0.00.8 I sched_reserve: graph splits = 1\n"
)

# SYNTHETIC and pathological: a dense model, every layer "offloaded", but the
# scheduler cut the graph into a split per attention layer because the fused
# attention node was not supported on the device and fell back to the CPU
# SILENTLY. That is roughly 2 extra splits per attention layer
# (GPU -> CPU -> GPU), so the count is far above the healthy baseline of 2.
# The engine prints no error for this; the count is the only evidence.
CPU_ATTENTION_FALLBACK = (
    "0.00.1 I print_info: n_expert              = 0\n"
    "0.00.2 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.00.3 I load_tensors:        ROCm0 model buffer size =  6539.67 MiB\n"
    "0.00.4 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB\n"
    "0.00.5 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB\n"
    "0.00.6 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.00.7 I sched_reserve: graph splits = 34\n"
)

# SYNTHETIC and LEGITIMATE: the owner's 35B MoE with expert weights kept on the
# CPU inside GPU layers (`--n-cpu-moe`). Many splits are expected here, and the
# log does not say how many expert layers were kept on the CPU, so the detector
# must refuse a verdict rather than cry wolf.
MOE_EXPERT_OFFLOAD = (
    "0.00.1 I print_info: n_expert              = 256\n"
    "0.00.2 I print_info: n_expert_used         = 8\n"
    "0.00.3 I load_tensors: offloaded 49/49 layers to GPU\n"
    "0.00.4 I load_tensors:        ROCm0 model buffer size = 18000.00 MiB\n"
    "0.00.5 I load_tensors:           CPU model buffer size =  2000.00 MiB\n"
    "0.00.6 I sched_reserve:      ROCm0 compute buffer size =   600.00 MiB\n"
    "0.00.7 I sched_reserve: graph splits = 98\n"
)

# SYNTHETIC and HEALTHY: the same dense 65/65 load split across TWO devices.
# The graph is the CPU embedding run plus one run per device — CPU -> ROCm0 ->
# ROCm1 = 3 splits — and that is the correct count, not a divergence. A17b's
# derived baseline knew `ngl` but not the device count, so it expected 2 and
# flagged this healthy load; the device count now comes from the load's own
# distinct device labels (`ROCm0` / `ROCm1`, `ROCm_Host` excluded).
TWO_DEVICE_DENSE = (
    "0.00.1 I print_info: n_expert              = 0\n"
    "0.00.2 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.00.3 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    "0.00.4 I load_tensors:        ROCm0 model buffer size =  3270.00 MiB\n"
    "0.00.5 I load_tensors:        ROCm1 model buffer size =  3269.67 MiB\n"
    "0.00.6 I llama_kv_cache:      ROCm0 KV buffer size =  1088.00 MiB\n"
    "0.00.7 I llama_kv_cache:      ROCm1 KV buffer size =  1088.00 MiB\n"
    "0.00.8 I sched_reserve:      ROCm0 compute buffer size =   205.14 MiB\n"
    "0.00.9 I sched_reserve:      ROCm1 compute buffer size =   205.14 MiB\n"
    "0.01.0 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.01.1 I sched_reserve: graph splits = 3\n"
)

# SYNTHETIC and pathological: the same two-device load, but the fused attention
# node fell back to the CPU per layer (GPU -> CPU -> GPU), so the count is far
# above the 3 the two-device placement expects.
TWO_DEVICE_CPU_ATTENTION_FALLBACK = TWO_DEVICE_DENSE.replace(
    "graph splits = 3", "graph splits = 35")

# SYNTHETIC and TRUNCATED: the log reports layers offloaded to a GPU but the
# buffer section names no device at all (only the host-mapped weight copy), so
# the number of devices — and therefore the number of GPU runs — cannot be
# derived. The honest answer is NOT COMPARABLE, not the 2 a single-device guess
# would produce (which would flag this load's `graph splits = 2`).
GPU_LAYERS_WITHOUT_DEVICE_LABEL = (
    "0.00.1 I print_info: n_expert              = 0\n"
    "0.00.2 I load_tensors: offloaded 65/65 layers to GPU\n"
    "0.00.3 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    "0.01.1 I sched_reserve: graph splits = 2\n"
)


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

def test_real_load_numbers_are_extracted():
    load = engine_log.parse_load(REAL_LOAD)

    assert load["found"] is True
    assert load["offloaded"] == "65/65"
    assert load["offloaded_layers"] == (65, 65)
    assert load["n_seq_max"] == 1
    assert load["graph_splits"] == 2
    assert load["n_expert"] == 0
    assert load["backend"] == "rocm"

    model = {b["label"]: b["mb"] for b in load["model_buffers"]}
    assert model["ROCm0"] == pytest.approx(6539.67)
    assert model["CPU_Mapped"] == pytest.approx(322.07)
    assert {b["label"]: b["mb"] for b in load["kv_buffers"]}["ROCm0"] == \
        pytest.approx(2176.00)
    assert {b["label"]: b["mb"] for b in load["rs_buffers"]}["ROCm0"] == \
        pytest.approx(149.62)
    compute = {b["label"]: b["mb"] for b in load["compute_buffers"]}
    assert compute["ROCm0"] == pytest.approx(410.28)
    assert compute["ROCm_Host"] == pytest.approx(84.28)


def test_two_loads_report_the_final_load_not_the_fitting_pass():
    loads = engine_log.parse_loads(FITTING_PASS + REAL_LOAD)

    assert len(loads) == 2
    # the fitting pass is first and its model buffer is zero ...
    assert loads[0]["model_buffers"][0]["mb"] == pytest.approx(0.00)
    # ... the reported load is the last one, with real weights
    last = engine_log.parse_load(FITTING_PASS + REAL_LOAD)
    assert last["model_buffers"][-1]["mb"] == pytest.approx(6539.67)


def test_a_load_with_no_buffer_lines_is_found_but_has_nothing_to_sum():
    # `offloaded` alone is not memory data: the parser must not invent a 0.
    load = engine_log.parse_load(
        "0.0 I load_tensors: offloaded 65/65 layers to GPU\n")

    assert load["found"] is True
    assert load["model_buffers"] == []


def test_a_log_with_no_load_at_all_is_not_found():
    load = engine_log.parse_load(NO_BUFFERS)

    assert load["found"] is False
    assert load["graph_splits"] is None
    assert load["n_seq_max"] is None


def test_other_backend_labels_are_classified_host_vs_device():
    load = engine_log.parse_load(OTHER_BACKENDS)

    model = {b["label"]: b for b in load["model_buffers"]}
    assert model["CUDA0"]["host"] is False
    assert model["Metal"]["host"] is False
    assert model["CPU"]["host"] is True
    assert {b["label"]: b for b in load["compute_buffers"]}["Vulkan_Host"][
        "host"] is True


def test_n_expert_attaches_to_the_load_after_it_and_does_not_leak():
    # The hparams dump precedes the marker, so the value is carried forward; it
    # must not survive into a later load that has no dump of its own.
    text = ("0.0 I print_info: n_expert = 4\n"
            + MOE_EXPERT_OFFLOAD.split("\n", 2)[2])
    loads = engine_log.parse_loads(text)

    assert loads[-1]["n_expert"] == 4
    # a tail with no hparams dump is UNKNOWN, never the previous model's count
    assert engine_log.parse_load(
        "0.0 I load_tensors: offloaded 65/65 layers to GPU\n"
        "0.1 I load_tensors: ROCm0 model buffer size = 10.00 MiB\n")["n_expert"] \
        is None


# ---------------------------------------------------------------------------
# The split expectation comes from the PLAN
# ---------------------------------------------------------------------------

def test_the_real_loads_plan_fields_come_from_its_own_log():
    fields = engine_log.plan_fields_from_load(engine_log.parse_load(REAL_LOAD))

    # `offloaded 65/65` counts the output layer, so the model has 65 - 1 = 64
    # layers and all 65 offload slots are on the GPU.
    assert fields == {"ngl": 65, "n_layers": 64, "n_cpu_moe": 0,
                      "backend": "rocm"}


def test_a_dense_load_with_any_gpu_layers_expects_two_splits_on_one_device():
    # all-GPU ...
    assert engine_log.expected_splits(ngl=65, n_layers=64, n_cpu_moe=0,
                                      n_devices=1) == 2
    # ... and a contiguous partial offload (first layers on the CPU) are the
    # same two runs: the CPU embedding is contiguous with the CPU layers.
    assert engine_log.expected_splits(ngl=40, n_layers=64, n_cpu_moe=0,
                                      n_devices=1) == 2


def test_a_dense_load_without_a_device_count_is_not_comparable():
    """A17e: the same rule as the MoE case. `ngl` alone does not determine the
    number of backend runs — one per device — so a plan that does not carry the
    device count gets NO verdict rather than the single-device 2."""
    assert engine_log.expected_splits(ngl=65, n_layers=64, n_cpu_moe=0) is None
    assert engine_log.expected_splits(ngl=65, n_layers=64, n_cpu_moe=0,
                                      n_devices=0) is None


def test_a_dense_load_across_two_devices_expects_one_run_per_device():
    # CPU embedding run + ROCm0 run + ROCm1 run = 3.
    assert engine_log.expected_splits(ngl=65, n_layers=64, n_cpu_moe=0,
                                      n_devices=2) == 3
    assert engine_log.expected_splits(ngl=65, n_layers=64, n_cpu_moe=0,
                                      n_devices=4) == 5


def test_a_cpu_only_plan_expects_one_split():
    assert engine_log.expected_splits(ngl=0, n_layers=64, n_cpu_moe=0) == 1
    assert engine_log.expected_splits(ngl=64, n_layers=64, n_cpu_moe=0,
                                      backend="cpu") == 1


def test_moe_expert_offload_has_no_derivable_split_count():
    # n_cpu_moe > 0: expert weights on the CPU inside GPU layers force a new
    # split per affected layer, so `ngl` does not determine the count.
    assert engine_log.expected_splits(ngl=49, n_layers=48, n_cpu_moe=12) is None
    # n_cpu_moe unknown (what the log gives for any MoE): also not comparable.
    assert engine_log.expected_splits(ngl=49, n_layers=48,
                                      n_cpu_moe=None) is None
    # no layer count: nothing to derive from.
    assert engine_log.expected_splits(ngl=65, n_layers=None,
                                      n_cpu_moe=0) is None


def test_device_labels_are_the_distinct_non_host_labels_in_the_load():
    """A17e: the device count comes from the load's own buffer labels. Host
    buffers (`CPU_Mapped`, `ROCm_Host`) are not devices, and a device label is
    counted once however many buffers it carries."""
    two = engine_log.parse_load(TWO_DEVICE_DENSE)
    assert engine_log.device_labels(two) == ["ROCm0", "ROCm1"]
    # the one-device real load: ROCm_Host is staging RAM, not a second device
    assert engine_log.device_labels(engine_log.parse_load(REAL_LOAD)) == ["ROCm0"]
    # the other families the engine names in the same position are devices too
    assert engine_log.device_labels(engine_log.parse_load(OTHER_BACKENDS)) == [
        "CUDA0", "Metal", "Vulkan0"]


def test_a_two_device_dense_load_expects_three_splits():
    load = engine_log.parse_load(TWO_DEVICE_DENSE)

    assert engine_log.expected_splits_for_load(load) == 3
    # the log's own placement is named in the sentence, not a single-device 2
    r = engine_log.compare_plan(load)
    assert r["expected_splits"] == 3
    assert r["split_verdict"] == "ok"
    assert r["unexpected_splits"] is False
    assert r["diverges"] is False


def test_a_gpu_load_whose_device_count_cannot_be_derived_is_not_comparable():
    """A17e: the load says layers went to a GPU but names no device, so the
    number of GPU runs is unknown. NOT COMPARABLE — never the single-device 2
    that would flag this (healthy-looking) `graph splits = 2`."""
    load = engine_log.parse_load(GPU_LAYERS_WITHOUT_DEVICE_LABEL)

    assert engine_log.device_labels(load) == []
    assert engine_log.expected_splits_for_load(load) is None

    r = engine_log.compare_plan(load)
    assert r["graph_splits"] == 2
    assert r["expected_splits"] is None
    assert r["split_verdict"] == "not_comparable"
    assert r["unexpected_splits"] is None
    assert r["diverges"] is False
    assert "NOT COMPARABLE" in r["detail"]
    assert "device" in r["detail"]


def test_an_overridden_expectation_is_described_consistently():
    """Verifier nit: `_split_sentence` used to recompute the parenthetical from
    the LOAD, so `expected_splits=1` on an all-GPU load read "ABOVE the 1 ...
    (an all-GPU dense load)" — a sentence that contradicts its own number. The
    words now follow the `expected_splits` argument."""
    load = engine_log.parse_load(REAL_LOAD)

    r = engine_log.compare_plan(load, expected_splits=1)

    assert r["split_verdict"] == "diverges"
    assert "the 1 the plan expects" in r["detail"]
    assert "all-GPU dense load" not in r["detail"]
    assert "CPU-only plan" in r["detail"]


def test_the_old_constant_baseline_of_one_would_flag_the_healthy_load():
    """The defect, pinned. `expected_splits = 1` (the old default) made the real
    fixture diverge; the derived baseline does not."""
    load = engine_log.parse_load(REAL_LOAD)

    old = engine_log.compare_plan(load, PLAN_PREDICTION_MB, expected_splits=1)
    now = engine_log.compare_plan(load, PLAN_PREDICTION_MB)

    assert old["unexpected_splits"] is True
    assert now["unexpected_splits"] is False


# ---------------------------------------------------------------------------
# Plan vs reality
# ---------------------------------------------------------------------------

def test_the_real_load_is_healthy():
    r = engine_log.compare_plan(engine_log.parse_load(REAL_LOAD),
                                PLAN_PREDICTION_MB)

    assert r["known"] is True
    assert r["actual_vram_mb"] == pytest.approx(9275.57)
    assert r["host_ram_mb"] == pytest.approx(406.35)
    # graph splits = 2 is the DENSE baseline, not a divergence
    assert r["graph_splits"] == 2
    assert r["expected_splits"] == 2
    assert r["split_verdict"] == "ok"
    assert r["unexpected_splits"] is False
    assert r["vram_verdict"] == "ok"
    assert r["diverges"] is False
    assert "diverges" not in r["detail"].lower()


def test_divergence_is_measured_against_device_vram_only():
    # Device total: 6539.67 + 2176.00 + 149.62 + 410.28 = 9275.57 MiB.
    # Host RAM (CPU_Mapped 322.07 + ROCm_Host 84.28) is NOT VRAM and is
    # excluded; including it would report 9681.92 and overstate the overrun.
    r = engine_log.compare_plan(engine_log.parse_load(REAL_LOAD),
                                PLAN_PREDICTION_MB)

    assert r["actual_vram_mb"] == pytest.approx(9275.57)
    assert r["host_ram_mb"] == pytest.approx(406.35)
    # against the plan's OWN prediction the gap is the compute buffer and the
    # RS buffer minus the host-mapped weights, ~2.5% — not the 32% a bare
    # budget produced.
    assert r["divergence_mb"] == pytest.approx(227.23, abs=0.01)
    assert r["divergence_pct"] == pytest.approx(2.5113, abs=1e-3)


def test_a_bare_file_size_would_report_the_kv_cache_as_a_divergence():
    """GUIDANCE entry 5, pinned as a test.

    The engine's 9275.57 MiB "actual" includes the KV cache at ctx 65536.
    Compared against the weight file alone (6872.34 MiB) that reads as a +35%
    divergence that is entirely the KV cache; against the plan's own prediction
    for the same ctx / cache / slots it is +2.5%.
    """
    load = engine_log.parse_load(REAL_LOAD)

    file_only = engine_log.compare_plan(load, GGUF_BYTES / 2**20)
    with_plan = engine_log.compare_plan(load, PLAN_PREDICTION_MB)

    assert file_only["divergence_pct"] == pytest.approx(34.970, abs=1e-2)
    assert file_only["vram_verdict"] == "diverges"
    assert with_plan["vram_verdict"] == "ok"


def test_without_a_plan_prediction_the_vram_axis_is_not_comparable():
    # The `/api/server/findings` surface has the log but not the plan. Saying
    # "not comparable" is the honest answer; inventing a number is not.
    r = engine_log.compare_plan(engine_log.parse_load(REAL_LOAD))

    assert r["vram_verdict"] == "not_comparable"
    assert r["divergence_mb"] is None
    assert r["divergence_pct"] is None
    assert "NOT COMPARABLE" in r["detail"]
    # ... and the split axis, which the log DOES determine, still has a verdict
    assert r["split_verdict"] == "ok"
    assert r["diverges"] is False


def test_a_cpu_attention_fallback_diverges():
    load = engine_log.parse_load(CPU_ATTENTION_FALLBACK)

    r = engine_log.compare_plan(load)

    assert r["graph_splits"] == 34
    assert r["expected_splits"] == 2
    assert r["split_verdict"] == "diverges"
    assert r["unexpected_splits"] is True
    assert r["diverges"] is True
    assert "CPU attention fallback" in r["detail"]


def test_a_two_device_cpu_attention_fallback_still_diverges():
    """A17e: widening the baseline to one run per device must NOT hide a genuine
    CPU attention fallback. On two devices the healthy count is 3; a per-layer
    GPU -> CPU -> GPU fallback pushes it far above that."""
    load = engine_log.parse_load(TWO_DEVICE_CPU_ATTENTION_FALLBACK)

    r = engine_log.compare_plan(load)

    assert r["graph_splits"] == 35
    assert r["expected_splits"] == 3
    assert r["split_verdict"] == "diverges"
    assert r["unexpected_splits"] is True
    assert r["diverges"] is True
    assert "CPU attention fallback" in r["detail"]


def test_moe_expert_offload_is_not_comparable_not_a_divergence():
    load = engine_log.parse_load(MOE_EXPERT_OFFLOAD)

    r = engine_log.compare_plan(load)

    assert r["graph_splits"] == 98
    assert r["expected_splits"] is None
    assert r["split_verdict"] == "not_comparable"
    assert r["unexpected_splits"] is None
    assert r["diverges"] is False
    assert "NOT COMPARABLE" in r["detail"]
    assert "n_expert = 256" in r["detail"]


def test_fewer_splits_than_expected_is_not_flagged():
    # The engine doing BETTER than the plan (e.g. the embedding on the device)
    # is not a defect, and a count below the baseline is not the fallback shape.
    load = engine_log.parse_load(
        REAL_LOAD.replace("graph splits = 2", "graph splits = 1"))

    r = engine_log.compare_plan(load)

    assert r["graph_splits"] == 1
    assert r["split_verdict"] == "ok"
    assert r["unexpected_splits"] is False


def test_no_memory_data_is_explicitly_unknown_never_zero():
    r = engine_log.compare_plan(engine_log.parse_load(NO_BUFFERS), 7000)

    assert r["known"] is False
    assert r["actual_vram_mb"] is None
    assert r["divergence_mb"] is None
    assert r["divergence_pct"] is None
    assert "unknown" in r["detail"].lower()
    # a silent 0 would read as "it fits"
    assert r["actual_vram_mb"] != 0
    # S2b: an unknown load has no split verdict either — `False` would read as
    # "no split problem" to a caller looking at this key alone.
    assert r["unexpected_splits"] is None
    assert r["split_verdict"] == "not_comparable"
    assert r["vram_verdict"] == "not_comparable"
    assert r["diverges"] is None


def test_a_found_load_with_no_buffer_lines_is_unknown_not_zero():
    """S2b: `compare_plan`'s OTHER unknown branch. `found` is True (the
    `offloaded` marker was there) but no buffer line followed, so there is
    nothing to sum — it must be as explicitly unknown as a log with no load at
    all, including `unexpected_splits`."""
    load = engine_log.parse_load(
        "0.0 I load_tensors: offloaded 65/65 layers to GPU\n")
    assert load["found"] is True            # the marker was there ...
    assert load["model_buffers"] == []      # ... and no buffer line followed

    r = engine_log.compare_plan(load, 7000)

    assert r["known"] is False
    assert r["actual_vram_mb"] is None
    assert r["divergence_mb"] is None
    assert r["unexpected_splits"] is None


def test_compare_plan_accepts_the_full_load_list():
    loads = engine_log.parse_loads(FITTING_PASS + REAL_LOAD)

    assert engine_log.compare_plan(loads, PLAN_PREDICTION_MB)[
        "actual_vram_mb"] == pytest.approx(9275.57)


# ---------------------------------------------------------------------------
# Wiring: /api/server/findings is `engine_log.findings`, so the load
# accounting must add a finding ONLY when it really diverges.
# ---------------------------------------------------------------------------

def test_a_healthy_load_yields_no_finding():
    """The whole point of the fix: the owner's real load is healthy and must
    produce NO finding on every launch."""
    assert engine_log.findings(REAL_LOAD) == []


def test_a_healthy_two_device_load_yields_no_finding():
    """A17e acceptance criterion: a healthy all-GPU dense load split across TWO
    devices (CPU embedding + ROCm0 + ROCm1 = 3 splits) must produce NO finding.
    Before the fix the baseline was 2 for any dense load, so this healthy load
    was flagged `diverges`."""
    assert engine_log.findings(TWO_DEVICE_DENSE) == []


def test_a_gpu_load_with_an_underivable_device_count_yields_no_finding():
    # "not comparable" is not a finding; the device count is unknown here.
    assert engine_log.findings(GPU_LAYERS_WITHOUT_DEVICE_LABEL) == []


def test_a_cpu_attention_fallback_load_yields_a_finding():
    got = engine_log.findings(CPU_ATTENTION_FALLBACK)

    assert [f["id"] for f in got] == ["plan_divergence"]
    assert got[0]["severity"] == "warn"
    assert got[0]["confirmed_here"] is False
    assert "CPU attention fallback" in got[0]["message"]
    assert "graph splits = 34" in got[0]["example"]


def test_a_two_device_cpu_attention_fallback_yields_a_finding():
    got = engine_log.findings(TWO_DEVICE_CPU_ATTENTION_FALLBACK)

    assert [f["id"] for f in got] == ["plan_divergence"]
    assert "graph splits = 35" in got[0]["example"]


def test_a_moe_expert_offload_load_yields_no_finding():
    # "not comparable" is not a finding; the owner's 35B MoE must not be flagged.
    assert engine_log.findings(MOE_EXPERT_OFFLOAD) == []


def test_the_real_fitting_pass_and_load_together_yield_no_finding():
    assert engine_log.findings(FITTING_PASS + REAL_LOAD) == []


# ---------------------------------------------------------------------------
# The existing five patterns must keep working
# ---------------------------------------------------------------------------

def test_existing_findings_are_unaffected_by_memory_lines():
    text = (REAL_LOAD + "0.05 W srv load_model: cache_reuse is not supported "
            "by this context, it will be disabled\n")

    got = engine_log.findings(text)

    assert [f["id"] for f in got] == ["cache_reuse_disabled"]


# ---------------------------------------------------------------------------
# A17e nit: the engine's DIRECT device statements are a second label source
# ---------------------------------------------------------------------------
#
# Buffer labels are not the only place the engine names a device. Per load it
# also prints, per layer (65 lines for the real load):
#   D load_tensors: layer   0 assigned to device ROCm0, is_swa = 0
# (prism-v.log:2510-2574) and once per model:
#   I llama_prepare_model_devices: using device ROCm0 (AMD Radeon RX 9070 XT) ...
# (prism-v.log:81). Both name a DEVICE directly, so they are a second source for
# the device count — and the only one when the buffer section is truncated.

_PREPARE_ROCm0 = (
    "0.00.264.167 I llama_prepare_model_devices: using device ROCm0 "
    "(AMD Radeon RX 9070 XT) (0000:04:00.0) - 16368 MiB\n")


def _assigned(dev, n, start=0):
    """`n` verbatim `assigned to device` lines, as the engine prints them."""
    return "".join(
        "0.00.521.%03d D load_tensors: layer %3d assigned to device %s, "
        "is_swa = 0\n" % (i, start + i, dev) for i in range(n))


DIRECT_DEVICE_LINES = _PREPARE_ROCm0 + _assigned("ROCm0", 65)

# The real load with its device buffer line CUT OFF — only the host-mapped
# weight copy survives — but the direct statements still name ROCm0.
TRUNCATED_BUFFERS_WITH_DIRECT_DEVICE = (
    DIRECT_DEVICE_LINES
    + "0.00.987.636 I print_info: n_expert              = 0\n"
    + "0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU\n"
    + "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    + "0.04.417.939 I sched_reserve: graph splits = 2\n")

# Two devices named ONLY by the direct statements (the buffer section is host
# RAM only), so the count cannot come from the buffer labels.
TWO_DEVICE_TRUNCATED_BUFFERS = (
    _PREPARE_ROCm0
    + "0.00.264.168 I llama_prepare_model_devices: using device ROCm1 "
      "(AMD Radeon RX 9070 XT) (0000:05:00.0) - 16368 MiB\n"
    + _assigned("ROCm0", 33, start=0)
    + _assigned("ROCm1", 32, start=33)
    + "0.00.987.636 I print_info: n_expert              = 0\n"
    + "0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU\n"
    + "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    + "0.04.417.939 I sched_reserve: graph splits = 3\n")

# Only the `using device` line survives: no buffer label and no per-layer line.
PREPARED_ONLY_TRUNCATED = (
    _PREPARE_ROCm0
    + "0.00.987.636 I print_info: n_expert              = 0\n"
    + "0.01.522.670 I load_tensors: offloaded 65/65 layers to GPU\n"
    + "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =   322.07 MiB\n"
    + "0.04.417.939 I sched_reserve: graph splits = 2\n")


def test_direct_device_statements_agree_with_the_buffer_labels():
    """The union of the two sources must not move the single-device reading."""
    with_direct = engine_log.parse_load(DIRECT_DEVICE_LINES + REAL_LOAD)
    without = engine_log.parse_load(REAL_LOAD)

    assert engine_log.device_labels(with_direct) == ["ROCm0"]
    assert engine_log.expected_splits_for_load(with_direct) == 2
    # byte-identical to the buffer-label-only reading: same expected count, same
    # sentence, same divergence
    assert engine_log.compare_plan(with_direct, PLAN_PREDICTION_MB) == \
        engine_log.compare_plan(without, PLAN_PREDICTION_MB)


def test_a_truncated_buffer_section_still_has_a_device_count():
    """The device buffer line was cut off, but the direct per-layer statements
    still name ROCm0, so the count is derivable — this reads HEALTHY instead of
    NOT COMPARABLE."""
    load = engine_log.parse_load(TRUNCATED_BUFFERS_WITH_DIRECT_DEVICE)

    assert engine_log.device_labels(load) == ["ROCm0"]
    assert engine_log.expected_splits_for_load(load) == 2

    r = engine_log.compare_plan(load)
    assert r["split_verdict"] == "ok"
    assert r["expected_splits"] == 2
    assert engine_log.findings(TRUNCATED_BUFFERS_WITH_DIRECT_DEVICE) == []


def test_two_devices_named_only_by_the_direct_statements_are_counted():
    load = engine_log.parse_load(TWO_DEVICE_TRUNCATED_BUFFERS)

    assert engine_log.device_labels(load) == ["ROCm0", "ROCm1"]
    assert engine_log.expected_splits_for_load(load) == 3
    assert engine_log.compare_plan(load)["split_verdict"] == "ok"


def test_using_device_is_a_fallback_when_no_precise_evidence_exists():
    load = engine_log.parse_load(PREPARED_ONLY_TRUNCATED)

    assert load["devices"] == []           # no per-layer line in this tail
    assert load["prepared_devices"] == ["ROCm0"]
    assert engine_log.device_labels(load) == ["ROCm0"]
    assert engine_log.expected_splits_for_load(load) == 2


def test_using_device_cannot_raise_the_count_above_the_layer_evidence():
    """`assigned to device` names the device that HOLDS a layer, so each label
    is provably a backend run; `using device` names the model's prepared device
    list, which can include a device that received no layers. The precise source
    wins, so a phantom prepared device cannot raise the expectation and MASK a
    real divergence."""
    text = (_PREPARE_ROCm0
            + "0.00.264.168 I llama_prepare_model_devices: using device ROCm1 "
              "(AMD Radeon RX 9070 XT) (0000:05:00.0) - 16368 MiB\n"
            + _assigned("ROCm0", 65)
            + REAL_LOAD)
    load = engine_log.parse_load(text)

    assert load["prepared_devices"] == ["ROCm0", "ROCm1"]
    assert engine_log.device_labels(load) == ["ROCm0"]
    assert engine_log.expected_splits_for_load(load) == 2


def test_mapped_and_pinned_labels_are_host_not_devices():
    """Verifier nit 3: `CUDA_Mapped` / `CUDA0_pinned` do not end `_Host`, so they
    used to be counted as devices — an over-count that raises the expectation
    and so can MASK a real divergence. No llama.cpp label takes those forms, but
    the classification is now closed for them."""
    text = ("0.00.1 I print_info: n_expert              = 0\n"
            "0.00.2 I load_tensors: offloaded 65/65 layers to GPU\n"
            "0.00.3 I load_tensors:    CUDA_Mapped model buffer size =   322.07 MiB\n"
            "0.00.4 I load_tensors:  CUDA0_pinned model buffer size =    10.00 MiB\n"
            "0.00.5 I load_tensors:        CUDA0 model buffer size =  6539.67 MiB\n"
            "0.01.1 I sched_reserve: graph splits = 2\n")
    load = engine_log.parse_load(text)

    assert engine_log.device_labels(load) == ["CUDA0"]
    assert engine_log.expected_splits_for_load(load) == 2


# ---------------------------------------------------------------------------
# DR2-1: the plan's prediction is the WHOLE FILE, so it is only a device-side
# figure when the engine put the whole file on the device.
#
# `memtruth.planned_mb`'s weight term is `plan.gguf.bytes / 2**20` — the whole
# GGUF file. `compare_plan`'s `actual_vram_mb` sums DEVICE buffers only. For a
# plan that does not place the whole file on the GPU — a dense spill
# (`ngl < n_layers`, a first-class plan shape) or MoE expert offload
# (`n_cpu_moe > 0`) — the whole-file prediction overstates the device figure by
# exactly the RAM-resident weight bytes, and above the slack that is a
# `plan_divergence` on EVERY launch of a healthy, intended configuration.
#
# Every `findings()` test above omits `expected_vram_mb`, so none of them ever
# reached this state. These do.
# ---------------------------------------------------------------------------

# A 64-layer dense model with 32 layers spilled (offloaded 33/65), `graph
# splits = 2` = healthy. Device model buffer 3281.20 + KV 2176.00 + RS 149.62 +
# compute 410.28 = 6017.10 MiB; against the plan's whole-file prediction
# 9037.70 MiB that is -3020.60 MiB / -33.4%, i.e. a `plan_divergence` before the
# fix. The host `CPU_Mapped` holds the 32 CPU-resident layers' weights.
SPILL_PREDICTION_MB = 9037.70
SPILLED_LOAD = (
    "0.00.987.636 I print_info: n_expert              = 0\n"
    "0.01.522.670 I load_tensors: offloaded 33/65 layers to GPU\n"
    "0.01.522.675 I load_tensors:   CPU_Mapped model buffer size =  3300.00 MiB\n"
    "0.01.522.676 I load_tensors:        ROCm0 model buffer size =  3281.20 MiB\n"
    "0.04.285.857 I llama_context: n_seq_max             = 1\n"
    "0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  2176.00 MiB\n"
    "0.04.369.729 I llama_memory_recurrent:      ROCm0 RS buffer size =   149.62 MiB\n"
    "0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =   410.28 MiB\n"
    "0.04.417.938 I sched_reserve:  ROCm_Host compute buffer size =    84.28 MiB\n"
    "0.04.417.939 I sched_reserve: graph splits = 2\n"
)


def test_a_spilled_load_is_not_a_device_side_prediction():
    load = engine_log.parse_load(SPILLED_LOAD)

    assert load["offloaded_layers"] == (33, 65)
    assert engine_log.weights_are_device_resident(load) is False

    r = engine_log.compare_plan(load, SPILL_PREDICTION_MB)

    assert r["actual_vram_mb"] == pytest.approx(6017.10)
    assert r["vram_verdict"] == "not_comparable", r
    assert r["vram_why"] == "not_device_resident"
    assert r["divergence_mb"] is None
    assert r["diverges"] is False
    assert "offloaded 33/65 layers to GPU" in r["detail"]


def test_a_healthy_spilled_load_yields_no_finding():
    """DR2-1 acceptance, direction one: the owner's intended spill must produce
    NO finding when the plan's own prediction is supplied. Before the fix this
    same log read `['plan_divergence']`."""
    assert engine_log.findings(SPILLED_LOAD,
                               expected_vram_mb=SPILL_PREDICTION_MB) == []


def test_a_moe_expert_offload_yields_no_finding_with_a_plan_prediction():
    """DR2-1 acceptance: the owner's 35B MoE shape. The load offloads every
    layer but keeps expert weights in a `CPU` model buffer, so the whole-file
    prediction is not a device-side figure either. Before the fix, supplying the
    prediction fired `plan_divergence` (device 18600 vs prediction 9048)."""
    load = engine_log.parse_load(MOE_EXPERT_OFFLOAD)

    assert engine_log.weights_are_device_resident(load) is False
    assert engine_log.findings(MOE_EXPERT_OFFLOAD,
                               expected_vram_mb=PLAN_PREDICTION_MB) == []


def test_a_genuinely_wrong_prediction_is_still_caught_on_a_resident_load():
    """DR2-1 acceptance, direction two: the gate keys on RESIDENCY, not on the
    sign of the divergence. The same low device figure as SPILLED_LOAD, but the
    engine reports EVERY layer offloaded, so the plan's whole-file prediction IS
    a device-side figure and a -33% gap is a real divergence."""
    resident = SPILLED_LOAD.replace("offloaded 33/65 layers to GPU",
                                    "offloaded 65/65 layers to GPU")
    load = engine_log.parse_load(resident)

    assert engine_log.weights_are_device_resident(load) is True

    r = engine_log.compare_plan(load, SPILL_PREDICTION_MB)

    assert r["actual_vram_mb"] == pytest.approx(6017.10)
    assert r["vram_verdict"] == "diverges"
    assert r["divergence_pct"] == pytest.approx(-33.42, abs=0.01)
    assert r["diverges"] is True
    assert engine_log.findings(
        resident, expected_vram_mb=SPILL_PREDICTION_MB
    )[0]["id"] == "plan_divergence"


def test_the_healthy_all_gpu_load_is_still_compared_and_still_healthy():
    """The gate must not suppress the case it was built for: the real all-GPU
    load has no `CPU` model buffer, so it stays comparable and stays `ok`."""
    load = engine_log.parse_load(REAL_LOAD)

    assert engine_log.weights_are_device_resident(load) is True

    r = engine_log.compare_plan(load, PLAN_PREDICTION_MB)

    assert r["vram_verdict"] == "ok"
    assert r["vram_why"] is None
    assert engine_log.findings(REAL_LOAD,
                               expected_vram_mb=PLAN_PREDICTION_MB) == []


def test_a_load_with_no_placement_line_is_not_a_device_side_prediction():
    """A truncated log with buffer lines but no `offloaded N/M` line cannot say
    whether the whole file is resident; "cannot tell" must not produce a verdict
    (and must not fire a divergence)."""
    load = engine_log.parse_load(
        "0.01.522.676 I load_tensors: ROCm0 model buffer size = 3281.20 MiB\n"
        "0.04.364.938 I llama_kv_cache: ROCm0 KV buffer size = 2176.00 MiB\n")

    assert engine_log.weights_are_device_resident(load) is False
    assert engine_log.compare_plan(load, SPILL_PREDICTION_MB)[
        "vram_verdict"] == "not_comparable"

