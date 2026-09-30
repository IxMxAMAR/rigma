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
"""
import pytest

from rigma import engine_log

# ---------------------------------------------------------------------------
# Fixtures: verbatim lines from .scratch/prism-v.log (whitespace preserved).
# ---------------------------------------------------------------------------

# The fitting pass: prism-v.log:2159-2340. Model buffers are ZERO because this
# pass only reserves; the real weights land in the second load.
FITTING_PASS = (
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

# The real load: prism-v.log:4485-4672.
REAL_LOAD = (
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


# ---------------------------------------------------------------------------
# Plan vs reality
# ---------------------------------------------------------------------------

def test_divergence_is_measured_against_device_vram_only():
    # Device total: 6539.67 + 2176.00 + 149.62 + 410.28 = 9275.57 MiB.
    # Host RAM (CPU_Mapped 322.07 + ROCm_Host 84.28) is NOT VRAM and is
    # excluded; including it would report 9681.92 and overstate the overrun.
    r = engine_log.compare_plan(engine_log.parse_load(REAL_LOAD), 7000)

    assert r["known"] is True
    assert r["actual_vram_mb"] == pytest.approx(9275.57)
    assert r["host_ram_mb"] == pytest.approx(406.35)
    assert r["divergence_mb"] == pytest.approx(2275.57)
    assert r["divergence_pct"] == pytest.approx(32.5081, abs=1e-3)
    # more than one graph split means some ops were assigned to a backend the
    # plan did not assume; the plan's charge assumes a single split.
    assert r["unexpected_splits"] is True


def test_single_graph_split_is_not_unexpected():
    load = engine_log.parse_load(
        REAL_LOAD.replace("graph splits = 2", "graph splits = 1"))

    r = engine_log.compare_plan(load, 7000)

    assert r["graph_splits"] == 1
    assert r["unexpected_splits"] is False


def test_expected_splits_argument_raises_the_bar():
    load = engine_log.parse_load(REAL_LOAD)

    assert engine_log.compare_plan(load, 7000, expected_splits=2)[
        "unexpected_splits"] is False


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

    assert engine_log.compare_plan(loads, 7000)["actual_vram_mb"] == \
        pytest.approx(9275.57)


# ---------------------------------------------------------------------------
# The existing five patterns must keep working
# ---------------------------------------------------------------------------

def test_existing_findings_are_unaffected_by_memory_lines():
    text = (REAL_LOAD + "0.05 W srv load_model: cache_reuse is not supported "
            "by this context, it will be disabled\n")

    got = engine_log.findings(text)

    assert [f["id"] for f in got] == ["cache_reuse_disabled"]
