"""R3-MEM-1: verify the plan against the engine's own measurement.

The strings below are VERBATIM from the pinned build (b9867, rocm-mainline) on
the owner's machine, captured by running `llama-fit-params` directly. They are
fixtures rather than live calls because the parser is the fragile part and it
must be pinned to real output — but `test_live_oracle_*` at the bottom runs the
real binary, so a fixture that drifts from reality is caught too.
"""
from __future__ import annotations

import subprocess

import pytest

from rigma import memtruth

# `llama-fit-params -m SmolLM2-135M-Instruct-Q2_K.gguf -c 32768 -ngl 99 --parallel 2 -fa on -lv 4`
SMOL_C32768 = """
0.00.186.414 I common_memory_breakdown_print: | memory breakdown [MiB] | total    free    self   model   context   compute    unaccounted |
0.00.186.419 I common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + ( 899 =    82 +     720 +      97) +        -735 |
0.00.186.420 I common_memory_breakdown_print: |   - Host               |                    46 =    28 +       0 +      18                |
0.00.199.677 I common_params_fit_impl: projected to use 899 MiB of device memory vs. 16140 MiB of free device memory
0.00.199.680 I common_params_fit_impl: will leave 15240 >= 1024 MiB of free device memory, no changes needed
0.00.199.683 I common_fit_params: successfully fit params to free device memory
"""
SMOL_STDOUT = "-c 32768 -ngl 99"

# `-c 0` (let the model decide) — note the smaller context term.
SMOL_C0 = """
0.00.204.418 I common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + ( 359 =    82 +     180 +      97) +        -195 |
0.00.218.438 I common_params_fit_impl: projected to use 359 MiB of device memory vs. 16140 MiB of free device memory
0.00.218.442 I common_params_fit_impl: will leave 15780 >= 1024 MiB of free device memory, no changes needed
"""

# `-c 2000000` — the oracle printed the breakdown and then ABORTED, with no fit
# line and no stdout. This is the case that must never read as agreement.
IMPOSSIBLE = """
0.00.218.242 I common_memory_breakdown_print: | memory breakdown [MiB] | total    free     self   model   context   compute    unaccounted |
0.00.218.246 I common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + (45996 =    82 +   43948 +    1965) +      -45832 |
0.00.218.246 I common_memory_breakdown_print: |   - Host               |                   1984 =    28 +       0 +    1955                |
0.00.236.174
"""

# A refusal: the target could not be met.
REFUSED = """
0.00.100.000 I common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) | 16304 = 16140 + (9000 =  6000 +    2500 +     500) +       -3536 |
0.00.110.000 I common_params_fit_impl: cannot meet free memory target of 3296 MiB
"""


# --- the parser, against real output ---------------------------------------

def test_parses_the_real_device_line():
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    assert r.ok
    d = r.primary
    assert d.device == "ROCm0 (RX 9070 XT)"
    assert (d.total, d.free) == (16304, 16140)
    assert (d.model, d.context, d.compute) == (82, 720, 97)
    assert d.unaccounted == -735
    assert d.self_mb == 899          # 82 + 720 + 97, the number Rigma estimates
    assert r.projected_mb == 899
    assert r.leaves_mb == 15240
    assert r.target_mb == 1024
    assert r.fitted_args == "-c 32768 -ngl 99"


def test_the_host_line_is_not_mistaken_for_a_device():
    """`Host` has the same leading columns and no self/model/context/compute
    group. Requiring the group is what excludes it — without that, Rigma would
    read the host's numbers as GPU memory."""
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    assert len(r.devices) == 1
    assert "Host" not in r.primary.device


def test_negative_unaccounted_is_parsed():
    """It was -735 on a real run and -45832 on an impossible one; a parser that
    only accepted digits would drop every overcommitted line — i.e. exactly the
    lines worth seeing."""
    assert memtruth.parse_fit_output(IMPOSSIBLE).primary.unaccounted == -45832


def test_the_aborted_oversized_request_is_not_reported_as_fitting():
    """THE case this module exists for. The oracle died without a fit line; the
    breakdown it printed first is evidence of overcommitment, not of success.

    The reason now names the overcommit directly rather than the missing fit
    line, because the arithmetic check outranks it and says more.
    """
    r = memtruth.parse_fit_output(IMPOSSIBLE)
    assert r.ok is False
    assert r.primary is not None            # the evidence survived
    assert r.primary.self_mb == 45995       # 82 + 43948 + 1965, vs 16140 free
    assert r.primary.self_mb > r.primary.free
    assert "does not fit" in r.reason


def test_a_refusal_is_reported_as_a_refusal_with_its_target():
    r = memtruth.parse_fit_output(REFUSED)
    assert r.ok is False
    assert r.target_mb == 3296
    assert "cannot meet" in r.reason
    # and the measurement is still there to report
    assert r.primary.self_mb == 9000


def test_empty_output_degrades_to_no_evidence_not_to_agreement():
    r = memtruth.parse_fit_output("")
    assert r.ok is False
    assert r.devices == []
    assert "no memory breakdown" in r.reason


def test_an_unrecognised_format_degrades_rather_than_raising():
    """A llama.cpp upgrade that renames these strings must not crash Rigma, and
    must not read as a pass. Silence is not agreement."""
    r = memtruth.parse_fit_output(
        "0.00.1 I common_memory_breakdown_print: | something entirely new |\n")
    assert r.ok is False and r.devices == []
    assert r.reason


# A model the pinned engine cannot load at all. VERBATIM from the owner's
# machine: Ternary-Bonsai-2-27B-PQ2_0 is ggml type 142 and the pin is b9867,
# which accepts [0, 42). Rigma's resolver happily planned this model — this is
# the oracle catching a plan its own engine cannot serve.
CANNOT_LOAD = """
0.00.085.166 E gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142. should be in [0, 42)
0.00.085.176 E gguf_init_from_reader: failed to read tensor info
0.00.089.939 E llama_model_load: error loading model: llama_model_loader: failed to load model from C:\\models\\Ternary-Bonsai-2-27B-PQ2_0.gguf
0.00.090.584 E common_fit_params: encountered an error while trying to fit params to free device memory: failed to load model
"""


def test_an_unloadable_model_is_reported_as_such_not_as_a_parse_problem():
    """The most actionable answer this module gives. Reporting it as "printed no
    memory breakdown" would blame the parser for a real incompatibility."""
    r = memtruth.parse_fit_output(CANNOT_LOAD)
    assert r.ok is False
    assert r.devices == []
    assert "cannot load this model" in r.reason
    assert "invalid ggml type 142" in r.reason, r.reason
    assert r.load_error


def test_a_load_error_outranks_a_present_breakdown():
    """A build can print a breakdown and still fail the load; the load failure is
    the thing the user needs, so it wins."""
    r = memtruth.parse_fit_output(CANNOT_LOAD + SMOL_C32768, SMOL_STDOUT)
    assert r.ok is False
    assert "cannot load this model" in r.reason


def test_a_claimed_fit_that_exceeds_free_memory_is_refused():
    """R3-MEM-1, the Windows trap, and the reason this check is arithmetic rather
    than a trust in the engine's verdict.

    On Windows (WDDM, and NVIDIA's Sysmem Fallback Policy from driver 536.40) an
    allocation LARGER than dedicated VRAM SUCCEEDS out of system RAM: the model
    runs over PCIe at a fraction of the speed and llama.cpp prints NO error. So
    "it started" is not evidence that it fits, and a verdict that rested on the
    allocation succeeding would be wrong in the silent direction.

    Here the oracle reports success while holding more than is free. Rigma must
    refuse anyway.
    """
    claimed_ok_but_over = (
        "0.00.1 I common_memory_breakdown_print: |   - ROCm0 (RX 9070 XT) "
        "| 16304 = 4000 + (9000 =  6000 +    2500 +     500) +       -3536 |\n"
        "0.00.2 I common_params_fit_impl: projected to use 9000 MiB of device "
        "memory vs. 4000 MiB of free device memory\n"
        "0.00.3 I common_params_fit_impl: will leave 1000 >= 1024 MiB of free "
        "device memory, no changes needed\n")
    r = memtruth.parse_fit_output(claimed_ok_but_over)
    assert r.primary.self_mb == 9000
    assert r.primary.free == 4000
    assert r.ok is False, "a plan holding more than is free was reported as fitting"
    assert "does not fit" in r.reason
    assert "system RAM" in r.reason      # says WHY it is refused, not just that


def test_a_plan_within_free_memory_is_still_a_fit():
    """The control: the overcommit guard must not refuse everything."""
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    assert r.primary.self_mb == 899 < r.primary.free == 16140
    assert r.ok is True


# --- the structured JSONL path ---------------------------------------------

JSONL = (
    '{"type":"fit_memory_breakdown","data":{"unit":"MiB","rows":['
    '{"kind":"device","name":"CUDA0","description":"RTX 4090",'
    '"total":24564,"free":23000,"self":1500,"model":1200,'
    '"context":200,"compute":100,"unaccounted":-64},'
    '{"kind":"host","name":"Host","self":46}]}}\n')


def test_the_structured_breakdown_is_preferred_when_present():
    """Newer builds emit JSONL; it is strictly better than scraping a table."""
    r = memtruth.parse_any("", JSONL)
    assert r.devices and r.primary.device == "CUDA0"
    assert (r.primary.model, r.primary.context, r.primary.compute) == (1200, 200, 100)
    assert r.primary.self_mb == 1500


def test_the_jsonl_host_row_is_not_a_device():
    r = memtruth.parse_any("", JSONL)
    assert len(r.devices) == 1


def test_a_build_without_jsonl_falls_through_to_the_table():
    """b9867 has no `--log-jsonl` (verified: it answers `invalid argument`), so
    the table path is the one that must keep working."""
    r = memtruth.parse_any(SMOL_C32768, SMOL_STDOUT)
    assert r.primary.device == "ROCm0 (RX 9070 XT)"
    assert r.primary.self_mb == 899


def test_malformed_jsonl_does_not_break_the_table_fallback():
    r = memtruth.parse_any(SMOL_C32768, '{"type":"fit_memory_breakdown","data":{')
    assert r.primary is not None and r.primary.self_mb == 899


def test_jsonl_without_devices_falls_through():
    r = memtruth.parse_any(SMOL_C32768, '{"type":"something_else"}\n')
    assert r.primary is not None and r.primary.self_mb == 899


# --- comparison ------------------------------------------------------------

def test_comparison_is_silent_when_the_estimate_is_close():
    """Rigma's figure and the engine's are not the same quantity — Rigma's is
    weights+KV against a reserved budget, the engine's is a real allocation
    against real free memory — so a small gap is expected."""
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    assert memtruth.compare(899.0, r) is None
    assert memtruth.compare(800.0, r) is None      # within the slack


def test_comparison_names_both_numbers_when_they_disagree():
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    msg = memtruth.compare(300.0, r)
    assert msg is not None
    assert "300" in msg and "899" in msg, msg


def test_comparison_reports_the_engine_measuring_LESS_too():
    """Not only overcommitment: a plan that wildly over-reserves wastes the card
    and should be visible."""
    r = memtruth.parse_fit_output(SMOL_C32768, SMOL_STDOUT)
    msg = memtruth.compare(4000.0, r)
    assert msg is not None and "less" in msg, msg


def test_comparison_says_nothing_without_a_measurement():
    assert memtruth.compare(1000.0, memtruth.parse_fit_output("")) is None


# --- the plan-side helpers --------------------------------------------------

def _a_plan(slug="qwen3-0.6b", ctx=8192, k="f16", v="f16"):
    """A real plan off the BUNDLED registry, so the spec is the one that ships.

    The bundled registry has only a few models (the full one is downloaded), so
    this uses the smallest rather than a model from a developer's install.
    """
    from rigma.registry import Registry
    spec = Registry.load().models[slug]
    gguf = spec.ggufs[0]
    from rigma.models import ComboFlags
    return type("P", (), {
        "model_slug": slug, "gguf": gguf,
        "flags": ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=v),
    })()


def _expected_mb(plan):
    """Recompute weights+KV independently of the function under test."""
    from rigma.resolve import kv_bytes_per_token, swa_kv_bytes
    from rigma.registry import Registry
    spec = Registry.load().models[plan.model_slug]
    f = plan.flags
    kv = (f.ctx * kv_bytes_per_token(spec, f.cache_type_k, f.cache_type_v)
          + swa_kv_bytes(spec, f.cache_type_k, f.cache_type_v, f.ctx)) / 2**20
    return plan.gguf.bytes / 2**20 + kv


def test_planned_mb_is_weights_plus_kv_and_never_silently_zero():
    """An earlier version read `plan.spec` behind a `hasattr`, so it returned 0.0
    for every real plan and compared nothing — a silent zero makes every plan look
    catastrophically under-budgeted."""
    plan = _a_plan()
    mb = memtruth.planned_mb(plan)
    assert mb > 0, "planned_mb returned a silent zero"
    assert mb == pytest.approx(_expected_mb(plan))


def test_planned_mb_raises_rather_than_returning_a_wrong_number():
    plan = _a_plan()
    plan.model_slug = "no-such-model-in-the-registry"
    with pytest.raises(ValueError):
        memtruth.planned_mb(plan)


def test_planned_mb_scales_with_context():
    """KV is linear in ctx, so doubling ctx must add KV, not nothing."""
    a, b = memtruth.planned_mb(_a_plan(ctx=8192)), memtruth.planned_mb(_a_plan(ctx=16384))
    assert b > a
    assert b - a == pytest.approx(_expected_mb(_a_plan(ctx=16384))
                                  - _expected_mb(_a_plan(ctx=8192)))


def test_fit_argv_carries_the_flags_being_verified():
    """Built from the plan's own flags, so a flag added to the launch path is
    verified automatically rather than silently omitted."""
    argv = memtruth.fit_argv(_a_plan(), "m.gguf")
    assert argv[:2] == ["-m", "m.gguf"]
    for pair in (["-c", "8192"], ["-ngl", "99"], ["-fa", "on"],
                 ["--cache-type-k", "f16"], ["--cache-type-v", "f16"],
                 ["--parallel", "2"], ["-lv", "4"]):
        assert pair == argv[argv.index(pair[0]):argv.index(pair[0]) + 2], pair


def test_fit_argv_matches_what_the_engine_is_actually_launched_with():
    """The oracle reports on what it is GIVEN, so verifying a different argument
    set than the launch uses would verify a plan nobody runs. `--parallel 2` is
    the case that matters: it is what Rigma launches with."""
    argv = memtruth.fit_argv(_a_plan(), "m.gguf")
    assert "--parallel" in argv and argv[argv.index("--parallel") + 1] == "2"


def test_verify_plan_reports_a_disagreement_without_refusing(tmp_path):
    """`verify_plan` never refuses anything itself — whether a modelling
    difference should block a launch is policy that belongs with the launch."""
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    (tmp_path / "llama-fit-params.exe").write_bytes(b"x")

    def fake(*a, **k):
        # the engine says 8179 MiB where Rigma's plan is far smaller
        return subprocess.CompletedProcess(a[0], 0, SMOL_STDOUT,
                                           "0.00.1 I common_memory_breakdown_print: "
                                           "|   - ROCm0 (RX 9070 XT) | 16304 = 16140 "
                                           "+ (8179 =    82 +    8000 +      97) +      -4015 |\n"
                                           "0.00.2 I common_params_fit_impl: projected to use "
                                           "8179 MiB of device memory vs. 16140 MiB of free "
                                           "device memory\n"
                                           "0.00.3 I common_params_fit_impl: will leave "
                                           "7961 >= 1024 MiB of free device memory, no "
                                           "changes needed\n")

    res, disagree = memtruth.verify_plan(
        _a_plan(), "m.gguf", tmp_path / "llama-server.exe", popen=fake)
    assert res.primary is not None and res.primary.self_mb == 8179
    assert disagree is not None and "8179" in disagree
    assert res.ok, "a plan that fits must still be reported as fitting"


# --- binary resolution -----------------------------------------------------

def test_the_oracle_is_found_beside_the_server(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    (tmp_path / "llama-fit-params.exe").write_bytes(b"x")
    got = memtruth.fit_params_bin(tmp_path / "llama-server.exe")
    assert got is not None and got.name == "llama-fit-params.exe"


def test_a_build_without_the_oracle_returns_none_not_a_guess(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    assert memtruth.fit_params_bin(tmp_path / "llama-server.exe") is None


def test_a_missing_oracle_is_reported_as_no_evidence(tmp_path):
    """`ok=False` and a reason — never as "fits"."""
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    r = memtruth.run_fit(tmp_path / "llama-server.exe", ["-m", "x"])
    assert r.ok is False
    assert "no llama-fit-params" in r.reason
    assert r.devices == []


def test_a_failed_launch_is_reported_not_raised(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    (tmp_path / "llama-fit-params.exe").write_bytes(b"x")

    def boom(*a, **k):
        raise OSError("missing dll")

    r = memtruth.run_fit(tmp_path / "llama-server.exe", ["-m", "x"], popen=boom)
    assert r.ok is False and "could not run" in r.reason


def test_nonzero_exit_keeps_the_evidence(tmp_path):
    """The oracle aborts on an impossible request AFTER printing the breakdown.
    Treating a non-zero exit as a plain failure would throw away the one piece
    of evidence that explains why."""
    (tmp_path / "llama-server.exe").write_bytes(b"x")
    (tmp_path / "llama-fit-params.exe").write_bytes(b"x")

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 1, "", IMPOSSIBLE)

    r = memtruth.run_fit(tmp_path / "llama-server.exe", ["-m", "x"], popen=fake)
    assert r.ok is False
    assert r.primary is not None and r.primary.self_mb == 45995


# --- live: the real binary --------------------------------------------------
#
# The suite's own `_never_touch_the_real_rigma_home` fixture (session-scoped,
# autouse) redirects RIGMA_HOME to a throwaway directory so no test can write to
# the user's install — correct, and it is why the live check below has to reach
# the REAL home explicitly rather than through `runtime.rigma_home()`. Captured
# at import, before any fixture runs, because importing happens first.
_REAL_HOME = __import__("os").environ.get("RIGMA_HOME")


def _real_home():
    from pathlib import Path
    if _REAL_HOME:
        return Path(_REAL_HOME)
    return Path.home() / ".rigma"


def _pinned_server():
    from rigma import engines, runtime
    man = runtime._engines_manifest()
    name = ("llama-server.exe" if engines._os_name() == "windows"
            else "llama-server")
    for backend in ("rocm-mainline", "rocm", "vulkan", "cuda", "cpu"):
        p = _real_home() / "engines" / man["version"] / backend / name
        if p.exists():
            return p
    return None


def _a_loadable_model():
    """Smallest real model in the install. mmproj files are skipped: they are
    vision projectors, not models, and the oracle cannot load one alone."""
    root = _real_home() / "models"
    if not root.exists():
        return None
    for p in sorted(root.rglob("*.gguf"), key=lambda x: x.stat().st_size):
        if "mmproj" in p.name.lower():
            continue
        return p
    return None


@pytest.mark.hardware
def test_live_oracle_measures_a_real_model():
    """Not a fixture: the real pinned binary, the real device.

    Skipped rather than failed when the engine or a model is absent, because a
    missing download is not a code defect — but when both are present this
    asserts the parser still matches the build, which is the only thing that
    catches a llama.cpp upgrade renaming these strings.
    """
    srv = _pinned_server()
    model = _a_loadable_model()
    if srv is None or model is None:
        pytest.skip(f"no pinned engine or no local model to measure "
                    f"(srv={srv}, model={model})")
    r = memtruth.run_fit(srv, ["-m", str(model), "-c", "8192", "-ngl", "99",
                               "-fa", "on", "-lv", "4"])
    assert r.primary is not None, r.reason
    d = r.primary
    assert d.total > 0 and d.free > 0
    assert d.self_mb == d.model + d.context + d.compute
    assert d.self_mb > 0, "a loaded model cannot occupy zero bytes"
    assert r.ok, r.reason
