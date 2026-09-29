"""A launch never pages a cache the fit rejected.

The owner's model pins `q8_0` KV and the resolver's ladder was being defeated
two ways at once:

  * `cache_type_policy.pinned` (documented as the Models-page EXPLORER's knob,
    "never set on a stored spec") made `_cache_candidates` return only q8_0;
  * `launch.kv` was applied to the plan AFTER the fit, so even the ladder's
    fully-resident q5_1 was overwritten by q8_0.

On a 16GB card at ctx 262144 that is 15,576MB against a 14,954MB budget:
Windows/WDDM accepts the allocation, pages 622MB to system RAM and prints no
error, and decode drops from 54.5 t/s to ~12.6 (findings-r3/36). These tests
pin the corrected behaviour: the requested type is a CEILING, fitted in, and a
step-down is reported rather than silent.

Geometry is the owner's real model (Ternary-Bonsai-2-27B, qwen35 hybrid).
"""
import json
import os

import pytest

from rigma import server_ops
from rigma import state as st
from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, LaunchDefaults, ModelSpec)
from rigma.registry import Registry
from rigma.resolve import (_budgets, fit_for_launch, fit_gguf,
                           kv_bytes_per_token, quant_verdicts)

SLUG = "ternary-bonsai-2-27b-uncensored-heretic-pq2-0"
FILE_BYTES = 7206168928


def _profile(vram_mb: int = 16304) -> HardwareProfile:
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=vram_mb,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=30000,
                           cpu=CpuInfo(cores=10), name="", os="windows",
                           disk_free_gb=500.0, vram_used_mb=None)


def _spec(*, pinned: bool = True, launch: LaunchDefaults | None = None):
    return ModelSpec(
        slug=SLUG, family="qwen35", kind="dense", n_layers=64,
        full_attn_layers=16, kv_heads=4, head_dim=256, native_ctx=262144,
        custom=True,
        ggufs=[GgufFile(repo="local", file="b.gguf", bytes=FILE_BYTES,
                        quant="Q2_0")],
        use_cases=["general"],
        cache_type_policy=CachePolicy(k="q8_0", v="q8_0", pinned=pinned),
        launch=launch)


def _model_plus_kv_mb(spec, flags) -> float:
    return (spec.ggufs[0].bytes / 2**20
            + flags.ctx * kv_bytes_per_token(spec, flags.cache_type_k,
                                             flags.cache_type_v) / 2**20)


# --- the fit itself -----------------------------------------------------------

def test_a_pinned_policy_that_spills_names_the_rung_that_fits():
    """The plan is allowed to spill — that is the question the explorer asked —
    but it must say which rung the pin removed, and how many layers that is."""
    spec, prof = _spec(pinned=True), _profile()
    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], prof, 262144, explain)
    assert flags.cache_type_k == "q8_0"
    assert flags.ngl == 58                     # 7 of 64 transformer layers on CPU
    line = next(e for e in explain if "pinned" in e)
    assert "7 of 64 layers" in line            # ngl counts the output layer
    assert "q5_1 fits fully on the GPU" in line


def test_an_unpinned_spec_does_not_get_the_warning():
    spec, prof = _spec(pinned=False), _profile()
    explain: list[str] = []
    fit_gguf(spec, spec.ggufs[0], prof, 262144, explain)
    assert not any("pinned" in e for e in explain)


def test_quant_verdicts_carries_the_same_fact():
    """The Models page reads quant_verdicts, not the plan's explain."""
    spec, prof = _spec(pinned=True), _profile()
    v = quant_verdicts(spec, prof, kv="q8_0", vision=False,
                       grow="context")[0]
    assert v["ctx"] == 262144 and v["kv"] == "q8_0"
    assert "q5_1 fits fully on the GPU" in v.get("note", "")
    assert "7 of 64 layers" in v["note"]


# --- the launch fit (the shared ceiling rule) ---------------------------------

def test_the_launch_fit_steps_down_instead_of_paging():
    spec, prof = _spec(pinned=True), _profile()
    flags, stepped = fit_for_launch(spec, spec.ggufs[0], prof, 262144,
                                    kv="q8_0", vision=False)
    assert stepped == "q8_0"
    assert flags.cache_type_k == "q5_1" and flags.cache_type_v == "q5_1"
    assert flags.ngl == 99                     # nothing on the CPU
    budget, _ = _budgets(prof)
    assert _model_plus_kv_mb(spec, flags) <= budget


def test_a_stored_pin_does_not_defeat_the_launch_ladder():
    """`pinned` is the explorer's knob; a stored spec that carries it anyway is
    read as "no opinion" on the launch path, or the ladder is dead."""
    spec, prof = _spec(pinned=True), _profile()
    flags, stepped = fit_for_launch(spec, spec.ggufs[0], prof, 262144,
                                    kv="", vision=False)
    assert stepped == ""
    assert flags.cache_type_k == "q5_1" and flags.ngl == 99


def test_a_requested_cache_that_fits_is_kept():
    spec, prof = _spec(pinned=True), _profile()
    flags, stepped = fit_for_launch(spec, spec.ggufs[0], prof, 65536,
                                    kv="q8_0", vision=False)
    assert stepped == "" and flags.cache_type_k == "q8_0"


def test_the_launch_fit_never_exceeds_the_budget_at_any_rung():
    spec, prof = _spec(pinned=True), _profile()
    budget, _ = _budgets(prof)
    for ctx in (65536, 131072, 196608, 245760, 262144):
        flags, _ = fit_for_launch(spec, spec.ggufs[0], prof, ctx, kv="q8_0",
                                  vision=False)
        assert flags is not None, ctx
        assert _model_plus_kv_mb(spec, flags) <= budget, ctx


# --- the launch path, end to end ----------------------------------------------

@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = _spec(pinned=True,
                 launch=LaunchDefaults(ctx=262144, kv="q8_0", vision=False))
    (tmp_path / "models").mkdir(parents=True)
    (tmp_path / "models" / "b.gguf").write_bytes(b"x")
    reg = Registry([], {SLUG: spec}, {})
    # a DIFFERENT model is running, so the same-model guard does not fire
    st.write_state("other", "Q0", 18500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    seen = {}

    class _SP:
        proc = type("P", (), {"pid": 4242})()

    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda b, o: "llama-server.exe")

    def fake_launch(exe, rp, path, port, extra_args=None):
        seen["plan"] = rp
        return _SP()

    monkeypatch.setattr("rigma.runtime.launch_server", fake_launch)
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    monkeypatch.setattr(server_ops, "_await_port_free", lambda *a, **k: True)
    return reg, seen


def test_perform_switch_yields_the_all_gpu_plan_and_a_notice(env):
    """The owner's exact scenario: stored ctx 262144 + kv q8_0 on a 16GB card.
    It must come up on q5_1 with every layer on the GPU, and say why."""
    reg, seen = env
    out = server_ops.perform_switch(SLUG, reg, _profile())
    f = seen["plan"].flags
    assert f.ctx == 262144
    assert (f.cache_type_k, f.cache_type_v) == ("q5_1", "q5_1")
    assert f.ngl == 99
    budget, _ = _budgets(_profile())
    assert _model_plus_kv_mb(_spec(), f) <= budget
    assert "q5_1" in out.get("notice", "")
    assert out["kv_cache"] == "q5_1"


def test_perform_switch_keeps_a_cache_that_fits(env):
    reg, seen = env
    out = server_ops.perform_switch(SLUG, reg, _profile(), ctx=8192,
                                    kv="q8_0")
    assert seen["plan"].flags.cache_type_k == "q8_0"
    assert "notice" not in out


def test_perform_switch_still_reaches_the_launch_flags(env):
    """Regression guard for test_server_ops_ctx: an explicit kv must still land
    on the plan when it fits."""
    reg, seen = env
    server_ops.perform_switch(SLUG, reg, _profile(), ctx=16384, kv="q4_0")
    assert seen["plan"].flags.cache_type_k == "q4_0"


def test_the_stored_spec_is_not_mutated(env):
    """fit_for_launch works on copies; the registry object is process-wide."""
    reg, _ = env
    spec = reg.models[SLUG]
    before = json.dumps(spec.model_dump(), sort_keys=True)
    server_ops.perform_switch(SLUG, reg, _profile())
    assert json.dumps(reg.models[SLUG].model_dump(), sort_keys=True) == before
