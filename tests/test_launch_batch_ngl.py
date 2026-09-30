"""C10: `-b`/`-ub`/`-ngl` are settable per model, and the fit still rules.

Tier C's premise correction (BACKLOG.md): Rigma could already EMIT `-b`, `-ub`
and `-ngl` (`ComboFlags.batch`/`ubatch`/`ngl`, `RunPlan.server_args`), and the
exhaustive sweep set batch 16384 / ubatch 2048 — but no caller could ASK for
them, so the owner could not launch a model at a chosen batch/ubatch/ngl. This
adds the lever; it does not add the flag.

The hard part is that the lever must not bypass the fit:

* `ngl` is the fit's own OUTPUT. A requested value is a CAP, clamped down to
  what fits at the chosen ctx/cache and reported — never accepted blindly,
  because more GPU layers than the budget holds is the silent WDDM paging the
  whole resolver exists to prevent.
* `ubatch` raises llama.cpp's compute buffer, so the fit's compute term is
  charged from it (`resolve.compute_buffer_mb`). The default constant is
  untouched: with no override every number is byte-identical (see the
  before/after dump in the impl report).

Nothing here launches an engine: resolve/fit are pure arithmetic over a
synthetic spec, and the API tests are storage only.
"""
import pytest

import rigma.resolve as resolve_mod
from rigma.models import (ComboFlags, CpuInfo, GgufFile, GpuInfo,
                          HardwareProfile, LaunchDefaults, ModelSpec)
from rigma.registry import Registry
from rigma.resolve import _budgets, _spilled, fit_gguf, resolve


def compute_buffer_mb(ubatch: int = 0) -> float:
    """Resolved at call time, not import time, so this module still COLLECTS
    against the pre-change tree — each test below must fail on its own
    behaviour (missing field, missing kwarg), not on a collection error that
    would hide which behaviour is unpinned."""
    return resolve_mod.compute_buffer_mb(ubatch)

# The card the owner runs: 16GB Windows, so the desktop reserve is 1200 and the
# compute constant 150. `slug` names no combo, so `resolve` takes the calculator
# path and the arithmetic below is the whole answer.
VRAM_MB = 16368
USABLE_MB = 15018.0
UB2048_COMPUTE_MB = 600.0


def _profile(vram=VRAM_MB, ram_free=9100, slug="w10a-card"):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug=slug, backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=16234, ram_free_mb=ram_free,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _spec(**over):
    """A 64-layer dense model whose FILE cannot be fully resident at 16K, so the
    fit has to place layers rather than say "all" — which is what makes the ngl
    cap observable. Pinned by .w10a-evidence/probe_fixture.py: the fit lands on
    q4_0 with ngl 59 (9% spilled) at the default 16384, and ngl 57 once a 2048
    ubatch is charged."""
    base = dict(slug="w10a-fixed", family="f", kind="dense", n_layers=64,
                full_attn_layers=64, kv_heads=4, head_dim=256,
                native_ctx=262144,
                ggufs=[GgufFile(repo="r", file="fixed.gguf",
                                bytes=15000 * 2**20, quant="Q4")],
                use_cases=["general"])
    base.update(over)
    return ModelSpec(**base)


def _plan(launch, profile=None):
    spec = _spec(launch=launch)
    reg = Registry([], {"w10a-fixed": spec}, {})
    return spec, resolve(profile or _profile(), reg, use_case="general",
                         model_override="w10a-fixed")


# --- the stored default -----------------------------------------------------

def test_defaults_carry_batch_ubatch_ngl_and_survive_json():
    """They live in the spec file, so they must round-trip like every other
    stored field — and `as_overrides` must carry them, or the launcher never
    sees them."""
    d = LaunchDefaults(ctx=8192, batch=16384, ubatch=2048, ngl=40)
    assert d.batch == 16384 and d.ubatch == 2048 and d.ngl == 40
    back = ModelSpec.model_validate_json(
        _spec(launch=d).model_dump_json()).launch
    assert (back.batch, back.ubatch, back.ngl) == (16384, 2048, 40)
    over = d.as_overrides()
    assert (over["batch"], over["ubatch"], over["ngl"]) == (16384, 2048, 40)
    # …and an untouched default is "no opinion", not 0/-1 leaking into a launch
    assert LaunchDefaults().as_overrides() == {}
    assert LaunchDefaults(batch=0, ubatch=0, ngl=-1).as_overrides() == {}


def test_ngl_zero_is_an_opinion_and_minus_one_is_not():
    """`-ngl 0` is a real request (every layer on the CPU), so ngl cannot use
    the falsy 0 sentinel the other fields use. -1 is the sentinel."""
    assert LaunchDefaults(ngl=0).is_set("ngl") is True
    assert LaunchDefaults(ngl=0).as_overrides()["ngl"] == 0
    assert LaunchDefaults(ngl=-1).is_set("ngl") is False
    assert LaunchDefaults().is_set("ngl") is False


# --- refusing what the engine refuses ---------------------------------------

def test_ubatch_above_batch_is_refused():
    """llama.cpp will not start when the physical batch exceeds the logical
    one, so a stored default that says so is a 400 at write time, not a dead
    engine at load time."""
    with pytest.raises(ValueError):
        LaunchDefaults(batch=4096, ubatch=8192)


def test_ubatch_above_the_ENGINE_default_batch_is_refused_when_batch_is_unset():
    """`batch` unset means 2048, not 0. A stored ubatch 4096 is still illegal
    and must not be waved through by comparing against the sentinel."""
    with pytest.raises(ValueError):
        LaunchDefaults(ubatch=4096)
    assert LaunchDefaults(ubatch=2048).ubatch == 2048   # equal is fine
    # a small batch is fine only if the (default) physical batch still fits
    assert LaunchDefaults(batch=512).batch == 512
    with pytest.raises(ValueError):
        LaunchDefaults(batch=256)          # default ubatch 512 > 256
    assert LaunchDefaults(batch=256, ubatch=256).ubatch == 256


def test_a_negative_batch_size_is_refused():
    with pytest.raises(ValueError):
        LaunchDefaults(batch=-1)
    with pytest.raises(ValueError):
        LaunchDefaults(ubatch=-1)


def test_combo_flags_refuse_the_same_pair():
    """`server_args` writes -b/-ub straight from ComboFlags, so the argv carrier
    must refuse it too — a hand-edited calibration row is the reachable case."""
    with pytest.raises(ValueError):
        ComboFlags(ctx=4096, batch=2048, ubatch=4096)
    assert ComboFlags(ctx=4096, batch=2048, ubatch=2048).ubatch == 2048


# --- the compute buffer -----------------------------------------------------

def test_compute_buffer_is_unchanged_at_the_default():
    """The constant is the figure measured at ubatch 512, so 0 (no opinion) and
    512 must both return it exactly — that is what keeps every existing plan
    byte-identical."""
    assert compute_buffer_mb(0) == 150.0
    assert compute_buffer_mb(512) == 150.0


def test_compute_buffer_scales_with_a_larger_ubatch():
    """PREDICTION, linear from the one measured point: 2048/512 * 150 = 600."""
    assert compute_buffer_mb(2048) == UB2048_COMPUTE_MB
    assert compute_buffer_mb(1024) == 300.0


def test_compute_buffer_never_shrinks_below_the_constant():
    """No measurement exists below 512, so a smaller ubatch must not make the
    fit MORE optimistic than the measured configuration."""
    assert compute_buffer_mb(256) == 150.0


def test_the_requested_ubatch_is_charged_against_the_budget():
    p = _profile()
    assert _budgets(p)[0] == USABLE_MB
    assert _budgets(p, ubatch=2048)[0] == USABLE_MB - (
        UB2048_COMPUTE_MB - 150.0)


# --- end to end: the plan and the argv --------------------------------------

def test_a_requested_ubatch_reaches_the_fit_and_the_argv():
    """The value is in the argv AND the fit has already paid for it: charging
    600MB instead of 150 costs two GPU layers on this fixture (59 -> 57)."""
    base_spec, base = _plan(None)
    spec, plan = _plan(LaunchDefaults(ubatch=2048))
    assert "-ub" in plan.server_args("C:/m/f.gguf", 8080)
    argv = plan.server_args("C:/m/f.gguf", 8080)
    assert argv[argv.index("-ub") + 1] == "2048"
    assert plan.flags.ngl == 57
    assert plan.flags.ngl < base.flags.ngl
    assert _spilled(spec, plan.flags) > _spilled(base_spec, base.flags)


def test_a_requested_batch_reaches_the_argv():
    _spec2, plan = _plan(LaunchDefaults(batch=8192, ubatch=2048))
    argv = plan.server_args("C:/m/f.gguf", 8080)
    assert argv[argv.index("-b") + 1] == "8192"
    assert argv[argv.index("-ub") + 1] == "2048"


def test_a_requested_ngl_above_the_fit_is_clamped_and_the_plan_says_so():
    """The policy: CLAMP WITH A NOTE, never refuse. `999` is clamped to the
    fit's own 59 — the fit rules — and the plan reports what it actually used,
    so the argv and the explain cannot disagree."""
    _spec2, plan = _plan(LaunchDefaults(ngl=999))
    argv = plan.server_args("C:/m/f.gguf", 8080)
    assert plan.flags.ngl == 59
    assert argv[argv.index("-ngl") + 1] == "59"
    assert any("launch default ngl 999 exceeds what fits" in e
               for e in plan.explain), plan.explain


def test_a_requested_ngl_below_the_fit_is_honoured_and_named():
    """The safe direction is not clamped away: asking for FEWER GPU layers only
    frees VRAM, so the request stands and the note says what the fit allowed."""
    _spec2, plan = _plan(LaunchDefaults(ngl=40))
    assert plan.flags.ngl == 40
    assert any("launch default ngl 40 (the fit allows 59)" in e
               for e in plan.explain), plan.explain


def test_an_unplaceable_ngl_keeps_the_resolvers_placement():
    """A request that cannot be placed at all (the absolute CPU floor) must not
    turn into a refusal: the resolver's own placement stands and the plan says
    why, exactly like every other memory lever here."""
    p = _profile(vram=2048, ram_free=2500, slug="w10a-tiny")
    _spec2, plan = _plan(LaunchDefaults(ngl=12), profile=p)
    assert plan.flags.ngl == 0          # the floor plan, untouched
    assert any("cannot be placed" in e for e in plan.explain), plan.explain


# --- no override, no change -------------------------------------------------

def test_no_batch_override_leaves_the_plan_byte_identical():
    """A model that pins something else (ctx) must not thereby pin a batch: the
    flags, the argv and the explain are exactly the resolver's."""
    _s1, plain = _plan(None)
    _s2, pinned = _plan(LaunchDefaults(ctx=32768))
    assert plain.flags.model_dump() == pinned.flags.model_dump()
    assert plain.server_args("C:/m/f.gguf", 8080) == \
        pinned.server_args("C:/m/f.gguf", 8080)
    assert plain.explain == pinned.explain


def test_fit_gguf_without_an_override_prices_the_default_constant():
    """The fit reads the ubatch from the spec, but a spec with no launch object
    must price the measured 150MB constant — no drift in the default path."""
    spec = _spec()
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 16384, [])
    assert flags is not None and flags.ngl == 59
    assert _budgets(_profile())[0] == USABLE_MB


# --- the launch path the UI actually reaches --------------------------------

def test_perform_switch_launches_with_the_stored_batch_ubatch_ngl(
        tmp_path, monkeypatch):
    """The API/storage is not the deliverable: the OWNER's load button is. A
    stored default must reach the argv `perform_switch` hands to the launcher —
    that is the route the UI's load button takes, and the one the brief is
    about. No engine is started: `launch_server` is replaced."""
    import os
    from types import SimpleNamespace

    from rigma import server_ops, state
    from rigma.registry import Registry

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    home = tmp_path / "home"
    (home / "models").mkdir(parents=True, exist_ok=True)
    (home / "models" / "fixed.gguf").write_text("x", encoding="utf-8")
    spec = _spec(launch=LaunchDefaults(batch=8192, ubatch=2048, ngl=40))
    reg = Registry([], {"w10a-fixed": spec}, {})
    p = _profile(vram=24576)
    state.write_state("other", "Q0", 18500, engine_pid=999999,
                      ui_pid=os.getpid(), backend="vulkan", ctx=4096)
    monkeypatch.setattr("rigma.state.kill_pid", lambda pid: None)
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: home / "llama-server.exe")
    seen = {}

    def _launch(exe, plan, mp, port=0, timeout=300.0, extra_args=None):
        seen["plan"] = plan
        return SimpleNamespace(proc=SimpleNamespace(pid=4242))

    monkeypatch.setattr("rigma.runtime.launch_server", _launch)
    server_ops.perform_switch("w10a-fixed", registry=reg, profile=p)
    plan = seen["plan"]
    assert (plan.flags.batch, plan.flags.ubatch) == (8192, 2048)
    assert plan.flags.ngl == 40            # below the fit's 99: honoured
    argv = plan.server_args("C:/m/f.gguf", 8080)
    assert argv[argv.index("-b") + 1] == "8192"
    assert argv[argv.index("-ub") + 1] == "2048"
    assert argv[argv.index("-ngl") + 1] == "40"
