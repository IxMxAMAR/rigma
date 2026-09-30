"""A17d: make the VRAM divergence axis live on the findings routes.

A17c made `engine_log.compare_plan` judge the engine's reported actual against
the PLAN'S OWN prediction for the same ctx / cache / slots, and report
`not_comparable` when it has none. But the live route passed nothing, so the axis
was inert on the owner's machine — it could never fire.

`server_ops.planned_vram_mb(state)` rebuilds the running plan from the state the
launch wrote and returns `memtruth.planned_mb(plan)`, the exact function the fit
(`memtruth.verify_plan`) compares its oracle against. It is deliberately NOT a
bare file size: the engine's actual includes the KV cache, so the file size alone
would read as a divergence that is only the cache (GUIDANCE 5).

Both `/api/server/findings` and `/api/server` (via `_engine_extras`) pass it; the
route tests below pin the two outcomes the owner sees: a plan in state gives a
real verdict, and no plan gives `not_comparable` (no finding).

DR2-1-res adds the other axis the plan never recorded: its device-side
PLACEMENT. `planned_vram_mb` used to rebuild the plan with `ComboFlags`'
defaults (`ngl=99, n_cpu_moe=0`), i.e. "fully resident"; for a dense spill or a
MoE expert offload the whole-file prediction is not comparable to the engine's
device buffers, so `compare_plan` suppressed the VRAM axis and a real divergence
was blind. The launch now writes `placement = {"ngl", "n_cpu_moe"}` into
state.json, `planned_vram_mb` scales the weight term to the device with it, and
`compare_plan` compares it directly. An OLD record with no `placement` reads as
UNKNOWN (None), never as the confident zero.
"""
import json
import os

import pytest
from fastapi.testclient import TestClient

from rigma import engine_log, memtruth, server_ops
from rigma import state as st
from rigma.models import ComboFlags, RunPlan
from rigma.registry import Registry
from rigma.serve import build_app

SLUG = "qwen3-0.6b"
QUANT = "Q8_0"
CTX = 32768

# The owner's 35B MoE — the shape DR2-1-res exists for (expert weights kept in
# RAM inside GPU layers, `--n-cpu-moe`).
MOE_SLUG = "qwen3.6-35b-a3b"
MOE_QUANT = "UD-Q4_K_XL"
MOE_CTX = 16384
RESIDENT = {"ngl": 99, "n_cpu_moe": 0}
MOE_12 = {"ngl": 99, "n_cpu_moe": 12}

# DR21R-n1: a DEEP dense spill — a small DEVICE weight term, so the 15% slack is
# under the 512 MiB floor and the engine's compute buffer alone used to exceed
# the whole slack. `SPILL_CTX` keeps the KV term modest too, so the pre-fix
# expected figure really is small.
SPILL_NGL = 5
SPILL_CTX = 8192
SPILL_COMPUTE_MB = 600.0


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture(autouse=True)
def _reset_work_route_limiter():
    """The work-route limiter is PROCESS-GLOBAL (`serve._rate_hits`, 6 GETs per
    10 s per path). This file GETs `/api/server/findings` several times; without
    a reset those hits spend the budget of whatever test file runs next and it
    gets a 429 (`test_engine_log_findings_route.py`). Reset before AND after, so
    this file leaves no residue. Same idea as test_audit_sec13_net.py:87."""
    from rigma import serve
    serve._rate_hits.clear()
    yield
    serve._rate_hits.clear()


def _client(home):
    # No `__enter__`: the lifespan warms the memory embedder, and these tests
    # must not load a model (§0.1).
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def _state() -> dict:
    return {"model": SLUG, "quant": QUANT, "ctx": CTX, "kv_cache": "f16",
            "backend": "rocm", "placement": dict(RESIDENT)}


def _moe_state() -> dict:
    return {"model": MOE_SLUG, "quant": MOE_QUANT, "ctx": MOE_CTX,
            "kv_cache": "q8_0", "backend": "rocm", "placement": dict(MOE_12)}


def _write_log(home, text):
    logs = home / "logs"
    logs.mkdir(exist_ok=True)
    (logs / "server-1.log").write_text(text, encoding="utf-8")


def _divergent_log(expected_mb: float = 40000.0) -> str:
    """A dense all-GPU load (n_expert = 0, offloaded 29/29 => expected 2 splits)
    whose device model buffer is far above any plan's prediction, so ONLY the
    VRAM axis can diverge. The default is deliberately a fixed huge number so the
    route tests do not need the helper to build their own fixture."""
    return (
        "0.00.987.636 I print_info: n_expert              = 0\n"
        "0.01.522.670 I load_tensors: offloaded 29/29 layers to GPU\n"
        "0.01.522.676 I load_tensors:        ROCm0 model buffer size = "
        f"{expected_mb * 2:.2f} MiB\n"
        "0.04.364.938 I llama_kv_cache:      ROCm0 KV buffer size =  100.00 MiB\n"
        "0.04.417.931 I sched_reserve:      ROCm0 compute buffer size =  10.00 MiB\n"
        "0.04.417.939 I sched_reserve: graph splits = 2\n"
    )


# ---------------------------------------------------------------------------
# The helper: the plan's own prediction, or None.
# ---------------------------------------------------------------------------

def test_planned_vram_mb_is_the_plans_own_weights_kv_and_compute_charge():
    """DR21R-n1: the DEVICE figure is weights + KV + the plan's own compute
    charge.

    `memtruth.planned_mb` deliberately omits the compute term — the FIT
    comparison needs to SEE an under-count — but the VRAM axis compares against
    the engine's device buffers, which INCLUDE it. So `planned_vram_mb` adds
    DR2-3's `resolve.compute_buffer_mb` and the two sides share one basis at
    every spill depth (before this, a deep spill's small expected figure fell
    below the 512 MiB slack floor and a healthy load reported
    `plan_divergence`)."""
    from rigma.resolve import compute_buffer_mb, launch_ubatch

    got = server_ops.planned_vram_mb(_state())

    spec = Registry.load().models[SLUG]
    gguf = next(g for g in spec.ggufs if g.quant == QUANT)
    plan = RunPlan(model_slug=SLUG, gguf=gguf, backend="rocm",
                   flags=ComboFlags(ctx=CTX, cache_type_k="f16",
                                    cache_type_v="f16"), origin="state")
    compute = compute_buffer_mb(launch_ubatch(spec))
    assert got == pytest.approx(memtruth.planned_mb(plan) + compute)
    # weights + KV + compute, NOT the bare file size: the engine's actual
    # includes the KV cache and its own compute scratch, so a file size alone
    # would read as a divergence that is only them.
    assert got > gguf.bytes / 2**20 + 1000


@pytest.mark.parametrize("bad", [
    {"model": SLUG, "quant": QUANT, "ctx": 0, "kv_cache": "f16"},
    {"model": SLUG, "quant": "", "ctx": CTX},
    {"model": "", "quant": QUANT, "ctx": CTX},
    {"model": "nope", "quant": QUANT, "ctx": CTX},
    {"model": SLUG, "quant": "NOPE", "ctx": CTX},
])
def test_planned_vram_mb_is_none_when_not_recoverable(bad):
    assert server_ops.planned_vram_mb(bad) is None


# ---------------------------------------------------------------------------
# DR2-2: ONE registry, and no registry parse per poll.
#
# The weight term follows `plan.gguf.bytes` (the caller's registry) but the KV
# term used to come from a fresh global `Registry.load()` inside
# `memtruth.planned_mb`, so the prediction was a chimera of two registries
# whenever they disagreed. On the live path `registry is None`, so every poll
# also paid that parse.
# ---------------------------------------------------------------------------

def _custom_registry_doubling_kv_heads():
    """The real registry with one model's KV geometry changed, so the two
    sources provably disagree on the KV term."""
    reg = Registry.load()
    spec = reg.models[SLUG]
    changed = spec.model_copy(update={"kv_heads": spec.kv_heads * 2})
    custom = Registry(gpus=reg.gpus, models={**reg.models, SLUG: changed},
                      combos=reg.combos, use_cases=reg.use_cases)
    return custom, spec, changed


def test_planned_vram_mb_takes_its_kv_from_the_registry_it_was_handed(home):
    custom, spec, changed = _custom_registry_doubling_kv_heads()

    got = server_ops.planned_vram_mb(_state(), custom)

    from rigma.resolve import (compute_buffer_mb, kv_bytes_per_token,
                               launch_ubatch, swa_kv_bytes)
    gguf = next(g for g in changed.ggufs if g.quant == QUANT)
    want_kv = (CTX * kv_bytes_per_token(changed, "f16", "f16")
               + swa_kv_bytes(changed, "f16", "f16", CTX)) / 2**20
    global_kv = (CTX * kv_bytes_per_token(spec, "f16", "f16")
                 + swa_kv_bytes(spec, "f16", "f16", CTX)) / 2**20
    # DR21R-n1: the compute charge follows the same registry's spec (the launch
    # ubatch is spec data), so it is the caller's registry on both sides too.
    compute = compute_buffer_mb(launch_ubatch(changed))

    # BOTH terms follow the registry the caller handed...
    assert got == pytest.approx(gguf.bytes / 2**20 + want_kv + compute)
    # ...and the KV term is NOT the process-global one (the pre-fix chimera).
    assert got != pytest.approx(gguf.bytes / 2**20 + global_kv + compute)


def test_the_memo_does_not_serve_one_registrys_prediction_for_another(home):
    custom, _spec, _changed = _custom_registry_doubling_kv_heads()

    custom_mb = server_ops.planned_vram_mb(_state(), custom)
    global_mb = server_ops.planned_vram_mb(_state())

    assert custom_mb != pytest.approx(global_mb)


def test_the_plan_prediction_is_memoised_not_reloaded_on_every_poll(
        home, monkeypatch):
    """`/api/server` is polled every 5 s from every open tab and the live app is
    built with `registry=None`, so an unmemoised prediction paid a full
    `Registry.load()` per poll — twice, because `planned_mb` loaded its own."""
    from rigma.registry import Registry as _Registry
    calls = {"n": 0}
    real = _Registry.load

    def counting(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    monkeypatch.setattr(_Registry, "load", staticmethod(counting))

    first = server_ops.planned_vram_mb(_state())
    second = server_ops.planned_vram_mb(_state())

    assert first == second
    assert calls["n"] == 1, (
        f"the registry was parsed {calls['n']} times for two polls")


# ---------------------------------------------------------------------------
# The route: a plan in state makes the axis live; no plan keeps it honest.
# ---------------------------------------------------------------------------

def test_the_findings_route_reports_a_real_vram_verdict_with_a_plan(home):
    _write_log(home, _divergent_log())
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=CTX,
                   kv_cache="f16", placement=dict(RESIDENT))

    r = _client(home).get("/api/server/findings")

    assert r.status_code == 200
    ids = [f["id"] for f in r.json()["findings"]]
    assert ids == ["plan_divergence"], ids


def test_the_findings_route_is_not_comparable_with_no_plan(home):
    _write_log(home, _divergent_log())
    # no state.json at all -> no running plan -> the axis must stay
    # `not_comparable` rather than comparing against a bare file size.

    r = _client(home).get("/api/server/findings")

    assert r.status_code == 200
    assert r.json()["findings"] == []


def test_server_info_carries_the_same_verdict(home):
    """`_engine_extras` is the other call site and must be wired too."""
    _write_log(home, _divergent_log())
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=CTX,
                   kv_cache="f16", placement=dict(RESIDENT))

    info = _client(home).get("/api/server").json()

    assert "plan_divergence" in [f["id"] for f in info["engine_findings"]]


# ---------------------------------------------------------------------------
# DR2-1-res: the plan PERSISTS the placement it used, and the axis READS it.
#
# Before this, state.json carried no ngl / n_cpu_moe, so `planned_vram_mb`
# rebuilt the plan with `ComboFlags`' defaults (ngl=99, n_cpu_moe=0) — the
# confident "fully resident" assumption. The whole-file prediction is then not
# comparable to a MoE / spilled load's device buffers, so `compare_plan`
# suppressed the VRAM axis entirely and a real divergence was blind
# (deep-review-3.md, DR2-1-res).
# ---------------------------------------------------------------------------

def test_the_launch_placement_round_trips_through_state(home):
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=CTX,
                   kv_cache="f16", placement={"ngl": 99, "n_cpu_moe": 12})

    disk = st.read_state()
    assert disk["placement"] == {"ngl": 99, "n_cpu_moe": 12}
    # DR21RN1-n1: the reader now also reports the recorded ubatch. This record
    # predates that key, so it reads as None ("unknown") — the on-disk dict is
    # untouched and the placement itself is still a placement.
    assert server_ops.recorded_placement(disk) == {
        "ngl": 99, "n_cpu_moe": 12, "ubatch": None}

    # ...and a merge write (an unload) must carry it, not drop it.
    st.update_state(engine_pid=-1, unloaded=True)
    assert server_ops.recorded_placement(st.read_state()) == {
        "ngl": 99, "n_cpu_moe": 12, "ubatch": None}


def test_an_old_record_without_placement_reads_as_unknown(home):
    # A record written before `placement` existed: no key at all.
    st.state_path().write_text(json.dumps({
        "model": SLUG, "quant": QUANT, "ctx": CTX, "kv_cache": "f16",
        "backend": "rocm", "engine_pid": os.getpid(), "ui_pid": os.getpid()}),
        encoding="utf-8")
    s = st.read_state()
    assert "placement" not in s

    assert server_ops.recorded_placement(s) is None
    # UNKNOWN, not a confident whole-file ("fully resident") prediction: before
    # the fix this returned ~the whole GGUF, which is what the axis trusted.
    assert server_ops.planned_vram_mb(s) is None
    # A partial or non-dict placement is equally not a placement, so an absent
    # field can never be read as the confident `n_cpu_moe = 0`.
    assert server_ops.recorded_placement({"placement": {"ngl": 99}}) is None
    assert server_ops.recorded_placement({"placement": 0}) is None


def test_the_prediction_scales_the_weight_term_by_the_recorded_placement(home):
    resident = server_ops.planned_vram_mb({**_moe_state(),
                                           "placement": dict(RESIDENT)})
    moe = server_ops.planned_vram_mb(_moe_state())

    spec = Registry.load().models[MOE_SLUG]
    gguf = next(g for g in spec.ggufs if g.quant == MOE_QUANT)
    from rigma.resolve import _spilled
    spill = _spilled(spec, ComboFlags(ctx=MOE_CTX, ngl=99, n_cpu_moe=12,
                                      cache_type_k="q8_0", cache_type_v="q8_0"))
    assert spill > 0

    # The device-side figure is the whole-file one minus exactly the share the
    # plan's own recorded placement leaves in system RAM.
    assert moe == pytest.approx(resident - gguf.bytes / 2**20 * spill)
    assert moe < resident - 5000


def _moe_log(device_model_mb: float) -> str:
    return (
        "0.00.1 I print_info: n_expert              = 256\n"
        "0.00.2 I print_info: n_expert_used         = 8\n"
        "0.00.3 I load_tensors: offloaded 49/49 layers to GPU\n"
        "0.00.4 I load_tensors:        ROCm0 model buffer size = "
        f"{device_model_mb:.2f} MiB\n"
        "0.00.5 I load_tensors:           CPU model buffer size =  5430.00 MiB\n"
        "0.00.6 I sched_reserve:      ROCm0 compute buffer size =   600.00 MiB\n"
        "0.00.7 I sched_reserve: graph splits = 98\n"
    )


def test_a_recorded_moe_placement_makes_the_vram_axis_live(home):
    device_side = server_ops.planned_vram_mb(_moe_state())
    _write_log(home, _moe_log(device_side - 600))   # actual == prediction
    st.write_state(MOE_SLUG, MOE_QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=MOE_CTX,
                   kv_cache="q8_0", placement=dict(MOE_12))

    r = _client(home).get("/api/server/findings")

    assert r.status_code == 200
    assert r.json()["findings"] == []   # healthy, judged, no false alarm


def test_a_divergent_moe_load_is_caught_once_the_placement_is_recorded(home):
    device_side = server_ops.planned_vram_mb(_moe_state())
    _write_log(home, _moe_log(device_side + 6000))  # far above the device figure
    st.write_state(MOE_SLUG, MOE_QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=MOE_CTX,
                   kv_cache="q8_0", placement=dict(MOE_12))

    r = _client(home).get("/api/server/findings")

    assert r.status_code == 200
    ids = [f["id"] for f in r.json()["findings"]]
    assert ids == ["plan_divergence"], ids


def test_a_recorded_placement_compares_the_device_figure_directly(home):
    """The decision path consumes the persisted placement: the same MoE log is
    `diverges` against the device-side figure, and the plan's own placement is
    named in the detail. A whole-file figure (no placement) stays suppressed."""
    device_side = server_ops.planned_vram_mb(_moe_state())
    load = engine_log.parse_load(_moe_log(device_side + 6000))

    r = engine_log.compare_plan(load, device_side,
                                expected_placement=dict(MOE_12))
    assert r["vram_verdict"] == "diverges"
    assert r["vram_why"] is None
    assert "ngl=99, n_cpu_moe=12" in r["detail"]

    # DR2-1 is retained for a caller that has only a whole-file figure: it is
    # still not comparable to a non-resident load, so no false alarm.
    whole_file = server_ops.planned_vram_mb({**_moe_state(),
                                             "placement": dict(RESIDENT)})
    r2 = engine_log.compare_plan(load, whole_file)
    assert r2["vram_verdict"] == "not_comparable"
    assert r2["vram_why"] == "not_device_resident"

    # And with no prediction at all the axis stays unjudged.
    r3 = engine_log.compare_plan(load, None)
    assert r3["vram_verdict"] == "not_comparable"
    assert r3["vram_why"] == "no_prediction"


# ---------------------------------------------------------------------------
# DR21R-n1: the two sides must share ONE basis at EVERY spill depth.
#
# The engine's device figure is `device model + KV + RS + device compute`
# (`compare_plan`); `memtruth.planned_mb` deliberately leaves the compute term
# out, so the plan side used to be short by the engine's own compute buffer. A
# 512 MiB slack floor absorbed that for a large (resident) plan, but a DEEP
# spill's device prediction is small — the 15% term is under the floor — so a
# compute buffer larger than 512 MiB made a HEALTHY load report
# `plan_divergence`: the same class DR2-1 was fixed for. The plan's own charge
# for that term (DR2-3's `resolve.compute_buffer_mb`) is now on the plan side at
# every depth, so BOTH directions hold: healthy -> no finding, wrong -> finding.
# ---------------------------------------------------------------------------

def _spill_state() -> dict:
    return {"model": SLUG, "quant": QUANT, "ctx": SPILL_CTX, "kv_cache": "f16",
            "backend": "rocm", "placement": {"ngl": SPILL_NGL, "n_cpu_moe": 0}}


def _spill_flags() -> ComboFlags:
    return ComboFlags(ctx=SPILL_CTX, cache_type_k="f16", cache_type_v="f16",
                      ngl=SPILL_NGL, n_cpu_moe=0)


def _plan_terms(flags: ComboFlags):
    """(whole MiB, device weights MiB, KV MiB) for `flags`.

    Built from the plan's OWN arithmetic (`_spilled`, `memtruth.planned_mb`), so
    the "healthy" log below is a load that MATCHES the plan — not a round number
    that happens to fit."""
    from rigma.resolve import _spilled
    spec = Registry.load().models[SLUG]
    gguf = next(g for g in spec.ggufs if g.quant == QUANT)
    whole = gguf.bytes / 2**20
    plan = RunPlan(model_slug=SLUG, gguf=gguf, backend="rocm", flags=flags,
                   origin="test")
    kv = memtruth.planned_mb(plan, spec) - whole
    return whole, whole * (1.0 - _spilled(spec, flags)), kv


def _load_log(device_model_mb: float, kv_mb: float, *, ngl: int,
              compute_mb: float = SPILL_COMPUTE_MB) -> str:
    """A one-device load with an explicit `offloaded ngl/N` and compute buffer."""
    offloadable = Registry.load().models[SLUG].n_layers + 1
    return (
        "0.00.1 I print_info: n_expert              = 0\n"
        f"0.00.2 I load_tensors: offloaded {ngl}/{offloadable} layers to GPU\n"
        "0.00.3 I load_tensors:        ROCm0 model buffer size = "
        f"{device_model_mb:.2f} MiB\n"
        "0.00.4 I llama_kv_cache:      ROCm0 KV buffer size = "
        f"{kv_mb:.2f} MiB\n"
        "0.00.5 I sched_reserve:      ROCm0 compute buffer size = "
        f"{compute_mb:.2f} MiB\n"
        "0.00.6 I sched_reserve: graph splits = 2\n"
    )


def test_a_deep_spill_with_a_compute_buffer_larger_than_the_old_slack_is_healthy(
        home):
    """Direction 1: the verifier's shape is HEALTHY and must add no finding.

    The device prediction is small (the 15% slack is under the 512 MiB floor),
    and the engine's compute buffer is 600 MiB — larger than the WHOLE old
    slack. Before the fix that alone reported `plan_divergence`."""
    _whole, dev_w, kv = _plan_terms(_spill_flags())
    expected = server_ops.planned_vram_mb(_spill_state())

    _write_log(home, _load_log(dev_w, kv, ngl=SPILL_NGL))
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=SPILL_CTX,
                   kv_cache="f16", placement={"ngl": SPILL_NGL, "n_cpu_moe": 0})

    load = engine_log.parse_load(_load_log(dev_w, kv, ngl=SPILL_NGL))
    r = engine_log.compare_plan(load, expected,
                                expected_placement={"ngl": SPILL_NGL,
                                                    "n_cpu_moe": 0})
    assert r["vram_verdict"] == "ok", r["detail"]
    assert r["vram_why"] is None
    assert abs(r["divergence_mb"]) < 512.0

    # The fix must be what closes the gap, not a coincidence: the plan's own
    # charge is 150 MiB, the engine's buffer 600, and 600 > 512 > 600 - 150.
    assert expected - (dev_w + kv) == pytest.approx(150.0)
    assert SPILL_COMPUTE_MB > 512.0
    assert SPILL_COMPUTE_MB - (expected - dev_w - kv) < 512.0

    resp = _client(home).get("/api/server/findings")
    assert resp.status_code == 200
    assert resp.json()["findings"] == []


def test_a_deep_spill_with_genuinely_wrong_vram_still_fires(home):
    """Direction 2: the SAME shape with the device model buffer 6000 MiB above
    the plan still fires, so the healthy case was not bought by widening the
    slack."""
    _whole, dev_w, kv = _plan_terms(_spill_flags())
    _write_log(home, _load_log(dev_w + 6000.0, kv, ngl=SPILL_NGL))
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=SPILL_CTX,
                   kv_cache="f16", placement={"ngl": SPILL_NGL, "n_cpu_moe": 0})

    resp = _client(home).get("/api/server/findings")
    assert resp.status_code == 200
    ids = [f["id"] for f in resp.json()["findings"]]
    assert ids == ["plan_divergence"], ids


def test_a_fully_resident_load_with_a_large_compute_buffer_is_also_healthy(home):
    """The same window at spill depth 0 — why the charge is added at EVERY
    depth, not only when `_spilled > 0`.

    qwen3-0.6b fully resident at ctx 8192 predicts ~1.5 GiB, so the 15% slack is
    under the floor and a 600 MiB compute buffer alone exceeded the whole slack.
    Scoping the fix to spills would have left this false positive reachable."""
    from rigma.resolve import _spilled
    flags = ComboFlags(ctx=SPILL_CTX, cache_type_k="f16", cache_type_v="f16",
                       ngl=99, n_cpu_moe=0)
    whole, dev_w, kv = _plan_terms(flags)
    assert _spilled(Registry.load().models[SLUG], flags) == 0.0
    assert dev_w == pytest.approx(whole)

    _write_log(home, _load_log(whole, kv, ngl=29))
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=SPILL_CTX,
                   kv_cache="f16", placement=dict(RESIDENT))

    resp = _client(home).get("/api/server/findings")
    assert resp.status_code == 200
    assert resp.json()["findings"] == []


@pytest.mark.parametrize("ngl", [1, 5, 14, 27, 29])
def test_both_directions_hold_at_every_spill_depth(home, ngl):
    """The invariant, swept from a total offload (ngl=1) to fully resident
    (ngl=29): a load that MATCHES the plan is `ok`, and the same load with the
    device model buffer 6000 MiB above the plan still `diverges`. The 600 MiB
    compute buffer is present in both, so a healthy load is never the one that
    fires."""
    flags = ComboFlags(ctx=SPILL_CTX, cache_type_k="f16", cache_type_v="f16",
                       ngl=ngl, n_cpu_moe=0)
    _whole, dev_w, kv = _plan_terms(flags)
    placement = {"ngl": ngl, "n_cpu_moe": 0}
    state = {"model": SLUG, "quant": QUANT, "ctx": SPILL_CTX,
             "kv_cache": "f16", "backend": "rocm", "placement": placement}
    expected = server_ops.planned_vram_mb(state)

    healthy = engine_log.compare_plan(
        engine_log.parse_load(_load_log(dev_w, kv, ngl=ngl)), expected,
        expected_placement=placement)
    assert healthy["vram_verdict"] == "ok", (ngl, healthy["detail"])
    assert abs(healthy["divergence_mb"]) < 512.0

    wrong = engine_log.compare_plan(
        engine_log.parse_load(_load_log(dev_w + 6000.0, kv, ngl=ngl)),
        expected, expected_placement=placement)
    assert wrong["vram_verdict"] == "diverges", (ngl, wrong["detail"])


def test_a_device_resident_load_is_judged_exactly_as_before(home):
    """Direction 3: the device-resident path is UNCHANGED.

    The engine's own figures are byte-identical and the verdict is the same
    (`diverges` against `_divergent_log`'s 80000 MiB device model buffer). The
    plan-side figure gains the plan's own compute charge — the deliberate basis
    change, pinned by
    `test_planned_vram_mb_is_the_plans_own_weights_kv_and_compute_charge` — and
    nothing else moves."""
    load = engine_log.parse_load(_divergent_log())
    r = engine_log.compare_plan(load, server_ops.planned_vram_mb(_state()),
                                expected_placement=dict(RESIDENT))

    assert r["actual_vram_mb"] == pytest.approx(80110.0)
    assert r["host_ram_mb"] == 0.0
    assert r["graph_splits"] == 2
    assert r["vram_verdict"] == "diverges"
    assert r["vram_why"] is None
    assert r["diverges"] is True


# ---------------------------------------------------------------------------
# DR21RN1-n1: the compute charge follows the ubatch the LAUNCH used, not the
# spec's launch default.
#
# DR21R-n1 put `resolve.compute_buffer_mb(launch_ubatch(spec))` on the plan side,
# but an explicit `rigma up --ubatch N` (cli.py) is a REQUEST on the plan flags,
# not a stored model default: the spec the reader later resolves still says
# 0/512, while the engine sizes its compute buffer for N. At N=2048 that is
# 150 MiB charged against the engine's ~1641.12 MiB, and a HEALTHY override load
# reported `plan_divergence` (+1491.12 > the 512 MiB slack) — the item's own
# false-positive class, on a shipped CLI path. The launch now records the ubatch
# it emitted in the `placement` dict, and the reader charges THAT.
# ---------------------------------------------------------------------------

UBATCH = 2048
# compute_buffer_mb(2048) = MEASURED_COMPUTE_BUFFER_MB * 2048 / 512.
UBATCH_COMPUTE_MB = 1641.12


def _override_plan(ubatch: int) -> RunPlan:
    """The plan `rigma up --ubatch <ubatch>` builds.

    The override is a REQUEST on the flags — `RunPlan.server_args` emits `-ub`
    from exactly `flags.ubatch` — and is not written into the spec, which is why
    the reader cannot recover it from the registry afterwards."""
    spec = Registry.load().models[SLUG]
    gguf = next(g for g in spec.ggufs if g.quant == QUANT)
    return RunPlan(model_slug=SLUG, gguf=gguf, backend="rocm",
                   flags=ComboFlags(ctx=CTX, cache_type_k="f16",
                                    cache_type_v="f16", ngl=99, n_cpu_moe=0,
                                    ubatch=ubatch), origin="state")


def _resident_flags() -> ComboFlags:
    return ComboFlags(ctx=CTX, cache_type_k="f16", cache_type_v="f16",
                      ngl=99, n_cpu_moe=0)


def _spec_with_launch_ubatch(ubatch: int):
    """The packaged registry with `launch.ubatch` set on SLUG.

    Zero shipped models set it (the verifier's NIT 3), so this is the only way
    to exercise a spec-level ubatch above 512."""
    from rigma.models import LaunchDefaults
    reg = Registry.load()
    spec = reg.models[SLUG]
    base = getattr(spec, "launch", None) or LaunchDefaults()
    changed = spec.model_copy(
        update={"launch": base.model_copy(update={"ubatch": ubatch})})
    return Registry(gpus=reg.gpus, models={**reg.models, SLUG: changed},
                    combos=reg.combos, use_cases=reg.use_cases)


def test_a_launch_with_an_explicit_ubatch_records_it_and_the_healthy_load_is_ok(
        home):
    """(a) The override reaches the argv AND the record, and a load that matches
    the plan at that ubatch adds no finding."""
    from rigma.resolve import compute_buffer_mb, launch_ubatch

    plan = _override_plan(UBATCH)
    argv = plan.server_args("model.gguf", 1)
    assert argv[argv.index("-ub") + 1] == str(UBATCH)   # what the engine gets
    rec = server_ops.plan_placement(plan)
    assert rec == {"ngl": 99, "n_cpu_moe": 0, "ubatch": UBATCH}

    # The spec the READER resolves still says 0 (the 150 MiB floor), which is
    # exactly why the record has to carry the override.
    assert launch_ubatch(Registry.load().models[SLUG]) == 0
    state = {**_state(), "placement": dict(rec)}
    expected = server_ops.planned_vram_mb(state)

    _whole, dev_w, kv = _plan_terms(_resident_flags())
    assert expected == pytest.approx(
        dev_w + kv + compute_buffer_mb(UBATCH), abs=0.01)
    assert compute_buffer_mb(UBATCH) == pytest.approx(UBATCH_COMPUTE_MB)

    log = _load_log(dev_w, kv, ngl=29, compute_mb=compute_buffer_mb(UBATCH))
    _write_log(home, log)
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=CTX,
                   kv_cache="f16", placement=dict(rec))

    direct = engine_log.compare_plan(engine_log.parse_load(log), expected,
                                     expected_placement=dict(rec))
    assert direct["vram_verdict"] == "ok", direct["detail"]
    assert abs(direct["divergence_mb"]) < 512.0

    resp = _client(home).get("/api/server/findings")
    assert resp.status_code == 200
    assert resp.json()["findings"] == []


def test_an_explicit_ubatch_override_with_genuinely_wrong_vram_still_fires(home):
    """(b) The same shape with the device model buffer 6000 MiB above the plan
    still yields `plan_divergence`: (a) was not bought by widening the slack."""
    from rigma.resolve import compute_buffer_mb

    rec = server_ops.plan_placement(_override_plan(UBATCH))
    _whole, dev_w, kv = _plan_terms(_resident_flags())
    _write_log(home, _load_log(dev_w + 6000.0, kv, ngl=29,
                               compute_mb=compute_buffer_mb(UBATCH)))
    st.write_state(SLUG, QUANT, 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), backend="rocm", ctx=CTX,
                   kv_cache="f16", placement=dict(rec))

    resp = _client(home).get("/api/server/findings")
    assert resp.status_code == 200
    ids = [f["id"] for f in resp.json()["findings"]]
    assert ids == ["plan_divergence"], ids


def test_an_old_record_without_a_recorded_ubatch_uses_the_spec_fallback(home):
    """(c) A placement written before the ubatch was persisted still predicts,
    and the compute charge falls back to the spec's launch ubatch — whose floor
    is absorbed by the 512 MiB slack."""
    from rigma.resolve import MEASURED_COMPUTE_BUFFER_MB, compute_buffer_mb

    state = {**_state(), "placement": {"ngl": 99, "n_cpu_moe": 0}}
    assert server_ops.recorded_placement(state) == {
        "ngl": 99, "n_cpu_moe": 0, "ubatch": None}

    _whole, dev_w, kv = _plan_terms(_resident_flags())
    expected = server_ops.planned_vram_mb(state)
    assert expected is not None                      # still loads / predicts
    assert expected == pytest.approx(dev_w + kv + compute_buffer_mb(0))

    # The engine really allocates 410.28 MiB at ubatch 512 while the fallback
    # charges the 150 floor; the 512 MiB slack absorbs that gap, so a healthy
    # 512 load is still `ok`.
    gap = MEASURED_COMPUTE_BUFFER_MB - compute_buffer_mb(0)
    assert 0.0 < gap < 512.0
    log = _load_log(dev_w, kv, ngl=29, compute_mb=MEASURED_COMPUTE_BUFFER_MB)
    r = engine_log.compare_plan(engine_log.parse_load(log), expected,
                                expected_placement=server_ops.recorded_placement(
                                    state))
    assert r["vram_verdict"] == "ok", r["detail"]

    # Absent OR corrupt reads as UNKNOWN, never as the confident 0 (a real
    # "no -ub" launch): only a genuine non-negative integer is a claim.
    for junk in ("junk", -1, True, None, 2048.5):
        p = server_ops.recorded_placement(
            {"placement": {"ngl": 99, "n_cpu_moe": 0, "ubatch": junk}})
        assert p["ubatch"] is None, junk
    assert server_ops.recorded_placement(
        {"placement": {"ngl": 99, "n_cpu_moe": 0, "ubatch": 0}})["ubatch"] == 0
    # JSON has no integer type, so a whole-number float is a legitimate read.
    assert server_ops.recorded_placement(
        {"placement": {"ngl": 99, "n_cpu_moe": 0,
                       "ubatch": 2048.0}})["ubatch"] == 2048


def test_a_spec_ubatch_above_512_is_charged_correctly(home):
    """(d) A spec whose own `launch.ubatch` is above 512 is charged at that
    ubatch — and a RECORDED ubatch still beats the spec when they disagree."""
    from rigma.resolve import compute_buffer_mb

    custom = _spec_with_launch_ubatch(UBATCH)
    _whole, dev_w, kv = _plan_terms(_resident_flags())

    # An OLD record (no ubatch) with a spec that sets launch.ubatch 2048: the
    # fallback charges the spec's ubatch, and a healthy load there is `ok`.
    old_state = {**_state(), "placement": {"ngl": 99, "n_cpu_moe": 0}}
    old_expected = server_ops.planned_vram_mb(old_state, custom)
    assert old_expected == pytest.approx(
        dev_w + kv + compute_buffer_mb(UBATCH), abs=0.01)
    log = _load_log(dev_w, kv, ngl=29, compute_mb=compute_buffer_mb(UBATCH))
    r = engine_log.compare_plan(
        engine_log.parse_load(log), old_expected,
        expected_placement=server_ops.recorded_placement(old_state))
    assert r["vram_verdict"] == "ok", r["detail"]

    # The RECORD wins over the spec: a launch that overrode the spec down to 512
    # charges the 150 floor, not the spec's 1641.12 MiB.
    rec_state = {**_state(),
                 "placement": {"ngl": 99, "n_cpu_moe": 0, "ubatch": 512}}
    rec_expected = server_ops.planned_vram_mb(rec_state, custom)
    assert rec_expected == pytest.approx(dev_w + kv + compute_buffer_mb(512))
    assert rec_expected < old_expected - 1000.0
