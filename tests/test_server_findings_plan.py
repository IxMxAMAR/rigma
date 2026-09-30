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

def test_planned_vram_mb_is_the_plans_own_weights_plus_kv():
    got = server_ops.planned_vram_mb(_state())

    spec = Registry.load().models[SLUG]
    gguf = next(g for g in spec.ggufs if g.quant == QUANT)
    plan = RunPlan(model_slug=SLUG, gguf=gguf, backend="rocm",
                   flags=ComboFlags(ctx=CTX, cache_type_k="f16",
                                    cache_type_v="f16"), origin="state")
    assert got == pytest.approx(memtruth.planned_mb(plan))
    # weights + KV, NOT the bare file size: the engine's actual includes the KV
    # cache, so a file size alone would read as a divergence that is only it.
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

    from rigma.resolve import kv_bytes_per_token, swa_kv_bytes
    gguf = next(g for g in changed.ggufs if g.quant == QUANT)
    want_kv = (CTX * kv_bytes_per_token(changed, "f16", "f16")
               + swa_kv_bytes(changed, "f16", "f16", CTX)) / 2**20
    global_kv = (CTX * kv_bytes_per_token(spec, "f16", "f16")
                 + swa_kv_bytes(spec, "f16", "f16", CTX)) / 2**20

    # BOTH terms follow the registry the caller handed...
    assert got == pytest.approx(gguf.bytes / 2**20 + want_kv)
    # ...and the KV term is NOT the process-global one (the pre-fix chimera).
    assert got != pytest.approx(gguf.bytes / 2**20 + global_kv)


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
    assert server_ops.recorded_placement(disk) == {"ngl": 99, "n_cpu_moe": 12}

    # ...and a merge write (an unload) must carry it, not drop it.
    st.update_state(engine_pid=-1, unloaded=True)
    assert server_ops.recorded_placement(st.read_state()) == {
        "ngl": 99, "n_cpu_moe": 12}


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
