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
"""
import os

import pytest
from fastapi.testclient import TestClient

from rigma import memtruth, server_ops
from rigma import state as st
from rigma.models import ComboFlags, RunPlan
from rigma.registry import Registry
from rigma.serve import build_app

SLUG = "qwen3-0.6b"
QUANT = "Q8_0"
CTX = 32768


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _client(home):
    # No `__enter__`: the lifespan warms the memory embedder, and these tests
    # must not load a model (§0.1).
    return TestClient(build_app(upstream_port=1, default_prompt=""))


def _state() -> dict:
    return {"model": SLUG, "quant": QUANT, "ctx": CTX, "kv_cache": "f16",
            "backend": "rocm"}


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
                   kv_cache="f16")

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
                   kv_cache="f16")

    info = _client(home).get("/api/server").json()

    assert "plan_divergence" in [f["id"] for f in info["engine_findings"]]
