"""What the DESKTOP is holding, measured instead of assumed.

Measured 2026-08-21 on the owner's machine: rigma budgets 1200 MB for
everything that is not the model; an idle Windows desktop with a browser and an
editor open held 3,955 MB. So rigma planned a "fully GPU-resident" config that
could not be resident, reported 0% offload, and Windows WDDM silently paged
4,107 MB of weights to system RAM — reached over PCIe on every token. The engine
log cannot show this: llama.cpp asked the driver for the memory and the driver
said yes.
"""
from rigma.probe import _sum_other_processes
from rigma.resolve import VRAM_RESERVE_MB, _budgets

# Not `from tests.test_backend_choice` — that needs the repo ROOT on sys.path,
# which `python -m pytest` provides (it prepends the working directory) and the
# bare `pytest` CI runs does not. pytest puts THIS directory on the path either
# way, so the flat name is the one that works in both.
from test_backend_choice import _prof


def _s(pairs):
    return [(f"pid_{p}_luid_0x00000000_0x00010a67_phys_0", mb * 2**20)
            for p, mb in pairs]


def test_sums_every_process_holding_vram():
    assert _sum_other_processes(_s([(1, 1668), (2, 340), (3, 230)]), set()) \
        == 2238


def test_excludes_our_own_engine():
    """The engine we are about to REPLACE still holds its VRAM while we plan
    the replacement. Counting it would reserve the same gigabytes twice and
    plan a far smaller model than the machine can hold."""
    assert _sum_other_processes(_s([(1, 1668), (7560, 12000)]), {7560}) == 1668


def test_ignores_instances_it_cannot_parse():
    weird = [("Total", 999 * 2**20), ("pid_5_luid_x_phys_0", 100 * 2**20)]
    assert _sum_other_processes(weird, set()) == 100


def test_no_samples_means_no_answer():
    assert _sum_other_processes([], set()) is None


def test_the_old_constant_is_a_floor_not_a_target():
    """A measurement must never make rigma MORE optimistic than the constant —
    the constant was conservative for a bare desktop and is still the right
    answer there."""
    prof = _prof(["vulkan"])
    baseline, _ = _budgets(prof)
    assert _budgets(prof, other_vram_mb=0)[0] == baseline
    assert _budgets(prof, other_vram_mb=200)[0] == baseline


def test_a_heavier_desktop_shrinks_the_budget():
    prof = _prof(["vulkan"])
    baseline, _ = _budgets(prof)
    tighter, _ = _budgets(prof, other_vram_mb=3955)
    assert tighter == baseline - (3955 - VRAM_RESERVE_MB["windows"])


def test_an_impossible_reading_is_ignored():
    """The per-process counter once reported 359,777 MiB on a 16GB card. A
    reading at or above the card's own capacity is a broken counter, not a
    busy desktop, and must not shrink the budget to zero."""
    prof = _prof(["vulkan"])
    baseline, _ = _budgets(prof)
    assert _budgets(prof, other_vram_mb=999_999)[0] == baseline
    assert _budgets(prof, other_vram_mb=16304)[0] == baseline


def test_the_measurement_is_cached_between_calls(monkeypatch):
    """probe_hardware runs on every /api/models request. A 1.3s PowerShell
    call per request would be worse than the bug this fixes."""
    from rigma import probe
    calls = []

    def _fake(*a, **k):
        calls.append(1)
        class R:
            stdout = str(2000 * 2**20)
        return R()
    monkeypatch.setattr(probe.subprocess, "run", _fake)
    monkeypatch.setattr(probe, "_os_name", lambda: "windows")
    probe._VRAM_CACHE.clear()
    first = probe.gpu_used_mb()
    second = probe.gpu_used_mb()
    assert first == second == 2000
    assert len(calls) == 1, f"measured {len(calls)} times, should have cached"


def test_the_running_engine_is_credited_back_by_its_OWN_file():
    """`_free_current` credited back the LARGEST gguf in the spec as a stand-in
    for the running one. That was harmless while a model listed a handful of
    files; after the 2026-08-21 file-list refresh the same model listed 103,
    including a 50GB BF16 — so it credited back three times the card's capacity,
    concluded the GPU was empty, and went straight back to planning against VRAM
    that does not exist. state["gguf"] names the actual file; use it."""
    from rigma.models import CpuInfo, GgufFile, GpuInfo, HardwareProfile, ModelSpec
    from rigma.server_ops import _free_current

    spec = ModelSpec(slug="m", family="qwen35", kind="dense", n_layers=64,
                     full_attn_layers=16, kv_heads=4, head_dim=256,
                     native_ctx=262144, custom=True,
                     ggufs=[GgufFile(repo="a/b", file="small.gguf",
                                     bytes=13_301_433_856, quant="Q3_K_M"),
                            GgufFile(repo="a/b", file="huge-BF16.gguf",
                                     bytes=54_000_000_000, quant="BF16")])

    class _Reg:
        models = {"m": spec}

    prof = HardwareProfile(
        os="windows", cpu=CpuInfo(cores=16),
        gpus=[GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16304,
                      arch="rdna4", slug="s", backends=["vulkan"])],
        ram_mb=32133, ram_free_mb=20000, disk_free_gb=500.0,
        vram_used_mb=15212)

    state = {"model": "m", "quant": "Q3_K_M", "gguf": "small.gguf"}
    out = _free_current(prof, state, _Reg())
    # 15,212 held minus the 12,686 MiB file that is actually loaded
    assert 2000 < out.vram_used_mb < 3200, (
        f"credited back the wrong file: {out.vram_used_mb}")


def test_crediting_falls_back_to_the_largest_when_the_file_is_unknown():
    """State written before the filename was recorded has no `gguf`. Guessing
    the largest is still better than crediting nothing, which would make a
    ctx change on the running model plan against its own occupied card."""
    from rigma.models import CpuInfo, GgufFile, GpuInfo, HardwareProfile, ModelSpec
    from rigma.server_ops import _free_current

    spec = ModelSpec(slug="m", family="qwen35", kind="dense", n_layers=64,
                     full_attn_layers=16, kv_heads=4, head_dim=256,
                     native_ctx=262144, custom=True,
                     ggufs=[GgufFile(repo="a/b", file="only.gguf",
                                     bytes=13_301_433_856, quant="Q3_K_M")])

    class _Reg:
        models = {"m": spec}

    prof = HardwareProfile(
        os="windows", cpu=CpuInfo(cores=16),
        gpus=[GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16304,
                      arch="rdna4", slug="s", backends=["vulkan"])],
        ram_mb=32133, ram_free_mb=20000, disk_free_gb=500.0,
        vram_used_mb=15212)
    out = _free_current(prof, {"model": "m", "quant": "Q3_K_M"}, _Reg())
    assert 2000 < out.vram_used_mb < 3200


# --- what the PANEL reports, as opposed to what the resolver plans with ------

def test_a_loaded_model_is_not_reported_as_other_apps(monkeypatch, tmp_path):
    """The adapter counter includes our own engine.

    Reported by the owner 2026-08-25 with a 15.7GB model loaded: the panel read
    "other apps hold 15.7 GB of VRAM - leaving 0.1 GB of 15.9 GB for the model"
    and advised closing a browser. The resolver had credited the running model
    back since 2026-08-21; this read had not, so the number shown and the
    number planned against disagreed by the size of the model.
    """
    from rigma import server_ops
    from rigma.models import GgufFile, ModelSpec

    spec = ModelSpec(slug="m", family="qwen35", kind="dense", n_layers=64,
                     full_attn_layers=16, kv_heads=4, head_dim=256,
                     native_ctx=262144,
                     ggufs=[GgufFile(repo="r", file="m.gguf",
                                     bytes=12 * 2**30, quant="Q4")])

    class _Reg:
        gpus = []
        models = {"m": spec}

    prof = _prof(["vulkan"])
    # 13.2GB held: 12GB of it is the model we are running, 1.2GB is everything
    # else on the card.
    prof = prof.model_copy(update={"vram_used_mb": 13_500.0})
    monkeypatch.setattr(server_ops, "probe_hardware", lambda gpus: prof,
                        raising=False)
    monkeypatch.setattr("rigma.probe.probe_hardware", lambda gpus: prof)
    monkeypatch.setattr("rigma.state.read_state",
                        lambda: {"model": "m", "gguf": "m.gguf",
                                 "unloaded": False, "public_port": 11500})

    snap = server_ops.vram_snapshot(_Reg())

    assert snap is not None
    # ~1.2GB of genuine other-app usage, not 13.5
    assert snap["desktop_mb"] < 2000, snap
    # and the model gets a real budget rather than 0.1GB
    assert snap["usable_mb"] > 10_000, snap


def test_an_unloaded_engine_leaves_the_reading_alone(monkeypatch):
    """Nothing of ours is on the card, so every megabyte really is someone
    else's and the warning should stand."""
    from rigma import server_ops
    from rigma.models import GgufFile, ModelSpec

    spec = ModelSpec(slug="m", family="qwen35", kind="dense", n_layers=64,
                     full_attn_layers=16, kv_heads=4, head_dim=256,
                     native_ctx=262144,
                     ggufs=[GgufFile(repo="r", file="m.gguf",
                                     bytes=12 * 2**30, quant="Q4")])

    class _Reg:
        gpus = []
        models = {"m": spec}

    prof = _prof(["vulkan"]).model_copy(update={"vram_used_mb": 4_000.0})
    monkeypatch.setattr("rigma.probe.probe_hardware", lambda gpus: prof)
    monkeypatch.setattr("rigma.state.read_state",
                        lambda: {"model": "m", "gguf": "m.gguf",
                                 "unloaded": True, "public_port": 11500})

    snap = server_ops.vram_snapshot(_Reg())

    assert snap is not None
    assert snap["desktop_mb"] == 4000
    assert snap["pressured"] is True
