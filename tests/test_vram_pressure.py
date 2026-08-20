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

from tests.test_backend_choice import _prof


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
