"""R3-CAL-1 — the calibration cache is keyed by the hardware it was measured on.

The unit rules live in `test_r3_hwid.py`. This file covers the INTEGRATION: that
the bench-side helpers actually use identity, that entries written before identity
existed are still honoured, and that a cache miss re-measures instead of erroring.
"""
from __future__ import annotations

import json

import pytest

from rigma import bench, hwid


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    bench._IDENT_CACHE.clear()
    yield tmp_path / "home"
    bench._IDENT_CACHE.clear()


@pytest.fixture
def fixed_identity(monkeypatch):
    """A deterministic identity, so the tests never depend on the machine running
    them — the whole point of this feature is that hardware matters.

    Built through `bench.current_identity` (not `hwid.identity_from_gpu` directly)
    so it is the SAME construction path production uses, and so it is normalised
    the way a saved entry is. Building it the other way made the fixture and the
    entry disagree about spelling while describing one card.
    """
    raw = {"vendor_id": 0x1002, "device_id": 0x7550,
           "device_uuid": "aa" * 16, "driver_version": 0x800184,
           "name": "AMD Radeon RX 9070 XT"}
    monkeypatch.setattr(bench, "current_identity",
                        lambda backend="", gpu=None: hwid.identity_from_gpu("vulkan", raw))
    bench._IDENT_CACHE.clear()
    return hwid.identity_from_gpu("vulkan", raw)


def _write(cal: dict) -> None:
    p = bench.calibration_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cal), encoding="utf-8")


def test_the_key_includes_the_hardware(home, fixed_identity):
    k = bench.calibration_key("qwen3-0.6b", "Q8_0", "vulkan")
    assert k == f"qwen3-0.6b:Q8_0:vulkan:{fixed_identity.digest}"
    assert k != bench.legacy_key("qwen3-0.6b", "Q8_0", "vulkan")


def test_a_new_entry_records_the_hardware(home, fixed_identity):
    bench.save_calibration(bench.calibration_key("m", "q", "vulkan"),
                           {"tg_tps": 30.0}, identity=fixed_identity)
    entry = bench.load_calibration()[bench.calibration_key("m", "q", "vulkan")]
    assert entry["hardware"]["id"] == fixed_identity.digest
    assert entry["hardware"]["uuid"] == "aa" * 16
    assert entry["hardware"]["vendor_id"] == "0x1002"
    assert entry["schema"] == 3


def test_a_legacy_entry_is_still_found(home, fixed_identity):
    """Every calibration on every existing machine predates identity. The machine
    that wrote one is still the machine reading it, so it must keep working."""
    _write({bench.legacy_key("m", "q", "vulkan"): {"measured": {"tg_tps": 30.0},
                                                   "calibrated": True}})
    key, entry = bench.calibration_entry(bench.load_calibration(), "m", "q", "vulkan")
    assert key == bench.legacy_key("m", "q", "vulkan")
    assert entry["measured"]["tg_tps"] == 30.0
    assert bench.is_calibrated("m", "q", "vulkan") is True


def test_a_legacy_entry_without_hardware_is_not_a_mismatch(home, fixed_identity):
    """No recorded hardware is not recorded-and-different. Invalidating every old
    entry at once helps nobody."""
    entry = {"measured": {"tg_tps": 30.0}, "engine": "b9867"}
    assert hwid.hard_mismatch(entry, fixed_identity) is None


def test_the_identity_key_wins_over_the_legacy_key(home, fixed_identity):
    """Once a measurement exists for THIS card it must be preferred, or a stale
    legacy number would shadow a fresh one forever."""
    _write({bench.legacy_key("m", "q", "vulkan"): {"measured": {"tg_tps": 1.0}},
            bench.calibration_key("m", "q", "vulkan"): {"measured": {"tg_tps": 30.0}}})
    key, entry = bench.calibration_entry(bench.load_calibration(), "m", "q", "vulkan")
    assert entry["measured"]["tg_tps"] == 30.0


def test_calibrated_on_another_card_is_not_calibrated_here(home, monkeypatch):
    """THE defect. Without this a 3090 inherits a 4090's tune and reports a
    throughput it cannot reach — silently, because nothing about the number looks
    wrong."""
    other = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x10DE, "device_id": 0x2684,
                                              "device_uuid": "bb" * 16})
    mine = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                             "device_uuid": "aa" * 16})
    _write({hwid.calibration_key("m", "q", other): {"measured": {"tg_tps": 99.0},
                                                    "calibrated": True,
                                                    "hardware": other.as_dict()}})
    monkeypatch.setattr(bench, "current_identity", lambda backend="", gpu=None: mine)
    bench._IDENT_CACHE.clear()
    assert bench.is_calibrated("m", "q", "vulkan") is False


def test_clearing_removes_both_keys(home, fixed_identity):
    """"Forget this tune" must mean forgotten: leaving the legacy entry behind
    would make the next lookup find it and report the model as calibrated."""
    _write({bench.legacy_key("m", "q", "vulkan"): {"calibrated": True},
            bench.calibration_key("m", "q", "vulkan"): {"calibrated": True}})
    assert bench.clear_calibration("m", "q", "vulkan") is True
    assert bench.load_calibration() == {}
    assert bench.is_calibrated("m", "q", "vulkan") is False


def test_a_driver_change_makes_an_entry_stale_but_not_wrong(home, fixed_identity):
    """Soft: re-measure, do not discard. A driver update made non-FA PP 5% and FA
    15% faster, which is real but is not "this is another machine"."""
    entry = {"hardware": hwid.identity_from_gpu("vulkan", {
        "vendor_id": 0x1002, "device_id": 0x7550, "device_uuid": "aa" * 16,
        "driver_version": 0x800184}).as_dict(), "engine": "b9867"}
    newer = hwid.identity_from_gpu("vulkan", {
        "vendor_id": 0x1002, "device_id": 0x7550, "device_uuid": "aa" * 16,
        "driver_version": 0x900000})
    assert hwid.hard_mismatch(entry, newer) is None, "only the driver changed"
    why = bench.calibration_stale(entry, 1100, "b9867", 0, newer)
    assert why and "driver" in why
    assert "uuid" not in why, "the card did not change, only the driver"


def test_a_different_card_makes_an_entry_stale_with_the_hard_reason(home):
    other = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x10DE, "device_id": 0x2684,
                                              "device_uuid": "bb" * 16})
    mine = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                             "device_uuid": "aa" * 16})
    why = bench.calibration_stale({"hardware": other.as_dict()}, 1100, "", 0, mine)
    assert why and "measured on" in why


def test_stale_still_reports_the_vram_drift_it_always_did(home, fixed_identity):
    """Identity must not have replaced the existing check — it is additive."""
    entry = {"hardware": fixed_identity.as_dict(), "vram_used_mb": 1100,
             "engine": "b9867"}
    assert bench.calibration_stale(entry, 1150, "b9867", 0, fixed_identity) is None
    why = bench.calibration_stale(entry, 4000, "b9867", 0, fixed_identity)
    assert why and "VRAM" in why


def test_stale_without_identity_behaves_exactly_as_before(home):
    """The old signature is still supported, so nothing that called it breaks."""
    entry = {"engine": "b9867", "ctx": 8192, "vram_used_mb": 1100}
    assert bench.calibration_stale(entry, 1150, "b9867", 8192) is None
    assert "engine" in bench.calibration_stale(entry, 1150, "b9999", 8192)
    assert "ctx" in bench.calibration_stale(entry, 1150, "b9867", 4096)


# --- pruning ----------------------------------------------------------------

def test_pruning_keeps_the_newest_per_identity(home):
    cal = {
        "m:q:vulkan:old": {"date": "2026-01-01", "hardware": {"id": "AAA"}},
        "m:q:vulkan:new": {"date": "2026-09-01", "hardware": {"id": "AAA"}},
        "m:q:vulkan:other": {"date": "2026-05-01", "hardware": {"id": "BBB"}},
    }
    out = bench.prune_calibration(cal)
    assert set(out) == {"m:q:vulkan:new", "m:q:vulkan:other"}


def test_pruning_keeps_entries_that_cannot_be_attributed(home):
    """Pre-identity entries have no identity to prune against, so they are kept
    rather than guessed at."""
    cal = {"legacy": {"date": "2026-01-01", "measured": {"tg_tps": 30.0}},
           "a:q:vulkan:x": {"date": "2026-09-01", "hardware": {"id": "AAA"}},
           "a:q:vulkan:y": {"date": "2026-08-01", "hardware": {"id": "AAA"}}}
    out = bench.prune_calibration(cal)
    assert "legacy" in out and set(out) == {"legacy", "a:q:vulkan:x"}


def test_saving_prunes_the_file(home, fixed_identity):
    """The file is JSON a human reads, so it must not grow without bound across GPU
    swaps and driver experiments.

    The stale entry must share the identity to be pruned — a different identity is
    a different card, and its measurement is not superseded by this one.
    """
    stale = dict(fixed_identity.as_dict())
    _write({f"m:q:vulkan:{fixed_identity.digest}": {"date": "2020-01-01",
                                                    "hardware": stale}})
    bench.save_calibration(bench.calibration_key("m", "q", "vulkan"),
                           {"tg_tps": 30.0}, identity=fixed_identity)
    cal = bench.load_calibration()
    assert len(cal) == 1, cal
    assert cal[bench.calibration_key("m", "q", "vulkan")]["measured"]["tg_tps"] == 30.0


# --- the cache must not poison itself ---------------------------------------

def test_an_unreadable_identity_is_not_cached(home, monkeypatch):
    """A transient probe failure would otherwise be frozen for the life of the
    process, and every key built afterwards would silently lose its hardware
    component — the exact silent degradation this feature prevents."""
    calls = {"n": 0}

    def flaky(backend="", gpu=None):
        calls["n"] += 1
        if calls["n"] == 1:
            return hwid.HardwareIdentity(backend=backend)     # empty: a failure
        return hwid.identity_from_gpu(backend, {"vendor_id": 0x1002,
                                                "device_id": 0x7550,
                                                "device_uuid": "aa" * 16})

    monkeypatch.setattr(bench, "current_identity", flaky)
    bench._IDENT_CACHE.clear()
    assert bench._identity_cache_key("vulkan").uuid == ""      # degraded, not fatal
    assert bench._identity_cache_key("vulkan").uuid == "aa" * 16   # recovers
    assert bench._identity_cache_key("vulkan").uuid == "aa" * 16   # then cached
    assert calls["n"] == 2


def test_a_cache_miss_is_not_an_error(home, fixed_identity):
    """Invalidate and re-measure; never error. Refusing to run because a
    calibration belongs to another GPU would make the tool worse, not safer."""
    key, entry = bench.calibration_entry(bench.load_calibration(), "nope", "q", "vulkan")
    assert (key, entry) == ("", {})
    assert bench.is_calibrated("nope", "q", "vulkan") is False
