"""A calibration is only valid for the VRAM pressure it was measured under.

Measured 2026-08-21 on the owner's machine, same model, same quant, same engine,
same context — only the desktop's VRAM footprint differed:

    desktop 3,955 MiB   ->   9.95 tok/s   (4,107 MiB of the model paged to RAM)
    desktop 1,131 MiB   ->  37.59 tok/s   (52 MiB paged)

A 3.8x difference with nothing in the calibration key to distinguish them. The
stored 8.16 t/s then became rigma's idea of "expected", so a machine running
correctly at 37 t/s would have been judged as wildly over-performing, and one
running at 9 would have been called normal.
"""
import pytest

from rigma.bench import calibration_stale


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "home"


def _entry(**over):
    e = {"measured": {"tg_tps": 30.0}, "schema": 2, "engine": "b9867",
         "ctx": 65536, "vram_used_mb": 1100}
    e.update(over)
    return e


def test_same_pressure_is_fresh():
    assert calibration_stale(_entry(), 1150, "b9867", 65536) is None


def test_a_much_busier_desktop_invalidates_it():
    reason = calibration_stale(_entry(), 3955, "b9867", 65536)
    assert reason is not None and "VRAM" in reason


def test_a_much_emptier_desktop_invalidates_it_too():
    """Both directions. An entry measured under pressure is just as wrong once
    the pressure is gone — that is the case that actually happened."""
    reason = calibration_stale(_entry(vram_used_mb=3955), 1131, "b9867", 65536)
    assert reason is not None and "VRAM" in reason


def test_small_drift_is_tolerated():
    """The desktop moves by tens of MB constantly. Invalidating on that would
    mean never having a calibration at all."""
    assert calibration_stale(_entry(), 1300, "b9867", 65536) is None


def test_an_engine_bump_still_invalidates():
    assert calibration_stale(_entry(), 1100, "b9999", 65536) is not None


def test_entries_from_before_this_existed_are_read_leniently():
    """Old entries carry no vram_used_mb. Treating that as "0 MiB of pressure"
    would invalidate every one of them at once."""
    old = {"measured": {"tg_tps": 30.0}}
    assert calibration_stale(old, 3955, "b9867", 65536) is None


def test_an_unmeasurable_desktop_does_not_invalidate():
    """gpu_used_mb returns None off Windows. No reading is not a reading of
    zero, and must not throw away a good calibration."""
    assert calibration_stale(_entry(), None, "b9867", 65536) is None


def test_expected_tg_refuses_a_calibration_taken_under_different_pressure(
        home, monkeypatch):
    """The verdict in the engine room is only as good as this. With the stale
    8.16 t/s entry in play, a machine running correctly at 37 would have been
    reported as three times faster than expected, and one crawling at 9 would
    have been called normal."""
    import json

    from rigma import bench, probe, server_ops
    (home / "calibration.json").write_text(json.dumps({
        "m:Q3_K_M:vulkan": {"measured": {"tg_tps": 8.16}, "schema": 2,
                            "engine": bench._engine_version(),
                            "vram_used_mb": 3955},
    }), encoding="utf-8")

    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 1131.0)
    assert server_ops.expected_tg("m", "Q3_K_M", "vulkan") is None

    monkeypatch.setattr(probe, "gpu_used_mb", lambda: 3900.0)
    assert server_ops.expected_tg("m", "Q3_K_M", "vulkan") == 8.16
