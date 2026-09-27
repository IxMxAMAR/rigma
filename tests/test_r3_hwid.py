"""R3-CAL-1 — keying a calibration by the hardware it was measured on.

Why the key needs hardware identity (llama.cpp CUDA scoreboard, Llama-2-7B Q4_0):
nine distinct 24 GB cards span tg128 54.74 -> 189.33, so VRAM size is not an
identity proxy; 3090 -> 4090 is pp512 +132%; and merely having 4 GPUs visible cut
a 7900 XTX's pp512 from 2,023 to 1,545. The failure that matters is the SILENT
one: a 3090 inherits a 4090's number and the UI reports a throughput the machine
cannot reach.

Why identity is NOT sufficient, which is why the hard/soft split exists: power
limits alone moved a 3090's pp512 from 4,175 to 5,406 (+29%) with no identity
change, and the same 7900 XTX varies 2.4x across driver/OS/overclock.
"""
from __future__ import annotations

import pytest

from rigma import hwid


def _amd(uuid="00000000040000000000000000000000", driver="0x800184"):
    return hwid.identity_from_gpu("vulkan", {
        "vendor_id": 0x1002, "device_id": 0x7550, "device_uuid": uuid,
        "driver_version": driver, "name": "AMD Radeon RX 9070 XT"})


# --- what identifies a card -------------------------------------------------

def test_two_cards_of_the_same_model_are_distinguished_by_uuid():
    """`deviceID` identifies the MODEL — "the same device ID should be used for all
    physical implementations of that device version" — so four 7900 XTXes report
    identically and only `deviceUUID` separates them."""
    a = _amd(uuid="aa" * 16)
    b = _amd(uuid="bb" * 16)
    assert a.device_id == b.device_id
    assert a.digest != b.digest


def test_the_driver_version_does_not_change_the_identity():
    """A driver update must not throw away a still-valid measurement. The driver is
    recorded as a FIELD and reported as a soft reason, never keyed on: it made
    non-FA PP 5% and FA 15% faster, which validation catches better than cache
    fragmentation does."""
    assert _amd(driver="0x800184").digest == _amd(driver="0x999999").digest


def test_the_marketing_name_does_not_change_the_identity():
    """`deviceName` embeds the driver ("... (RADV NAVI31)" vs "... (AMD proprietary
    driver)"), which is why Ollama has to match it heuristically. It is a label."""
    a = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                          "name": "AMD Radeon RX 9070 XT (RADV NAVI31)"})
    b = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                          "name": "AMD Radeon RX 9070 XT"})
    assert a.digest == b.digest


def test_the_vram_size_does_not_change_the_identity():
    """A card does not change identity because a browser freed memory."""
    a = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                          "vram_mb": 16304})
    b = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                          "vram_mb": 24564})
    assert a.digest == b.digest


def test_the_backend_does_change_the_identity():
    """cuda and vulkan on one card are different measurement conditions — that is
    why the backend was already in the key before identity existed."""
    assert (hwid.identity_from_gpu("cuda", {"vendor_id": 0x10DE, "device_id": 1}).digest
            != hwid.identity_from_gpu("vulkan", {"vendor_id": 0x10DE, "device_id": 1}).digest)


def test_different_cards_differ():
    assert (hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550}).digest
            != hwid.identity_from_gpu("vulkan", {"vendor_id": 0x10DE, "device_id": 0x2684}).digest)


def test_identity_is_stable_across_calls():
    assert _amd().digest == _amd().digest


# --- normalisation ----------------------------------------------------------

@pytest.mark.parametrize("a,b", [(0x7550, "0x7550"), (0x7550, "30032"),
                                 (0x7550, " 0X7550 "), ("0x7550", 30032)])
def test_one_card_cannot_produce_two_identities(a, b):
    """The probe, a saved entry and a hand-edited file can spell the same device
    three ways. If those hash differently the cache silently misses forever."""
    x = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": a})
    y = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": b})
    assert x.digest == y.digest


def test_a_missing_uuid_is_empty_not_a_placeholder():
    """An absent component must not collide with a real one, and a driver that
    never reports a UUID must still give a stable identity."""
    a = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550})
    b = hwid.identity_from_gpu("vulkan", {"vendor_id": 0x1002, "device_id": 0x7550,
                                          "device_uuid": ""})
    assert a.uuid == "" and a.digest == b.digest


def test_cpu_identity_is_the_backend_alone():
    """There is no card to identify, so keying CPU on hardware would invent a
    distinction that does not exist."""
    assert hwid.HardwareIdentity(backend="cpu").digest == \
        hwid.HardwareIdentity(backend="cpu").digest
    assert hwid.HardwareIdentity(backend="cpu").digest != \
        hwid.HardwareIdentity(backend="vulkan").digest
    assert not hwid.HardwareIdentity(backend="cpu").is_gpu


# --- the key ----------------------------------------------------------------

def test_the_key_names_the_hardware():
    k = hwid.calibration_key("qwen3-0.6b", "Q8_0", _amd())
    assert k.startswith("qwen3-0.6b:Q8_0:vulkan:")
    assert k.endswith(_amd().digest)


def test_two_cards_do_not_share_a_calibration_key():
    assert (hwid.calibration_key("m", "q", _amd(uuid="aa" * 16))
            != hwid.calibration_key("m", "q", _amd(uuid="bb" * 16)))


# --- hard vs soft -----------------------------------------------------------

def test_a_different_card_is_a_hard_mismatch():
    """The cached number is about other hardware and must not be shown at all."""
    entry = {"hardware": _amd(uuid="aa" * 16).as_dict()}
    why = hwid.hard_mismatch(entry, _amd(uuid="bb" * 16))
    assert why and "uuid" in why
    assert "moves slots" in why, "the weak component should say it is weak"


def test_a_driver_change_is_soft_not_hard():
    """Real but not a lie: re-measure, do not discard. This is the distinction the
    whole hard/soft split exists for."""
    entry = {"hardware": _amd(driver="0x800184").as_dict()}
    now = _amd(driver="0x900000")
    assert hwid.hard_mismatch(entry, now) is None
    reasons = hwid.soft_reasons(entry, now)
    assert reasons and "driver" in reasons[0]


def test_an_entry_from_before_identity_is_read_leniently():
    """Every existing calibration on every machine predates this. Invalidating them
    all at upgrade would make the tool worse for the person upgrading, which is not
    what fixing a cache key is supposed to do."""
    assert hwid.hard_mismatch({}, _amd()) is None
    assert hwid.hard_mismatch({"engine": "b9867"}, _amd()) is None
    assert hwid.soft_reasons({}, _amd()) == []


def test_a_cpu_entry_never_hard_mismatches():
    """A CPU calibration has no card, so there is nothing to be the wrong card."""
    entry = {"hardware": hwid.HardwareIdentity(backend="cpu").as_dict()}
    assert hwid.hard_mismatch(entry, hwid.HardwareIdentity(backend="cpu")) is None


def test_soft_reasons_do_not_repeat_what_hard_mismatch_already_said():
    """The two are asked in sequence by `calibration_stale`; overlapping them would
    report the same change twice."""
    entry = {"hardware": _amd().as_dict(), "engine": "b9867", "ctx": 8192}
    assert hwid.hard_mismatch(entry, _amd()) is None
    assert hwid.soft_reasons(entry, _amd(), engine="b9867", ctx=8192) == []
    assert len(hwid.soft_reasons(entry, _amd(), engine="b9999", ctx=8192)) == 1


def test_soft_reasons_report_engine_and_ctx_without_hardware():
    """The pre-identity reasons must survive: they are the ones already in use."""
    entry = {"engine": "b9867", "ctx": 8192}
    r = hwid.soft_reasons(entry, _amd(), engine="b9999", ctx=16384)
    assert any("engine" in x for x in r) and any("ctx" in x for x in r)
