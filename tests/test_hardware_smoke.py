"""Smoke tests that require the reference machine (real GPU). Excluded in CI."""
import pytest

from rigma.probe import enumerate_vulkan, probe_hardware
from rigma.registry import Registry
from rigma.resolve import resolve


@pytest.mark.hardware
def test_real_probe_finds_a_gpu():
    assert enumerate_vulkan(), "no Vulkan device found on this machine"


@pytest.mark.hardware
def test_real_plan_resolves_from_registry():
    reg = Registry.load()
    p = probe_hardware(reg.gpus)
    plan = resolve(p, reg, use_case="coding")
    assert plan.origin.startswith(("combo:", "class:", "calculator"))
    assert plan.gguf.bytes > 0
