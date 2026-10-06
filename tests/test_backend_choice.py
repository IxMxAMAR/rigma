"""Choosing the compute backend, instead of always taking the first one.

`_backend` returned `gpu.backends[0]` and nothing could say otherwise, so an
RX 9070 XT — whose table row has read ["vulkan", "rocm"] all along — could only
ever run Vulkan. ROCm 7 added rocWMMA flash-attention for RDNA4 and the picture
for dense models changed, but rigma had no way to even try it (owner request,
2026-08-21).
"""
import pytest

from rigma.models import CpuInfo, GpuInfo, HardwareProfile
from rigma.resolve import ResolveError, _backend


def _prof(backends, os_name="windows"):
    return HardwareProfile(
        os=os_name, cpu=CpuInfo(cores=16), gpus=[GpuInfo(
            vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=16304,
            arch="rdna4", slug="amd-radeon-rx-9070-xt-16g", backends=backends)],
        ram_mb=32133, ram_free_mb=20000, disk_free_gb=500.0)


def test_default_is_still_the_first_backend():
    assert _backend(_prof(["vulkan", "rocm"])) == "vulkan"


def test_an_override_the_card_supports_is_honoured():
    assert _backend(_prof(["vulkan", "rocm"]), "rocm") == "rocm"


def test_an_override_the_card_does_not_support_is_refused():
    """Silently falling back would launch Vulkan while the UI said ROCm, and
    the resulting benchmark would be attributed to the wrong backend."""
    with pytest.raises(ResolveError, match="rocm"):
        _backend(_prof(["vulkan"]), "rocm")


def test_no_gpu_still_means_cpu():
    prof = HardwareProfile(os="windows", cpu=CpuInfo(cores=8), gpus=[],
                           ram_mb=8000, ram_free_mb=4000, disk_free_gb=100.0)
    assert _backend(prof) == "cpu"


def test_an_empty_override_means_no_opinion():
    """The UI sends "" for 'whatever you'd normally pick'."""
    assert _backend(_prof(["vulkan", "rocm"]), "") == "vulkan"


def test_linux_skips_a_backend_that_has_no_pinned_build():
    """Runpod is Linux-only, and that is where the card's list and the manifest
    disagree: every NVIDIA row reads ["cuda", "vulkan"] while the pinned
    manifest ships linux/vulkan and linux/cpu and NO cuda. Choosing "cuda" made
    `ensure_engine` raise (runtime.py:278-279) and cli.py's ladder end on the
    CPU floor with the GPU idle."""
    assert _backend(_prof(["cuda", "vulkan"], os_name="linux")) == "vulkan"
    # Windows keeps its CUDA build, so the preference order is untouched there.
    assert _backend(_prof(["cuda", "vulkan"], os_name="windows")) == "cuda"


def test_a_card_with_nothing_pinned_still_answers_the_first():
    """When NO listed backend has a build, the old answer stands — the error
    then names the missing engine instead of the plan silently changing."""
    assert _backend(_prof(["rocm"], os_name="linux")) == "rocm"
