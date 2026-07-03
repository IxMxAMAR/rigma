from rigma.probe import classify_gpu, probe_hardware

GPU_TABLE = [
    {"match": "RX 9070", "vendor": "amd", "arch": "rdna4",
     "backends_windows": ["vulkan", "rocm"], "backends_linux": ["vulkan", "rocm"]},
    {"match": "RTX 3060", "vendor": "nvidia", "arch": "ampere",
     "backends_windows": ["cuda", "vulkan"], "backends_linux": ["cuda", "vulkan"]},
]


def test_classify_known_amd():
    g = classify_gpu({"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT",
                      "vram_mb": 16368}, GPU_TABLE, "windows")
    assert (g.vendor, g.arch, g.slug) == ("amd", "rdna4", "amd-radeon-rx-9070-xt-16g")
    assert g.backends == ["vulkan", "rocm"]


def test_classify_unknown_falls_back():
    g = classify_gpu({"vendor_id": 0x10DE, "name": "GeForce FUTURE 9999",
                      "vram_mb": 32768}, GPU_TABLE, "linux")
    assert g.vendor == "nvidia" and g.arch == "unknown"
    assert g.backends == ["vulkan"]  # safe default when unmatched


def test_probe_with_injected_gpus():
    p = probe_hardware(GPU_TABLE, raw_gpus=[
        {"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT", "vram_mb": 16368}])
    assert p.primary_gpu.arch == "rdna4"
    assert p.ram_mb > 0 and p.cpu.cores >= 1 and p.disk_free_gb > 0
    assert p.os in ("windows", "linux", "darwin")
