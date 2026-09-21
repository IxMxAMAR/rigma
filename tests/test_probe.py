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
    # unknown NVIDIA still prefers CUDA — Vulkan-only left big perf on the table
    assert g.backends == ["cuda", "vulkan"]


def test_probe_with_injected_gpus():
    p = probe_hardware(GPU_TABLE, raw_gpus=[
        {"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT", "vram_mb": 16368}])
    assert p.primary_gpu.arch == "rdna4"
    assert p.ram_mb > 0 and p.cpu.cores >= 1 and p.disk_free_gb > 0
    assert p.os in ("windows", "linux", "darwin")


# --- AUDIT F56: the Vulkan properties struct must match the C ABI ------------
#
# This needs NO Vulkan device, which is the point. `vkGetPhysicalDeviceProperties`
# writes sizeof(VkPhysicalDeviceProperties) bytes into whatever buffer it is
# handed, so a struct that is 8 bytes short is an 8-byte overrun on EVERY
# hardware probe. It never crashed because a release allocator absorbs those bytes
# into malloc rounding slack, and the fields Rigma reads all sit below the damage
# — so neither a wrong answer nor a crash would ever have reported it.

def test_the_vulkan_properties_struct_matches_the_c_abi():
    import ctypes

    from rigma import probe

    s = probe._VkPhysicalDeviceProperties
    assert ctypes.sizeof(s) == 824, (
        "the driver writes sizeof(VkPhysicalDeviceProperties) bytes; a smaller "
        "struct is an overrun and a larger one shifts every field after it")
    # `limits` is a VkPhysicalDeviceLimits, whose first member is 64-bit, so the
    # C ABI puts it at 296 with 4 bytes of padding after the UUID. Declaring it
    # as a byte array gave ctypes alignment 1 and landed it at 292.
    assert s.limits.offset == 296
    assert s.sparseProperties.offset == 800
    # The fields Rigma actually reads, which is why the bug was invisible.
    assert s.vendorID.offset == 8
    assert s.deviceType.offset == 16
    assert s.deviceName.offset == 20

