from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil
import subprocess
import time

import psutil

from .models import CpuInfo, GpuInfo, HardwareProfile

VENDOR_IDS = {0x1002: "amd", 0x10DE: "nvidia", 0x8086: "intel", 0x106B: "apple"}


def _os_name() -> str:
    return {"Windows": "windows", "Linux": "linux", "Darwin": "darwin"}.get(
        platform.system(), "linux")


def _slugify(name: str, vram_mb: int) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return f"{s}-{round(vram_mb / 1024)}g"


def classify_gpu(raw: dict, gpu_table: list[dict], os_name: str) -> GpuInfo:
    vendor = VENDOR_IDS.get(raw["vendor_id"], "unknown")
    name, vram = raw["name"], raw["vram_mb"]
    for row in gpu_table:
        if row["match"].lower() in name.lower():
            backends = row.get(f"backends_{os_name}",
                               row.get("backends_windows", ["vulkan"]))
            return GpuInfo(vendor=row["vendor"], name=name, vram_mb=vram,
                           arch=row["arch"], slug=_slugify(name, vram),
                           backends=backends)
    # unknown card: pick a sane backend by vendor. A new NVIDIA card missing
    # from the table should still try CUDA (Vulkan leaves big perf on the table)
    default_backends = {"nvidia": ["cuda", "vulkan"],
                        "amd": ["vulkan"],
                        "intel": ["vulkan"]}.get(vendor, ["vulkan"])
    return GpuInfo(vendor=vendor, name=name, vram_mb=vram,
                   slug=_slugify(name, vram), backends=default_backends)


_PID_INSTANCE = re.compile(r"^pid_(\d+)_luid_", re.I)


def _sum_other_processes(samples, exclude: set[int]) -> float | None:
    """MiB of VRAM held by processes other than `exclude`.

    Windows names GPU memory counter instances `pid_<N>_luid_<hi>_<lo>_phys_<n>`.
    Anything that does not parse as one is an aggregate row, not a process, and
    double-counts if summed.

    None (not 0) when there was nothing to read: "unmeasured" and "the desktop
    is using nothing" are very different claims, and only one of them is safe
    to plan a 13GB allocation against.

    NOTE: kept for the exclude-by-pid logic, but `gpu_used_mb` no longer uses
    the per-process counter — see the comment there.
    """
    total, seen = 0.0, False
    for name, value in samples:
        m = _PID_INSTANCE.match(str(name))
        if not m:
            continue
        seen = True
        if int(m.group(1)) in exclude:
            continue
        total += float(value) / 2**20
    return total if seen else None


_PS_ADAPTER = (
    r"$s=(Get-Counter '\GPU Adapter Memory(*)\Dedicated Usage' "
    r"-EA SilentlyContinue).CounterSamples;"
    r"$t=0; foreach($x in $s){$t+=$x.CookedValue}; $t")


_VRAM_CACHE: dict[str, tuple[float, float | None]] = {}
_VRAM_TTL_S = 10.0


def gpu_used_mb() -> float | None:
    """Dedicated VRAM in use on the adapter right now, across ALL processes.

    Adapter-level on purpose. The PER-PROCESS counter is not trustworthy:
    measured 2026-08-21 on this machine, one browser instance reported
    359,777 MiB of dedicated VRAM on a 16GB card. Summing those gives a number
    that would refuse to run anything. The adapter total read 3,955 MiB at the
    same moment, which matches what the desktop actually holds.

    Why this matters at all: Windows overcommits VRAM instead of refusing an
    allocation, so a model planned against a fixed "the desktop uses 1200MB"
    assumption gets paged to system RAM with no error anywhere (measured:
    4,107 MiB paged, 29% of the weights, decode at 21% of card bandwidth).

    Every failure path returns None and the caller keeps the old constant —
    this is cheap to be wrong about, and expensive to be confidently wrong.
    """
    if _os_name() != "windows":
        return None                     # Linux path not implemented yet
    # probe_hardware runs on every /api/models request; a 1.3s PowerShell call
    # per request would cost more than the bug this fixes. The desktop's VRAM
    # footprint does not move fast enough for a stale-by-10s answer to matter.
    hit = _VRAM_CACHE.get("adapter")
    if hit and time.monotonic() - hit[0] < _VRAM_TTL_S:
        return hit[1]
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             _PS_ADAPTER],
            capture_output=True, text=True, timeout=20)
        got: float | None = float((out.stdout or "").strip()) / 2**20
    except Exception:
        got = None
    _VRAM_CACHE["adapter"] = (time.monotonic(), got)
    return got


# --- Vulkan enumeration (ctypes; no SDK needed, the ICD ships with GPU drivers) ---

_VK_MEMORY_HEAP_DEVICE_LOCAL_BIT = 0x1


class _VkAppInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("pApplicationName", ctypes.c_char_p),
                ("applicationVersion", ctypes.c_uint32),
                ("pEngineName", ctypes.c_char_p), ("engineVersion", ctypes.c_uint32),
                ("apiVersion", ctypes.c_uint32)]


class _VkInstanceCreateInfo(ctypes.Structure):
    _fields_ = [("sType", ctypes.c_int), ("pNext", ctypes.c_void_p),
                ("flags", ctypes.c_uint32), ("pApplicationInfo", ctypes.c_void_p),
                ("enabledLayerCount", ctypes.c_uint32),
                ("ppEnabledLayerNames", ctypes.c_void_p),
                ("enabledExtensionCount", ctypes.c_uint32),
                ("ppEnabledExtensionNames", ctypes.c_void_p)]


class _VkPhysicalDeviceProperties(ctypes.Structure):
    _fields_ = [("apiVersion", ctypes.c_uint32), ("driverVersion", ctypes.c_uint32),
                ("vendorID", ctypes.c_uint32), ("deviceID", ctypes.c_uint32),
                ("deviceType", ctypes.c_int), ("deviceName", ctypes.c_char * 256),
                ("pipelineCacheUUID", ctypes.c_uint8 * 16),
                ("limits", ctypes.c_uint8 * 504),
                ("sparseProperties", ctypes.c_uint8 * 20)]


class _VkMemoryHeap(ctypes.Structure):
    _fields_ = [("size", ctypes.c_uint64), ("flags", ctypes.c_uint32)]


class _VkMemoryType(ctypes.Structure):
    _fields_ = [("propertyFlags", ctypes.c_uint32), ("heapIndex", ctypes.c_uint32)]


class _VkPhysicalDeviceMemoryProperties(ctypes.Structure):
    _fields_ = [("memoryTypeCount", ctypes.c_uint32),
                ("memoryTypes", _VkMemoryType * 32),
                ("memoryHeapCount", ctypes.c_uint32),
                ("memoryHeaps", _VkMemoryHeap * 16)]


def enumerate_vulkan() -> list[dict]:
    """Enumerate GPUs via the Vulkan loader. Returns [] on any failure."""
    try:
        lib = ctypes.CDLL("vulkan-1" if _os_name() == "windows" else
                          ("libvulkan.dylib" if _os_name() == "darwin"
                           else "libvulkan.so.1"))
        app = _VkAppInfo(sType=0, pApplicationName=b"rigma", apiVersion=(1 << 22))
        info = _VkInstanceCreateInfo(sType=1, pApplicationInfo=ctypes.cast(
            ctypes.pointer(app), ctypes.c_void_p))
        inst = ctypes.c_void_p()
        if lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(inst)) != 0:
            return []
        try:
            n = ctypes.c_uint32(0)
            lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), None)
            devs = (ctypes.c_void_p * n.value)()
            lib.vkEnumeratePhysicalDevices(inst, ctypes.byref(n), devs)
            out = []
            for d in devs:
                props = _VkPhysicalDeviceProperties()
                lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(d),
                                                  ctypes.byref(props))
                mem = _VkPhysicalDeviceMemoryProperties()
                lib.vkGetPhysicalDeviceMemoryProperties(ctypes.c_void_p(d),
                                                        ctypes.byref(mem))
                local = [mem.memoryHeaps[i].size for i in range(mem.memoryHeapCount)
                         if mem.memoryHeaps[i].flags & _VK_MEMORY_HEAP_DEVICE_LOCAL_BIT]
                if props.deviceType == 4:  # VK_PHYSICAL_DEVICE_TYPE_CPU
                    continue
                out.append({"vendor_id": props.vendorID,
                            "name": props.deviceName.decode(errors="replace"),
                            "vram_mb": int(max(local, default=0) / (1024 * 1024))})
            return out
        finally:
            lib.vkDestroyInstance(inst, None)
    except Exception:
        return []


def _nvml_gpus() -> list[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        out = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h)
            if isinstance(name, bytes):
                name = name.decode()
            out.append({"vendor_id": 0x10DE, "name": name,
                        "vram_mb": int(pynvml.nvmlDeviceGetMemoryInfo(h).total / 2**20)})
        pynvml.nvmlShutdown()
        return out
    except Exception:
        return []


def probe_hardware(gpu_table: list[dict],
                   raw_gpus: list[dict] | None = None) -> HardwareProfile:
    os_name = _os_name()
    raw = raw_gpus if raw_gpus is not None else (enumerate_vulkan() or _nvml_gpus())
    vm = psutil.virtual_memory()
    return HardwareProfile(
        gpus=[classify_gpu(r, gpu_table, os_name) for r in raw],
        ram_mb=int(vm.total / 2**20),
        ram_free_mb=int(vm.available / 2**20),
        cpu=CpuInfo(cores=os.cpu_count() or 1, name=platform.processor() or ""),
        os=os_name,
        disk_free_gb=shutil.disk_usage(os.path.expanduser("~")).free / 2**30,
        vram_used_mb=gpu_used_mb(),
    )
