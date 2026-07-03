from __future__ import annotations

import ctypes
import os
import platform
import re
import shutil

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
    return GpuInfo(vendor=vendor, name=name, vram_mb=vram,
                   slug=_slugify(name, vram), backends=["vulkan"])


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
    )
