from rigma.models import (
    ComboFlags,
    CpuInfo,
    GgufFile,
    GpuInfo,
    HardwareProfile,
    RunPlan,
    ram_tier,
)


def _profile(vram=16368, ram_mb=16234, os="windows"):
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="rx-9070-xt-16g", backends=["vulkan"])
    return HardwareProfile(gpus=[gpu], ram_mb=ram_mb, ram_free_mb=9100,
                           cpu=CpuInfo(cores=16), os=os, disk_free_gb=400.0)


def test_ram_tier_snaps_to_standard():
    assert ram_tier(16234) == 16
    assert ram_tier(32500) == 32
    assert ram_tier(7900) == 8


def test_fingerprint():
    assert _profile().fingerprint == "amd-rx-9070-xt-16g/ram-16/windows"


def test_primary_gpu_is_biggest_vram():
    p = _profile()
    p.gpus.append(GpuInfo(vendor="amd", name="iGPU", vram_mb=512))
    assert p.primary_gpu.name == "RX 9070 XT"


def test_server_args_full():
    plan = RunPlan(
        model_slug="qwen3.6-35b-a3b",
        gguf=GgufFile(repo="unsloth/Qwen3.6-35B-A3B-GGUF",
                      file="Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf",
                      bytes=18038862848, quant="UD-Q3_K_XL"),
        backend="vulkan",
        flags=ComboFlags(ctx=32768, n_cpu_moe=10,
                         cache_type_k="q8_0", cache_type_v="q8_0"),
        origin="combo:amd/rx-9070-xt-16g/ram-16/coding.json",
    )
    args = plan.server_args("C:/m.gguf", 11500)
    assert args[:2] == ["-m", "C:/m.gguf"]
    s = " ".join(args)
    for chunk in ("--port 11500", "-ngl 99", "-c 32768", "--n-cpu-moe 10",
                  "-fa on", "--cache-type-k q8_0", "--cache-type-v q8_0",
                  "--parallel 1"):  # single-user: 1 slot, not llama's default 4
        assert chunk in s
    bare = RunPlan(model_slug="x", gguf=plan.gguf, backend="vulkan",
                   flags=ComboFlags(ctx=8192), origin="calculator")
    assert "--n-cpu-moe" not in " ".join(bare.server_args("m", 11500))
