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


def test_server_args_reasoning_flag():
    from rigma.models import ComboFlags, GgufFile, RunPlan
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan",
                   flags=ComboFlags(ctx=4096, reasoning="off"), origin="test")
    args = plan.server_args("model.gguf", 11499)
    i = args.index("--reasoning")
    assert args[i + 1] == "off"
    plan2 = RunPlan(model_slug="m",
                    gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                    backend="vulkan",
                    flags=ComboFlags(ctx=4096), origin="test")
    assert "--reasoning" not in plan2.server_args("model.gguf", 11499)


def test_modelspec_optional_mmproj():
    from rigma.models import GgufFile, ModelSpec
    spec = ModelSpec(slug="v", family="f", kind="dense", n_layers=2,
                     full_attn_layers=2, kv_heads=2, head_dim=64,
                     native_ctx=8192,
                     ggufs=[GgufFile(repo="r", file="m.gguf", bytes=1, quant="Q4")],
                     mmproj=GgufFile(repo="r", file="mmproj.gguf", bytes=1,
                                     quant="F16"))
    assert spec.mmproj.file == "mmproj.gguf"
    spec2 = ModelSpec(slug="t", family="f", kind="dense", n_layers=2,
                      full_attn_layers=2, kv_heads=2, head_dim=64,
                      native_ctx=8192,
                      ggufs=[GgufFile(repo="r", file="m.gguf", bytes=1,
                                      quant="Q4")])
    assert spec2.mmproj is None


def test_flash_attn_tristate_and_bool_coercion():
    from rigma.models import ComboFlags, GgufFile, RunPlan
    assert ComboFlags(ctx=1024, flash_attn=True).flash_attn == "on"
    assert ComboFlags(ctx=1024, flash_attn=False).flash_attn == "off"
    assert ComboFlags(ctx=1024, flash_attn="auto").flash_attn == "auto"
    import pytest as _p
    with _p.raises(Exception):
        ComboFlags(ctx=1024, flash_attn="sideways")
    for mode in ("on", "off", "auto"):
        plan = RunPlan(model_slug="m",
                       gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                       backend="vulkan",
                       flags=ComboFlags(ctx=1024, flash_attn=mode),
                       origin="t")
        args = plan.server_args("m.gguf", 1)
        assert args[args.index("-fa") + 1] == mode


def test_spec_decode_and_cache_reuse_args():
    from rigma.models import ComboFlags, GgufFile, RunPlan
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan",
                   flags=ComboFlags(ctx=1024, spec_type="draft-mtp",
                                    spec_n_max=4), origin="t")
    args = plan.server_args("m.gguf", 1)
    assert args[args.index("--spec-type") + 1] == "draft-mtp"
    assert args[args.index("--spec-draft-n-max") + 1] == "4"
    assert args[args.index("--cache-reuse") + 1] == "256"
    plain = RunPlan(model_slug="m",
                    gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                    backend="vulkan", flags=ComboFlags(ctx=1024), origin="t")
    a2 = plain.server_args("m.gguf", 1)
    assert "--spec-type" not in a2 and "--cache-reuse" in a2
