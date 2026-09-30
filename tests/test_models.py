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
                  # 2 slots (user + aux) sharing ONE ctx-sized KV pool — aux
                  # calls stop evicting the conversation's prompt cache
                  "--parallel 2", "--kv-unified",
                  "--checkpoint-min-step 4096"):
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
    from pydantic import ValidationError
    with _p.raises(ValidationError, match="flash_attn"):
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


def test_server_args_answers_to_the_slug_name():
    """B7: llama-server ids the model by its gguf PATH, so an OpenAI-compatible
    client configured with the slug — the name Rigma itself writes into its
    harness configs — asks for a name the server does not answer to. `--alias`
    adds it. Flag verified at both pins: common/arg.cpp `{"-a", "--alias"},
    "STRING"`."""
    plan = RunPlan(model_slug="qwen3.6-35b-a3b",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096), origin="t")
    args = plan.server_args("m.gguf", 1)
    assert args[args.index("--alias") + 1] == "qwen3.6-35b-a3b"
    # an explicit alias still wins, for a plan served under another name
    named = RunPlan(model_slug="slug", gguf=plan.gguf, backend="vulkan",
                    flags=ComboFlags(ctx=4096, alias="rigma"), origin="t")
    nargs = named.server_args("m.gguf", 1)
    assert nargs[nargs.index("--alias") + 1] == "rigma"


def test_a_comma_in_an_explicit_alias_is_rejected():
    """B7e: `--alias` is comma-separated at both pins
    (PrismML-Eng 87268f77 common/arg.cpp:2997, ggml-org b9867 :2707:
    `string_split(value, ',')` into a `std::set<std::string>`), and the served id
    is `*model_alias.begin()` — the lexicographically first name. A hand-written
    alias with a comma therefore becomes several names and the one served is not
    the one written; an auto-slug cannot contain a comma (`hangar._slugify`), so
    only this path needs the guard."""
    import pytest as _p
    from pydantic import ValidationError

    with _p.raises(ValidationError, match="alias"):
        ComboFlags(ctx=4096, alias="rigma,other")
    # The two legal shapes still parse: empty (use the plan's slug) and a plain
    # single name (which reaches argv verbatim).
    assert ComboFlags(ctx=4096).alias == ""
    assert ComboFlags(ctx=4096, alias="rigma").alias == "rigma"


def test_the_alias_guard_is_construction_only_by_design():
    """B7E-n1: the validator runs at construction, and the source note says so.

    Pydantic's `validate_assignment` is off, so `model_copy(update=...)` and a
    plain attribute assignment reach a comma alias unvalidated. No production
    call site sets `alias` after construction, so the gap is unreachable today;
    this pins that scope and the provenance note, so enabling
    `validate_assignment` (which the note forbids) or adding a post-construction
    writer surfaces here instead of as a silently unguarded served name.
    """
    import pathlib

    import pytest as _p
    from pydantic import ValidationError

    with _p.raises(ValidationError, match="alias"):
        ComboFlags(ctx=4096, alias="a,b")
    # The documented bypass, pinned: the guard is construction-only.
    assert ComboFlags(ctx=4096).model_copy(
        update={"alias": "a,b"}).alias == "a,b"
    flags = ComboFlags(ctx=4096)
    flags.alias = "a,b"
    assert flags.alias == "a,b"
    # The provenance NOTE is the deliverable; it must stay with the validator.
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "src" / "rigma" / "models.py").read_text(encoding="utf-8")
    assert "CONSTRUCTION-ONLY (B7E-n1)" in src


def test_server_args_pins_the_deepseek_reasoning_format():
    """B7: `--reasoning-format deepseek` keeps thought tags in
    `message.reasoning_content` (streaming deltas included) instead of leaking
    them into `message.content`. Both pins already compile that default in
    (common/common.h: `reasoning_format = COMMON_REASONING_FORMAT_DEEPSEEK`),
    so this pins the wire contract against the help text's "auto" and the
    LLAMA_ARG_THINK env rather than changing today's output."""
    plan = RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096), origin="t")
    args = plan.server_args("m.gguf", 1)
    assert args[args.index("--reasoning-format") + 1] == "deepseek"
    # an explicit plan value wins, and a typo never reaches the engine
    other = RunPlan(model_slug="m", gguf=plan.gguf, backend="vulkan",
                    flags=ComboFlags(ctx=4096, reasoning_format="none"),
                    origin="t")
    oargs = other.server_args("m.gguf", 1)
    assert oargs[oargs.index("--reasoning-format") + 1] == "none"
    import pytest as _p
    from pydantic import ValidationError
    with _p.raises(ValidationError, match="reasoning_format"):
        ComboFlags(ctx=4096, reasoning_format="deepseek-legacy ")
