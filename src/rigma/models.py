from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

STANDARD_GB = [4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def ram_tier(mb: int) -> int:
    gb = mb / 1024
    return min(STANDARD_GB, key=lambda s: abs(s - gb))


class GpuInfo(BaseModel):
    vendor: str
    name: str
    vram_mb: int
    arch: str = "unknown"
    slug: str = "unknown"
    backends: list[str] = Field(default_factory=list)


class CpuInfo(BaseModel):
    cores: int
    name: str = ""


class HardwareProfile(BaseModel):
    gpus: list[GpuInfo]
    ram_mb: int
    ram_free_mb: int
    cpu: CpuInfo
    os: str  # "windows" | "linux" | "darwin"
    disk_free_gb: float

    @property
    def primary_gpu(self) -> GpuInfo | None:
        return max(self.gpus, key=lambda g: g.vram_mb) if self.gpus else None

    @property
    def ram_tier_gb(self) -> int:
        return ram_tier(self.ram_mb)

    @property
    def fingerprint(self) -> str:
        gpu = self.primary_gpu
        head = f"{gpu.vendor}-{gpu.slug}" if gpu else "cpu-only"
        return f"{head}/ram-{self.ram_tier_gb}/{self.os}"


class MoESpec(BaseModel):
    total_b: float
    active_b: float
    expert_weight_fraction: float


class CachePolicy(BaseModel):
    k: str = "f16"
    v: str = "f16"
    reason: str = ""


class GgufFile(BaseModel):
    repo: str
    file: str
    bytes: int
    quant: str
    sha256: str | None = None


class UseCase(BaseModel):
    name: str
    system_prompt: str
    description: str = ""


class ModelSpec(BaseModel):
    slug: str
    family: str
    kind: str  # "dense" | "moe"
    n_layers: int
    full_attn_layers: int
    kv_heads: int
    head_dim: int
    native_ctx: int
    ggufs: list[GgufFile]
    moe: MoESpec | None = None
    cache_type_policy: CachePolicy = CachePolicy()
    license: str = ""
    use_cases: list[str] = Field(default_factory=list)
    capabilities: list[str] = Field(default_factory=list)  # tools|vision|thinking
    mmproj: GgufFile | None = None    # multimodal projector (vision models)
    sources: list[str] = Field(default_factory=list)
    custom: bool = False              # user-installed (Hangar), not registry
    # model-card recommended sampling (e.g. Qwen: temp .7 + DRY for quantized
    # builds). Weakest layer: session > preset > these.
    default_params: dict[str, float] = Field(default_factory=dict)


class ComboFlags(BaseModel):
    ctx: int
    ngl: int = 99
    n_cpu_moe: int = 0
    flash_attn: str = "on"   # on | off | auto (legacy bools coerced)
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    reasoning: str = ""   # ""(engine default) | on | off | auto
    spec_type: str = "none"   # none | draft-mtp | ngram-simple | ... (engine list)
    spec_n_max: int = 3
    batch: int = 0        # -b logical batch (0 = engine default 2048)
    ubatch: int = 0       # -ub physical batch (0 = engine default 512)
    env: dict[str, str] = Field(default_factory=dict)  # engine-spawn env overrides

    @field_validator("flash_attn", mode="before")
    @classmethod
    def _coerce_fa(cls, v):
        if isinstance(v, bool):
            return "on" if v else "off"
        if v not in ("on", "off", "auto"):
            raise ValueError("flash_attn must be on, off, or auto")
        return v

    @field_validator("spec_type")
    @classmethod
    def _known_spec(cls, v):
        ok = {"none", "draft-simple", "draft-eagle3", "draft-mtp", "draft-dflash",
              "ngram-simple", "ngram-map-k", "ngram-map-k4v", "ngram-mod",
              "ngram-cache"}
        if v not in ok:
            raise ValueError(f"spec_type must be one of {sorted(ok)}")
        return v

    @model_validator(mode="after")
    def _symmetric_kv(self):
        # llama.cpp's fused flash-attn kernel only fires when ctk==ctv; a
        # mismatch SILENTLY drops to a slow non-fused path (RDNA4 finding
        # 2026-07-17). Normalize both to the more-precise type to keep the
        # fast path without silently degrading quality.
        rank = {"f16": 3, "q8_0": 2, "q4_0": 1}
        if self.cache_type_k != self.cache_type_v:
            best = self.cache_type_k if rank.get(self.cache_type_k, 0) >= \
                rank.get(self.cache_type_v, 0) else self.cache_type_v
            self.cache_type_k = self.cache_type_v = best
        return self


class Budget(BaseModel):
    vram_mb: int
    ram_mb: int


class Combo(BaseModel):
    model: str
    quant: str
    backend: str
    flags: ComboFlags
    budget: Budget | None = None
    expected: dict | None = None
    verified: dict | None = None
    notes: str = ""
    sources: list[str] = Field(default_factory=list)


class RunPlan(BaseModel):
    model_slug: str
    gguf: GgufFile
    backend: str
    flags: ComboFlags
    origin: str  # "combo:<path>" | "class:<path>" | "calculator"
    explain: list[str] = Field(default_factory=list)

    def server_args(self, model_path: str, port: int) -> list[str]:
        # --parallel 1: Rigma serves one user. llama-server defaults to 4
        # slots, each allocating a full ctx of KV cache — 4x the memory the
        # resolver budgeted, which silently overflows VRAM into system RAM.
        args = ["-m", model_path, "--port", str(port), "--host", "127.0.0.1",
                "-ngl", str(self.flags.ngl), "-c", str(self.flags.ctx),
                "--parallel", "1"]
        if self.flags.n_cpu_moe > 0:
            args += ["--n-cpu-moe", str(self.flags.n_cpu_moe)]
        if self.flags.batch > 0:
            args += ["-b", str(self.flags.batch)]
        if self.flags.ubatch > 0:
            args += ["-ub", str(self.flags.ubatch)]
        args += ["-fa", self.flags.flash_attn]
        args += ["--cache-type-k", self.flags.cache_type_k,
                 "--cache-type-v", self.flags.cache_type_v]
        if self.flags.reasoning:
            args += ["--reasoning", self.flags.reasoning]
        if self.flags.spec_type and self.flags.spec_type != "none":
            args += ["--spec-type", self.flags.spec_type,
                     "--spec-draft-n-max", str(self.flags.spec_n_max)]
        # reuse unchanged KV prefixes on edit/regenerate/compact turns
        args += ["--cache-reuse", "256"]
        return args
