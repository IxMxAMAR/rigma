from __future__ import annotations

from pydantic import BaseModel, Field

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
    sources: list[str] = Field(default_factory=list)


class ComboFlags(BaseModel):
    ctx: int
    ngl: int = 99
    n_cpu_moe: int = 0
    flash_attn: bool = True
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"


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
        if self.flags.flash_attn:
            args += ["-fa", "on"]
        args += ["--cache-type-k", self.flags.cache_type_k,
                 "--cache-type-v", self.flags.cache_type_v]
        return args
