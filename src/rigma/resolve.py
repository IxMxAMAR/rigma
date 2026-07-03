from __future__ import annotations

import math

from .models import ComboFlags, GgufFile, HardwareProfile, ModelSpec, RunPlan
from .registry import Registry

VRAM_RESERVE_MB = {"windows": 1200, "linux": 400, "darwin": 0}
RAM_RESERVE_MB = 2048
COMPUTE_BUFFER_MB = 900
CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q4_0": 0.5625}
CTX_DEFAULT = {"coding": 32768}
CTX_FLOOR = 8192


class ResolveError(RuntimeError):
    pass


def kv_bytes_per_token(spec: ModelSpec, k: str, v: str) -> float:
    per_side = spec.full_attn_layers * spec.kv_heads * spec.head_dim
    return per_side * CACHE_BYTES[k] + per_side * CACHE_BYTES[v]


def _budgets(profile: HardwareProfile) -> tuple[float, float]:
    gpu = profile.primary_gpu
    vram = (gpu.vram_mb if gpu else 0) - VRAM_RESERVE_MB[profile.os] - COMPUTE_BUFFER_MB
    return max(vram, 0), max(profile.ram_free_mb - RAM_RESERVE_MB, 0)


def fit_gguf(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
             ctx: int, explain: list[str]) -> ComboFlags | None:
    usable_vram, usable_ram = _budgets(profile)
    k, v = spec.cache_type_policy.k, spec.cache_type_policy.v
    file_mb = gguf.bytes / 2**20
    kv_mb = ctx * kv_bytes_per_token(spec, k, v) / 2**20
    explain.append(f"{gguf.quant}@ctx{ctx}: file={file_mb:.0f}MB kv={kv_mb:.0f}MB "
                   f"vs vram={usable_vram:.0f}MB ram={usable_ram:.0f}MB")
    if spec.moe is None:
        if file_mb + kv_mb <= usable_vram:
            return ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=v)
        return None
    expert_mb = file_mb * spec.moe.expert_weight_fraction
    per_layer = expert_mb / spec.n_layers
    need_off = max(0.0, file_mb + kv_mb - usable_vram)
    n_off = math.ceil(need_off / per_layer) if need_off else 0
    if n_off <= spec.n_layers and n_off * per_layer <= usable_ram:
        return ComboFlags(ctx=ctx, n_cpu_moe=n_off, cache_type_k=k, cache_type_v=v)
    return None


def _backend(profile: HardwareProfile) -> str:
    gpu = profile.primary_gpu
    return gpu.backends[0] if gpu and gpu.backends else "cpu"


def _calculate(profile: HardwareProfile, registry: Registry,
               use_case: str) -> RunPlan | None:
    explain: list[str] = []
    pool = [m for m in registry.models.values() if use_case in m.use_cases] or \
        list(registry.models.values())

    def total_b(m: ModelSpec) -> float:
        return m.moe.total_b if m.moe else m.n_layers

    for spec in sorted(pool, key=total_b, reverse=True):
        for gguf in spec.ggufs:  # registry order: largest quant first
            ctx = min(CTX_DEFAULT.get(use_case, 16384), spec.native_ctx)
            while ctx >= CTX_FLOOR:
                flags = fit_gguf(spec, gguf, profile, ctx, explain)
                if flags:
                    return RunPlan(model_slug=spec.slug, gguf=gguf,
                                   backend=_backend(profile), flags=flags,
                                   origin="calculator", explain=explain)
                ctx //= 2
    return None


def resolve(profile: HardwareProfile, registry: Registry,
            use_case: str = "general", model_override: str | None = None) -> RunPlan:
    if not registry.models:
        raise ResolveError("registry has no models")
    gpu = profile.primary_gpu
    if gpu and model_override is None:
        hit = registry.find_combo(gpu.vendor, gpu.slug, round(gpu.vram_mb / 1024),
                                  profile.ram_tier_gb, use_case)
        if hit:
            combo, rel = hit
            spec = registry.models[combo.model]
            gguf = next(g for g in spec.ggufs if g.quant == combo.quant)
            kind = "class" if rel.startswith("_class/") else "combo"
            return RunPlan(model_slug=combo.model, gguf=gguf, backend=combo.backend,
                           flags=combo.flags, origin=f"{kind}:{rel}",
                           explain=[f"registry match: {rel}"] + combo.sources)
    if model_override:
        registry = Registry(registry.gpus,
                            {model_override: registry.models[model_override]},
                            registry.combos)
    plan = _calculate(profile, registry, use_case)
    if plan:
        return plan
    # absolute floor: smallest model, smallest quant, CPU
    spec = min(registry.models.values(), key=lambda m: m.ggufs[-1].bytes)
    return RunPlan(model_slug=spec.slug, gguf=spec.ggufs[-1], backend="cpu",
                   flags=ComboFlags(ctx=CTX_FLOOR, ngl=0), origin="calculator",
                   explain=["floor: nothing larger fits"])
