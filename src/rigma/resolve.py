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


def _apply_calibration(plan: RunPlan) -> RunPlan:
    from .bench import load_calibration
    key = f"{plan.model_slug}:{plan.gguf.quant}:{plan.backend}"
    entry = load_calibration().get(key)
    if entry and entry.get("flags"):
        plan.flags = plan.flags.model_copy(update=entry["flags"])
        plan.origin += "+calibrated"
        plan.explain.append(f"calibration override applied: {entry['flags']} "
                            f"(measured {entry.get('date', '?')})")
    return plan


def kv_bytes_per_token(spec: ModelSpec, k: str, v: str) -> float:
    per_side = spec.full_attn_layers * spec.kv_heads * spec.head_dim
    return per_side * CACHE_BYTES[k] + per_side * CACHE_BYTES[v]


def _budgets(profile: HardwareProfile) -> tuple[float, float]:
    # llama.cpp splits tensors across all GPUs, so budget the SUM of their
    # VRAM (reserving per-card overhead), not just the primary
    gpus = profile.gpus or []
    total_vram = sum(g.vram_mb for g in gpus)
    reserve = VRAM_RESERVE_MB[profile.os] * max(1, len(gpus)) + COMPUTE_BUFFER_MB
    vram = total_vram - reserve
    return max(vram, 0), max(profile.ram_free_mb - RAM_RESERVE_MB, 0)


def fit_gguf(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
             ctx: int, explain: list[str]) -> ComboFlags | None:
    usable_vram, usable_ram = _budgets(profile)
    k, v = spec.cache_type_policy.k, spec.cache_type_policy.v
    file_mb = gguf.bytes / 2**20
    # vision projector loads alongside the weights and can't be offloaded;
    # it counts against VRAM but not the MoE expert math below
    mm_mb = spec.mmproj.bytes / 2**20 if spec.mmproj else 0.0
    kv_mb = ctx * kv_bytes_per_token(spec, k, v) / 2**20
    explain.append(f"{gguf.quant}@ctx{ctx}: file={file_mb:.0f}MB kv={kv_mb:.0f}MB "
                   + (f"mmproj={mm_mb:.0f}MB " if mm_mb else "")
                   + f"vs vram={usable_vram:.0f}MB ram={usable_ram:.0f}MB")
    if spec.moe is None:
        if file_mb + mm_mb + kv_mb <= usable_vram:
            return ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=v)
        # partial offload: keep as many layers on the GPU as fit, spill the
        # rest to system RAM. A 24B dense model on 16GB runs at ~15 t/s this
        # way instead of returning "doesn't fit" and dropping to CPU/a smaller
        # model. mmproj + kv must stay on the GPU, so they eat the VRAM budget.
        if spec.n_layers <= 0:
            return None
        per_layer = file_mb / spec.n_layers
        gpu_room = usable_vram - mm_mb - kv_mb
        n_gpu = int(gpu_room // per_layer) if per_layer else 0
        if n_gpu <= 0:
            return None                      # not even one layer + kv fits
        n_gpu = min(n_gpu, spec.n_layers)
        spilled = (spec.n_layers - n_gpu) * per_layer
        if spilled > usable_ram:
            return None
        explain.append(f"dense partial offload: {n_gpu}/{spec.n_layers} layers "
                       f"on GPU ({spilled:.0f}MB to RAM)")
        return ComboFlags(ctx=ctx, ngl=n_gpu, cache_type_k=k, cache_type_v=v)
    expert_mb = file_mb * spec.moe.expert_weight_fraction
    per_layer = expert_mb / spec.n_layers
    need_off = max(0.0, file_mb + mm_mb + kv_mb - usable_vram)
    n_off = math.ceil(need_off / per_layer) if need_off else 0
    if n_off <= spec.n_layers and n_off * per_layer <= usable_ram:
        return ComboFlags(ctx=ctx, n_cpu_moe=n_off, cache_type_k=k, cache_type_v=v)
    return None


def _backend(profile: HardwareProfile) -> str:
    gpu = profile.primary_gpu
    return gpu.backends[0] if gpu and gpu.backends else "cpu"


def _grow_ctx(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
              flags: ComboFlags, explain: list[str]) -> ComboFlags:
    """Calculator plans only: double ctx while it still fits, up to native.

    CTX_DEFAULT is a starting probe, not a ceiling (owner finding 2026-07-16:
    the old cap silently wasted VRAM that could hold 4-8x more context)."""
    best = flags
    ctx = best.ctx * 2
    while ctx <= spec.native_ctx:
        grown = fit_gguf(spec, gguf, profile, ctx, explain)
        if grown is None:
            break
        # never trade decode speed for context: stop if the larger window
        # forces more CPU offload (MoE) or fewer GPU layers (dense)
        if grown.n_cpu_moe > best.n_cpu_moe or grown.ngl < best.ngl:
            explain.append(f"grow-to-fit: stop at ctx {best.ctx} "
                           f"(ctx {ctx} would force more offload)")
            break
        explain.append(f"grow-to-fit: ctx {best.ctx} -> {ctx} "
                       f"(n_cpu_moe {best.n_cpu_moe} -> {grown.n_cpu_moe})")
        best = grown
        ctx *= 2
    return best


def _calculate(profile: HardwareProfile, registry: Registry,
               use_case: str) -> RunPlan | None:
    explain: list[str] = []
    pool = [m for m in registry.models.values() if use_case in m.use_cases] or \
        list(registry.models.values())
    if use_case == "coding":
        tooled = [m for m in pool if "tools" in m.capabilities]
        if tooled:
            explain.append("coding: restricting to tools-capable models "
                           f"({', '.join(m.slug for m in tooled)})")
            pool = tooled
        else:
            explain.append("coding: WARNING - no tools-capable model available; "
                           "agent tool calling will not work")

    def total_b(m: ModelSpec) -> float:
        # capability proxy = largest gguf size. n_layers was WRONG here: an 8B
        # with 36 layers outranked the 35B MoE (regression caught 2026-07-16)
        return max((g.bytes for g in m.ggufs), default=0)

    for spec in sorted(pool, key=total_b, reverse=True):
        for gguf in spec.ggufs:  # registry order: largest quant first
            ctx = min(CTX_DEFAULT.get(use_case, 16384), spec.native_ctx)
            while ctx >= CTX_FLOOR:
                flags = fit_gguf(spec, gguf, profile, ctx, explain)
                if flags:
                    flags = _grow_ctx(spec, gguf, profile, flags, explain)
                    return RunPlan(model_slug=spec.slug, gguf=gguf,
                                   backend=_backend(profile), flags=flags,
                                   origin="calculator", explain=explain)
                ctx //= 2
    return None


def fallback_plans(plan: RunPlan, registry: Registry,
                   profile: HardwareProfile) -> list[RunPlan]:
    out: list[RunPlan] = []
    spec = registry.models.get(plan.model_slug)
    if spec is not None:
        smaller = [g for g in spec.ggufs if g.bytes < plan.gguf.bytes]
        for gguf in smaller:  # registry order: largest first
            explain = [f"fallback: {plan.gguf.quant} failed to launch"]
            ctx = plan.flags.ctx
            flags = None
            while ctx >= CTX_FLOOR and flags is None:
                flags = fit_gguf(spec, gguf, profile, ctx, explain)
                if flags is None:
                    ctx //= 2
            if flags is not None:
                out.append(_apply_calibration(RunPlan(
                    model_slug=spec.slug, gguf=gguf, backend=plan.backend,
                    flags=flags, origin="fallback", explain=explain)))
    have_ggufs = [m for m in registry.models.values() if m.ggufs]
    if have_ggufs:
        floor_spec = min(have_ggufs, key=lambda m: m.ggufs[-1].bytes)
        if (floor_spec.slug, floor_spec.ggufs[-1].quant) != (plan.model_slug,
                                                             plan.gguf.quant):
            out.append(RunPlan(
                model_slug=floor_spec.slug, gguf=floor_spec.ggufs[-1],
                backend="cpu", flags=ComboFlags(ctx=CTX_FLOOR, ngl=0),
                origin="fallback:floor",
                explain=["fallback floor: smallest model on CPU"]))
    return out


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
            return _apply_calibration(RunPlan(
                model_slug=combo.model, gguf=gguf, backend=combo.backend,
                flags=combo.flags, origin=f"{kind}:{rel}",
                explain=[f"registry match: {rel}"] + combo.sources))
    if model_override:
        if model_override not in registry.models:
            raise ResolveError(
                f"unknown model: {model_override} — run `rigma update` "
                f"(your cached registry may predate it)")
        registry = Registry(registry.gpus,
                            {model_override: registry.models[model_override]},
                            registry.combos)
    plan = _calculate(profile, registry, use_case)
    if plan:
        return _apply_calibration(plan)
    # absolute floor: smallest model, smallest quant, CPU
    have_ggufs = [m for m in registry.models.values() if m.ggufs]
    if not have_ggufs:
        raise ResolveError("no model in the registry has a gguf to run")
    spec = min(have_ggufs, key=lambda m: m.ggufs[-1].bytes)
    return _apply_calibration(RunPlan(
        model_slug=spec.slug, gguf=spec.ggufs[-1], backend="cpu",
        flags=ComboFlags(ctx=CTX_FLOOR, ngl=0), origin="calculator",
        explain=["floor: nothing larger fits"]))
