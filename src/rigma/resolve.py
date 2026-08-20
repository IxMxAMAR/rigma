from __future__ import annotations

import math

from .models import ComboFlags, GgufFile, HardwareProfile, ModelSpec, RunPlan
from .registry import Registry

VRAM_RESERVE_MB = {"windows": 1200, "linux": 400, "darwin": 0}
RAM_RESERVE_MB = 2048
# llama.cpp's own scratch allocation, on top of weights and KV. Measured
# 2026-08-21 on a dense 27B at ubatch 512, across four fully-resident rungs:
# 43, 44, 55 and 81 MiB. The old 900 was a guess made when the Windows desktop
# reserve was also a guess, and the two were covering for each other; with the
# desktop now measured, this can be an estimate of the thing it names. Still
# ~5x the largest observation, because MoE and larger batches allocate more.
COMPUTE_BUFFER_MB = 400
CACHE_BYTES = {"f16": 2.0, "q8_0": 1.0625, "q5_1": 0.75, "q4_0": 0.5625}
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


def _budgets(profile: HardwareProfile,
             other_vram_mb: float | None = None) -> tuple[float, float]:
    """Usable VRAM and RAM for a plan.

    `other_vram_mb` is what the desktop is measured to be holding. Without it
    the reserve is a constant, and on Windows that constant was a fiction:
    1200MB budgeted against 3,955MB actually held by a browser and an editor.
    Windows does not refuse the resulting overcommit — it pages the difference
    to system RAM, so the plan reports 0% offload while 29% of the weights ride
    PCIe on every token (measured 2026-08-21, decode at 21% of card bandwidth).

    The constant stays a FLOOR. A measurement may only make the budget
    smaller, never larger: it is a snapshot, the user can open something a
    second later, and being optimistic here is what caused the bug.
    """
    # llama.cpp splits tensors across all GPUs, so budget the SUM of their
    # VRAM (reserving per-card overhead), not just the primary
    gpus = profile.gpus or []
    total_vram = sum(g.vram_mb for g in gpus)
    floor = VRAM_RESERVE_MB[profile.os] * max(1, len(gpus))
    # A reading at or above the card's own capacity is a broken counter, not a
    # busy desktop (Windows' per-process counter once reported 359,777MB on a
    # 16GB card). Discard it rather than budget zero.
    measured = (profile.vram_used_mb if other_vram_mb is None
                else other_vram_mb) or 0.0
    if measured >= total_vram:
        measured = 0.0
    reserve = max(floor, measured) + COMPUTE_BUFFER_MB
    vram = total_vram - reserve
    return max(vram, 0), max(profile.ram_free_mb - RAM_RESERVE_MB, 0)


def fit_gguf(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
             ctx: int, explain: list[str]) -> ComboFlags | None:
    usable_vram, usable_ram = _budgets(profile)
    # Two passes, and the order matters: try EVERY cache type fully on the GPU
    # before letting any of them spill weights to RAM. Quantising the cache
    # costs ~8.5 effective bits; pushing layers (or experts) to system RAM costs
    # real tokens/sec. Doing this in one pass picked f16-with-offload over
    # q8_0-fully-resident, which is strictly the worse trade.
    for strict in (True, False):
        best = None
        for k, v in _cache_candidates(spec):
            got = _fit_with_cache(spec, gguf, profile, ctx, k, v,
                                  usable_vram, usable_ram, explain, strict)
            if got is None:
                continue
            if strict:
                return got      # fully resident: the most precise cache wins
            # Offloading is unavoidable at this ctx, so now the cache's job is
            # to MINIMISE it. Taking the first that merely fits picked f16 and
            # spilled 22% of a dense model's weights, where q8_0 spills 8% —
            # ~0.06% perplexity against a PCIe round trip on every token, for
            # every offloaded layer. The pass above already guaranteed nothing
            # here can be fully resident, so precision is no longer free.
            if best is None or _spilled(spec, got) < _spilled(spec, best):
                best = got
        if best is not None:
            if best.cache_type_k != spec.cache_type_policy.k:
                explain.append(
                    f"cache {best.cache_type_k} over "
                    f"{spec.cache_type_policy.k}: keeps "
                    f"{_spilled(spec, best):.0%} of the weights off system RAM")
            return best
    return None


def _spilled(spec: ModelSpec, flags: ComboFlags) -> float:
    """Fraction of the model left in system RAM under this plan. Dense counts
    layers; MoE counts only the expert share of an offloaded layer, since
    sparse activation makes that far cheaper."""
    n = spec.n_layers or 0
    if not n:
        return 0.0
    if spec.moe is None:
        return max(0, n - min(flags.ngl, n)) / n
    return (min(flags.n_cpu_moe, n) / n) * spec.moe.expert_weight_fraction


def _cache_candidates(spec: ModelSpec):
    """Cache types to try, best quality first.

    The policy default (f16) is tried first, then q8_0. q8_0 stores 32 values as
    int8 plus one f16 scale — 1.0625 bytes/element vs 2.0, so it HALVES the KV
    cache for ~8.5 effective bits. That is far more precision than the weights
    themselves carry (Q6_K ~6.5 bits, IQ3_M ~3.5), so it is not the accuracy
    bottleneck — but dropping context to 8K to protect it very much is a real
    cost. Trying it before giving up context is close to free."""
    k, v = spec.cache_type_policy.k, spec.cache_type_policy.v
    if spec.cache_type_policy.pinned:
        return [(k, v)]          # explorer: answer the question that was asked
    out = [(k, v)]
    # Down to q5_1 and q4_0 before giving up and spilling weights. The ladder
    # used to stop at q8_0, so a 27B at 64K "did not fit" and nine of its
    # sixty-four layers went to the CPU — while q5_1 fit on the GPU with room
    # to spare and ran at 38.27 tok/s. Measured on the same machine the same
    # day, offloading half a gigabyte cost 60% of throughput (32.47 -> 15.86
    # tok/s) and 80% of prefill. One more step of cache quantisation costs a
    # fraction of a percent of perplexity. The trade is not close.
    for step in (("q8_0", "q8_0"), ("q5_1", "q5_1"), ("q4_0", "q4_0")):
        if step != (k, v):
            out.append(step)
    return out


def _fit_with_cache(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
                    ctx: int, k: str, v: str, usable_vram: float,
                    usable_ram: float, explain: list[str],
                    strict: bool = False) -> ComboFlags | None:
    file_mb = gguf.bytes / 2**20
    # vision projector loads alongside the weights and can't be offloaded;
    # it counts against VRAM but not the MoE expert math below
    mm_mb = spec.mmproj.bytes / 2**20 if spec.mmproj else 0.0
    kv_mb = ctx * kv_bytes_per_token(spec, k, v) / 2**20
    explain.append(f"{gguf.quant}@ctx{ctx} kv={k}: file={file_mb:.0f}MB kv={kv_mb:.0f}MB "
                   + (f"mmproj={mm_mb:.0f}MB " if mm_mb else "")
                   + f"vs vram={usable_vram:.0f}MB ram={usable_ram:.0f}MB")
    if spec.moe is None:
        if file_mb + mm_mb + kv_mb <= usable_vram:
            return ComboFlags(ctx=ctx, cache_type_k=k, cache_type_v=v)
        if strict:
            return None            # try the next cache type before offloading
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
    if strict and file_mb + mm_mb + kv_mb > usable_vram:
        return None                # ditto for MoE expert offload
    expert_mb = file_mb * spec.moe.expert_weight_fraction
    per_layer = expert_mb / spec.n_layers
    need_off = max(0.0, file_mb + mm_mb + kv_mb - usable_vram)
    n_off = math.ceil(need_off / per_layer) if need_off else 0
    if n_off <= spec.n_layers and n_off * per_layer <= usable_ram:
        return ComboFlags(ctx=ctx, n_cpu_moe=n_off, cache_type_k=k, cache_type_v=v)
    return None


def _backend(profile: HardwareProfile, override: str | None = None) -> str:
    """Which compute backend to launch on.

    The card's table row lists what it CAN run, best-first, and taking [0] was
    the whole selection policy — so an RX 9070 XT, whose row has read
    ["vulkan", "rocm"] since it was added, could never run ROCm. ROCm 7 brought
    rocWMMA flash-attention to RDNA4 and the answer for dense models is now an
    open question, which cannot be measured without being selectable first
    (owner request, 2026-08-21).

    An override the card does not list is an ERROR, never a quiet fallback:
    launching Vulkan while the UI says ROCm would file the resulting benchmark
    under the wrong backend, which is worse than not running it.
    """
    gpu = profile.primary_gpu
    if not gpu or not gpu.backends:
        return "cpu"
    if not override:
        return gpu.backends[0]
    if override not in gpu.backends:
        raise ResolveError(
            f"{gpu.name} cannot run {override} on {profile.os} — it supports "
            f"{', '.join(gpu.backends)}")
    return override


def _grow_ctx(spec: ModelSpec, gguf: GgufFile, profile: HardwareProfile,
              flags: ComboFlags, explain: list[str],
              layer_budget: float = 0.0) -> ComboFlags:
    """Calculator plans only: double ctx while it still fits, up to native.

    CTX_DEFAULT is a starting probe, not a ceiling (owner finding 2026-07-16:
    the old cap silently wasted VRAM that could hold 4-8x more context)."""
    best = start = flags
    ctx = best.ctx * 2
    while ctx <= spec.native_ctx:
        grown = fit_gguf(spec, gguf, profile, ctx, explain)
        if grown is None:
            break
        # dense: fewer GPU layers is a real per-token cost, so stop before the
        # larger window forces weights off the GPU. MoE: expert offload is
        # cheap (sparse activation) and users want the context — grow to native
        # as long as it still fits (the 35B is verified healthy at 262K ctx
        # with expert offload: 56.7 t/s). Guarding MoE here capped qwen3.6 at
        # 32K when it runs fine at 256K (owner, 2026-07-18).
        # layer_budget > 0 buys the doubling anyway, up to that share of the
        # layers. The default 0.0 is the historical rule: never trade a layer.
        # It was set when a KV cache cost 4x what a hybrid-attention model's
        # actually does, and it silently reports the smaller window — UD-Q3_K_XL
        # missed 16K by 46MB and simply displayed 8K (owner, 2026-07-30).
        #
        # Measured against `start`, not the previous step: a per-doubling budget
        # is spent again at every rung, so 8K->64K quietly offloaded 26 layers
        # on a 15% allowance. And ngl is a SENTINEL (99 = "all"), so it has to
        # be clamped to n_layers before differencing or "all -> 62 of 65" reads
        # as 37 layers lost instead of 3.
        cap = spec.n_layers or 99
        lost = max(0, min(start.ngl, cap) - min(grown.ngl, cap))
        allowed = int(cap * layer_budget)
        if (lost > allowed) or (spec.moe is None
                                and grown.n_cpu_moe > best.n_cpu_moe):
            explain.append(f"grow-to-fit: stop at ctx {best.ctx} "
                           f"(ctx {ctx} would push weights off the GPU)")
            break
        if lost:
            explain.append(f"grow-to-fit: spending {lost} of {cap} GPU layers "
                           f"to reach ctx {ctx} (budget {allowed})")
        explain.append(f"grow-to-fit: ctx {best.ctx} -> {ctx} "
                       f"(n_cpu_moe {best.n_cpu_moe} -> {grown.n_cpu_moe})")
        best = grown
        ctx *= 2
    return best


def quant_verdicts(spec: ModelSpec, profile: HardwareProfile, *,
                   kv: str = "", vision: bool = True,
                   grow: str = "speed") -> list[dict]:
    """Per-quant verdicts, with the three things that were fixed constants.

    Every number this returned was computed under one hidden configuration —
    the spec's own cache policy, the vision projector always resident, and a
    growth rule that never trades a GPU layer for context. Each of those is a
    real choice with a real cost, and presenting one of them as the answer made
    the page read as a wall (owner, 2026-07-30).

      kv      "" keeps the spec's policy ladder; or force f16/q8_0/q5_1/q4_0,
              or "k,v" for an asymmetric pair such as "q8_0,q4_0".
      vision  False drops the mmproj. It is permanently resident and counted
              whether or not an image is ever sent — 888MB on Qwen3.8-27B,
              which is 4x the context on a 16GB card.
      grow    "speed" stops growing context at the first GPU layer it would
              cost (the long-standing default). "context" spends up to
              _GROW_LAYER_BUDGET of the layers for each doubling.
    ONE implementation, deliberately: this ran only inside hf_browse's
    pre-download browser, so the Models page — the page you actually pick a
    quant from — showed a bare size and nothing else. Two copies of this
    arithmetic would drift, and a fit verdict that disagrees with itself
    between two screens is worse than none.
    """
    spec = _configured(spec, kv=kv, vision=vision)
    usable_vram, _ = _budgets(profile)
    mm_mb = spec.mmproj.bytes / 2**20 if spec.mmproj else 0.0
    out = []
    for g in spec.ggufs:
        flags = None
        for ctx in (8192, 4096, 2048):
            flags = fit_gguf(spec, g, profile, ctx, [])
            if flags:
                flags = _grow_ctx(spec, g, profile, flags, [],
                                  layer_budget=(_GROW_LAYER_BUDGET
                                                if grow == "context" else 0.0))
                break
        if flags is None:
            out.append({"ok": False, "speed": "no", "offload_pct": 100,
                        "budget": _budget_rows(spec, g, mm_mb, 8192,
                                               usable_vram)})
            continue
        # The spill fraction comes from the PLAN the resolver actually made —
        # ngl for dense, n_cpu_moe for MoE — not from comparing the file to
        # VRAM. The old file-size guess called a quant "gpu" while the very
        # same verdict carried ngl=56 of 65 layers: it ignored the KV cache,
        # which is precisely what a big context window spends VRAM on.
        spill = 0.0
        if spec.moe is None:
            if spec.n_layers > 0 and flags.ngl < spec.n_layers:
                spill = (spec.n_layers - max(0, flags.ngl)) / spec.n_layers
        elif spec.n_layers > 0 and flags.n_cpu_moe > 0:
            # only the EXPERT weights of those layers leave the GPU, and expert
            # activation is sparse, so the same fraction costs far less here
            spill = (min(flags.n_cpu_moe, spec.n_layers) / spec.n_layers
                     * spec.moe.expert_weight_fraction)
        speed = "gpu" if spill <= 0.001 else ("light" if spill <= 0.15
                                              else "offload")
        out.append({"ok": True, "ctx": flags.ctx, "n_cpu_moe": flags.n_cpu_moe,
                    "ngl": flags.ngl, "kv": flags.cache_type_k,
                    "kv_v": flags.cache_type_v,
                    "offload_pct": round(spill * 100), "speed": speed,
                    "budget": _budget_rows(spec, g, mm_mb, flags.ctx,
                                           usable_vram,
                                           flags.cache_type_k,
                                           flags.cache_type_v)})
    return out


# How much of the model the "context" growth policy may push off the GPU for
# each doubling of the window. 0.15 = up to 15% of the layers; the "speed"
# policy uses 0.0, which is the historical behaviour (never trade a layer).
_GROW_LAYER_BUDGET = 0.15


def _configured(spec: ModelSpec, *, kv: str = "",
                vision: bool = True) -> ModelSpec:
    """A copy of `spec` with the cache policy and vision projector overridden.

    A copy, not a mutation: these come from a query string and must not leak
    into the registry object the rest of the process is sharing."""
    if not kv and vision:
        return spec
    s = spec.model_copy(deep=True)
    if kv:
        # ONE type: K and V always match (ComboFlags._symmetric_kv — llama.cpp's
        # fused flash-attention needs ctk == ctv). A "k,v" pair used to be
        # parsed here and then silently normalised downstream, so an asymmetric
        # request looked honoured and wasn't. The API rejects that form now;
        # this only has to not crash on one.
        k = kv.split(",", 1)[0].strip()
        if k:
            s.cache_type_policy.k = s.cache_type_policy.v = k
        # asked for explicitly, so do not quietly fall back to q8_0 — that made
        # f16 / q5_1 / q4_0 all report the identical verdict
        s.cache_type_policy.pinned = True
    if not vision:
        s.mmproj = None
    return s


def _budget_rows(spec: ModelSpec, gguf: GgufFile, mm_mb: float, ctx: int,
                 usable_vram: float, k: str = "", v: str = "") -> dict:
    """Where the VRAM actually goes, in MB. The whole point of the explorer is
    that a single "8K" tells you nothing about WHY — this is the arithmetic
    behind it, so a 46MB near-miss reads as a near-miss."""
    k = k or spec.cache_type_policy.k
    v = v or spec.cache_type_policy.v
    kv_mb = ctx * kv_bytes_per_token(spec, k, v) / 2**20
    file_mb = gguf.bytes / 2**20
    return {"file_mb": round(file_mb), "mmproj_mb": round(mm_mb),
            "kv_mb": round(kv_mb), "budget_mb": round(usable_vram),
            "over_mb": round(file_mb + mm_mb + kv_mb - usable_vram),
            "ctx": ctx, "kv_type": k}


def recommended_quant(quants: list[dict]) -> str | None:
    """Best QUALITY that still runs at GPU speed. `quants` are largest-first, so
    quality descends down the list; prefer a quant that fits on the GPU (or only
    lightly offloads) over a bigger one that spills to RAM and crawls."""
    fast = [q for q in quants
            if q["fit"].get("ok") and q["fit"].get("speed") in ("gpu", "light")]
    if fast:
        return fast[0]["quant"]      # largest fast-enough = best quality @ speed
    fits = [q for q in quants if q["fit"].get("ok")]
    return fits[-1]["quant"] if fits else None   # else the least-offloaded


def _calculate(profile: HardwareProfile, registry: Registry,
               use_case: str, backend: str | None = None) -> RunPlan | None:
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

    def _on_disk(g: GgufFile) -> bool:
        from .runtime import rigma_home
        try:
            return (rigma_home() / "models" / g.file).exists()
        except Exception:
            return False

    for spec in sorted(pool, key=total_b, reverse=True):
        # LOCALITY BEFORE QUALITY. Live 2026-07-20: `up -y` on a model whose
        # 14.4GB quant sat on disk silently began downloading the 26GB quant,
        # because registry order (largest first) was the only order. Rule: any
        # on-disk quant that fits beats any remote one; remote quants are only
        # considered when nothing local fits. Cold-starting a fresh machine
        # still works — the local list is empty and the order degenerates to
        # the old one.
        local = [g for g in spec.ggufs if _on_disk(g)]
        ordered = local + [g for g in spec.ggufs if g not in local]
        if local:
            explain.append(f"{spec.slug}: preferring on-disk quant(s) "
                           f"{', '.join(g.quant for g in local)}")
        for gguf in ordered:  # on-disk first, then registry order
            ctx = min(CTX_DEFAULT.get(use_case, 16384), spec.native_ctx)
            while ctx >= CTX_FLOOR:
                flags = fit_gguf(spec, gguf, profile, ctx, explain)
                if flags:
                    flags = _grow_ctx(spec, gguf, profile, flags, explain)
                    return RunPlan(model_slug=spec.slug, gguf=gguf,
                                   backend=_backend(profile, backend),
                                   flags=flags,
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
            use_case: str = "general", model_override: str | None = None,
            backend_override: str | None = None) -> RunPlan:
    if not registry.models:
        raise ResolveError("registry has no models")
    gpu = profile.primary_gpu
    # A curated combo pins its own backend, so honouring an explicit request
    # means skipping the combo path — otherwise asking for ROCm would silently
    # return a Vulkan combo and the UI would report a backend it never ran.
    if gpu and model_override is None and not backend_override:
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
    plan = _calculate(profile, registry, use_case, backend_override)
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
