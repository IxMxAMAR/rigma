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
    # VRAM every process on the machine currently holds, measured. None when
    # unmeasurable. Windows overcommits VRAM rather than refusing, so planning
    # against a fixed "the desktop uses 1200MB" gets the difference silently
    # paged to system RAM (measured 2026-08-21: 4,107MB, 29% of the weights).
    vram_used_mb: float | None = None

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
    # True = use exactly these types, no q8_0 fallback. Set only by the Models
    # page explorer, where the whole point is to answer "what if the cache were
    # q4_0" — silently falling back to q8_0 made every choice show the same
    # answer. Never set on a stored spec; a real launch keeps the ladder so a
    # too-large f16 cache degrades instead of failing.
    pinned: bool = False


class GgufFile(BaseModel):
    repo: str
    file: str
    bytes: int
    quant: str
    sha256: str | None = None
    # Does THIS file carry the multi-token-prediction tensors? Per-file, not
    # per-model: whether a draft head survives is a decision the quantiser makes
    # per artefact, and asking llama.cpp for draft-mtp without them resets the
    # Vulkan driver rather than erroring. None = not probed yet.
    mtp: bool | None = None


class UseCase(BaseModel):
    name: str
    system_prompt: str
    description: str = ""


class LaunchDefaults(BaseModel):
    """How this model should be launched when nothing else says otherwise.

    Owner request 2026-08-19: "I have to change context then KV and model
    reloads 2 times to get me to my idle config." Applying them in one relaunch
    was fixed then; remembering them was not, so every plain load still landed
    wherever the resolver put it.

    It matters more than a preference. Measured on this machine 2026-08-21, the
    same model and quant ran at 9.95 tok/s in one configuration and 38.27 in
    another; the resolver cannot know which the user wants because the trade
    (context against speed against draft cache) is a judgement, not arithmetic.

    Unset means NO OPINION, and that is why every field has a falsy sentinel
    rather than a plausible-looking default: a model that only wants to pin its
    context must not thereby also pin its cache type.
    """
    # which gguf to prefer, by the label the Models page shows
    quant: str = ""
    ctx: int = 0
    kv: str = ""
    # tri-state: None keeps whatever the last launch used. Distinct from False,
    # which is an explicit "run this vision model text-only" — conflating them
    # would silently reload a projector the user turned off to free VRAM.
    vision: bool | None = None
    # speculative decoding, e.g. "draft-mtp". Only meaningful on a gguf that
    # actually carries the draft head; asking for it otherwise resets the
    # Vulkan driver rather than erroring.
    spec_type: str = ""
    spec_n_max: int = 0

    def is_set(self, field: str) -> bool:
        """Whether this field carries an opinion. `vision` is the odd one: its
        unset value is None, while everything else uses 0 or ""."""
        value = getattr(self, field)
        if field == "vision":
            return value is not None
        return bool(value)

    def as_overrides(self) -> dict:
        """Only the fields that were actually set, for merging over a request."""
        return {f: getattr(self, f) for f in
                ("quant", "ctx", "kv", "vision", "spec_type", "spec_n_max")
                if self.is_set(f)}


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
    # ---- probed facts (custom imports; registry specs are hand-authored) ----
    # Exact parameter count, summed from the tensor table's dims. Identical
    # across every quant of a model, so one probe prices them all: bits-per-
    # weight = bytes*8/params is then arithmetic on two measured quantities
    # rather than a guess from the filename.
    params: int = 0
    # Extra decoder blocks that exist only to draft tokens. NOT part of n_layers
    # — see gguf_meta for why block_count is the wrong number to store.
    mtp_layers: int = 0
    # Every Nth layer keeps a growing KV cache (Qwen3.5/3.8 hybrid attention).
    # Stored so a spec written by an older probe can be re-derived without the
    # multi-GB file being on disk.
    full_attention_interval: int = 0
    # AUDIT F19: docs/audit-2026-09-04-full.md — a sliding-window model keeps a
    # SECOND, window-sized KV cache for the layers `full_attn_layers` excludes.
    # Budgeting those layers as zero is ~400 MiB unaccounted on a 27B-class SWA
    # model, and it errs toward accepting a plan the card cannot hold — the one
    # approximation in the fit math that was unsafe in that direction. Defaults
    # of 0 mean "not a windowed model", which charges nothing.
    swa_layers: int = 0
    swa_kv_heads: int = 0
    swa_window: int = 0
    # False = the gguf shipped no tokenizer.chat_template, so an empty
    # capability list is missing evidence rather than a finding about the model.
    has_template: bool = True
    # The configuration this model should come up in. None = no opinion, let
    # the resolver choose as before.
    launch: LaunchDefaults | None = None
    # Bumped when the probe learns to read something it used to get wrong;
    # specs below PROBE_VERSION are re-derived on read (hangar.heal_spec).
    probe_version: int = 0


# Injected right before the end-of-thinking tag when --reasoning-budget runs
# out. Written as an instruction the model can act on in a few tokens: state the
# decision, do not restart the analysis.
BUDGET_EXHAUSTED = (
    "You have used your thinking budget. Stop analysing now and write your "
    "answer. Begin it by stating, briefly, the decisions you reached and the "
    "single next step you are taking — anything you do not write down here is "
    "lost to the next turn."
)


class ComboFlags(BaseModel):
    ctx: int
    ngl: int = 99
    n_cpu_moe: int = 0
    flash_attn: str = "on"   # on | off | auto (legacy bools coerced)
    cache_type_k: str = "f16"
    cache_type_v: str = "f16"
    reasoning: str = ""   # ""(engine default) | on | off | auto
    # Max thinking tokens per turn. Back to -1 (engine default, unlimited)
    # after a 16384 default coincided with decode collapsing from ~26 t/s to
    # 0.59 t/s on a fresh 4K-context chat (2026-08-28). It was the only launch
    # flag that changed, so it is the only one that gets reverted until it is
    # measured on its own. The runaway-deliberation problem it was meant to
    # solve is real, but it is worth less than 50x the speed.
    reasoning_budget: int = -1
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
        # --parallel 2 + --kv-unified: one slot for the user's conversation,
        # one for Rigma's own aux calls (auto-title, compaction, delegate,
        # memory harvest). With ONE slot every aux call evicted the main
        # trajectory's prompt cache, forcing a full re-prefill of 20-60K
        # tokens (~30-90s dead) before the next real turn — the single
        # biggest hidden latency tax on agent runs (audit 2026-07-21).
        # --kv-unified keeps ONE shared KV pool of size ctx, so the memory
        # the resolver budgeted is unchanged (non-unified would allocate a
        # full ctx PER SLOT — the 4x overflow the old --parallel 1 avoided).
        args = ["-m", model_path, "--port", str(port), "--host", "127.0.0.1",
                "-ngl", str(self.flags.ngl), "-c", str(self.flags.ctx),
                "--parallel", "2", "--kv-unified"]
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
        if self.flags.reasoning_budget >= 0:
            args += ["--reasoning-budget", str(self.flags.reasoning_budget)]
            # What the model reads AS it is cut off. Without it the thinking
            # block is truncated mid-sentence and whatever it had worked out is
            # lost; with it, the budget ends in a conclusion instead of a
            # guillotine.
            args += ["--reasoning-budget-message", BUDGET_EXHAUSTED]
        if self.flags.spec_type and self.flags.spec_type != "none":
            args += ["--spec-type", self.flags.spec_type,
                     "--spec-draft-n-max", str(self.flags.spec_n_max)]
        # reuse unchanged KV prefixes on edit/regenerate/compact turns.
        # (No effect on DeltaNet hybrids — KV shifting is unsupported there,
        # llama.cpp #18497 — but harmless, and it still helps pure
        # transformers.)
        args += ["--cache-reuse", "256"]
        # hybrid/recurrent models can't rewind KV freely: any edit deep in
        # history rolls back to the nearest checkpoint or reprocesses from
        # scratch. Denser checkpoints (default spacing 8192) make observation
        # masking and compaction edits cheap; harmless on pure transformers.
        args += ["--checkpoint-min-step", "4096"]
        return args
