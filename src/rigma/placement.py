"""The tier plan: what goes on the card, what goes to RAM, and what it costs.

STOLEN FROM RADIANCE, and improved on where llama.cpp lets us.

Radiance (codeberg.org/StillDeadcode/radiance, Apache-2.0) splits a card into
resident weights and a paged KV pool and *derives* the split instead of asking
for it — "the engine derives the split from the card and the model; no flag
states either pool" (docs/GUIDE.md §4.3). Its inputs are a headroom it leaves
unclaimed (`--gpu-headroom-mib`, default 96 MiB), a bound on the cache
(`--max-model-len` x `--max-num-seqs`), and a pinned host tier
(`--host-pool-mib`); and it guarantees the cache "never gets less than one
`--max-model-len` sequence". It prints every term before it allocates anything.

Two things are worth taking, and one is worth fixing:

TAKE 1 — the budget is a RECEIPT, not a verdict. Radiance prints each term by
name, so a bad launch is diagnosable from the log. Rigma reduced the same
arithmetic to a single `explain` line, which is why "it OOMs sometimes" was
never answerable from a bug report.

TAKE 2 — the host tier is a BUDGET, not a byproduct. Rigma asks llama.cpp to
offload `--n-cpu-moe N` layers and never states the RAM that implies. Radiance
states the pool it is filling. `host_mb` below is that number.

FIX — size the slab from BYTES, not parameters. Rigma computed the expert slab
as `file_bytes * expert_params / total_params` (hangar.py builds the ratio, and
`_fit_with_cache` multiplied by it). That is a PARAMETER ratio applied to a BYTE
total: exact only when every tensor in the file shares one quant type. A real
MoE GGUF does not — Q4_K experts beside a Q8_0 trunk is ordinary, and PrismML's
ternary experts beside ordinary attention is the extreme case. The GGUF tensor
table carries each tensor's byte offset, so the true split is free to compute
(`gguf_meta.TensorIndex.expert_bytes`); this module prefers it and falls back to
the ratio only when the table could not be read whole.

WHAT WE DO NOT TAKE: Radiance's actual tiering mechanism. It lends idle KV
blocks back to the expert slab in place (a virtual-memory aliasing trick), keeps
a pinned host pool, and can stream weights from disk with O_DIRECT. llama.cpp
has none of those: `--n-cpu-moe N` moves the experts of the first N blocks to
system RAM and nothing else. Copying the *accounting* without pretending to
copy the *mechanism* is the whole point of this module — every number here is
something llama.cpp will really do.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# The KV floor: llama.cpp allocates the whole `-c` window per parallel slot up
# front, so a plan that cannot hold one sequence's cache cannot serve at all.
# Radiance states the same rule ("the cache never gets less than one
# `--max-model-len` sequence"); here it is simply the first term the split is
# tested against, because llama.cpp has no paging to make it elastic.


@dataclass(frozen=True)
class TierPlan:
    """Where every byte goes, for one launch.

    All figures are MiB. `gpu_mb` and `host_mb` are the weight split; `kv_mb`,
    `extra_mb` (mmproj, recurrent state) are charged to the card only, because
    llama.cpp cannot put either in host RAM.
    """
    n_layers: int
    n_off: int                 # -> --n-cpu-moe
    gpu_weights_mb: float
    host_weights_mb: float
    kv_mb: float
    extra_mb: float
    claimable_mb: float
    usable_ram_mb: float
    expert_mb: float
    exact: bool                # weights measured from the tensor table, or ratioed
    slab_note: str             # how `expert_mb` was obtained, for the receipt
    reserved_mb: float = 0.0   # card space taken before the plan starts

    @property
    def card_mb(self) -> float:
        return self.gpu_weights_mb + self.kv_mb + self.extra_mb

    @property
    def headroom_mb(self) -> float:
        """What is left of the claimable budget. Negative means it does not fit."""
        return self.claimable_mb - self.card_mb

    @property
    def fits(self) -> bool:
        return self.headroom_mb >= 0 and self.host_weights_mb <= self.usable_ram_mb

    @property
    def expert_share_on_card(self) -> float:
        """Fraction of expert bytes resident. The one number that predicts decode
        speed on a MoE, and the one the sweep can check against a measurement."""
        if self.expert_mb <= 0 or self.n_layers <= 0:
            return 1.0
        return (self.n_layers - self.n_off) / self.n_layers

    @property
    def summary(self) -> str:
        """One line for `explain`, which is printed per fit attempt.

        `receipt()` is the full block and belongs in a place that prints once;
        `explain` is appended to on every rung of the cache ladder, so the
        detail goes here as a single line or `rigma up` prints four of them per
        attempt. What the reader needs from a launch log is the split and the
        RAM it costs, and that is one line.
        """
        return (f"placement: {self.n_layers - self.n_off}/{self.n_layers} layers' "
                f"experts on the card ({self.expert_share_on_card:.0%} of expert "
                f"bytes), {self.host_weights_mb:.0f}MB to RAM "
                f"(slab {self.expert_mb:.0f}MB, {self.slab_note})")

    def receipt(self) -> tuple[str, ...]:
        """The plan, one term per line — the thing a bug report can be read from."""
        out = [
            f"placement: {self.n_layers - self.n_off}/{self.n_layers} layers' experts "
            f"on the card ({self.expert_share_on_card:.0%} of expert bytes)"
            + ("" if self.n_off == 0 else f", {self.n_off} offloaded"),
            # The reserve is named because it is the term most often mistaken for
            # free memory: on Windows it is max(1200MB floor, measured desktop) +
            # the compute buffer, and a plan that ignores it is the overcommit
            # `_budgets` exists to prevent.
            f"  card   {self.claimable_mb:.0f}MB claimable of "
            f"{self.claimable_mb + self.reserved_mb:.0f}MB "
            f"({self.reserved_mb:.0f}MB reserved for desktop + compute buffer)",
            f"  use    {self.card_mb:.0f}MB: weights {self.gpu_weights_mb:.0f} + "
            f"kv {self.kv_mb:.0f}"
            + (f" + other {self.extra_mb:.0f}" if self.extra_mb else "")
            + f", headroom {self.headroom_mb:.0f}MB",
            f"  host   {self.host_weights_mb:.0f}MB of {self.usable_ram_mb:.0f}MB RAM",
            f"  slab   {self.expert_mb:.0f}MB experts ({self.slab_note})",
        ]
        return tuple(out)


def from_offload(n_off: int, expert_mb: float, file_mb: float, kv_mb: float,
                 extra_mb: float, claimable_mb: float, usable_ram_mb: float,
                 n_layers: int, exact: bool, slab_note: str,
                 reserved_mb: float = 0.0) -> TierPlan | None:
    """The plan implied by a GIVEN offload, rather than by solving for one.

    This is the reporting half of `split`. A placement that Rigma did not choose
    still has a budget, and the interesting case is the one where it does not
    fit: a curated registry combo pins `n_cpu_moe` for a hardware class, not for
    this machine on this afternoon, so `fits` being False is the honest answer
    to "does the combo still fit here" — the same question `_combo_rejection`
    asks about ctx, answered about the weights.
    """
    if n_layers <= 0 or not 0 <= n_off <= n_layers:
        return None
    host_mb = expert_mb * n_off / n_layers
    return TierPlan(
        n_layers=n_layers,
        n_off=n_off,
        gpu_weights_mb=file_mb - host_mb,
        host_weights_mb=host_mb,
        kv_mb=kv_mb,
        extra_mb=extra_mb,
        claimable_mb=claimable_mb,
        usable_ram_mb=usable_ram_mb,
        expert_mb=expert_mb,
        exact=exact,
        slab_note=slab_note,
        reserved_mb=reserved_mb,
    )


def split(expert_mb: float, file_mb: float, kv_mb: float, extra_mb: float,
          claimable_mb: float, usable_ram_mb: float, n_layers: int,
          exact: bool, slab_note: str,
          reserved_mb: float = 0.0) -> TierPlan | None:
    """Choose `--n-cpu-moe` for a MoE launch. None when no split fits.

    llama.cpp keeps the attention, norms and embeddings of EVERY block on the
    card (`-ngl 99`) and moves only the expert tensors of the first `n_off`
    blocks to system RAM. So the card pays the non-expert weights in full, plus
    a proportional share of the expert slab, plus KV and the projectors — which
    is exactly the sum tested here.

    `n_off` is the SMALLEST offload that fits, because every layer brought back
    onto the card is decode speed: on a MoE the routed experts run per token,
    and an expert in system RAM is read over the link on every one.
    """
    if n_layers <= 0:
        return None
    # The whole file is charged first (that is the un-offloaded cost); each
    # offloaded layer then gives back `expert_mb / n_layers` of it. So the
    # numerator is the FULL weight cost, not the resident part.
    need = file_mb + kv_mb + extra_mb - claimable_mb
    if need <= 0:
        n_off = 0
    else:
        per_layer = expert_mb / n_layers
        if per_layer <= 0:
            return None
        n_off = math.ceil(need / per_layer)
    if n_off > n_layers:
        return None                          # even a full offload does not fit
    host_mb = expert_mb * n_off / n_layers
    if host_mb > usable_ram_mb:
        return None
    return from_offload(n_off, expert_mb, file_mb, kv_mb, extra_mb,
                        claimable_mb, usable_ram_mb, n_layers, exact, slab_note,
                        reserved_mb)


@lru_cache(maxsize=8)
def _index_for(path_str: str, size: int, mtime_ns: int):
    """The tensor index for one file, memoised on (size, mtime).

    The fit runs `_fit_with_cache` once per rung of the cache ladder, per ctx
    candidate, and again per strict pass — dozens of times for one launch. The
    table walk is a few hundred KB of reads, so it is cached; keying on size and
    mtime means a re-quantised file (same path, new bytes) is re-read rather
    than silently planned against the old inventory.
    """
    from .gguf_meta import read_tensor_index
    return read_tensor_index(Path(path_str))


def expert_slab_mb(spec, gguf) -> tuple[float, bool, str]:
    """Expert-slab size in MiB, whether it was measured, and how.

    Measured beats ratioed whenever the file is on disk and its table was read
    whole. The fallback is the old behaviour, kept deliberately: a model that is
    still downloading, or served from a ranged HTTP read, has no local table to
    walk, and refusing to plan it would be worse than planning it approximately.
    """
    file_mb = gguf.bytes / 2**20
    if spec.moe is None:
        return 0.0, False, "not a MoE"
    ratio_mb = file_mb * spec.moe.expert_weight_fraction
    try:
        from .runtime import rigma_home
        path = rigma_home() / "models" / gguf.file
        st = path.stat()
        idx = _index_for(str(path), st.st_size, st.st_mtime_ns)
    except Exception:
        return ratio_mb, False, f"param ratio {spec.moe.expert_weight_fraction:.3f}"
    if not idx.bytes_complete:
        return ratio_mb, False, f"param ratio {spec.moe.expert_weight_fraction:.3f} (no table)"
    measured = idx.expert_bytes / 2**20
    if measured <= 0:
        # The table read fine and holds no `_exps.` tensor: the header calls this
        # a MoE and the file disagrees. The table wins (same rule the rest of
        # gguf_meta applies), and a zero slab means "nothing to offload".
        return 0.0, True, "tensor table says no expert tensors"
    return measured, True, "tensor table"
