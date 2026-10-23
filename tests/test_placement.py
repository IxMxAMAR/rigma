"""The tier plan: expert slab from bytes, and the split that reaches `--n-cpu-moe`.

Two separate claims are under test here and they fail differently:

1. `gguf_meta` can say how many BYTES the expert tensors occupy, not just how
   many parameters they hold. That is what makes the slab exact for a MoE whose
   experts and trunk are quantised differently.
2. `placement.split` picks the smallest offload that fits, and its receipt
   accounts for every byte. A plan that "fits" while its own receipt sums past
   the budget is the failure this test exists to prevent.
"""
from __future__ import annotations

import struct
from types import SimpleNamespace

from rigma import placement
from rigma.gguf_meta import read_tensor_index


def _s(text: bytes) -> bytes:
    return struct.pack("<Q", len(text)) + text


def _tensor(name: bytes, dims: list[int], ttype: int, offset: int) -> bytes:
    return (_s(name) + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<I", ttype) + struct.pack("<Q", offset))


def _gguf(tensors: list[bytes], data_len: int) -> bytes:
    return (b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", len(tensors))
            + struct.pack("<Q", 0) + b"".join(tensors) + b"\0" * data_len)


# Offsets are relative to the data section, so the table below describes:
#   token_embd            0     .. 1000   other
#   blk.0.ffn_gate_exps    1000  .. 5000   expert
#   blk.0.attn_q           5000  .. 6000   other
#   blk.1.nextn.eh_proj    6000  .. 7000   mtp
_LAYOUT = [
    _tensor(b"token_embd.weight", [100], 0, 0),
    _tensor(b"blk.0.ffn_gate_exps.weight", [100], 0, 1000),
    _tensor(b"blk.0.attn_q.weight", [100], 0, 5000),
    _tensor(b"blk.1.nextn.eh_proj.weight", [100], 0, 6000),
]


def _write(tmp_path, name="m.gguf", data_len=7000):
    p = tmp_path / name
    p.write_bytes(_gguf(_LAYOUT, data_len))
    return p


def test_tensor_offsets_become_byte_buckets(tmp_path):
    """The expert slab is measured in BYTES, from the table's own offsets."""
    idx = read_tensor_index(_write(tmp_path))
    assert idx.bytes_complete
    assert idx.expert_bytes == 4000          # the gap 1000..5000
    assert idx.other_bytes == 2000           # token_embd + attn_q
    # The last tensor's span runs to the end of the FILE, so it absorbs the
    # header — deliberately, because the buckets must sum to the size the fit
    # compares against. 1000 bytes of tensor plus 240 bytes of header here.
    assert idx.mtp_bytes >= 1000
    assert idx.expert_bytes + idx.mtp_bytes + idx.other_bytes == (tmp_path / "m.gguf").stat().st_size


def test_bytes_are_not_the_parameter_ratio(tmp_path):
    """The whole point: equal parameter counts, wildly unequal byte counts.

    A ratio built from parameters cannot see this file's layout at all — every
    tensor here holds 100 elements, so the ratio would put 50% of the bytes on
    the experts. The table says 4000/7000.
    """
    idx = read_tensor_index(_write(tmp_path))
    ratio = idx.expert_params / idx.params
    assert ratio == 0.25
    assert idx.expert_bytes / (tmp_path / "m.gguf").stat().st_size > 0.5


def test_a_file_like_source_reports_no_bytes(tmp_path):
    """A ranged read has no length to charge the last tensor against.

    `bytes_complete` must stay False so the caller falls back to the ratio
    instead of planning against a partial sum.
    """
    blob = _gguf(_LAYOUT, 7000)
    import io
    idx = read_tensor_index(io.BytesIO(blob))
    assert not idx.bytes_complete
    assert idx.expert_bytes == 0


def _spec(fraction=0.5):
    return SimpleNamespace(moe=SimpleNamespace(expert_weight_fraction=fraction))


def _gguf_file(name, size):
    return SimpleNamespace(file=name, bytes=size)


def test_slab_comes_from_the_table_when_the_file_is_present(tmp_path, monkeypatch):
    p = _write(tmp_path)
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path.parent))
    (tmp_path.parent / "models").mkdir(exist_ok=True)
    (tmp_path.parent / "models" / "m.gguf").write_bytes(p.read_bytes())

    mb, exact, note = placement.expert_slab_mb(
        _spec(0.9), _gguf_file("m.gguf", (tmp_path / "m.gguf").stat().st_size))
    assert exact and note == "tensor table"
    # 4000 bytes, not 90% of the file — the ratio would have said ~6300.
    assert round(mb, 4) == round(4000 / 2**20, 4)


def test_slab_falls_back_to_the_ratio_without_the_file():
    mb, exact, note = placement.expert_slab_mb(_spec(0.5), _gguf_file("absent.gguf", 2**20))
    assert not exact
    assert "param ratio" in note
    assert mb == 0.5


def test_split_takes_the_smallest_offload_that_fits():
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=0,
                           claimable_mb=8000, usable_ram_mb=16000, n_layers=40,
                           exact=True, slab_note="tensor table")
    # need = 10000 + 1000 - 8000 = 3000MB; each layer gives back 8000/40 = 200MB
    assert plan is not None and plan.n_off == 15
    assert plan.host_weights_mb == 3000
    assert plan.gpu_weights_mb == 7000
    assert plan.card_mb == 8000
    assert plan.headroom_mb == 0
    assert plan.fits


def test_split_keeps_everything_resident_when_it_fits():
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=0,
                           claimable_mb=11000, usable_ram_mb=16000, n_layers=40,
                           exact=True, slab_note="tensor table")
    assert plan is not None and plan.n_off == 0
    assert plan.host_weights_mb == 0
    assert plan.expert_share_on_card == 1.0


def test_split_refuses_when_no_offload_can_fit():
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=0,
                           claimable_mb=2000, usable_ram_mb=16000, n_layers=40,
                           exact=True, slab_note="tensor table")
    assert plan is None


def test_split_refuses_when_the_host_pool_is_too_small():
    """The host tier is a budget, not a byproduct: 3000MB will not fit 1000MB."""
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=0,
                           claimable_mb=8000, usable_ram_mb=1000, n_layers=40,
                           exact=True, slab_note="tensor table")
    assert plan is None


def test_receipt_states_every_term():
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=500,
                           claimable_mb=8000, usable_ram_mb=16000, n_layers=40,
                           exact=True, slab_note="tensor table")
    lines = plan.receipt()
    assert len(lines) == 5
    joined = "\n".join(lines)
    # need = 10000 + 1000 + 500 - 8000 = 3500MB, at 200MB a layer -> 18 of 40
    assert "22/40 layers' experts" in joined
    assert "card   8000MB claimable of 8000MB" in joined
    assert "use    7900MB: weights 6400 + kv 1000 + other 500" in joined
    assert "headroom 100MB" in joined
    assert "host   3600MB of 16000MB RAM" in joined
    assert "(tensor table)" in joined
    # `explain` takes the one-liner, never the block: it is printed once per
    # fit attempt, and the fit runs once per rung of the cache ladder.
    assert "\n" not in plan.summary
    assert plan.summary.startswith("placement: 22/40 layers' experts")
    assert "3600MB to RAM" in plan.summary


# --- the resolver actually uses it -------------------------------------------

def _moe_registry(size_gb=20.0, fraction=0.93):
    """A MoE big enough that a 16 GB card MUST offload some of it.

    The fraction is deliberately the one Ornith-1.5-35B-A3B declares (0.93) so
    the test would fail if the ratio were still what reached `--n-cpu-moe`.
    """
    from rigma.models import CachePolicy, GgufFile, ModelSpec, MoESpec
    from rigma.registry import Registry
    spec = ModelSpec(slug="moe", family="qwen3", kind="moe", n_layers=40,
                     full_attn_layers=40, kv_heads=4, head_dim=128,
                     native_ctx=8192, use_cases=["general"],
                     cache_type_policy=CachePolicy(),
                     moe=MoESpec(total_b=35.0, active_b=3.0,
                                 expert_weight_fraction=fraction),
                     ggufs=[GgufFile(repo="r", file="moe.gguf",
                                     bytes=int(size_gb * 2**30), quant="Q4_K_M")])
    return Registry([], {"moe": spec}, {})


def _card(vram_mb=16304):
    """A 16 GB RDNA4 card with a stated desktop load.

    `vram_used_mb` is set explicitly because the probe otherwise reports what
    this machine's desktop is holding RIGHT NOW — and with a model loaded that
    is most of the card, which would make the budget (and this test) depend on
    whether a server happened to be running.
    """
    from rigma.probe import probe_hardware
    p = probe_hardware([], raw_gpus=[
        {"vendor_id": 0x1002, "name": "AMD Radeon RX 9070 XT",
         "vram_mb": vram_mb, "device_type": 2}])
    p.vram_used_mb = 512.0
    # RAM as well. The probe reports what this machine has FREE right now, so a
    # model load — the owner's, or a benchmark running in another window — moves
    # these tests' verdicts: with 4.8GB free instead of 7GB the curated combo
    # stopped passing its own budget check and three tests failed for a reason
    # that had nothing to do with the code. Pin both and the budget is a
    # constant.
    p.ram_mb = 32768.0
    p.ram_free_mb = 30000.0
    return p


def test_the_resolver_states_its_split(tmp_path, monkeypatch):
    """End to end: the plan reaches `explain`, so `rigma up` prints it."""
    from rigma.resolve import resolve
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = resolve(_card(), _moe_registry(), use_case="general")
    assert plan.flags.n_cpu_moe > 0, "a 20GB MoE fit a 16GB card with no offload"
    lines = [line for line in plan.explain if line.startswith("placement:")]
    assert len(lines) == 1, f"expected one placement line, got {plan.explain}"
    assert "param ratio" in lines[0], "no local file: the ratio is the honest source"


def test_a_local_file_makes_the_split_exact(tmp_path, monkeypatch):
    """With the GGUF on disk, the slab is measured instead of ratioed.

    The file is laid out so the two answers differ by a mile: 8 MB total, of
    which the expert span is 6 MB (75%). The spec's ratio says 93%, which would
    claim 7.4 MB. The line must carry the measured figure.
    """
    from rigma.resolve import resolve
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    models = tmp_path / "models"
    models.mkdir()
    blob = _gguf([
        _tensor(b"token_embd.weight", [100], 0, 0),
        _tensor(b"blk.0.ffn_gate_exps.weight", [100], 0, 1 * 2**20),
        _tensor(b"output.weight", [100], 0, 7 * 2**20),
    ], data_len=8 * 2**20)
    (models / "moe.gguf").write_bytes(blob)
    plan = resolve(_card(), _moe_registry(size_gb=len(blob) / 2**30), use_case="general")
    lines = [line for line in plan.explain if line.startswith("placement:")]
    assert len(lines) == 1, plan.explain
    assert "tensor table" in lines[0], lines[0]
    assert "param ratio" not in lines[0]
    # 6 MB of experts, not 93% of 8 MB (7.4 MB).
    assert "slab 6MB" in lines[0], lines[0]


# --- a placement Rigma did not choose: the registry combo --------------------

def _run_plan(spec, gguf, n_cpu_moe, ctx=4096):
    from rigma.models import ComboFlags, RunPlan
    return RunPlan(
        model_slug=spec.slug, gguf=gguf, backend="vulkan",
        flags=ComboFlags(ctx=ctx, n_cpu_moe=n_cpu_moe,
                         cache_type_k="f16", cache_type_v="f16"),
        origin="combo:_class/test", explain=[])


def test_the_receipt_names_the_reserved_card_space():
    """The reserve is the term most often mistaken for free memory.

    It is max(per-OS floor, measured desktop) + compute buffer, and a receipt
    that reports only "claimable" leaves the reader unable to check it against
    the number Task Manager shows.
    """
    plan = placement.split(expert_mb=8000, file_mb=10000, kv_mb=1000, extra_mb=0,
                           claimable_mb=14592, usable_ram_mb=16000, n_layers=40,
                           exact=True, slab_note="tensor table",
                           reserved_mb=1712)
    joined = "\n".join(plan.receipt())
    assert "card   14592MB claimable of 16304MB (1712MB reserved for desktop + compute buffer)" \
        in joined


def test_the_plan_carries_the_full_receipt(tmp_path, monkeypatch):
    """`rigma plan --explain` prints `plan.placement`; `rigma up` prints only the
    one-line summary in `explain`."""
    from rigma.resolve import resolve
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    plan = resolve(_card(), _moe_registry(), use_case="general")
    assert plan.placement and plan.placement[0].startswith("placement:")
    assert len(plan.placement) == 5
    assert sum(1 for ln in plan.explain if ln.startswith("placement:")) == 1


def test_a_fallback_states_its_split_too(tmp_path, monkeypatch):
    """A fallback is a launch candidate, not a label.

    `fallback_plans` built its plans with no budget at all, so the one path that
    runs AFTER a launch has already failed was the one that could not say
    whether it would fit either.
    """
    from rigma.models import CachePolicy, GgufFile, ModelSpec, MoESpec
    from rigma.registry import Registry
    from rigma.resolve import fallback_plans
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = ModelSpec(
        slug="moe", family="qwen3", kind="moe", n_layers=40,
        full_attn_layers=40, kv_heads=4, head_dim=128, native_ctx=8192,
        use_cases=["general"], cache_type_policy=CachePolicy(),
        moe=MoESpec(total_b=35.0, active_b=3.0, expert_weight_fraction=0.93),
        ggufs=[GgufFile(repo="r", file="big.gguf", bytes=20 * 2**30,
                        quant="Q5_K_M"),
               GgufFile(repo="r", file="small.gguf", bytes=14 * 2**30,
                        quant="Q4_K_M")])
    reg = Registry([], {"moe": spec}, {})
    primary = _run_plan(spec, spec.ggufs[0], n_cpu_moe=20, ctx=8192)
    out = fallback_plans(primary, reg, _card())
    assert out, "no fallback produced"
    fb = out[0]
    assert fb.origin == "fallback" and fb.gguf.quant == "Q4_K_M"
    assert fb.placement and fb.placement[0].startswith("placement:"), fb.explain


def test_an_over_budget_combo_falls_back_to_the_calculator(tmp_path, monkeypatch):
    """A combo pins `n_cpu_moe` for a hardware CLASS, not for this afternoon.

    It still wins whenever it fits — its sources are measured benchmarks, and
    that is worth more than arithmetic. But when its own placement is over
    budget, the calculator's arithmetic has to take over: the alternative is a
    launch Windows answers by paging weights over PCIe on every token.
    """
    from rigma import resolve as R
    from rigma.registry import Registry
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    # Let the combo through its own checks (declared budget, ctx placement) so
    # the ONLY thing rejecting it is the placement this test is about.
    monkeypatch.setattr(R, "_combo_rejection", lambda *a, **k: "")
    profile = _card()
    profile.vram_used_mb = 14000.0     # same 16GB card, a desktop that is using it
    plan = R.resolve(profile, Registry.load(), use_case="coding")
    assert not plan.origin.startswith("combo:"), plan.origin
    assert any("NOT used" in ln and "does not fit" in ln for ln in plan.explain), \
        plan.explain


def test_a_curated_combo_states_its_split(tmp_path, monkeypatch):
    """The combo path is the DEFAULT on verified hardware, and it used to say
    nothing about the placement it pinned — so an over-committed combo and a
    comfortable one looked identical in the log."""
    from rigma.resolve import _note_tier_plan
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = _moe_registry().models["moe"]
    gguf = spec.ggufs[0]
    plan = _run_plan(spec, gguf, n_cpu_moe=20)
    _note_tier_plan(plan, spec, _card())
    lines = [ln for ln in plan.explain if ln.startswith("placement:")]
    assert len(lines) == 1, plan.explain
    assert "20/40 layers' experts" in lines[0]
    assert not any("WARNING" in ln for ln in plan.explain), plan.explain


def test_a_combo_that_no_longer_fits_says_so(tmp_path, monkeypatch):
    """Same combo, an 8 GB card: the placement it pinned is now over budget,
    and the log must name the overcommit rather than launch into paging."""
    from rigma.resolve import _note_tier_plan
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = _moe_registry().models["moe"]
    plan = _run_plan(spec, spec.ggufs[0], n_cpu_moe=2)
    _note_tier_plan(plan, spec, _card(vram_mb=8192))
    warnings = [ln for ln in plan.explain if ln.startswith("WARNING:")]
    assert len(warnings) == 1, plan.explain
    assert "does not fit right now" in warnings[0]
    assert "over by" in warnings[0]


def test_the_real_registry_combo_reports_its_placement():
    """The shipped registry is what `rigma up` uses with no arguments."""
    from rigma.registry import Registry
    from rigma.resolve import resolve
    plan = resolve(_card(), Registry.load(), use_case="coding")
    assert plan.origin.startswith(("combo:", "class:")), plan.origin
    assert any(ln.startswith("placement:") for ln in plan.explain), plan.explain
