"""The fit charges the hybrid's per-sequence recurrent-state buffer.

llama.cpp allocates a separate RS buffer for Mamba/DeltaNet-style layers, one
per sequence (`--parallel`), and Rigma budgeted exactly zero for it. Ground
truth is a real load of Ternary-Bonsai-2-27B (qwen35, 64 layers, 48 recurrent)
in .scratch/prism-v.log: at n_seq_max=1 the engine logged

    llama_memory_recurrent: ROCm0 RS buffer size = 149.62 MiB
    size = 149.62 MiB (1 cells, 64 layers, 1 seqs 0 rs_seq),
           R (f32): 5.62 MiB, S (f32): 144.00 MiB

Rigma launches `--parallel 2`, so the plan must reserve two of those.
"""
import pytest

from rigma.models import (LAUNCH_PARALLEL, CachePolicy, CpuInfo, GgufFile,
                          GpuInfo, HardwareProfile, ModelSpec)
from rigma.resolve import (fit_gguf, kv_bytes_per_token, recurrent_state_mb,
                           recurrent_state_unknown)

MIB = 2 ** 20

# The exact qwen35.ssm.* values read from the file's header.
SSM = {"ssm_state_size": 128, "ssm_inner_size": 6144,
       "ssm_conv_kernel": 4, "ssm_group_count": 16}


def _profile(vram=16368):
    gpu = GpuInfo(vendor="amd", name="AMD Radeon RX 9070 XT", vram_mb=vram,
                  arch="rdna4", slug="amd-radeon-rx-9070-xt-16g",
                  backends=["vulkan", "rocm"])
    return HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=24000,
                           cpu=CpuInfo(cores=16), os="windows",
                           disk_free_gb=400.0)


def _bonsai(**over) -> ModelSpec:
    """The owner's real hybrid: 64 layers, 48 recurrent, PQ2_0 file."""
    fields = dict(
        slug="bonsai", family="qwen35", kind="dense", n_layers=64,
        full_attn_layers=16, kv_heads=4, head_dim=256, native_ctx=262144,
        custom=True, full_attention_interval=4, recurrent_layers=48,
        cache_type_policy=CachePolicy(k="f16", v="f16"),
        ggufs=[GgufFile(repo="local", file="b.gguf", bytes=7206168928,
                        quant="Q2_0")],
        **SSM)
    fields.update(over)
    return ModelSpec(**fields)


def _reserve_mb(profile) -> float:
    from rigma.resolve import COMPUTE_BUFFER_MB, VRAM_RESERVE_MB
    return VRAM_RESERVE_MB[profile.os] + COMPUTE_BUFFER_MB


def test_the_recurrent_state_is_the_buffer_the_engine_measured():
    """48 x (S + R), from llama-hparams.cpp:
        n_embd_s() = ssm_d_state * ssm_d_inner
        n_embd_r() = (ssm_d_conv - 1) * (ssm_d_inner + 2*ssm_n_group*ssm_d_state)
    both f32, over the 48 recurrent layers of 64."""
    spec = _bonsai()
    per_seq = recurrent_state_mb(spec)
    assert per_seq == pytest.approx(149.62, abs=0.01)
    s = 48 * (128 * 6144) * 4 / MIB
    r = 48 * ((4 - 1) * (6144 + 2 * 16 * 128)) * 4 / MIB
    assert s == pytest.approx(144.00, abs=0.01)
    assert r == pytest.approx(5.62, abs=0.01)
    assert per_seq == pytest.approx(s + r, abs=0.01)
    # ... and the launch's --parallel is the multiplier, from ONE constant
    assert LAUNCH_PARALLEL == 2
    assert per_seq * LAUNCH_PARALLEL == pytest.approx(299.25, abs=0.02)


def test_a_dense_model_is_charged_nothing():
    spec = _bonsai(recurrent_layers=0, full_attention_interval=0,
                   **{k: 0 for k in SSM})
    assert recurrent_state_mb(spec) == 0.0
    assert not recurrent_state_unknown(spec)
    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    assert not any("rs=" in ln for ln in explain), explain


def test_the_fit_reserves_two_sequences_of_recurrent_state():
    """The term has to land in the fit, not only in the helper: a card with
    room for the file, the KV and 150 MiB — but not for the RS buffer — must
    not be promised full residency."""
    spec = _bonsai()
    ctx = 8192
    kv_mb = ctx * kv_bytes_per_token(spec, "f16", "f16") / MIB
    prof = _profile()
    room = spec.ggufs[0].bytes / MIB + kv_mb + 150
    prof = prof.model_copy(update={"gpus": [prof.gpus[0].model_copy(update={
        "vram_mb": int(room + _reserve_mb(prof))})]})
    blind = spec.model_copy(update={"recurrent_layers": 0})
    explain: list[str] = []
    honest = fit_gguf(spec, spec.ggufs[0], prof, ctx, explain)
    assert any("rs=299MB" in ln for ln in explain), explain
    assert honest is not None, "the honest fit produced no plan at all"
    # the RS buffer is what stops f16 being fully resident here
    fits_blind = fit_gguf(blind, blind.ggufs[0], prof, ctx, [])
    assert fits_blind is not None and fits_blind.ngl == 99
    assert honest.ngl < 99 or honest.cache_type_k != "f16", \
        "promised full residency for 299 MiB the driver will page"


def test_a_recurrent_model_without_ssm_geometry_is_charged_zero_and_says_so():
    """A hybrid header that omits `ssm.*` must not crash and must not read as a
    dense model: 0 MiB charged, and the explain line marks it unknown."""
    spec = _bonsai(**{k: 0 for k in SSM})
    assert recurrent_state_mb(spec) == 0.0
    assert recurrent_state_unknown(spec)
    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    assert any("rs=unknown" in ln for ln in explain), explain


def test_the_budget_table_charges_the_recurrent_state():
    """A2b: the fit charges RS (A2), but the display-only budget table did not —
    so an explorer row could read "fits" (`over_mb` <= 0) while the fit actually
    offloads a layer. The geometry is the owner's real hybrid, and every constant
    comes from the code: 48 recurrent layers of 64, ssm.* from the header,
    multiplied by LAUNCH_PARALLEL."""
    from rigma.resolve import LAUNCH_PARALLEL, _budget_rows, recurrent_state_mb
    spec = _bonsai()
    gguf = spec.ggufs[0]
    per_seq = recurrent_state_mb(spec)
    rs_mb = per_seq * LAUNCH_PARALLEL
    # 48 x (S + R) = 149.625 MiB per sequence; the launch runs two
    assert rs_mb == pytest.approx(299.25, abs=0.02)
    assert LAUNCH_PARALLEL == 2

    ctx, generous = 8192, 15000.0
    row = _budget_rows(spec, gguf, 0.0, ctx, generous)
    blind = spec.model_copy(update={"recurrent_layers": 0})
    blind_row = _budget_rows(blind, blind.ggufs[0], 0.0, ctx, generous)
    assert row["rs_mb"] == round(rs_mb)
    # the term is inside the arithmetic the page calls "over", not only in its
    # own column (both columns are rounded to whole MB, so allow the rounding)
    assert row["over_mb"] - blind_row["over_mb"] == pytest.approx(rs_mb, abs=1)

    # and the failure the finding describes: a budget with room for everything
    # EXCEPT the RS buffer. The old table said it fits; the fit offloads.
    budget = gguf.bytes / MIB + blind_row["kv_mb"] + 150.0
    fits = _budget_rows(blind, gguf, 0.0, ctx, budget)
    honest = _budget_rows(spec, gguf, 0.0, ctx, budget)
    assert fits["over_mb"] <= 0 < honest["over_mb"], (
        f"budget table disagrees with the fit: {fits} vs {honest}")


def test_an_unrecognised_recurrent_geometry_is_unknown_not_zero():
    """A2d: the probe reports `rs_geometry_unknown` for a pure-Mamba header, an
    explicit `attention.recurrent_layers` array, or a partial `ssm.*` set — the
    shapes whose count or buffer it cannot derive. The fit must say so instead of
    charging a confident zero."""
    spec = _bonsai(rs_geometry_unknown=True, recurrent_layers=0,
                   **{k: 0 for k in SSM})
    assert recurrent_state_unknown(spec)
    explain: list[str] = []
    flags = fit_gguf(spec, spec.ggufs[0], _profile(), 8192, explain)
    assert flags is not None
    assert any("rs=unknown" in ln for ln in explain), explain


def test_the_over_mb_delta_double_rounds_by_less_than_one_mb():
    """A2e, closed as COSMETIC: no change, and this pins the arithmetic.

    A2b's verifier measured that the `over_mb` delta reads 300 for a 299.25 MiB
    term. It is double rounding, not a wrong charge. `_budget_rows` reports each
    `over_mb` as `round(exact)`, and the delta differences TWO independently
    rounded numbers, so it carries up to one whole MB of rounding noise on top
    of the term it is meant to show:

        over_mb(row)   = round(X + 299.25)
        over_mb(blind) = round(X)
        delta          = round(X + 299.25) - round(X)

    with X = file + kv - budget = -7615.661773681641 for the constants below.
    round(-7316.411773681641) - round(-7615.661773681641) = -7316 - (-7616) = 300,
    while the charged term is 299.25 — a 0.25 MB (0.08%) overstatement in the
    DELTA. Neither row is wrong: each is the exact arithmetic rounded to the
    whole MB the display shows, and 0.25 MB is below that display's resolution.

    Making the delta exact is worse, not better: summing the already-rounded
    columns would let `over_mb`'s SIGN disagree with the exact arithmetic by up
    to 2 MB, and `over_mb > 0` is the page's "OVER by ... that much spills to
    RAM" vs "headroom" verdict. A wrong verdict is a wrong answer; 0.25 MB on a
    299 MB term is not. The fit's own arithmetic is untouched either way."""
    from rigma.resolve import LAUNCH_PARALLEL, _budget_rows, recurrent_state_mb
    spec = _bonsai()
    gguf = spec.ggufs[0]
    rs_mb = recurrent_state_mb(spec) * LAUNCH_PARALLEL
    assert rs_mb == pytest.approx(299.25, abs=0.02)

    ctx, budget = 8192, 15000.0
    row = _budget_rows(spec, gguf, 0.0, ctx, budget)
    blind = _budget_rows(spec.model_copy(update={"recurrent_layers": 0}),
                         gguf, 0.0, ctx, budget)
    assert row["rs_mb"] == 299            # the charge, rounded to whole MB
    delta = row["over_mb"] - blind["over_mb"]
    assert delta == 300                   # the measured double rounding
    # ... and it is bounded by the display's own resolution: no fractional part
    # of X can move the delta more than 0.75 MB away from the true term.
    assert abs(delta - rs_mb) <= 0.75

    # the two rows, reconstructed: each is the exact sum rounded once
    x = gguf.bytes / MIB + ctx * kv_bytes_per_token(spec, "f16", "f16") / MIB \
        - budget
    assert blind["over_mb"] == round(x)
    assert row["over_mb"] == round(x + rs_mb)
