"""The Models page's fit + quality columns.

Owner report 2026-07-30: 21 quants offered with nothing but a file size to
choose between them, no "does this fit my VRAM", no context per quant, and one
model whose three rows all read "GGUF".
"""
import pathlib

import pytest

from rigma import hangar
from rigma.quant_quality import quality_of, size_vs_bf16, tier_for


# --- the reference table -----------------------------------------------------

def test_known_formats_have_a_figure():
    for q in ("Q8_0", "Q6_K", "Q5_K_M", "Q4_K_M", "Q4_K_S", "IQ4_XS",
              "Q3_K_M", "Q2_K", "UD-Q4_K_XL", "UD-Q3_K_XL", "UD-IQ2_XXS"):
        assert quality_of(q) is not None, q


def test_quality_is_monotonic_in_bits():
    """More bits must never be scored as worse quality. A table hand-typed from
    several sources is exactly where an inverted pair hides."""
    order = ["Q8_0", "Q6_K", "Q5_K_M", "Q5_K_S", "Q4_K_M", "Q4_K_S",
             "IQ4_XS", "Q3_K_M", "Q3_K_S", "Q2_K"]
    losses = [quality_of(q)["ppl_pct"] for q in order]
    assert losses == sorted(losses), dict(zip(order, losses))
    bpws = [quality_of(q)["bpw"] for q in order]
    assert bpws == sorted(bpws, reverse=True), dict(zip(order, bpws))


def test_dynamic_quants_beat_the_plain_format_they_extend():
    # the whole point of UD-: wider bits on the tensors that matter
    for ud, plain in (("UD-Q4_K_XL", "Q4_K_M"), ("UD-Q3_K_XL", "Q3_K_M"),
                      ("UD-Q2_K_XL", "Q2_K")):
        assert quality_of(ud)["ppl_pct"] < quality_of(plain)["ppl_pct"]


def test_bf16_and_f16_are_the_zero_point():
    for q in ("BF16", "F16"):
        assert quality_of(q)["ppl_pct"] == 0.0
        assert size_vs_bf16(q, 1) == 1.0


def test_an_unknown_label_returns_none_rather_than_a_guess():
    """THE honest-answer test. A repo that ships I-Compact / I-Balanced has no
    published figure; deriving a percentage from the file size would be
    fabricating data, which is worse than an empty cell."""
    for q in ("GGUF", "COMPACT", "I-BALANCED", "APEX", "", "wat"):
        assert quality_of(q) is None
        assert size_vs_bf16(q, 1_000) is None


def test_an_unknown_ud_prefix_falls_back_to_the_plain_format():
    # UD-Q5_K_S isn't listed; it must read as Q5_K_S, not as nothing
    assert quality_of("UD-Q5_K_S") == quality_of("Q5_K_S")


def test_tier_names_track_the_loss():
    assert tier_for(0.0) == "lossless"
    assert tier_for(1.0) == "great"
    assert tier_for(9.0) == "poor"
    assert tier_for(50.0) == "damaged"


def test_size_ratio_is_bits_over_sixteen():
    # rounded to 4dp by size_vs_bf16 — a display ratio, not a precise quantity
    assert size_vs_bf16("Q8_0", 1) == pytest.approx(8.50 / 16, abs=1e-4)
    assert size_vs_bf16("Q4_K_M", 1) == pytest.approx(4.85 / 16, abs=1e-4)


# --- labels: "GGUF, GGUF, GGUF" ----------------------------------------------

def test_same_tag_files_get_distinguishing_labels():
    # the APEX repo: no standard quant tag anywhere in the names
    files = [
        "Qwen3.6-35B-A3B-heretic-APEX-I-Balanced.gguf",
        "Qwen3.6-35B-A3B-heretic-APEX-I-Quality.gguf",
        "Qwen3.6-35B-A3B-heretic-APEX-I-Compact.gguf",
    ]
    labels = hangar._distinct_quants(files)
    assert len(set(labels)) == 3, labels
    assert "GGUF" not in labels
    assert set(labels) == {"BALANCED", "QUALITY", "COMPACT"}


def test_real_quant_tags_are_left_alone():
    files = ["m-Q4_K_M.gguf", "m-Q6_K.gguf", "m-UD-Q3_K_XL.gguf"]
    assert hangar._distinct_quants(files) == ["Q4_K_M", "Q6_K", "UD-Q3_K_XL"]


def test_a_single_untagged_file_still_gets_a_label():
    assert hangar._distinct_quants(["mystery.gguf"]) == ["GGUF"]


# --- list_models wiring ------------------------------------------------------

def _spec(tmp_path, files):
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    return ModelSpec(
        slug="t", family="t", kind="dense", n_layers=32, full_attn_layers=32,
        kv_heads=8, head_dim=128, native_ctx=32768,
        cache_type_policy=CachePolicy(),
        ggufs=[GgufFile(repo="r", file=f, bytes=b, quant="GGUF")
               for f, b in files])


class _Reg:
    def __init__(self, spec):
        self.models = {"t": spec}
        self.gpus = []


def test_list_models_heals_stale_gguf_labels(tmp_path, monkeypatch):
    """A spec written before _distinct_quants existed has "GGUF" stored on every
    entry. Labels are derived on READ so those heal without rewriting specs."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = _spec(tmp_path, [("m-I-Quality.gguf", 3), ("m-I-Compact.gguf", 2)])
    assert [g.quant for g in spec.ggufs] == ["GGUF", "GGUF"]   # as stored
    out = hangar.list_models(_Reg(spec))
    got = [q["quant"] for q in out["models"][0]["quants"]]
    assert got == ["QUALITY", "COMPACT"], got


def test_list_models_attaches_quality_and_omits_fit_without_a_profile(
        tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    spec = _spec(tmp_path, [("m-Q4_K_M.gguf", 3), ("m-Q6_K.gguf", 4)])
    q = hangar.list_models(_Reg(spec))["models"][0]["quants"]
    assert q[0]["quality"]["tier"] == "great"       # Q4_K_M
    assert q[1]["quality"]["tier"] == "lossless"    # Q6_K
    # no profile passed -> no fit claimed, and no recommendation invented
    assert q[0]["fit"] == {}
    assert out_rec(spec, tmp_path) is None


def out_rec(spec, tmp_path):
    return hangar.list_models(_Reg(spec))["models"][0]["recommended"]


def test_fit_is_computed_when_a_profile_is_given(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    prof = HardwareProfile(
        gpus=[GpuInfo(vendor="amd", name="RX", vram_mb=16368,
                      backends=["vulkan"])],
        ram_mb=32768, ram_free_mb=24000, cpu=CpuInfo(cores=16),
        os="windows", disk_free_gb=400.0)
    spec = _spec(tmp_path, [("m-Q4_K_M.gguf", 8 * 2**30),
                            ("m-Q8_0.gguf", 60 * 2**30)])
    card = hangar.list_models(_Reg(spec), prof)["models"][0]
    small, huge = card["quants"]
    assert small["fit"]["ok"] is True
    assert small["fit"]["ctx"] >= 8192
    assert small["fit"]["speed"] in ("gpu", "light", "offload")
    assert huge["fit"]["ok"] is False          # 60GB on a 16GB card
    assert huge["fit"]["speed"] == "no"
    assert card["recommended"] == "Q4_K_M"     # the only one that runs


def test_the_models_page_and_the_hf_page_use_one_fit_implementation():
    """They disagreed by construction before: the fit math lived inside
    hf_browse's pre-download browser only."""
    import inspect

    from rigma import hf_browse, resolve
    assert hasattr(resolve, "quant_verdicts")
    assert "quant_verdicts" in inspect.getsource(hf_browse.inspect_repo)
    assert "quant_verdicts" in inspect.getsource(hangar.list_models)


# --- combined loss and the explorer knobs ------------------------------------

def test_total_loss_composes_weights_and_cache():
    from rigma.quant_quality import kv_loss, total_loss
    t = total_loss("Q4_K_M", "f16")
    assert t["kv_pct"] == 0.0
    assert t["total_pct"] == pytest.approx(t["weights_pct"], abs=0.01)
    # a lossy cache must make the TOTAL worse than the weights alone
    worse = total_loss("Q4_K_M", "q4_0")
    assert worse["total_pct"] > t["total_pct"]
    # ratios, not a plain sum
    w, c = worse["weights_pct"] / 100, worse["kv_pct"] / 100
    assert worse["total_pct"] == pytest.approx(((1 + w) * (1 + c) - 1) * 100,
                                               abs=0.01)
    assert kv_loss("q8_0") < kv_loss("q4_0")


def test_bf16_weights_with_f16_cache_is_the_zero_point():
    from rigma.quant_quality import total_loss
    assert total_loss("BF16", "f16")["total_pct"] == 0.0


def test_total_loss_is_none_for_an_unknown_label_or_cache():
    from rigma.quant_quality import total_loss
    assert total_loss("I-COMPACT", "q8_0") is None
    assert total_loss("Q4_K_M", "not_a_cache") is None


def _prof():
    from rigma.models import CpuInfo, GpuInfo, HardwareProfile
    return HardwareProfile(
        gpus=[GpuInfo(vendor="amd", name="RX", vram_mb=16304,
                      backends=["vulkan"])],
        ram_mb=32768, ram_free_mb=24000, cpu=CpuInfo(cores=16),
        os="windows", disk_free_gb=400.0)


def _hybrid_spec(file_gb, mmproj_gb=0.0):
    from rigma.models import CachePolicy, GgufFile, ModelSpec
    mm = (GgufFile(repo="r", file="mm.gguf", bytes=int(mmproj_gb * 2**30),
                   quant="mmproj") if mmproj_gb else None)
    return ModelSpec(
        slug="t", family="t", kind="dense", n_layers=65, full_attn_layers=16,
        kv_heads=4, head_dim=256, native_ctx=262144,
        cache_type_policy=CachePolicy(),
        ggufs=[GgufFile(repo="r", file="a.gguf",
                        bytes=int(file_gb * 2**30), quant="Q3_K_M")],
        mmproj=mm)


def test_dropping_vision_frees_context():
    """The projector is permanently resident and counted whether or not an image
    is ever sent — 888MB on Qwen3.8-27B, which was 4x the context."""
    from rigma.resolve import quant_verdicts
    spec, prof = _hybrid_spec(12.5, mmproj_gb=0.87), _prof()
    with_v = quant_verdicts(spec, prof)[0]
    without = quant_verdicts(spec, prof, vision=False)[0]
    assert without["ctx"] > with_v["ctx"]
    assert without["budget"]["mmproj_mb"] == 0
    assert with_v["budget"]["mmproj_mb"] > 800


def test_a_cheaper_cache_buys_context():
    from rigma.resolve import kv_bytes_per_token, quant_verdicts
    spec, prof = _hybrid_spec(11.0), _prof()
    f16 = quant_verdicts(spec, prof, kv="f16")[0]
    q40 = quant_verdicts(spec, prof, kv="q4_0")[0]
    assert q40["ctx"] > f16["ctx"]
    # per TOKEN it is cheaper; in absolute MB it is not, because the cheaper
    # cache is immediately spent on a bigger window
    assert (kv_bytes_per_token(spec, "q4_0", "q4_0")
            < kv_bytes_per_token(spec, "f16", "f16"))


def test_an_explicit_cache_choice_is_not_silently_replaced():
    """Without the pin, _cache_candidates falls back to q8_0 and f16 / q5_1 /
    q4_0 all reported the identical verdict — the explorer answered a question
    nobody asked."""
    from rigma.resolve import quant_verdicts
    spec, prof = _hybrid_spec(11.0), _prof()
    for want in ("f16", "q8_0", "q5_1", "q4_0"):
        v = quant_verdicts(spec, prof, kv=want)[0]
        assert v["kv"] == want and v["kv_v"] == want, (want, v)


def test_k_and_v_are_always_symmetric():
    """ComboFlags._symmetric_kv normalises them on purpose: llama.cpp's fused
    flash-attention kernel only fires when ctk == ctv, and a mismatch silently
    drops to a slow non-fused path (RDNA4, 2026-07-17). Every verdict must come
    back symmetric — an asymmetric one could never be launched."""
    from rigma.resolve import quant_verdicts
    for kv in ("", "f16", "q8_0", "q4_0"):
        for v in quant_verdicts(_hybrid_spec(11.0), _prof(), kv=kv):
            if v["ok"]:
                assert v["kv"] == v["kv_v"], (kv, v)


def test_kv_loss_takes_one_cache_type():
    """The module used to weight an asymmetric K:V pair 2:1 and the tooltip
    described it — arithmetic that could never run, about a trade rigma does
    not offer."""
    import inspect

    from rigma import quant_quality
    assert len(inspect.signature(quant_quality.kv_loss).parameters) == 1
    assert "_K_WEIGHT" not in dir(quant_quality)
    ui = (pathlib.Path(__file__).parent.parent / "frontend-v2" / "src"
          / "models" / "ModelsSurface.tsx").read_text(encoding="utf-8")
    assert "2:1" not in ui, "the tooltip still describes a weighting that cannot apply"


def test_context_policy_spends_layers_but_stays_bounded():
    """The budget is measured from the START, not per doubling — a per-rung
    allowance is spent again at every rung and quietly offloaded 26 layers."""
    from rigma.resolve import _GROW_LAYER_BUDGET, quant_verdicts
    spec, prof = _hybrid_spec(13.6), _prof()
    speed = quant_verdicts(spec, prof, grow="speed")[0]
    ctxp = quant_verdicts(spec, prof, grow="context")[0]
    assert ctxp["ctx"] >= speed["ctx"]
    assert ctxp["offload_pct"] <= round(_GROW_LAYER_BUDGET * 100) + 1


def test_the_ngl_sentinel_is_not_differenced_raw():
    """ngl=99 means "all layers", so 99 -> 62 of 65 is 3 layers lost, not 37.
    Differencing the sentinel made the context policy refuse every trade."""
    from rigma.resolve import quant_verdicts
    v = quant_verdicts(_hybrid_spec(12.5), _prof(), grow="context")[0]
    assert v["offload_pct"] <= 20, v


def test_budget_rows_add_up():
    from rigma.resolve import quant_verdicts
    b = quant_verdicts(_hybrid_spec(12.5, 0.87), _prof())[0]["budget"]
    total = b["file_mb"] + b["mmproj_mb"] + b["kv_mb"]
    assert b["over_mb"] == pytest.approx(total - b["budget_mb"], abs=2)


def test_overrides_never_mutate_the_shared_spec():
    """kv/vision come from a query string; the registry object is process-wide."""
    from rigma.resolve import quant_verdicts
    spec = _hybrid_spec(11.0, 0.87)
    quant_verdicts(spec, _prof(), kv="q4_0", vision=False)
    assert spec.mmproj is not None
    assert spec.cache_type_policy.k == "f16"


# --- labels carrying a quanter's own decoration -------------------------------

def test_a_prefixed_label_still_finds_its_format():
    """jaromer ships RVN-Q6_K.gguf. Demanding an exact table match left every
    row of that repo showing an em dash (owner, 2026-07-30). Enumerating every
    quanter's invented prefix is a losing race, so the format token is found
    INSIDE the label."""
    for label, want in [("RVN-Q6_K", "Q6_K"), ("Q4_K_M (RVN)", "Q4_K_M"),
                        ("FOO-IQ4_XS-BAR", "IQ4_XS"),
                        ("Q3_K_M (QWEN3.8-27B-HE)", "Q3_K_M")]:
        got, exp = quality_of(label), quality_of(want)
        assert got is not None, label
        assert got["ppl_pct"] == exp["ppl_pct"], (label, want)


def test_longest_token_wins_so_bf16_is_not_read_as_f16():
    # "F16" is a substring of "BF16"; they happen to share a loss figure, so
    # compare the bits instead — a shorter-first search would be a silent bug
    assert quality_of("RVN-BF16")["bpw"] == quality_of("BF16")["bpw"]
    assert quality_of("RVN-Q2_K_L")["bpw"] == quality_of("Q2_K_L")["bpw"]
    assert quality_of("RVN-Q2_K_L")["bpw"] != quality_of("Q2_K")["bpw"]


def test_a_name_with_no_format_token_is_still_unpriced():
    # the honest-answer guarantee must survive the substring search
    for label in ("GGUF", "COMPACT", "QUALITY", "BALANCED", "APEX", "mmproj"):
        assert quality_of(label) is None, label


def test_a_colliding_tag_keeps_the_tag_at_the_front():
    """Two files reducing to the same tag (RVN-Q4_K_M.gguf and
    Qwen3.8-27B-Heretic-Q4_K_M.gguf) used to fall back to a bare filename core,
    and truncating that to 24 chars ate the "_M" — leaving the row unreadable
    AND unpriceable."""
    labels = hangar._distinct_quants(
        ["RVN-Q4_K_M.gguf", "Qwen3.8-27B-Heretic-Q4_K_M.gguf"])
    assert len(set(labels)) == 2
    assert all(x.startswith("Q4_K_M") for x in labels), labels
    assert all(quality_of(x) is not None for x in labels)


def test_prefixed_files_keep_their_distinct_formats():
    files = ["RVN-Q6_K.gguf", "RVN-Q5_K_M.gguf", "RVN-IQ4_XS.gguf"]
    # no collision here, so the plain tags come through untouched
    assert hangar._distinct_quants(files) == ["Q6_K", "Q5_K_M", "IQ4_XS"]


# --- running a vision model text-only ----------------------------------------

def test_no_vision_is_persisted_and_sticky(tmp_path, monkeypatch):
    """A later ctx change must not silently reload the projector and eat the
    VRAM the user just freed, so the choice lives in state, not in the call."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import state as st
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=2, ctx=8192,
                   no_vision=True)
    assert st.read_state()["no_vision"] is True
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=2, ctx=8192)
    assert st.read_state()["no_vision"] is False      # default stays off


def test_state_without_the_key_reads_as_vision_on(tmp_path, monkeypatch):
    """State files written before this existed have no `no_vision` key; they
    must not be read as "vision was turned off"."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    import json

    from rigma import state as st
    st.state_path().parent.mkdir(parents=True, exist_ok=True)
    st.state_path().write_text(json.dumps({"model": "m", "ctx": 8192}),
                               encoding="utf-8")
    assert not bool((st.read_state() or {}).get("no_vision"))


def test_when_offloading_is_forced_the_cache_minimises_it():
    """Once nothing can be fully resident, cache precision stops being free.

    Live 2026-07-30: at 64K the resolver took f16 because it was first in the
    ladder and merely fit, spilling 22% of a dense model's weights; q8_0 spills
    8%. That trades ~0.06% perplexity against a PCIe round trip per offloaded
    layer on EVERY token — backwards.
    """
    from rigma.resolve import _spilled, fit_gguf
    spec, prof = _hybrid_spec(12.5), _prof()
    small = fit_gguf(spec, spec.ggufs[0], prof, 8192, [])
    big = fit_gguf(spec, spec.ggufs[0], prof, 65536, [])
    assert small and big
    # the small window still fits entirely, so it keeps the precise cache
    assert small.cache_type_k == "f16"
    assert _spilled(spec, small) == 0.0
    # the big one cannot, so it buys layers back with cache precision
    assert big.cache_type_k == "q8_0"
    assert _spilled(spec, big) < 0.20


def test_a_fully_resident_fit_still_prefers_the_precise_cache():
    """The minimise-offload rule must not leak into the strict pass — a cache
    that fits on the GPU is free, and quality wins there."""
    from rigma.resolve import _spilled, fit_gguf
    spec, prof = _hybrid_spec(6.0), _prof()
    f = fit_gguf(spec, spec.ggufs[0], prof, 8192, [])
    assert f.cache_type_k == "f16" and _spilled(spec, f) == 0.0


def test_spilled_counts_moe_experts_not_whole_layers():
    from rigma.models import MoESpec
    from rigma.resolve import _spilled
    dense = _hybrid_spec(10.0)
    moe = _hybrid_spec(10.0)
    moe.moe = MoESpec(total_b=35.0, active_b=3.0, expert_weight_fraction=0.85)
    from rigma.models import ComboFlags
    # 13 of 65 layers offloaded either way
    dense_f = ComboFlags(ctx=8192, ngl=52)
    moe_f = ComboFlags(ctx=8192, ngl=99, n_cpu_moe=13)
    assert _spilled(dense, dense_f) == pytest.approx(13 / 65)
    assert _spilled(moe, moe_f) == pytest.approx(13 / 65 * 0.85)


# --- measured bits per weight -------------------------------------------------
def test_measured_bpw_is_arithmetic_on_two_counted_quantities():
    from rigma.quant_quality import measured_bpw
    # 1 GB file, 2B parameters -> 4 bits each
    assert measured_bpw(2_000_000_000, 4_000_000_000) == 4.0


def test_measured_bpw_needs_both_terms():
    from rigma.quant_quality import measured_bpw
    assert measured_bpw(0, 1_000) is None
    assert measured_bpw(1_000, 0) is None


def test_a_label_with_no_known_format_still_reports_what_it_spends():
    """SC117's APEX quants are named I-Balanced / I-Quality / I-Compact, so
    quality_of returns None and the Models page rendered an em dash for the one
    MTP-capable model on this machine — three rows with no way to choose between
    them. Bits per weight is measured, not a quality figure invented from a
    size, so it can be shown where the reference table has nothing."""
    from rigma.quant_quality import measured_bpw, quality_of
    assert quality_of("I-COMPACT") is None
    assert measured_bpw(17_016_624_544, 35_505_251_456) == 3.83


def test_label_overstating_the_file_is_flagged():
    """The APEX I-Compact stamps general.file_type = 15 (Q4_K_M, 4.85 bpw) into
    a file that spends 3.83 — a custom mix inheriting its base ftype's name."""
    from rigma.quant_quality import label_overstates
    flag = label_overstates("Q4_K_M", 17_016_624_544, 35_505_251_456)
    assert flag is not None
    assert flag["measured_bpw"] == 3.83
    assert flag["label_bpw"] == 4.85
    assert flag["drift_pct"] < 0


def test_an_honest_label_is_not_flagged():
    from rigma.quant_quality import label_overstates
    # 4.85 bpw of a 10B model = 6.06 GB; the label describes the file
    assert label_overstates("Q4_K_M", int(4.85 * 10_000_000_000 / 8),
                            10_000_000_000) is None


def test_unknown_labels_have_nothing_to_contradict():
    from rigma.quant_quality import label_overstates
    assert label_overstates("I-COMPACT", 17_016_624_544, 35_505_251_456) is None


def test_size_vs_bf16_prefers_the_measurement_over_the_nominal_bpw():
    from rigma.quant_quality import size_vs_bf16
    nominal = size_vs_bf16("Q4_K_M", 17_016_624_544)
    measured = size_vs_bf16("Q4_K_M", 17_016_624_544, 35_505_251_456)
    assert nominal == round(4.85 / 16, 4)
    assert measured == round(3.83 / 16, 4)


def test_bpw_is_never_converted_into_a_quality_percentage():
    """A percentage would need a perplexity run. Bits per weight is the size of
    the budget, not the quality it buys — the module must not blur that."""
    from rigma.quant_quality import quality_of
    assert quality_of("I-COMPACT") is None      # still None, bpw notwithstanding
