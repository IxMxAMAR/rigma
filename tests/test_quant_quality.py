"""The Models page's fit + quality columns.

Owner report 2026-07-30: 21 quants offered with nothing but a file size to
choose between them, no "does this fit my VRAM", no context per quant, and one
model whose three rows all read "GGUF".
"""
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
