"""How much quality a quant gives up, and how many bits per weight it keeps.

WHAT THIS IS NOT: a measurement of the model in front of you. Real quality loss
is measured by running perplexity or KL-divergence over a corpus with the actual
weights loaded — it needs the multi-GB file downloaded and a full eval pass, and
it lands differently for every model. Nothing in a repo listing can tell you
that, and a per-model percentage invented from a file size would be a lie
dressed as data.

WHAT IT IS: published reference figures for the llama.cpp quant FORMATS. The
formats have fixed construction (bits per weight is a property of the format,
not the model), and their typical perplexity cost over BF16 has been measured
repeatedly on LLaMA-family models. Those numbers transfer well enough to rank
quants — which is the actual decision being made on the Models page — and badly
enough that they must be shown as approximate, which is why every value here is
a BAND and the UI renders a "≈".

Two systematic caveats, both stated in the UI:
  * bigger models degrade LESS than these figures at the same quant, and these
    are drawn largely from 7B-13B evals, so on a 27B they are pessimistic;
  * "UD-" (Unsloth Dynamic) quants mix per-tensor bit widths and beat the plain
    format of the same name, so they get their own, better, entries.

`ppl_pct` = typical % increase in perplexity vs BF16. Lower is better.
`bpw`     = nominal bits per weight of the format, used for the size math.
"""
from __future__ import annotations

# fmt: off
# format          bpw    ppl_pct   one-line character
_TABLE: dict[str, tuple[float, float, str]] = {
    "F32":        (32.0,  0.00, "unquantised"),
    "BF16":       (16.0,  0.00, "reference — no loss by definition"),
    "F16":        (16.0,  0.00, "reference — no loss by definition"),
    "Q8_0":       ( 8.50, 0.05, "effectively lossless"),
    "UD-Q8_K_XL": ( 8.75, 0.03, "lossless; keeps sensitive tensors wider"),
    "Q6_K":       ( 6.56, 0.10, "no loss you will notice"),
    "UD-Q6_K_XL": ( 6.80, 0.08, "no loss you will notice"),
    "Q5_K_M":     ( 5.69, 0.35, "very close to the original"),
    "UD-Q5_K_XL": ( 5.90, 0.25, "very close to the original"),
    "Q5_K_S":     ( 5.54, 0.50, "very close to the original"),
    "Q5_1":       ( 6.00, 0.60, "superseded by Q5_K_M"),
    "Q5_0":       ( 5.50, 0.80, "superseded by Q5_K_S"),
    "Q4_K_M":     ( 4.85, 0.90, "the usual sweet spot"),
    "UD-Q4_K_XL": ( 5.10, 0.65, "sweet spot, dynamic — best 4-bit"),
    "Q4_K_S":     ( 4.58, 1.50, "solid 4-bit"),
    "IQ4_NL":     ( 4.50, 1.30, "solid 4-bit, non-linear"),
    "IQ4_XS":     ( 4.25, 1.70, "leanest good 4-bit"),
    "Q4_1":       ( 4.78, 2.10, "legacy — prefer Q4_K_M"),
    "Q4_0":       ( 4.34, 2.70, "legacy — prefer Q4_K_S"),
    # bartowski's "_L"/"_XL" variants keep embeddings and the output tensor at
    # Q8_0 while the rest matches the base format. Slightly bigger, slightly
    # better than the base — those two tensors are disproportionately sensitive.
    "Q6_K_L":     ( 6.80, 0.08, "Q6_K with q8_0 embed/output"),
    "Q5_K_L":     ( 5.90, 0.28, "Q5_K_M with q8_0 embed/output"),
    "Q4_K_L":     ( 5.10, 0.70, "Q4_K_M with q8_0 embed/output"),
    "Q3_K_XL":    ( 4.10, 2.40, "Q3_K_L with q8_0 embed/output"),
    "Q2_K_L":     ( 2.90, 9.00, "Q2_K with q8_0 embed/output"),
    "IQ3_XS":     ( 3.30, 6.20, "degraded"),
    "IQ2_S":      ( 2.50, 14.0, "heavy damage"),
    "IQ2_XS":     ( 2.31, 16.0, "heavy damage"),
    "Q3_K_L":     ( 4.27, 2.60, "noticeable drift on long output"),
    "Q3_K_M":     ( 3.91, 3.60, "noticeable drift on long output"),
    "UD-Q3_K_XL": ( 4.10, 2.40, "dynamic — much better than plain Q3"),
    "Q3_K_S":     ( 3.50, 6.00, "degraded; last resort above Q2"),
    "UD-IQ3_XXS": ( 3.30, 4.50, "dynamic 3-bit — usable"),
    "IQ3_M":      ( 3.70, 4.20, "usable 3-bit"),
    "IQ3_S":      ( 3.44, 5.50, "degraded"),
    "IQ3_XXS":    ( 3.06, 7.50, "degraded"),
    "Q2_K":       ( 2.63, 14.0, "heavy damage — reasoning suffers first"),
    "UD-Q2_K_XL": ( 2.90, 8.50, "dynamic 2-bit — the only 2-bit worth running"),
    "UD-IQ2_M":   ( 2.70, 9.50, "dynamic 2-bit"),
    "IQ2_M":      ( 2.70, 12.0, "heavy damage"),
    "UD-IQ2_XXS": ( 2.20, 13.0, "heavy damage; fits when nothing else will"),
    "IQ2_XXS":    ( 2.06, 20.0, "severe damage"),
    "IQ1_M":      ( 1.75, 35.0, "experimental"),
    "IQ1_S":      ( 1.56, 45.0, "experimental"),
    "MXFP4":      ( 4.25, 1.60, "4-bit float"),
}
# fmt: on

# ppl_pct -> a word, so the UI can colour it without re-deciding the thresholds
_TIERS = ((0.15, "lossless"), (0.6, "excellent"), (1.2, "great"),
          (2.5, "good"), (5.0, "fair"), (10.0, "poor"))


def tier_for(ppl_pct: float) -> str:
    for limit, name in _TIERS:
        if ppl_pct <= limit:
            return name
    return "damaged"


def quality_of(quant: str) -> dict | None:
    """Reference quality for a quant LABEL, or None when it isn't a known
    format. None is the honest answer for a repo that names its files
    I-Compact / I-Balanced: nobody has published figures for those, and
    guessing from the size would be fabrication."""
    key = (quant or "").strip().upper()
    row = _TABLE.get(key)
    if row is None and key.startswith("UD-"):
        row = _TABLE.get(key[3:])          # fall back to the plain format
    if row is None:
        # an unlisted "_L"/"_XL" variant reads as its base format rather than
        # as nothing: every quanter invents suffixes, and enumerating them all
        # is a losing race. The base is a safe (slightly pessimistic) stand-in.
        for suffix in ("_XL", "_L"):
            if key.endswith(suffix):
                row = _TABLE.get(key[: -len(suffix)])
                if row:
                    break
    if row is None:
        return None
    bpw, ppl, note = row
    return {"bpw": bpw, "ppl_pct": ppl, "tier": tier_for(ppl), "note": note,
            "basis": "reference"}


# KV-cache quantisation, same footing as the weight table above: published
# reference figures for the FORMAT, as a % perplexity increase over an f16
# cache. Much smaller numbers than the weight table because the cache is a
# smaller share of the error budget — but they behave differently, and the
# difference matters when choosing:
#
#   * weight error is baked in once, offline, with per-block scales and
#     importance weighting from imatrix calibration;
#   * cache error is applied per token AS IT IS WRITTEN, with plain
#     round-to-nearest, and every later token attends over the whole degraded
#     history — so it compounds with sequence length, which is exactly when
#     you wanted the long context.
#
# K is weighted more heavily than V below because K feeds the softmax: K error
# perturbs WHICH token is attended to, while V error only blurs the value
# retrieved. llama.cpp practice is to hold K at q8_0 and quantise V harder.
_KV_PPL = {
    "f16":  0.00,
    "bf16": 0.00,
    "q8_0": 0.06,
    "q5_1": 0.35,
    "q5_0": 0.60,
    "q4_1": 0.90,
    "q4_0": 1.80,
}
_K_WEIGHT = 2 / 3      # K carries two-thirds of the cache's contribution


def kv_loss(k: str, v: str | None = None) -> float | None:
    """Reference % perplexity increase for a KV cache type, vs an f16 cache.
    Asymmetric pairs are weighted K:V = 2:1 — an approximation, and flagged as
    one wherever it is shown."""
    v = v or k
    a, b = _KV_PPL.get((k or "").lower()), _KV_PPL.get((v or "").lower())
    if a is None or b is None:
        return None
    return round(a * _K_WEIGHT + b * (1 - _K_WEIGHT), 3)


def total_loss(quant: str, kv_k: str, kv_v: str | None = None) -> dict | None:
    """Everything given up vs the true reference: BF16 weights + f16 cache.

    The two sources are independent, so their perplexity RATIOS compose rather
    than add: (1+w)(1+c) - 1. At these magnitudes that is within a whisker of
    w + c, but it stays correct if either term gets large (a 2-bit weight quant
    with a q4_0 cache is not a small correction).

    This is the number worth comparing across rows — a Q3 model with a q4_0
    cache and a Q6 model with an f16 cache are two different totals, and the
    weight column alone cannot tell you which you are actually running.
    """
    w = quality_of(quant)
    c = kv_loss(kv_k, kv_v)
    if w is None or c is None:
        return None
    total = ((1 + w["ppl_pct"] / 100) * (1 + c / 100) - 1) * 100
    return {"weights_pct": w["ppl_pct"], "kv_pct": c,
            "total_pct": round(total, 2), "tier": tier_for(total),
            "basis": "reference"}


def size_vs_bf16(quant: str, nbytes: int) -> float | None:
    """This file as a fraction of what BF16 would weigh, from the format's bits
    per weight. Exact arithmetic on a nominal bpw — so it is a good size ratio
    and says NOTHING about quality on its own."""
    q = quality_of(quant)
    if not q or not q["bpw"] or nbytes <= 0:
        return None
    return round(q["bpw"] / 16.0, 4)
