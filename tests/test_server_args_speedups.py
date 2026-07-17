import pytest

from rigma.models import ComboFlags, GgufFile, RunPlan


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096, **fl),
                   origin="calculator")


# --- Task 1: batch / ubatch -------------------------------------------------
def test_batch_flags_emitted_when_set():
    args = _plan(batch=16384, ubatch=2048).server_args("/m", 11500)
    assert args[args.index("-b") + 1] == "16384"
    assert args[args.index("-ub") + 1] == "2048"


def test_batch_flags_absent_by_default():
    args = _plan().server_args("/m", 11500)
    assert "-b" not in args and "-ub" not in args


# --- Task 2: symmetric KV ---------------------------------------------------
def test_asymmetric_kv_upgraded_to_more_precise():
    f = ComboFlags(ctx=4096, cache_type_k="q4_0", cache_type_v="f16")
    assert f.cache_type_k == "f16" and f.cache_type_v == "f16"


def test_symmetric_kv_untouched():
    f = ComboFlags(ctx=4096, cache_type_k="q8_0", cache_type_v="q8_0")
    assert f.cache_type_k == "q8_0" and f.cache_type_v == "q8_0"


# --- Task 5: spec_type validation ------------------------------------------
def test_spec_type_valid_accepted():
    ComboFlags(ctx=4096, spec_type="draft-mtp")
    ComboFlags(ctx=4096, spec_type="ngram-simple")


def test_spec_type_unknown_rejected():
    with pytest.raises(ValueError):
        ComboFlags(ctx=4096, spec_type="turbo-nonsense")


def test_ngram_spec_emitted():
    a = _plan(spec_type="ngram-simple").server_args("/m", 11500)
    assert a[a.index("--spec-type") + 1] == "ngram-simple"
