"""C1: `--ctx-checkpoints` / `-ctxcp`, the hybrid rewind-cost lever.

`--checkpoint-min-step` (already emitted, 4096) sets the SPACING between context
checkpoints; the COUNT is a separate flag. Both exist at the pinned engines:

  * fork     PrismML-Eng/llama.cpp 87268f77 `common/arg.cpp:1687`
  * mainline ggml-org/llama.cpp b9867       `common/arg.cpp:1449`

The count defaults to 32 per slot (`common/common.h` 87268f77:619, b9867:623)
and, unlike `--checkpoint-min-step`, the handler checks no range at all — it just
assigns `params.n_ctx_checkpoints = value` (87268f77:1690-1692, b9867:1452-1454).

The count is what bounds how far back a recurrent hybrid can rewind without
reprocessing; Rigma already emits the spacing for exactly that reason. These
tests assert on the built argv and never launch an engine.
"""
import pytest
from pydantic import ValidationError

from rigma.models import ComboFlags, GgufFile, RunPlan


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096, **fl),
                   origin="calculator")


def test_ctx_checkpoints_absent_by_default():
    """Unset must change nothing: the flag is omitted and the engine keeps 32."""
    args = _plan().server_args("/m", 11500)
    assert "--ctx-checkpoints" not in args
    assert "-ctxcp" not in args


def test_ctx_checkpoints_emitted_when_set():
    args = _plan(ctx_checkpoints=8).server_args("/m", 11500)
    assert args[args.index("--ctx-checkpoints") + 1] == "8"


def test_ctx_checkpoints_zero_is_a_value_not_an_unset_sentinel():
    """0 disables checkpoints and is distinct from the -1 'no opinion' default."""
    args = _plan(ctx_checkpoints=0).server_args("/m", 11500)
    assert args[args.index("--ctx-checkpoints") + 1] == "0"


def test_ctx_checkpoints_sits_beside_checkpoint_min_step():
    """Both halves of the checkpoint surface are emitted, and the count does not
    replace the spacing Rigma already relied on."""
    args = _plan(ctx_checkpoints=16).server_args("/m", 11500)
    assert args.index("--ctx-checkpoints") < args.index("--checkpoint-min-step")
    assert args[args.index("--checkpoint-min-step") + 1] == "4096"


def test_ctx_checkpoints_rejects_a_nonsense_negative():
    with pytest.raises(ValidationError):
        ComboFlags(ctx=4096, ctx_checkpoints=-2)


def test_ctx_checkpoints_accepts_the_unset_sentinel():
    assert ComboFlags(ctx=4096, ctx_checkpoints=-1).ctx_checkpoints == -1
