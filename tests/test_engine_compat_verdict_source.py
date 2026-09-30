"""A5: `_engine_compat_note` must follow the ENGINE's verdict and name its source.

The engine's own refusal is authoritative: a registered fork may load a file the
model-side heuristic calls unloadable. When the engine cannot be asked at all, the
model-side heuristic is the only evidence left — and the note must admit that,
instead of reusing the model-side guess as if the engine had said it.
"""
from __future__ import annotations

from rigma import cli, engine_compat

BONSAI = {0: 1, 30: 1, 142: 100}   # fork type 142 (PQ2_0)


class _Idx:
    def __init__(self, counts, complete=True):
        self.type_counts = counts
        self.types_complete = complete


def _model_side_says_cannot_load(monkeypatch):
    monkeypatch.setattr(engine_compat, "check_gguf",
                        lambda p: engine_compat.check_types(BONSAI))


def test_engine_loadable_overrides_the_model_side_verdict(monkeypatch):
    """The engine ran and did NOT refuse the file, so it loads — authoritative, even
    though the model-side heuristic flags fork type 142."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: None)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_engine_unavailable_uses_the_model_side_check_and_says_so(monkeypatch):
    """No oracle / failed probe: the model-side heuristic is all there is, and the
    note must name it as the source rather than implying the engine confirmed it."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count",
                        lambda e, m: cli._ENGINE_UNASKED)
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert "CANNOT LOAD" in note
    assert "model-side" in note
    assert "could not be asked" in note


def test_engine_refusal_names_the_engine_as_the_source(monkeypatch):
    """The engine refused, so the note is the engine's verdict and says so."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    import rigma.gguf_meta as gm
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(BONSAI))
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert "CANNOT LOAD" in note
    assert "source:" in note
    assert "type-table bound" in note


def test_a_failed_engine_explanation_is_not_the_model_side_verdict(monkeypatch):
    """A5's stale-`c` bug: the engine refused, but the histogram re-read failed, so the
    old code printed the model-side `check_gguf` result under the engine's banner. An
    unknown must stay unknown."""
    _model_side_says_cannot_load(monkeypatch)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    import rigma.gguf_meta as gm

    def boom(p):
        raise OSError("histogram read failed")

    monkeypatch.setattr(gm, "read_tensor_index", boom)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""
