"""R3-ENG-2/4 wiring — the pre-flight compat check must not cry wolf.

`_engine_compat_note` is what a user actually sees from `rigma plan --verify` and
`rigma up --verify`. It decides between three outcomes, and the middle one is the
whole reason it is not simply `engine_compat.check_gguf`:

  1. the model is fine for mainline          -> say nothing
  2. the model needs a fork AND the engine
     refuses it                              -> explain, and name the fix
  3. the model needs a fork BUT the engine
     is that fork and loads it                -> say nothing

Case 3 is a real bug that live testing caught: the model-side check alone reported a
problem for a model the registered PrismML fork loads perfectly well. A warning about
a model that loads is worse than no warning.
"""
from __future__ import annotations

from pathlib import Path

from rigma import cli, engine_compat

BONSAI = {0: 1, 30: 1, 142: 100}   # real Ternary-Bonsai-2-27B-PQ2_0 histogram
SMOL = {0: 1, 8: 12, 11: 40, 20: 5}


class _Idx:
    def __init__(self, counts, complete=True):
        self.type_counts = counts
        self.types_complete = complete


def _wire(monkeypatch, counts, *, count, complete=True):
    """Stub the two things the helper reads: the file, and the engine."""
    monkeypatch.setattr(engine_compat, "check_gguf",
                        lambda p: engine_compat.check_types(counts, complete=complete))
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: count)
    import rigma.gguf_meta as gm
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(counts, complete))


def test_an_ordinary_model_produces_no_note(monkeypatch):
    _wire(monkeypatch, SMOL, count=42)
    assert cli._engine_compat_note("m.gguf", "exe", "vulkan") == ""


def test_a_model_the_engine_refuses_is_explained(monkeypatch):
    """Case 2: the pinned b9867 fingerprints as type count 42 and cannot load 142."""
    _wire(monkeypatch, BONSAI, count=42)
    note = cli._engine_compat_note("m.gguf", "exe", "rocm")
    assert "CANNOT LOAD" in note
    assert "PQ2_0" in note
    assert "PrismML-Eng/llama.cpp" in note
    assert "fix:" in note


def test_a_model_the_engine_LOADS_produces_no_note(monkeypatch):
    """Case 3, the bug live testing caught. `_engine_type_count` returns None when
    the engine did not refuse the file — which is the authoritative answer and must
    override the model-side guess, or a registered fork would be warned about for
    every model it was installed to serve."""
    _wire(monkeypatch, BONSAI, count=None)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_an_unreadable_model_produces_no_note(monkeypatch):
    """Absence of evidence is not a warning. Claiming a model cannot load because
    its header could not be read would be a false alarm on a good file."""
    def boom(p):
        raise OSError("unreadable")

    monkeypatch.setattr(engine_compat, "check_gguf", boom)
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_a_truncated_model_read_produces_no_note(monkeypatch):
    """Same rule one level down: a partial histogram is not a verdict."""
    _wire(monkeypatch, BONSAI, count=42, complete=False)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_the_note_uses_the_engines_own_bound_not_a_constant(monkeypatch):
    """The R3-ENG-4 rule reaching the user: the message must describe THIS build.
    A Q2_0 file is refused by a build reporting 42 and accepted by one reporting 144.

    The model-side stub is given the same bound as the engine here, which is what
    `check_gguf` would do on a machine whose engine is already known — the point is
    that the ENGINE's number decides, not a module constant.
    """
    import rigma.gguf_meta as gm
    q2 = {0: 1, 42: 40}
    monkeypatch.setattr(
        engine_compat, "check_gguf",
        lambda p: engine_compat.check_types(q2, engine_type_count=42))
    monkeypatch.setattr(gm, "read_tensor_index", lambda p: _Idx(q2))
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 42)
    note = cli._engine_compat_note("m.gguf", "exe", "vulkan")
    assert "CANNOT LOAD" in note
    assert "Q2_0" in note

    # The same file on a build whose table reaches 144 is fine.
    monkeypatch.setattr(cli, "_engine_type_count", lambda e, m: 144)
    assert cli._engine_compat_note("m.gguf", "exe", "rocm") == ""


def test_vulkan_advice_is_included_when_the_model_needs_a_fork(monkeypatch):
    """PQ2_0 has no Vulkan kernel even on the fork that defines it."""
    _wire(monkeypatch, BONSAI, count=42)
    note = cli._engine_compat_note("m.gguf", "exe", "vulkan")
    assert "Vulkan" in note


# --- the engine fingerprint --------------------------------------------------

def test_a_missing_oracle_yields_no_fingerprint():
    """No oracle means no measurement, and no measurement must not become a verdict."""
    assert cli._engine_type_count(Path("nope") / "llama-server.exe",
                                  Path("model.gguf")) is None
