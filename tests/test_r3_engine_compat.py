"""R3-ENG-2 — decide whether an engine can load a model from the file itself.

The failure this prevents, reproduced on the owner's machine:

    gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 142.
                           should be in [0, 42)

Rigma planned `ternary-bonsai-2-27b`, downloaded it, launched the pinned engine, and
got that. The information needed to predict it was in the model's own tensor table
the whole time — `read_tensor_index` already read every tensor's ggml type id (it
sits between the dims and the offset, so it cannot be skipped) and threw it away.

The type histograms below are VERBATIM from real files on this machine.
"""
from __future__ import annotations

from rigma import engine_compat

# Real files, read with read_tensor_index.
SMOL = {0: 1, 8: 12, 11: 40, 20: 5}        # SmolLM2-135M-Instruct-Q2_K: loads fine
RVN = {0: 1, 8: 12, 11: 40, 12: 30, 13: 20, 14: 10}   # RVN-Q3_K_M-mtp: loads fine
BONSAI = {0: 1, 30: 1, 142: 100}           # Ternary-Bonsai-2-27B-PQ2_0: CANNOT load


def test_an_ordinary_model_passes():
    assert engine_compat.check_types(SMOL).ok
    assert engine_compat.check_types(RVN).ok


def test_the_real_failing_model_is_caught_from_the_header():
    """The whole feature. No engine is launched and no engine is downloaded — a few
    hundred KB of GGUF header is enough to know."""
    c = engine_compat.check_types(BONSAI)
    assert c.ok is False
    assert c.blocked
    assert 142 in c.fork_types


def test_the_message_names_the_fork_not_just_a_number():
    """'invalid ggml type 142' tells a user nothing they can act on. The enum name
    and the project that owns it do."""
    c = engine_compat.check_types(BONSAI)
    assert "PQ2_0" in c.reason
    assert "PrismML-Eng/llama.cpp" in c.reason


def test_the_advice_says_a_pin_bump_will_not_help():
    """The single most important sentence here. Type 142 is fork-private numbering
    far above mainline's maximum, so 'upgrade llama.cpp' — the obvious response, and
    the one Rigma's own error invites — cannot work."""
    c = engine_compat.check_types(BONSAI)
    assert "will not help" in c.advice
    assert "PrismML-Eng/llama.cpp" in c.advice


def test_the_reason_states_that_no_mainline_build_can_load_it():
    """Not 'your build is old' but 'this numbering is not mainline's at all'."""
    c = engine_compat.check_types(BONSAI)
    assert "NO mainline build" in c.reason


def test_ptq1_0_is_recognised_as_the_same_fork():
    c = engine_compat.check_types({0: 1, 143: 50})
    assert c.ok is False
    assert "PTQ1_0" in c.reason
    assert "PrismML-Eng/llama.cpp" in c.reason


def test_a_future_mainline_type_is_reported_differently_from_a_fork_type():
    """Both block, but they need different actions: a fork needs that fork, whereas
    an unrecognised high number might just be a newer mainline."""
    c = engine_compat.check_types({0: 1, 200: 5})
    assert c.ok is False
    assert "different engine" in c.reason or "newer engine" in c.reason
    assert not c.fork_types


def test_every_mainline_type_is_accepted():
    """The known-type table must not be stricter than mainline itself, or Rigma
    would refuse models that load perfectly well."""
    for tid in engine_compat.KNOWN_TYPES:
        assert engine_compat.check_types({0: 1, tid: 1}).ok, tid


def test_the_boundary_is_exactly_mainline_max():
    assert engine_compat.check_types({0: 1, 42: 1}).ok      # Q2_0: the last one
    assert engine_compat.check_types({0: 1, 43: 1}).ok is False


# --- honest degradation ------------------------------------------------------

def test_a_truncated_read_is_not_a_verdict():
    """A partial histogram cannot show a file is loadable, but it also must not
    manufacture an incompatibility — reporting a false block would refuse models
    that are fine."""
    c = engine_compat.check_types(SMOL, complete=False)
    assert c.ok is True


def test_an_unreadable_file_is_not_reported_as_incompatible(tmp_path):
    """Absent is not the same as broken. Claiming a model cannot load because its
    header could not be read would block perfectly good files."""
    c = engine_compat.check_gguf(tmp_path / "missing.gguf")
    assert c.ok is True
    assert "could not read" in c.reason


def test_an_empty_histogram_is_not_an_incompatibility():
    assert engine_compat.check_types({}).ok
    assert engine_compat.check_types(None).ok


def test_a_bad_type_beats_a_good_one():
    """One unloadable tensor makes the whole file unloadable, however many good
    tensors surround it."""
    c = engine_compat.check_types({**SMOL, 142: 1})
    assert c.ok is False


# --- the reader that feeds it -------------------------------------------------

def test_the_type_histogram_counts_every_tensor(tmp_path):
    """R3-ENG-2's enabling change: the ggml type id was already read and discarded.
    Capturing it must not change any other reading."""
    from rigma.gguf_meta import TensorIndex
    idx = TensorIndex(n_tensors=3, params=100, type_counts={0: 2, 142: 1})
    assert idx.types_complete is True
    assert idx.ggml_types == [0, 142]
    assert idx.params == 100, "capturing types must not disturb the existing reads"


def test_a_truncated_index_reports_no_types():
    """None, not a partial list: a partial list of types would read as a complete
    one and could hide the very type that makes the file unloadable."""
    from rigma.gguf_meta import TensorIndex
    idx = TensorIndex(n_tensors=99, type_counts={0: 5}, truncated=True)
    assert idx.ggml_types is None
    assert idx.types_complete is False
