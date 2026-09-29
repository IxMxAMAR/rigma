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


# --- the bound belongs to the ENGINE, not to a constant -----------------------
# The live false negative this fixes. MAINLINE_MAX_TYPE = 42 describes master, but
# the PINNED b9867 has GGML_TYPE_COUNT = 42 and so rejects id 42. A module-level
# bound therefore passed Q2_0 files that then died on the pinned engine with
# `invalid ggml type 42. should be in [0, 42)` — a check that says "fine" about a
# file the engine cannot open is worse than no check.

Q2_0_FILE = {0: 1, 8: 5, 42: 40}


def test_the_pinned_build_rejects_q2_0_even_though_master_has_it():
    """b9867 reports `[0, 42)`, i.e. COUNT = 42, i.e. it accepts 0..41 only."""
    c = engine_compat.check_types(Q2_0_FILE, engine_type_count=42)
    assert c.ok is False
    assert "Q2_0" in c.reason
    assert "42" in c.reason


def test_master_accepts_q2_0():
    """The same file on a build with COUNT = 43. Both answers are correct for their
    own build, which is exactly why the bound cannot be a constant."""
    assert engine_compat.check_types(Q2_0_FILE, engine_type_count=43).ok is True


def test_the_pinned_build_accepts_q1_0():
    """b9867 DOES have Q1_0 = 41 — the type added immediately before Q2_0. Getting
    this wrong in the other direction would refuse a file the pin loads."""
    assert engine_compat.check_types({0: 1, 41: 1}, engine_type_count=42).ok is True


def test_the_too_new_message_says_a_pin_bump_WILL_help():
    """The opposite advice to the fork case, and the distinction matters: a fork
    needs that fork, whereas a too-new mainline type needs a newer MAINLINE build.
    Telling a Q2_0 user to install a fork would send them somewhere useless."""
    c = engine_compat.check_types(Q2_0_FILE, engine_type_count=42)
    assert "no fork is needed" in c.advice
    assert not c.fork_types


def test_a_fork_type_IS_loadable_by_the_build_that_defines_it():
    """A fork type stops being a rejection once the engine's own table contains it.

    `engine_type_count` is the `N` from the engine's `should be in [0, N)` refusal, so it
    is a TABLE SIZE and a build with table 144 accepts every id below it — PQ2_0=142
    included. The old version of this test asserted the opposite ("never excused by a high
    count") and so pinned a FALSE NEGATIVE: it told a user running PrismML's own binary
    that their model could not load, and advised them to install PrismML's own binary.

    The naming fact it was reaching for is still asserted below, on the path where it is
    actually true — an engine that does NOT have the type.
    """
    c = engine_compat.check_types(BONSAI, engine_type_count=144)
    assert c.ok is True, c.reason


def test_a_fork_type_is_still_named_as_a_fork_when_the_engine_lacks_it():
    """The naming question survives independently of the capability question.

    A mainline build has no PQ2_0 at all, so the useful answer is not "upgrade" — no
    mainline version will ever have it — but "this needs PrismML's fork". That is what
    `forks` is for, and it must still fire here.
    """
    for bound in (42, 43):
        c = engine_compat.check_types(BONSAI, engine_type_count=bound)
        assert c.ok is False, bound
        assert "NO mainline build" in c.reason, c.reason
        assert c.fork_types.get(142), c.fork_types


def test_the_default_bound_describes_current_mainline():
    """With no engine context the checker is still useful, but it must be describing
    mainline's current table rather than pretending to know a specific build."""
    assert engine_compat.check_types(Q2_0_FILE).ok is True


def test_an_engine_count_below_every_used_type_rejects_the_file():
    """A very old build. The mechanism must be general, not special-cased to Q2_0."""
    c = engine_compat.check_types({0: 1, 14: 5}, engine_type_count=10)
    assert c.ok is False
    assert "Q6_K" in c.reason


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


# --- the engine side: fingerprinting a build from its own error ---------------
# Verbatim from llama.cpp gguf.cpp:
#   "%s: tensor '%s' has invalid ggml type %d. should be in [0, %d)\n"
# The bound is GGML_TYPE_COUNT. It is NOT a capability list (it counts seven
# permanently-unused holes), but it IS a precise readout of which enum the binary
# was compiled from, so it fingerprints mainline-vs-fork.

REAL_ERROR = ("0.00.128.269 E gguf_init_from_reader: tensor 'output.weight' has "
              "invalid ggml type 142. should be in [0, 42)")


def test_the_bound_is_parsed_out_of_a_real_error():
    assert engine_compat.parse_type_count(REAL_ERROR) == 42


def test_a_missing_bound_is_none_not_zero():
    """0 would read as a real (and absurd) count and fingerprint the engine as
    something it is not."""
    assert engine_compat.parse_type_count("") is None
    assert engine_compat.parse_type_count("some other error") is None


def test_b9867s_bound_fingerprints_it_as_mainline_before_q2_0():
    who, why = engine_compat.provenance(42)
    assert who == "mainline llama.cpp"
    assert "Q2_0" in why


def test_the_forks_bound_fingerprints_it_as_prismml():
    who, why = engine_compat.provenance(144)
    assert who == "PrismML-Eng/llama.cpp"
    assert "142" in why


def test_an_unrecognised_count_names_no_vendor():
    """A count we have never seen is evidence of nothing. Naming the wrong vendor
    would send a user to install the wrong engine — worse than saying nothing."""
    who, why = engine_compat.provenance(777)
    assert who == ""
    assert "777" in why


def test_a_missing_count_names_no_vendor():
    who, _ = engine_compat.provenance(None)
    assert who == ""


def test_the_verdict_now_says_which_build_the_engine_is():
    """The improvement over the raw error: not 'it refused' but 'this engine is
    mainline before Q2_0, and the model needs a fork'."""
    c = engine_compat.check_engine(BONSAI, engine_type_count=42)
    assert c.ok is False
    assert "42" in c.reason
    assert "mainline llama.cpp" in c.reason
    assert "PQ2_0" in c.reason


def test_an_engine_that_accepts_the_model_is_still_ok():
    c = engine_compat.check_engine(SMOL, engine_type_count=43)
    assert c.ok is True


def test_an_unknown_engine_count_still_reports_the_model_problem():
    """The model-side verdict must not depend on recognising the engine."""
    c = engine_compat.check_engine(BONSAI, engine_type_count=None)
    assert c.ok is False
    assert "PQ2_0" in c.reason


def test_vulkan_advice_no_longer_claims_a_missing_kernel():
    """The old advice said PQ2_0 has no Vulkan kernel. It does, and the claim was false.

    PrismML's fork README lists "Vulkan and SYCL backend support" and carries a Vulkan row
    in its build table; its release ships `bin-win-vulkan-x64`; and measured on an RX 9070
    XT that build loaded this file's 402 PQ2_0 tensors and served it. The old note was
    read from a card listing Metal/CUDA/HIP/CPU as *preferred* — a preference, not a
    capability — so this now says what is actually useful instead.
    """
    c = engine_compat.check_engine(BONSAI, engine_type_count=42, backend="vulkan")
    assert c.ok is False
    assert "Vulkan" in c.advice
    assert "no Vulkan kernel" not in c.advice
    # And the correction it should carry: the legacy file it used to recommend is the one
    # that will NOT load.
    assert "deprecated" in c.advice


def test_a_non_vulkan_backend_gets_no_vulkan_note():
    c = engine_compat.check_engine(BONSAI, engine_type_count=42, backend="rocm")
    assert "Vulkan" not in c.advice


def test_an_engine_whose_table_contains_a_fork_type_accepts_the_model():
    """The end-to-end consequence: PrismML's build loads a PrismML model, with no note.

    This is the pair that the two copies of the rule disagreed about —
    `engine_registry.select` said the registered engine "accepts every type this model
    uses" while `check_engine` said it could not, for the same file and the same build.
    """
    c = engine_compat.check_engine(BONSAI, engine_type_count=144, backend="vulkan")
    assert c.ok is True, c.reason
    assert c.advice == "", c.advice


def test_a_truncated_read_is_still_not_a_verdict_on_the_engine_path():
    """The engine path must inherit the same honesty rule as the model path."""
    c = engine_compat.check_engine(BONSAI, engine_type_count=42, complete=False)
    assert c.ok is True
