"""Speculative decoding: one vocabulary, three engines, honest refusals.

Every fact asserted here is read from upstream source or a measured binary, and
the ones that are NOT verified are asserted as unverified on purpose: the whole
design is that "we could not ask" must never be reported as "not supported".
"""
import json

import pytest

from rigma import engines
from rigma import spec_decode as sd
from rigma.models import ComboFlags, GgufFile, RunPlan

# --- the vocabulary ---------------------------------------------------------


def _plan(**fl):
    return RunPlan(model_slug="m",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=4096, **fl),
                   origin="calculator")


def test_a_mode_a_stored_spec_can_already_hold_is_still_known():
    """LEGACY_MODES is what shipped specs, registry combos and calibration rows
    contain. Dropping one would make an existing model unlaunchable."""
    for mode in sd.LEGACY_MODES:
        assert ComboFlags(ctx=4096, spec_type=mode).spec_type == mode


def test_every_mode_rigma_offers_passes_its_own_validation():
    for mode in sd.MODES:
        assert ComboFlags(ctx=4096, spec_type=mode).spec_type == mode


def test_the_legacy_spellings_are_a_subset_of_the_new_ones():
    assert sd.LEGACY_MODES <= set(sd.MODES)


def test_every_llamacpp_type_is_one_the_engine_documents():
    assert set(sd.LLAMACPP_TYPE.values()) <= sd.LLAMACPP_SPEC_TYPES


def test_peagle_is_the_only_mode_without_a_llamacpp_type():
    """llama.cpp has no P-EAGLE; mapping it onto draft-eagle3 would be a silent
    lie about what runs, so it is absent and `support` refuses it by name."""
    assert [m for m in sd.MODES if m not in sd.LLAMACPP_TYPE] == [sd.DRAFT_PEAGLE]


def test_every_mode_that_needs_a_draft_file_says_which_head_it_accepts():
    """`draft-simple` is the exception by definition: it is the "just give me a
    smaller model of the same family" mode, so any draft head goes."""
    for mode in sd.NEEDS_DRAFT_FILE - {sd.DRAFT_SIMPLE}:
        assert mode in sd.DRAFT_HEAD_ACCEPTS
    assert sd.DRAFT_SIMPLE not in sd.DRAFT_HEAD_ACCEPTS
    assert sd.head_mismatch(sd.DRAFT_SIMPLE, "whatever") is None


def test_mtp_is_the_only_target_side_head():
    assert sd.TARGET_HEAD == {sd.DRAFT_MTP: "mtp"}


# --- the llama.cpp argv -----------------------------------------------------


def test_dflash2_is_asked_for_as_dflash():
    """Upstream has ONE --spec-type for the family; the variant comes from the
    file. The alias exists so the user can SAY dflash2 and be checked."""
    args = sd.llamacpp_spec_args(sd.DRAFT_DFLASH2, n_max=15,
                                 draft="C:/m/dflash2.gguf")
    assert args[:2] == ["--spec-type", "draft-dflash"]


def test_a_file_draft_goes_through_md_and_a_repo_through_spec_draft_hf():
    a = sd.llamacpp_spec_args(sd.DRAFT_DFLASH, n_max=3, draft="C:/m/d.gguf")
    assert a[a.index("-md") + 1] == "C:/m/d.gguf"
    b = sd.llamacpp_spec_args(sd.DRAFT_DFLASH, n_max=3,
                              draft="z-lab/Qwen3.8-27B-DFlash2")
    assert b[b.index("--spec-draft-hf") + 1] == "z-lab/Qwen3.8-27B-DFlash2"
    assert "-md" not in b


def test_the_confidence_cut_is_emitted_only_when_it_is_on():
    on = sd.llamacpp_spec_args(sd.DRAFT_DSPARK, n_max=7, conf_min=0.4)
    assert on[on.index("--spec-draft-conf-min") + 1] == "0.4"
    off = sd.llamacpp_spec_args(sd.DRAFT_DSPARK, n_max=7, conf_min=0.0)
    assert "--spec-draft-conf-min" not in off


def test_no_speculation_emits_no_flags():
    assert sd.llamacpp_spec_args(sd.NONE, n_max=3) == []
    assert sd.llamacpp_spec_args("", n_max=3) == []


def test_an_unset_draft_depth_is_left_to_the_engine():
    """0 is the "no opinion" sentinel here (as for batch/ubatch/ngl), and a
    literal `--spec-draft-n-max 0` would mean "draft nothing"."""
    args = sd.llamacpp_spec_args(sd.DRAFT_DFLASH, n_max=0, draft="C:/m/d.gguf")
    assert "--spec-draft-n-max" not in args
    assert args[:2] == ["--spec-type", "draft-dflash"]


def test_peagle_refuses_rather_than_emitting_a_type_llama_cpp_lacks():
    with pytest.raises(ValueError, match="no --spec-type"):
        sd.llamacpp_spec_args(sd.DRAFT_PEAGLE, n_max=3)


def test_an_unknown_mode_refuses_too():
    with pytest.raises(ValueError):
        sd.llamacpp_spec_args("draft-turbo", n_max=3)


@pytest.mark.parametrize("value,kind", [
    ("C:/models/d.gguf", "path"),
    ("./d.gguf", "path"),
    ("models\\d.gguf", "path"),
    ("z-lab/Qwen3.8-27B-DFlash2", "hf"),
    ("", ""),
])
def test_draft_shape_decides_which_flag_is_used(value, kind):
    assert sd.draft_shape(value) == kind


def test_a_bare_word_is_neither_a_path_nor_a_repo_id():
    """The two shapes select two different engine flags, so a value that is
    neither is a typo rather than a guess Rigma should make."""
    with pytest.raises(ValueError, match="neither"):
        sd.draft_shape("somefile")


def test_the_flags_model_rejects_a_draft_it_cannot_classify():
    with pytest.raises(ValueError):
        ComboFlags(ctx=4096, spec_draft="somefile")


def test_the_flags_model_rejects_an_impossible_confidence():
    with pytest.raises(ValueError):
        ComboFlags(ctx=4096, spec_conf_min=1.5)
    with pytest.raises(ValueError):
        ComboFlags(ctx=4096, spec_conf_min=-0.1)
    assert ComboFlags(ctx=4096, spec_conf_min=0.25).spec_conf_min == 0.25


def test_the_launch_defaults_model_validates_the_mode_too():
    """`set_launch_defaults` is the API path; a stored mode it accepts must be
    one `--spec` would accept, or the UI could store a launch the CLI refuses."""
    from rigma.models import LaunchDefaults
    assert LaunchDefaults(spec_type="draft-dflash2").spec_type == "draft-dflash2"
    with pytest.raises(ValueError):
        LaunchDefaults(spec_type="turbo-nonsense")
    with pytest.raises(ValueError):
        LaunchDefaults(spec_draft="somefile")


def test_the_plan_argv_carries_the_translation_end_to_end():
    a = _plan(spec_type="draft-dflash2", spec_n_max=15,
              spec_draft="C:/m/dflash2.gguf").server_args("/m", 11500)
    assert a[a.index("--spec-type") + 1] == "draft-dflash"
    assert a[a.index("--spec-draft-n-max") + 1] == "15"
    assert a[a.index("-md") + 1] == "C:/m/dflash2.gguf"


def test_the_plan_argv_is_unchanged_when_speculation_is_off():
    a = _plan().server_args("/m", 11500)
    assert "--spec-type" not in a and "-md" not in a


# --- capability, measured ---------------------------------------------------

# Mainline master: every documented type, both DSpark lineages and DFlash2.
MASTER = sd.EngineSpecCaps(
    server_exe="master", spec_types=sd.LLAMACPP_SPEC_TYPES,
    flags=frozenset({"--spec-type", "--spec-draft-conf-min"}),
    markers=frozenset({"dflash2", "dspark", "dspark-prism"}),
    markers_probed=True)

# The pin, b9867: no draft-dspark at all, and its dflash.cpp is v1 (no selector
# and no convolution markers — measured).
B9867 = sd.EngineSpecCaps(
    server_exe="b9867",
    spec_types=frozenset({
        "none", "draft-simple", "draft-eagle3", "draft-dflash", "draft-mtp",
        "ngram-cache", "ngram-simple", "ngram-map-k", "ngram-map-k4v",
        "ngram-mod"}),
    flags=frozenset({"--spec-type", "--spec-draft-n-max"}),
    markers=frozenset(), markers_probed=True)

# A build with the mainline DSpark loader only.
MAINLINE_ONLY = sd.EngineSpecCaps(
    server_exe="mainline", spec_types=sd.LLAMACPP_SPEC_TYPES,
    markers=frozenset({"dflash2", "dspark"}), markers_probed=True)


def test_the_pin_cannot_be_asked_for_dspark():
    v = sd.support(sd.DRAFT_DSPARK, engines.LLAMACPP, B9867)
    assert v.supported is False
    assert "does not accept --spec-type draft-dspark" in v.reason
    assert "draft-dflash" in v.reason          # says what it DOES accept


def test_the_pin_accepts_the_type_but_cannot_load_a_dflash2_file():
    """This is the case a version-number check would get wrong: b9867 lists
    draft-dflash, so only the loader's own marker shows DFlash2 is absent."""
    v = sd.support(sd.DRAFT_DFLASH2, engines.LLAMACPP, B9867)
    assert v.supported is False
    assert "cannot load" in v.reason and "27342" in v.reason


def test_dflash_v1_still_runs_on_the_pin():
    assert sd.support(sd.DRAFT_DFLASH, engines.LLAMACPP, B9867).supported is True


def test_master_supports_the_whole_family():
    for mode in (sd.DRAFT_MTP, sd.DRAFT_DFLASH, sd.DRAFT_DFLASH2,
                 sd.DRAFT_DSPARK, sd.DRAFT_EAGLE3):
        assert sd.support(mode, engines.LLAMACPP, MASTER).supported is True


def test_an_unasked_engine_is_unverified_not_unsupported():
    """"No engine binary to ask" must warn, not refuse: refusing here would make
    every mode unusable before the pin is downloaded."""
    v = sd.support(sd.DRAFT_DSPARK, engines.LLAMACPP, None)
    assert v.supported is None
    assert not v                            # bool() is False, but not a refusal
    assert "unverified" in v.reason


def test_a_failed_probe_carries_its_reason_into_the_verdict():
    caps = sd.EngineSpecCaps(server_exe="x", error="no engine at x")
    v = sd.support(sd.DRAFT_MTP, engines.LLAMACPP, caps)
    assert v.supported is None and "no engine at x" in v.reason


def test_dspark_on_master_names_the_lineage_the_draft_must_match():
    v = sd.support(sd.DRAFT_DSPARK, engines.LLAMACPP, MAINLINE_ONLY)
    assert v.supported is True
    assert "mainline" in v.caveat


def test_a_build_with_no_dspark_marker_at_all_refuses_the_mode():
    caps = sd.EngineSpecCaps(server_exe="old", spec_types=sd.LLAMACPP_SPEC_TYPES,
                             markers=frozenset({"dflash2"}),
                             markers_probed=True)
    v = sd.support(sd.DRAFT_DSPARK, engines.LLAMACPP, caps)
    assert v.supported is False and "predates DSpark" in v.reason


def test_peagle_is_refused_on_llamacpp_whichever_build_is_asked():
    v = sd.support(sd.DRAFT_PEAGLE, engines.LLAMACPP, MASTER)
    assert v.supported is False and "no --spec-type" in v.reason


def test_an_invented_mode_is_refused_by_name():
    v = sd.support("draft-turbo", engines.LLAMACPP, MASTER)
    assert v.supported is False and "not a speculative mode" in v.reason


def test_capability_refusal_is_the_one_line_form_both_launch_paths_use():
    assert sd.capability_refusal(sd.DRAFT_DSPARK, B9867)
    assert sd.capability_refusal(sd.DRAFT_DSPARK, MASTER) == ""
    # Unverified is not a refusal — the UI relaunch must not be blocked by a
    # probe that could not run either.
    assert sd.capability_refusal(sd.DRAFT_DSPARK, None) == ""


def test_speculation_off_is_supported_everywhere():
    assert sd.support(sd.NONE, engines.LLAMACPP, B9867).supported is True
    assert sd.support(sd.NONE, engines.VLLM).supported is True


# --- the artefact gate ------------------------------------------------------


def test_a_plain_dflash_file_under_dflash2_is_refused_with_the_mode_that_fits():
    why = sd.head_mismatch(sd.DRAFT_DFLASH2, "dflash")
    assert why and "dflash2 draft head" in why
    assert "--spec draft-dflash" in why


def test_a_dflash2_file_under_dflash2_passes():
    assert sd.head_mismatch(sd.DRAFT_DFLASH2, "dflash2") is None


def test_a_dflash2_file_under_plain_dflash_runs_but_is_remarked_on():
    """Upstream loads it under the same type, so this is not a refusal — but the
    user asked for the older family member and is told what they got."""
    assert sd.head_mismatch(sd.DRAFT_DFLASH, "dflash2") is None
    assert "DFlash2" in sd.variant_note(sd.DRAFT_DFLASH, "dflash2")
    assert sd.variant_note(sd.DRAFT_DFLASH, "dflash") == ""


def test_a_draft_file_with_no_head_at_all_is_refused():
    why = sd.head_mismatch(sd.DRAFT_DSPARK, None)
    assert why and "no draft head" in why


def test_mtp_is_refused_when_the_target_file_carries_no_mtp_tensors():
    why = sd.head_mismatch(sd.DRAFT_MTP, None, target=True)
    assert why and "blk.N.nextn" in why
    # The same call on a non-target file is not this file's question.
    assert sd.head_mismatch(sd.DRAFT_MTP, None, target=False) is None


def test_mtp_passes_when_the_target_carries_them():
    assert sd.head_mismatch(sd.DRAFT_MTP, "mtp", target=True) is None


def test_ngram_modes_need_no_artefact():
    assert sd.head_mismatch(sd.NGRAM_SIMPLE, None) is None
    assert sd.head_mismatch(sd.NONE, None) is None


def test_a_prism_draft_on_a_mainline_build_is_refused():
    why = sd.artifact_lineage_ok(sd.DRAFT_DSPARK, "dspark-prism", MAINLINE_ONLY)
    assert why and "PrismML-lineage" in why


def test_a_mainline_draft_on_a_prism_only_build_is_refused():
    caps = sd.EngineSpecCaps(server_exe="prism", spec_types=sd.LLAMACPP_SPEC_TYPES,
                             markers=frozenset({"dspark-prism"}),
                             markers_probed=True)
    why = sd.artifact_lineage_ok(sd.DRAFT_DSPARK, "dspark", caps)
    assert why and "mainline-lineage" in why


def test_a_build_with_both_lineages_accepts_either():
    assert sd.artifact_lineage_ok(sd.DRAFT_DSPARK, "dspark", MASTER) is None
    assert sd.artifact_lineage_ok(sd.DRAFT_DSPARK, "dspark-prism", MASTER) is None


def test_an_unscanned_build_refuses_no_lineage():
    """Silence is not evidence: a build whose binary was never scanned must not
    have its draft refused on lineage grounds."""
    caps = sd.EngineSpecCaps(server_exe="x", spec_types=sd.LLAMACPP_SPEC_TYPES)
    assert sd.artifact_lineage_ok(sd.DRAFT_DSPARK, "dspark-prism", caps) is None


def test_lineage_is_only_a_question_for_dspark():
    assert sd.artifact_lineage_ok(sd.DRAFT_DFLASH2, "dflash2", MAINLINE_ONLY) is None


# --- vLLM -------------------------------------------------------------------


def test_peagle_is_eagle3_with_parallel_drafting():
    """vLLM's SpeculativeMethod literal has no `peagle`; its own table gives
    P-EAGLE as method eagle3 + parallel_drafting."""
    body = sd.vllm_speculative_config(sd.DRAFT_PEAGLE, n_max=4, draft="u/d")
    assert body == {"method": "eagle3", "parallel_drafting": True,
                    "num_speculative_tokens": 4, "model": "u/d"}


def test_dflash2_is_served_as_the_dflash_method():
    """DFlash2 is an ARCHITECTURE (qwen3_dflash2) selected by the draft
    checkpoint; there is no `dflash2` method to ask for."""
    body = sd.vllm_speculative_config(sd.DRAFT_DFLASH2, n_max=15,
                                      draft="./ckpt")
    assert body["method"] == "dflash"


def test_every_vllm_method_is_one_vllm_documents():
    for mode, method in sd.VLLM_METHOD.items():
        assert method in sd.VLLM_METHOD_VALUES, mode


def test_mtp_needs_no_draft_checkpoint_and_dflash_does():
    assert sd.vllm_needs_draft(sd.DRAFT_MTP) is False
    assert sd.vllm_needs_draft(sd.NGRAM_SIMPLE) is False
    assert sd.vllm_needs_draft(sd.DRAFT_DFLASH) is True
    assert sd.vllm_needs_draft(sd.DRAFT_DSPARK) is True
    body = sd.vllm_speculative_config(sd.DRAFT_MTP, n_max=1)
    assert "model" not in body and body["method"] == "mtp"


def test_the_vllm_flag_is_one_json_object_under_the_documented_flag():
    flag = sd.vllm_speculative_flag(sd.DRAFT_DSPARK, n_max=7, draft="u/d")
    assert flag[0] == "--speculative-config"
    assert json.loads(flag[1])["method"] == "dspark"


def test_an_unknown_mode_has_no_vllm_method():
    with pytest.raises(ValueError, match="no vLLM speculative method"):
        sd.vllm_speculative_config("draft-turbo", n_max=3)


def test_the_vllm_method_table_matches_the_verified_literal():
    """READ from vllm/config/speculative.py (main): `SpeculativeMethod` is
    `ngram, medusa, mlp_speculator, draft_model, suffix, custom_class,
    EagleModelTypes, NgramGPUTypes, DSparkModelTypes`, where the nested literals
    add `eagle`, `eagle3`, `extract_hidden_states`, `dflash`, 28 MTP spellings
    and `dspark`. `peagle` and `dflash2` are absent from the whole file, so a
    table that emitted either would be rejected by pydantic inside vLLM — this
    test is the pin, and it fails if someone "fixes" the mapping back."""
    assert {"dspark", "dflash", "eagle3", "mtp", "ngram"} <= sd.VLLM_METHOD_VALUES
    assert "peagle" not in sd.VLLM_METHOD_VALUES
    assert "dflash2" not in sd.VLLM_METHOD_VALUES
    assert sd.VLLM_METHOD[sd.DRAFT_DSPARK] == "dspark"
    assert sd.VLLM_METHOD[sd.DRAFT_DFLASH2] == "dflash"
    assert sd.VLLM_METHOD[sd.DRAFT_PEAGLE] == "eagle3"


def test_mtp_is_emitted_as_the_current_spelling_not_the_deprecated_one():
    """vLLM rewrites every MTPModelTypes value to `mtp` and warns that the rest
    are deprecated, so `deepseek_mtp` must never be what Rigma sends."""
    for mode in (sd.DRAFT_MTP,):
        body = sd.vllm_speculative_config(mode, n_max=3)
        assert body["method"] == "mtp" and "deepseek_mtp" not in str(body)


def test_an_unset_count_is_reported_where_the_checkpoint_may_not_carry_one():
    """A speculators checkpoint brings its own num_speculative_tokens; a bare
    draft directory does not, and the project's own DSpark example sets it."""
    assert sd.vllm_count_caveat(sd.DRAFT_DSPARK, 0)
    assert sd.vllm_count_caveat(sd.DRAFT_DFLASH2, 0)
    assert sd.vllm_count_caveat(sd.DRAFT_DSPARK, 7) == ""
    # MTP and the n-gram methods have no separate checkpoint to carry a count.
    assert sd.vllm_count_caveat(sd.DRAFT_MTP, 0) == ""
    assert sd.vllm_count_caveat(sd.NGRAM_SIMPLE, 0) == ""


def test_vllm_support_reports_the_method_and_the_caveat():
    v = sd.support(sd.DRAFT_PEAGLE, engines.VLLM)
    assert v.supported is True and "eagle3" in v.reason
    assert "parallel_drafting" in v.caveat


def test_vllm_argv_carries_the_speculative_config():
    argv = engines.vllm_argv("meta-llama/Llama-3-8B", port=8000,
                             spec_mode="draft-dspark", spec_n_max=7,
                             spec_draft="z-lab/dspark-draft")
    assert "--speculative-config" in argv
    body = json.loads(argv[argv.index("--speculative-config") + 1])
    assert body == {"method": "dspark", "num_speculative_tokens": 7,
                    "model": "z-lab/dspark-draft"}
    # No llama.cpp flag may leak into a vLLM command line.
    assert "--spec-type" not in argv and "-md" not in argv


def test_vllm_argv_refuses_a_model_based_method_with_no_draft():
    with pytest.raises(ValueError, match="draft checkpoint"):
        engines.vllm_argv("m", port=8000, spec_mode="draft-dflash")


def test_vllm_argv_without_speculation_is_unchanged():
    argv = engines.vllm_argv("m", port=8000)
    assert "--speculative-config" not in argv


# --- the probe --------------------------------------------------------------


def test_parse_help_reads_the_bracketed_type_list():
    text = ("  --spec-type [none|draft-simple|draft-dflash|draft-dspark|"
            "draft-mtp]      (default: none)\n"
            "  --spec-draft-conf-min P   confidence threshold\n")
    types, flags = sd.parse_help(text)
    assert "draft-dspark" in types and "ngram-mod" not in types
    assert "--spec-draft-conf-min" in flags and "--spec-type" in flags


def test_parse_help_reads_the_comma_separated_list_the_builds_actually_print():
    """MEASURED: b9867 and prism-b10743 print
    `--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,...`
    with no brackets. Reading only the bracketed doc form reported every mode as
    "unverified" on this machine, which is the failure this test pins."""
    text = ("-h, --help, --usage   print usage and exit\n"
            "--spec-type none,draft-simple,draft-eagle3,draft-mtp,draft-dflash,"
            "ngram-simple,ngram-mod\n"
            "                                        comma-separated list of "
            "types (default: none)\n"
            "--spec-draft-n-max N   number of tokens to draft (default: 3)\n")
    types, flags = sd.parse_help(text)
    assert types == frozenset({"none", "draft-simple", "draft-eagle3",
                               "draft-mtp", "draft-dflash", "ngram-simple",
                               "ngram-mod"})
    assert "--spec-draft-n-max" in flags


def test_parse_help_does_not_invent_types_from_a_neighbouring_flag():
    text = "--spec-type N   comma-separated list\n--spec-draft-n-max N\n"
    types, _ = sd.parse_help(text)
    assert types == frozenset()


def test_parse_help_on_a_build_with_no_spec_support_returns_nothing():
    types, flags = sd.parse_help("  -h, --help   show this help\n")
    assert types == frozenset() and "--spec-type" not in flags


def test_the_probe_reports_what_the_binary_actually_advertises(tmp_path):
    exe = tmp_path / "llama-server.exe"
    # A real binary carries the tensor names as string literals; that is what
    # makes the marker scan evidence rather than a version guess.
    exe.write_bytes(b"\x00" * 32 + b"selector_hidden" + b"\x00" * 8)

    class _Cp:
        returncode = 0
        stdout = "  --spec-type [none|draft-dflash]\n  --spec-draft-n-max N\n"
        stderr = ""

    caps = sd.probe_llamacpp(exe, popen=lambda *a, **k: _Cp())
    assert caps.known and caps.spec_types == frozenset({"none", "draft-dflash"})
    assert caps.markers == frozenset({"dflash2"}) and caps.markers_probed


def test_a_binary_that_cannot_be_asked_is_unknown_not_empty(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")

    def _boom(*a, **k):
        raise OSError("missing dll")

    caps = sd.probe_llamacpp(exe, popen=_boom)
    assert caps.spec_types is None and not caps.known
    assert "missing dll" in caps.error


def test_a_help_that_names_no_types_is_unknown_not_empty(tmp_path):
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")

    class _Cp:
        returncode = 0
        stdout = "usage: llama-server [options]\n"
        stderr = ""

    caps = sd.probe_llamacpp(exe, popen=lambda *a, **k: _Cp())
    assert caps.spec_types is None
    assert "names no --spec-type" in caps.error


def test_a_missing_binary_is_reported_by_path():
    caps = sd.probe_llamacpp("no/such/llama-server")
    assert caps.spec_types is None and "no engine at" in caps.error


def test_a_marker_split_across_a_read_boundary_is_still_found(tmp_path):
    """The scan reads in chunks, so a name straddling the boundary would be
    missed without the overlap."""
    exe = tmp_path / "blob.bin"
    exe.write_bytes(b"x" * (sd._MARKER_CHUNK - 3) + b"markov_head_a")
    assert "dspark-prism" in (sd.scan_markers(exe) or frozenset())


def test_an_unreadable_binary_reports_no_scan_rather_than_no_markers(tmp_path):
    assert sd.scan_markers(tmp_path / "nope.bin") is None


def test_a_marker_in_a_dll_beside_the_exe_is_found(tmp_path):
    """MEASURED layout: the shipped Windows engine is a 9 KB `llama-server.exe`
    shim plus the real code in DLLs. Scanning only the executable would report
    "scanned and absent" for a build that supports the mode — a false refusal,
    which is the one direction this gate must never fail in."""
    (tmp_path / "llama-server.exe").write_bytes(b"\x00" * 64)
    (tmp_path / "llama.dll").write_bytes(b"\x00" * 32 + b"selector_hidden")
    assert sd.scan_markers(tmp_path / "llama-server.exe") == frozenset({"dflash2"})


def test_markers_in_several_binaries_are_unioned(tmp_path):
    (tmp_path / "llama-server.exe").write_bytes(b"\x00")
    (tmp_path / "llama-common.dll").write_bytes(b"markov_w1")
    (tmp_path / "llama.dll").write_bytes(b"markov_head_a")
    assert sd.scan_markers(tmp_path / "llama-server.exe") == frozenset(
        {"dspark", "dspark-prism"})


def test_unrelated_files_beside_the_exe_are_not_scanned(tmp_path):
    """A model file dropped next to the engine is not evidence about the
    engine; only binaries are."""
    (tmp_path / "llama-server.exe").write_bytes(b"\x00")
    (tmp_path / "notes.txt").write_bytes(b"selector_hidden")
    (tmp_path / "model.gguf").write_bytes(b"markov_w1")
    assert sd.scan_markers(tmp_path / "llama-server.exe") == frozenset()


# --- the launch gate --------------------------------------------------------
#
# `cli._spec_verdict` is the one place every way of choosing a mode passes
# through (an explicit --spec, a stored launch default, the UI). Its contract:
# refuse ONLY what the engine is documented to fail on, warn about everything
# Rigma could not establish.


class _Flags:
    def __init__(self, **kw):
        self.spec_type = ""
        self.spec_draft = ""
        self.spec_conf_min = 0.0
        self.spec_p_min = 0.0
        self.spec_n_max = 3
        self.__dict__.update(kw)

    def model_copy(self, *, update=None):
        # Mirrors pydantic's: the gate assigns the result back, and a real
        # ComboFlags is immutable, so the stub must hand back a new object.
        clone = _Flags(**{k: v for k, v in self.__dict__.items()})
        clone.__dict__.update(update or {})
        return clone


class _Rp:
    def __init__(self, **kw):
        from types import SimpleNamespace
        self.flags = _Flags(**kw)
        self.gguf = SimpleNamespace(file="f.gguf", mtp=None)


def test_the_gate_refuses_a_draft_file_that_is_not_there():
    from rigma import cli
    refusal, _ = cli._spec_verdict(
        _Rp(spec_type="draft-dflash2", spec_draft="C:/nope/never.gguf"))
    assert refusal and "not on disk" in refusal


def test_the_gate_refuses_a_mode_the_engine_cannot_be_asked_for(monkeypatch):
    from rigma import cli
    monkeypatch.setattr(sd, "probe_llamacpp",
                        lambda *a, **k: B9867)
    refusal, _ = cli._spec_verdict(
        _Rp(spec_type="draft-dspark", spec_draft="C:/x/d.gguf"),
        engine_exe="llama-server.exe")
    assert refusal and "does not accept --spec-type draft-dspark" in refusal


def test_the_gate_warns_rather_than_refuses_when_the_engine_is_unknown(
        tmp_path, monkeypatch):
    """The pin may not be downloaded yet; that is not evidence of anything."""
    from rigma import cli
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    monkeypatch.setattr("rigma.hangar.draft_head_of_file", lambda p: "dflash2")
    refusal, warn = cli._spec_verdict(
        _Rp(spec_type="draft-dflash2", spec_draft=str(draft)))
    assert refusal == ""
    assert "unverified" in warn


def test_the_gate_lets_a_plain_dflash_file_through_for_plain_dflash(
        tmp_path, monkeypatch):
    from rigma import cli
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: MASTER)
    monkeypatch.setattr("rigma.hangar.draft_head_of_file", lambda p: "dflash2")
    refusal, warn = cli._spec_verdict(
        _Rp(spec_type="draft-dflash", spec_draft=str(draft)),
        engine_exe="llama-server.exe")
    assert refusal == ""
    assert "DFlash2" in warn


def test_the_gate_refuses_a_dflash2_mode_on_a_plain_dflash_file(
        tmp_path, monkeypatch):
    from rigma import cli
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: MASTER)
    monkeypatch.setattr("rigma.hangar.draft_head_of_file", lambda p: "dflash")
    refusal, _ = cli._spec_verdict(
        _Rp(spec_type="draft-dflash2", spec_draft=str(draft)),
        engine_exe="llama-server.exe")
    assert refusal and "--spec draft-dflash" in refusal


def test_a_draft_file_that_cannot_be_read_is_refused_not_assumed(
        tmp_path, monkeypatch):
    """A file whose tensors Rigma cannot read is not evidence that it matches;
    the unsafe direction is to launch it anyway."""
    from rigma import cli
    junk = tmp_path / "junk.gguf"
    junk.write_bytes(b"not a gguf at all")
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: MASTER)
    refusal, _ = cli._spec_verdict(
        _Rp(spec_type="draft-dspark", spec_draft=str(junk)),
        engine_exe="llama-server.exe")
    assert refusal and "no draft head" in refusal


def test_the_gate_refuses_mtp_on_a_target_without_the_tensors(monkeypatch):
    from rigma import cli
    from rigma import hangar
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: MASTER)
    monkeypatch.setattr(hangar, "file_has_mtp", lambda gguf: False)
    refusal, _ = cli._spec_verdict(_Rp(spec_type="draft-mtp"),
                                   engine_exe="llama-server.exe")
    assert refusal and "blk.N.nextn" in refusal


def test_the_gate_says_nothing_when_speculation_is_off():
    from rigma import cli
    assert cli._spec_verdict(_Rp()) == ("", "")


def test_the_gate_warns_that_a_confidence_cut_is_a_dspark_lever():
    from rigma import cli
    _, warn = cli._spec_verdict(
        _Rp(spec_type="ngram-simple", spec_conf_min=0.5))
    assert "DSpark lever" in warn


# --- flags the build does not have -----------------------------------------
#
# MEASURED on this machine (both `llama-server --help` dumps): the pinned
# mainline b9867 and PrismML prism-b10743 both advertise --spec-draft-p-min and
# NEITHER advertises --spec-draft-conf-min. A flag Rigma sends anyway is an
# argparse exit after the download, so the probe decides what may be emitted.

MEASURED_PIN_FLAGS = frozenset({
    "--spec-type", "--spec-draft-n-max", "--spec-draft-n-min",
    "--spec-draft-p-min", "--spec-draft-p-split", "--spec-draft-model",
    "--spec-draft-hf", "--spec-draft-ngl",
})


def test_the_p_min_lever_is_emitted_and_is_the_one_both_builds_have():
    args = sd.llamacpp_spec_args(sd.DRAFT_DSPARK, n_max=7, p_min=0.15)
    assert args[args.index("--spec-draft-p-min") + 1] == "0.15"
    assert "--spec-draft-p-min" in MEASURED_PIN_FLAGS


def test_a_flag_the_build_lacks_is_reported_not_sent():
    caps = sd.EngineSpecCaps(server_exe="pin", flags=MEASURED_PIN_FLAGS)
    missing = sd.unadvertised_flags(caps, conf_min=0.4, p_min=0.1)
    assert missing == [("spec_conf_min", "--spec-draft-conf-min")]


def test_nothing_is_dropped_when_the_binary_was_never_asked():
    """Silence is not evidence: an unprobed engine must not cost the user a
    setting. `flags` is None, not empty, in exactly that case."""
    caps = sd.EngineSpecCaps(server_exe="pin")
    assert sd.unadvertised_flags(caps, conf_min=0.4, p_min=0.1) == []
    assert sd.unadvertised_flags(None, conf_min=0.4, p_min=0.1) == []


def test_an_hf_draft_on_a_build_without_the_flag_is_reported():
    caps = sd.EngineSpecCaps(server_exe="old",
                             flags=frozenset({"--spec-type"}))
    assert sd.unadvertised_flags(caps, draft="z-lab/dflash2") == [
        ("spec_draft_hf", "--spec-draft-hf")]
    assert sd.unadvertised_flags(caps, draft="C:/m/d.gguf") == []


def test_the_gate_drops_an_unadvertised_confidence_cut_and_says_so(
        tmp_path, monkeypatch):
    from rigma import cli
    draft = tmp_path / "d.gguf"
    draft.write_bytes(b"GGUF")
    monkeypatch.setattr("rigma.hangar.draft_head_of_file", lambda p: "dspark")
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: sd.EngineSpecCaps(
        server_exe="pin", spec_types=sd.LLAMACPP_SPEC_TYPES,
        flags=MEASURED_PIN_FLAGS, markers=frozenset({"dspark"}),
        markers_probed=True))
    rp = _Rp(spec_type="draft-dspark", spec_draft=str(draft),
             spec_conf_min=0.4, spec_p_min=0.1)
    refusal, warn = cli._spec_verdict(rp, engine_exe="llama-server.exe")
    assert refusal == ""
    assert "--spec-draft-conf-min" in warn and "dropped" in warn
    # The setting is really gone, so the argv cannot carry the flag.
    assert rp.flags.spec_conf_min == 0.0
    assert rp.flags.spec_p_min == 0.1


def test_the_gate_refuses_an_hf_draft_the_build_cannot_fetch(monkeypatch):
    from rigma import cli
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: sd.EngineSpecCaps(
        server_exe="old", spec_types=sd.LLAMACPP_SPEC_TYPES,
        flags=frozenset({"--spec-type", "--spec-draft-n-max"}),
        markers=frozenset({"dspark"}), markers_probed=True))
    refusal, _ = cli._spec_verdict(
        _Rp(spec_type="draft-dspark", spec_draft="z-lab/dspark-draft"),
        engine_exe="llama-server.exe")
    assert refusal and "--spec-draft-hf" in refusal


def test_the_gate_says_a_draft_is_not_loaded_by_a_mode_that_drafts_itself(
        monkeypatch):
    """`--spec-draft x.gguf` with `--spec draft-mtp` used to reach the engine as
    `-md`, i.e. loading a model nothing uses. The flag is now omitted and the
    user is told, rather than the value vanishing."""
    from rigma import cli
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: B9867)
    rp = _Rp(spec_type="draft-mtp", spec_draft="C:/models/d.gguf")
    rp.gguf.mtp = True
    refusal, warn = cli._spec_verdict(rp, engine_exe="llama-server.exe")
    assert refusal == ""
    assert "drafts without a separate model" in warn
    assert sd.llamacpp_spec_args("draft-mtp", n_max=0,
                                 draft="C:/models/d.gguf") == [
        "--spec-type", "draft-mtp"]


# --- end to end: the command, not the helper --------------------------------
#
# `_spec_verdict` passing its own tests says nothing about whether `up` calls it,
# or whether a refusal reaches the user as an exit code. These drive the real
# command through the real gate with a plan built the way `up` builds one. Only
# two things are stubbed, and both are the machine: the hardware profile, and
# which binary the pin resolves to — the probe itself is fed the help text
# MEASURED on b9867, which has no draft-dspark at all.


def _up_engine_world(tmp_path, monkeypatch, *, mtp=True):
    """A one-model registry plus everything `up` touches before it launches."""
    from typer.testing import CliRunner
    from rigma import cli, server_ops
    from rigma import state as st
    from rigma.models import (CachePolicy, CpuInfo, GgufFile, GpuInfo,
                              HardwareProfile, ModelSpec)
    from rigma.registry import Registry

    gguf = GgufFile(repo="r", file="m.gguf", bytes=1 << 30, quant="Q4", mtp=mtp)
    spec = ModelSpec(slug="mtp-model", family="f", kind="dense", n_layers=32,
                     full_attn_layers=32, kv_heads=8, head_dim=128,
                     native_ctx=32768, ggufs=[gguf], use_cases=["general"],
                     capabilities=[], cache_type_policy=CachePolicy())
    (tmp_path / "models").mkdir(parents=True, exist_ok=True)
    (tmp_path / "models" / "m.gguf").write_text("x", encoding="utf-8")
    reg = Registry([], {"mtp-model": spec}, {})
    gpu = GpuInfo(vendor="amd", name="RX 9070 XT", vram_mb=16368, arch="rdna4",
                  slug="amd-radeon-rx-9070-xt-16g", backends=["vulkan"])
    profile = HardwareProfile(gpus=[gpu], ram_mb=32768, ram_free_mb=20000,
                              cpu=CpuInfo(cores=16), os="windows",
                              disk_free_gb=400.0)
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(Registry, "load", classmethod(lambda cls: reg))
    monkeypatch.setattr(cli, "probe_hardware",
                        lambda gpus, raw_gpus=None: profile)
    monkeypatch.setattr(cli, "_port_holder", lambda port: "")
    monkeypatch.setattr("rigma.runtime.ensure_engine",
                        lambda backend, os_name: tmp_path / "llama-server.exe")
    monkeypatch.setattr("rigma.runtime.ensure_model",
                        lambda g: tmp_path / "models" / g.file)
    monkeypatch.setattr("rigma.bench.is_calibrated", lambda *a, **k: True)
    monkeypatch.setattr(st, "kill_pid", lambda pid: None)
    monkeypatch.setattr(server_ops, "engine_binary_for_plan",
                        lambda plan, os_name: ("llama-server.exe", {}))
    return CliRunner()


def test_up_refuses_a_mode_the_pin_cannot_be_asked_for(tmp_path, monkeypatch):
    """b9867 has no `draft-dspark`, so the launch is an argparse exit minutes
    after the download. `up` must say so and stop with exit 2."""
    from rigma import cli
    runner = _up_engine_world(tmp_path, monkeypatch)
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: B9867)
    res = runner.invoke(cli.app, ["up", "--model", "mtp-model", "--spec",
                                  "draft-dspark", "--spec-draft", "C:/x/d.gguf",
                                  "--yes", "--no-browser", "--no-calibrate",
                                  "--dry-run"])
    assert res.exit_code == 2, res.output
    assert "does not accept --spec-type draft-dspark" in res.output
    assert "refusing to launch" in res.output


def test_up_runs_the_mode_the_pin_does_have(tmp_path, monkeypatch):
    """The same command with a mode b9867 DOES have reaches the argv, which is
    the positive half of the gate: a refusal that fired on everything would pass
    the test above and fail this one."""
    from rigma import cli
    runner = _up_engine_world(tmp_path, monkeypatch)
    monkeypatch.setattr(sd, "probe_llamacpp", lambda *a, **k: B9867)
    res = runner.invoke(cli.app, ["up", "--model", "mtp-model", "--spec",
                                  "draft-mtp", "--spec-n-max", "4", "--yes",
                                  "--no-browser", "--no-calibrate", "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "--spec-type draft-mtp" in res.output
    assert "--spec-draft-n-max 4" in res.output


def test_up_warns_instead_of_refusing_when_no_engine_can_be_asked(
        tmp_path, monkeypatch):
    """No engine binary yet is the first-launch case. It must launch (with a
    warning), not refuse: Rigma cannot know, and "unknown" is not "no"."""
    from rigma import cli
    from rigma import server_ops
    runner = _up_engine_world(tmp_path, monkeypatch)
    monkeypatch.setattr(server_ops, "engine_binary_for_plan",
                        lambda plan, os_name: (None, {}))
    res = runner.invoke(cli.app, ["up", "--model", "mtp-model", "--spec",
                                  "draft-dspark", "--spec-draft",
                                  "z-lab/dspark-draft",
                                  "--yes", "--no-browser", "--no-calibrate",
                                  "--dry-run"])
    assert res.exit_code == 0, res.output
    assert "spec: " in res.output and "unverified" in res.output
