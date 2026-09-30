"""The engine says these things once and then never again.

The cache_reuse line below is copied verbatim from this machine's own
server log, where it had appeared fifteen times across past launches and had
never been surfaced to anyone.
"""
from rigma import engine_log


REAL = (
    "0.00.123 I srv    load_model: loading model 'RVN-Q3_K_M-mtp.gguf'\n"
    "0.00.456 W srv    load_model: cache_reuse is not supported by this "
    "context, it will be disabled\n"
    "0.01.000 I srv    init: initializing slots, n_slots = 2\n"
)


def test_it_finds_the_warning_that_was_sitting_in_the_log():
    got = engine_log.findings(REAL)

    assert [f["id"] for f in got] == ["cache_reuse_disabled"]
    assert got[0]["severity"] == "warn"
    assert got[0]["confirmed_here"] is True
    assert "architecture limit" in got[0]["message"]


def test_a_clean_log_reports_nothing():
    assert engine_log.findings(
        "0.00.1 I srv init: initializing slots, n_slots = 2\n"
        "0.00.2 I srv main: server is listening on 127.0.0.1:11499\n") == []


def test_repeats_across_restarts_collapse_to_one_finding_with_a_count():
    # A log file spans several launches. The same fact reported five times is
    # one fact, or the panel becomes a wall.
    got = engine_log.findings(REAL * 5)

    assert len(got) == 1
    assert got[0]["count"] == 5


def test_the_example_is_the_most_recent_occurrence():
    # Earlier launches may have run a different model; the last line is the one
    # describing the engine that is up now. The marker has to be ON the matched
    # line: the loading-model line is not what this finding quotes, so putting
    # the model name there let the old assertion pass whichever occurrence
    # `findings` picked (AUDIT 02-7).
    text = (REAL.replace("it will be disabled",
                         "it will be disabled (OLD-MODEL)")
            + REAL.replace("it will be disabled",
                           "it will be disabled (CURRENT-MODEL)"))
    got = engine_log.findings(text)

    assert got[0]["count"] == 2
    assert "CURRENT-MODEL" in got[0]["example"]
    assert "OLD-MODEL" not in got[0]["example"], (
        "the panel would describe a previous launch's model as the one running")


# A new engine PROCESS starts at its parameter dump — the first line of every
# one of this machine's 16 `server-*.log` files.
BANNER = "0.00.038 I cmn common_param: common_params_print_info: verbosity = 3\n"


def test_a_warning_from_a_previous_launch_is_not_attributed_to_this_one():
    """06R3-10: `findings` scanned the whole log and quoted the last matching
    line, so a launch that came up clean still reported the PREVIOUS launch's
    warning as a fact about the engine that is up now. The engine's own
    parameter dump marks where each process starts; only the last process's
    lines belong to the running engine."""
    text = (BANNER + REAL
            + BANNER + "0.00.9 I srv init: initializing slots, n_slots = 2\n")

    assert engine_log.findings(text) == []


def test_a_warning_in_the_current_launch_is_reported_once():
    text = BANNER + REAL + BANNER + REAL

    got = engine_log.findings(text)

    assert [f["id"] for f in got] == ["cache_reuse_disabled"]
    # the CURRENT launch's one line, not the same warning from both launches
    assert got[0]["count"] == 1


def test_without_a_launch_banner_the_last_occurrence_is_still_used():
    """A tail that begins mid-launch carries no marker to segment on, so the
    whole text is scanned exactly as before — `findings` then makes no claim
    beyond "this text contains it" (see the docstring)."""
    got = engine_log.findings(REAL)

    assert [f["id"] for f in got] == ["cache_reuse_disabled"]


def test_empty_and_none_are_not_errors():
    assert engine_log.findings("") == []
    assert engine_log.findings(None) == []


def test_a_bare_n_swa_parameter_dump_is_not_a_warning():
    """AUDIT 02-6: the pattern used to match `n_swa = 0` on its own.

    That is a parameter dump, not a diagnostic, so every non-SWA model's
    hparams line reported a `--swa-full` no-op for a flag Rigma never passes —
    sending the reader looking for a setting they cannot have changed.
    """
    assert engine_log.findings(
        "0.00.1 I srv load_model: n_swa = 0\n") == []


def test_a_real_swa_disabled_warning_is_still_reported():
    # The warning shape (swa + disabled) is what the pattern is for.
    got = engine_log.findings(
        "0.00.2 W srv load_model: swa is not supported by this model, "
        "it will be disabled\n")
    assert [f["id"] for f in got] == ["swa_disabled"]


def test_a_full_context_shift_is_reported():
    got = engine_log.findings("0.1 W srv update_slots: KV cache is full - "
                              "shifting context\n")

    assert [f["id"] for f in got] == ["kv_cache_full"]
    # not confirmed on this machine, and the field says so rather than implying
    # the pattern has ever been seen to fire here
    assert got[0]["confirmed_here"] is False


def test_findings_keep_their_declared_order():
    # cache_reuse first: it is the one that silently changes behaviour for a
    # whole session.
    text = ("W srv update_slots: KV cache is full - shifting context\n"
            + REAL)
    assert [f["id"] for f in engine_log.findings(text)] == [
        "cache_reuse_disabled", "kv_cache_full"]
