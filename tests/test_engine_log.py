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
    # describing the engine that is up now.
    text = (REAL.replace("RVN-Q3_K_M", "OLD-MODEL")
            + REAL.replace("RVN-Q3_K_M", "CURRENT-MODEL"))
    got = engine_log.findings(text)

    assert got[0]["count"] == 2
    assert "cache_reuse" in got[0]["example"]


def test_empty_and_none_are_not_errors():
    assert engine_log.findings("") == []
    assert engine_log.findings(None) == []


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
