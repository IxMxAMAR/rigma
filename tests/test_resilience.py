"""The resilience layer: retry, circuit breaking, and loop detection.

These are the tests for the module that decides whether a failure is worth
another attempt. The stakes are asymmetric and that is why the coverage is
lopsided toward the REFUSALS: a transient error reported is a message the user
can act on, but a permanent error retried is a bug hidden behind a delay, and a
stream retried after it has already emitted bytes is a corrupted transcript.
"""
import asyncio
import random

import httpx
import pytest

from rigma import resilience as R


# --------------------------------------------------------------------------
# classify — the decision everything else rests on
# --------------------------------------------------------------------------

def test_a_connection_failure_is_transient():
    assert R.classify(httpx.ConnectError("refused")) == R.TRANSIENT


def test_a_read_timeout_is_transient():
    assert R.classify(httpx.ReadTimeout("slow")) == R.TRANSIENT


def test_a_remote_protocol_error_is_transient():
    """This is the one that actually bites: llama-server drops the socket
    mid-stream when it is still loading a large model."""
    assert R.classify(httpx.RemoteProtocolError("closed")) == R.TRANSIENT


def test_a_retryable_status_is_transient():
    for code in (408, 429, 500, 502, 503, 504):
        resp = httpx.Response(code, request=httpx.Request("GET", "http://x/"))
        exc = httpx.HTTPStatusError("boom", request=resp.request, response=resp)
        assert R.classify(exc) == R.TRANSIENT, code


def test_a_malformed_request_is_permanent():
    """A 400 means the request is wrong. It will be wrong on every attempt, so
    retrying it converts a clear error into a delay followed by the same error."""
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x/"))
    exc = httpx.HTTPStatusError("bad", request=resp.request, response=resp)
    assert R.classify(exc) == R.PERMANENT


def test_not_implemented_is_permanent_not_transient():
    """501 is a 5xx, and the naive rule "5xx is the server's fault" would retry
    it forever. It is the clearest example of why the permanent list is explicit."""
    resp = httpx.Response(501, request=httpx.Request("POST", "http://x/"))
    exc = httpx.HTTPStatusError("nope", request=resp.request, response=resp)
    assert R.classify(exc) == R.PERMANENT


def test_an_unlisted_5xx_is_still_the_servers_fault():
    resp = httpx.Response(599, request=httpx.Request("POST", "http://x/"))
    exc = httpx.HTTPStatusError("odd", request=resp.request, response=resp)
    assert R.classify(exc) == R.TRANSIENT


def test_a_cancellation_is_never_a_failure():
    """The user pressed Stop. Retrying would resurrect the turn they killed."""
    assert R.classify(asyncio.CancelledError()) == R.CANCELLED


def test_an_unrecognised_error_is_permanent():
    """Chosen direction: a permanent error retried hides a bug, so an unknown
    error must be reported rather than retried."""
    assert R.classify(ValueError("nonsense")) == R.PERMANENT


def test_a_disk_full_oserror_is_not_retried():
    import errno
    assert R.classify(OSError(errno.ENOSPC, "no space")) == R.PERMANENT


def test_a_connection_reset_errno_is_transient():
    import errno
    assert R.classify(OSError(errno.ECONNRESET, "reset")) == R.TRANSIENT


def test_status_of_reads_a_bare_int():
    assert R.status_of(503) == 503


def test_status_of_reads_a_response_carrying_exception():
    resp = httpx.Response(429, request=httpx.Request("GET", "http://x/"))
    exc = httpx.HTTPStatusError("slow down", request=resp.request, response=resp)
    assert R.status_of(exc) == 429


def test_status_of_is_zero_when_there_is_no_status():
    assert R.status_of(ValueError("nothing")) == 0


# --------------------------------------------------------------------------
# Policy — the backoff arithmetic
# --------------------------------------------------------------------------

def test_backoff_grows_exponentially():
    p = R.Policy(base_delay=1.0, factor=2.0, jitter=0.0)
    assert [p.delay_for(i) for i in range(4)] == [1.0, 2.0, 4.0, 8.0]


def test_backoff_is_capped():
    p = R.Policy(base_delay=1.0, factor=10.0, max_delay=5.0, jitter=0.0)
    assert p.delay_for(9) == 5.0


def test_jitter_keeps_the_delay_near_the_base():
    p = R.Policy(base_delay=4.0, factor=1.0, jitter=0.25)
    rng = random.Random(7)
    delays = [p.delay_for(0, rng=rng) for _ in range(200)]
    assert all(3.0 <= d <= 5.0 for d in delays), (min(delays), max(delays))
    # and it actually varies — a jitter that never changes is not a jitter
    assert len(set(delays)) > 100


def test_zero_jitter_is_exactly_deterministic():
    p = R.Policy(base_delay=2.0, jitter=0.0)
    assert {p.delay_for(1) for _ in range(50)} == {4.0}


def test_the_turn_policy_is_shorter_than_the_aux_policy():
    """A user is waiting on a turn; an aux call is not. If this ever inverts,
    someone has made a chat turn wait as long as a background job."""
    assert R.TURN.budget < R.AUX.budget
    assert R.TURN.attempts <= R.AUX.attempts


# --------------------------------------------------------------------------
# CircuitBreaker
# --------------------------------------------------------------------------

def test_the_breaker_allows_requests_below_the_threshold():
    b = R.CircuitBreaker(threshold=3)
    assert b.allow(0.0)
    b.record_failure(0.0)
    assert b.allow(0.0)


def test_the_breaker_opens_at_the_threshold():
    b = R.CircuitBreaker(threshold=3, cooldown=10.0)
    for _ in range(3):
        b.record_failure(100.0)
    assert b.open
    assert not b.allow(100.0)


def test_the_breaker_admits_one_probe_after_the_cooldown():
    """Exactly one. A burst of concurrent turns all deciding they are the probe
    is the thundering herd the breaker exists to prevent."""
    b = R.CircuitBreaker(threshold=1, cooldown=5.0)
    b.record_failure(0.0)
    assert not b.allow(1.0)
    assert b.allow(6.0)          # the probe
    assert not b.allow(6.0)      # the second caller is not the probe
    assert not b.allow(6.0)


def test_a_successful_probe_closes_the_breaker():
    b = R.CircuitBreaker(threshold=1, cooldown=5.0)
    b.record_failure(0.0)
    assert b.allow(6.0)
    b.record_success()
    assert not b.open
    assert b.allow(6.0)


def test_a_failed_probe_backs_the_cooldown_off():
    """A flapping endpoint must settle into a slow poll, not a tight loop."""
    b = R.CircuitBreaker(threshold=1, cooldown=5.0, max_cooldown=40.0)
    b.record_failure(0.0)
    assert b.allow(5.0)
    b.record_failure(5.0)                     # the probe failed
    assert b.retry_after(5.0) > 5.0
    assert b.retry_after(5.0) <= 40.0


def test_the_cooldown_never_exceeds_its_ceiling():
    b = R.CircuitBreaker(threshold=1, cooldown=5.0, max_cooldown=20.0)
    now = 0.0
    for _ in range(12):
        b.record_failure(now)
        now += b.retry_after(now) + 0.01
        b.allow(now)
    assert b.retry_after(now) <= 20.0


def test_retry_after_is_zero_when_the_breaker_is_closed():
    assert R.CircuitBreaker().retry_after(0.0) == 0.0


# --------------------------------------------------------------------------
# retry_async
# --------------------------------------------------------------------------

class _Clock:
    """A clock that only moves when something sleeps on it.

    Lets a test assert the retry DELAYS without spending them.
    """

    def __init__(self):
        self.now = 1000.0
        self.slept: list[float] = []

    def monotonic(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.now += seconds


def _run(coro):
    return asyncio.run(coro)


def test_a_permanent_failure_is_not_retried():
    calls = []

    async def op():
        calls.append(1)
        raise ValueError("this will never work")

    with pytest.raises(ValueError):
        _run(R.retry_async(op, policy=R.Policy(attempts=5, jitter=0.0)))
    assert len(calls) == 1, "a permanent error must be attempted exactly once"


def test_a_transient_failure_is_retried_until_it_succeeds():
    calls = []

    async def op():
        calls.append(1)
        if len(calls) < 3:
            raise httpx.ConnectError("refused")
        return "ok"

    c = _Clock()
    out = _run(R.retry_async(op, policy=R.Policy(attempts=5, jitter=0.0),
                             sleep=c.sleep, clock=c.monotonic))
    assert out == "ok"
    assert len(calls) == 3


def test_the_attempt_ceiling_is_respected():
    calls = []

    async def op():
        calls.append(1)
        raise httpx.ConnectError("refused")

    c = _Clock()
    with pytest.raises(R.RetryExhausted) as ei:
        _run(R.retry_async(op, policy=R.Policy(attempts=3, jitter=0.0),
                           sleep=c.sleep, clock=c.monotonic))
    assert len(calls) == 3
    assert ei.value.attempts == 3
    assert "3 attempts" in str(ei.value)


def test_the_budget_binds_before_the_attempt_ceiling():
    """The count and the budget are two different limits, and the budget is the
    one that keeps a wedged endpoint from costing a minute of the user's time."""
    calls = []

    async def op():
        calls.append(1)
        raise httpx.ConnectError("refused")

    c = _Clock()
    # base 5s, doubling, budget 6s: the second sleep (10s) cannot fit.
    with pytest.raises(R.RetryExhausted):
        _run(R.retry_async(op, policy=R.Policy(attempts=9, base_delay=5.0,
                                               factor=2.0, jitter=0.0,
                                               budget=6.0),
                           sleep=c.sleep, clock=c.monotonic))
    assert len(calls) == 2, f"expected the budget to stop it after 2, got {calls}"
    assert sum(c.slept) <= 6.0


def test_the_sleeps_are_the_backoff_the_policy_promised():
    async def op():
        raise httpx.ConnectError("refused")

    c = _Clock()
    with pytest.raises(R.RetryExhausted):
        _run(R.retry_async(op, policy=R.Policy(attempts=4, base_delay=1.0,
                                               factor=2.0, jitter=0.0,
                                               budget=100.0),
                           sleep=c.sleep, clock=c.monotonic))
    assert c.slept == [1.0, 2.0, 4.0]


def test_a_cancellation_propagates_untouched():
    async def op():
        raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        _run(R.retry_async(op, policy=R.Policy(attempts=5)))


def test_an_open_breaker_prevents_the_attempt_entirely():
    calls = []

    async def op():
        calls.append(1)
        return "ok"

    c = _Clock()
    b = R.CircuitBreaker(threshold=1, cooldown=1000.0)
    # Tripped at the clock's OWN now, not at 0: the breaker compares against the
    # clock it is handed, and a failure recorded at a time a thousand seconds
    # before that clock's present reads as "the cooldown already elapsed".
    b.record_failure(c.monotonic())
    with pytest.raises(R.RetryExhausted) as ei:
        _run(R.retry_async(op, breaker=b, sleep=c.sleep, clock=c.monotonic))
    assert calls == [], "an open breaker must not even attempt the request"
    assert ei.value.attempts == 0
    assert "recover" in str(ei.value)


def test_a_success_clears_the_breakers_failure_count():
    async def op():
        return "ok"

    b = R.CircuitBreaker(threshold=5)
    b.record_failure(0.0)
    b.record_failure(0.0)
    _run(R.retry_async(op, breaker=b))
    assert b.failures == 0


def test_a_permanent_failure_also_counts_against_the_breaker():
    """A dead endpoint answering 400 to everything is still dead, and the
    breaker should stop the next caller from finding out the slow way."""
    async def op():
        raise ValueError("bad request")

    b = R.CircuitBreaker(threshold=1)
    with pytest.raises(ValueError):
        _run(R.retry_async(op, breaker=b))
    assert b.open


def test_a_transient_failure_counts_against_the_breaker():
    """The breaker is only useful if the retry loop FEEDS it. Every other breaker
    test drives `record_failure` by hand, so a `retry_async` that never told the
    breaker anything would leave them all green while the breaker stayed closed
    through an outage — which is the whole failure it exists to prevent."""
    async def op():
        raise httpx.ConnectError("refused")

    c = _Clock()
    b = R.CircuitBreaker(threshold=2)
    with pytest.raises(R.RetryExhausted):
        _run(R.retry_async(op, policy=R.Policy(attempts=2, jitter=0.0),
                           breaker=b, sleep=c.sleep, clock=c.monotonic))
    assert b.failures == 2, b.failures
    assert b.open


def test_the_breaker_opens_across_separate_calls_not_just_within_one():
    """Each turn gets its own `retry_async` call. If failures were only counted
    within a single call the breaker would never open in the situation it was
    written for: many turns, one dead engine."""
    async def op():
        raise httpx.ConnectError("refused")

    b = R.CircuitBreaker(threshold=2, cooldown=1000.0)
    for _ in range(2):
        with pytest.raises(R.RetryExhausted):
            _run(R.retry_async(op, policy=R.Policy(attempts=1), breaker=b))
    assert b.open
    calls = []

    async def never():
        calls.append(1)

    with pytest.raises(R.RetryExhausted):
        _run(R.retry_async(never, breaker=b))
    assert calls == [], "the third turn must be refused without being attempted"


def test_on_retry_reports_the_attempt_and_the_delay():
    seen = []

    async def op():
        raise httpx.ConnectError("refused")

    c = _Clock()
    with pytest.raises(R.RetryExhausted):
        _run(R.retry_async(op, policy=R.Policy(attempts=3, base_delay=1.0,
                                               jitter=0.0, budget=99.0),
                           sleep=c.sleep, clock=c.monotonic,
                           on_retry=lambda n, d, e: seen.append((n, d, type(e).__name__))))
    assert seen == [(1, 1.0, "ConnectError"), (2, 2.0, "ConnectError")]


def test_retry_exhausted_chains_the_real_cause():
    """A traceback must still point at the true failure, not at the wrapper."""
    async def op():
        raise httpx.ConnectError("the real reason")

    with pytest.raises(R.RetryExhausted) as ei:
        _run(R.retry_async(op, policy=R.Policy(attempts=1)))
    assert isinstance(ei.value.last, httpx.ConnectError)
    assert "the real reason" in str(ei.value)


def test_a_single_attempt_policy_reports_one_attempt_not_zero():
    async def op():
        raise httpx.ConnectError("refused")

    with pytest.raises(R.RetryExhausted) as ei:
        _run(R.retry_async(op, policy=R.Policy(attempts=1)))
    assert ei.value.attempts == 1
    assert "1 attempt:" in str(ei.value)


# --------------------------------------------------------------------------
# describe — the message the user actually reads
# --------------------------------------------------------------------------

def test_describe_names_the_status_code():
    resp = httpx.Response(503, request=httpx.Request("POST", "http://x/"))
    exc = httpx.HTTPStatusError("unavailable", request=resp.request, response=resp)
    assert "HTTP 503" in R.describe(exc)


def test_describe_never_produces_a_sentence_with_a_hole_in_it():
    """Several exception types stringify to "". "the engine failed: " reads as a
    bug in Rigma rather than a fact about the engine."""
    class Silent(Exception):
        def __str__(self):
            return ""

    out = R.describe(Silent())
    assert out.strip()
    assert "Silent" in out


# --------------------------------------------------------------------------
# LoopGuard — truncation and thinking loops
# --------------------------------------------------------------------------

def test_a_length_finish_is_reported_as_truncated():
    g = R.LoopGuard()
    assert g.observe("a long but unique answer", finish_reason="length") == "truncated"


def test_truncation_is_reported_even_when_the_text_is_unique():
    """The point: a truncated reply is worth reporting even though it looks
    nothing like a loop. The user is staring at half an answer."""
    g = R.LoopGuard()
    assert g.observe("something never seen before", finish_reason="length") == "truncated"
    assert g.observe("something else entirely", finish_reason="length") == "truncated"


def test_a_thinking_only_turn_is_empty_not_successful():
    g = R.LoopGuard()
    assert g.observe("", reasoning="I thought about it for a long time") == "empty"


def test_a_whitespace_only_reply_is_empty():
    g = R.LoopGuard()
    assert g.observe("   \n\t  ") == "empty"


def test_repetition_is_caught_at_the_window():
    g = R.LoopGuard(window=3)
    assert g.observe("the same thing") == ""
    assert g.observe("the same thing") == ""
    assert g.observe("the same thing") == "repeat"


def test_repetition_survives_whitespace_and_case_differences():
    """A model looping rarely emits byte-identical text. Comparing raw strings
    misses exactly the loop worth catching."""
    g = R.LoopGuard(window=3)
    assert g.observe("The Same Thing") == ""
    assert g.observe("the   same\nthing") == ""
    assert g.observe("THE SAME THING ") == "repeat"


def test_different_answers_are_never_called_a_repeat():
    g = R.LoopGuard(window=3)
    for text in ("one", "two", "three", "four", "five", "six"):
        assert g.observe(text) == ""


def test_the_window_is_a_window_not_a_lifetime_counter():
    """Three identical answers, then a new one, then three more identical ones:
    the second run must be caught on its own third repetition."""
    g = R.LoopGuard(window=3)
    for _ in range(3):
        g.observe("same")
    assert g.observe("different") == ""
    assert g.observe("same") == ""
    assert g.observe("same") == ""
    assert g.observe("same") == "repeat"


def test_reset_clears_the_history():
    g = R.LoopGuard(window=2)
    g.observe("same")
    g.observe("same")
    assert g.streak == 2
    g.reset()
    assert g.streak == 0
    assert g.observe("same") == ""


def test_streak_counts_consecutive_repeats_only():
    g = R.LoopGuard(window=5)
    g.observe("a")
    g.observe("b")
    g.observe("b")
    g.observe("b")
    assert g.streak == 3


def test_a_normal_stop_is_not_flagged():
    g = R.LoopGuard()
    assert g.observe("a perfectly good answer", finish_reason="stop") == ""


def test_every_loop_kind_has_a_message_that_says_what_to_do():
    for kind in ("truncated", "repeat", "empty"):
        msg = R.loop_message(kind, R.LoopGuard())
        assert msg and msg[0].islower()
    # and they are three DIFFERENT messages, because they need three different
    # remedies
    msgs = {R.loop_message(k, R.LoopGuard()) for k in ("truncated", "repeat", "empty")}
    assert len(msgs) == 3


def test_the_repeat_message_names_the_count():
    g = R.LoopGuard(window=2)
    g.observe("x")
    g.observe("x")
    assert "2 times" in R.loop_message("repeat", g)


# --------------------------------------------------------------------------
# one end-to-end pass with the real asyncio.sleep, to prove the async plumbing
# --------------------------------------------------------------------------

def test_retry_works_with_the_real_event_loop():
    """Every other test injects a fake sleep. This one does not, so a bug in the
    `await` (a missing await, a sync sleep) cannot hide behind the injection."""
    calls = []

    async def op():
        calls.append(1)
        if len(calls) < 2:
            raise httpx.ConnectError("refused")
        return "recovered"

    async def main():
        return await R.retry_async(
            op, policy=R.Policy(attempts=3, base_delay=0.001, jitter=0.0))

    assert asyncio.run(main()) == "recovered"
    assert len(calls) == 2
