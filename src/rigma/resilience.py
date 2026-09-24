"""Survive the failures that are nobody's fault.

WHY THIS EXISTS. Every engine call Rigma makes is a single attempt. A local
llama-server that is still loading a 13 GB model, an OS that dropped a socket
mid-prefill, a Windows box that briefly ran out of handles, a proxy that answered
502 while an upstream restarted — all of them end the turn, and the user reads
"the engine is unreachable" about an engine that came back two seconds later.
The same is true one level up: an external agent backend (DSH, MiniMax Code) is
driven by a process that talks to a provider, and a provider hiccup kills a turn
that would have succeeded on the second try.

The autonomous run loop already has its own loop guards (`error_streak`,
`frozen_streak`, `lazy_streak` in `serve.py`), because a model that repeats itself
forever must be stopped. This module is the OTHER half: the part that decides a
failure was TRANSIENT and worth another attempt, versus REAL and worth reporting.
They are complements — this one must never turn a real error into an infinite
retry, and the run loop must never turn a transient one into a dead run.

THE ONE RULE THAT MATTERS. A stream that has already emitted bytes to the user
CANNOT be retried. Retrying would replay the same tokens on top of the partial
reply, and the transcript would show the answer twice — a corruption that looks
like a model bug and is actually ours. So `retry_stream` takes a `sent` callback:
the moment it reports True, the retry budget is spent and the failure is raised.
That is not a heuristic, it is the correctness condition of the whole module, and
it is the thing to preserve if you change anything here.

WHAT IT DELIBERATELY DOES NOT DO. It does not retry a 4xx that means the request
is wrong (400/401/403/404/422) — those fail identically forever and retrying them
just hides a bug behind a delay. It does not retry a cancellation. It does not
sleep without bound: every policy carries a total budget, and the budget is
checked before each sleep, so a wedged endpoint costs a bounded number of seconds
rather than a hung request.

Bounded, injectable, and testable: the clock and the sleep are parameters, so the
tests assert the retry COUNT and the DELAYS without spending real time.
"""
from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

# ---------------------------------------------------------------------------
# What went wrong, and whether it is worth another attempt.
# ---------------------------------------------------------------------------

TRANSIENT = "transient"      # try again — the same request may well succeed
PERMANENT = "permanent"      # the request or the state is wrong; retrying lies
CANCELLED = "cancelled"      # the caller asked us to stop; not a failure at all

# 5xx that mean "the server is having a moment" rather than "this request is
# unacceptable". 501 is deliberately absent: "not implemented" never becomes
# implemented on the second attempt.
_RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504, 507, 509})

# The opposite list, stated explicitly so the decision is readable rather than
# inferred from "not in the retryable set". A 400 is the engine telling us the
# request is malformed; it will say so every time.
_PERMANENT_STATUS = frozenset({400, 401, 403, 404, 405, 409, 413, 415, 422, 501})


def status_of(exc_or_status: Any) -> int:
    """The HTTP status carried by an exception or a response, or 0.

    httpx puts it on `response.status_code`; an exception raised from a response
    carries `response`. Both spellings appear in this codebase, so both are read
    rather than one being assumed.
    """
    if isinstance(exc_or_status, int):
        return exc_or_status
    code = getattr(exc_or_status, "status_code", None)
    if isinstance(code, int):
        return code
    resp = getattr(exc_or_status, "response", None)
    code = getattr(resp, "status_code", None)
    return code if isinstance(code, int) else 0


def _is_cancelled(exc: BaseException) -> bool:
    if isinstance(exc, asyncio.CancelledError):
        return True
    # A caller-side abort (the user pressed Stop) is not a transient fault and
    # must never be retried: the retry would resurrect a turn the user killed.
    return type(exc).__name__ in ("CancelledError", "AbortError")


# Exception type names that mean "the transport wobbled". Matched by NAME, not
# by isinstance, because httpx is imported lazily all over this project and a
# module-scope import here would drag it into every CLI path.
_TRANSIENT_EXC_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "TimeoutException", "ReadError", "WriteError",
    "RemoteProtocolError", "NetworkError", "ProtocolError", "ClosedResourceError",
    "BrokenResourceError", "IncompleteReadError", "ConnectionResetError",
    "ConnectionAbortedError", "ConnectionRefusedError", "ConnectionError",
    "TimeoutError", "socket.gaierror",
})


def classify(exc: BaseException) -> str:
    """TRANSIENT, PERMANENT or CANCELLED for one failure.

    Errs toward PERMANENT when it cannot tell. That direction is chosen on
    purpose: a permanent error retried is a bug hidden behind a delay, while a
    transient error reported is a message the user can act on. A wrong guess
    here should be visible, not silent.
    """
    if _is_cancelled(exc):
        return CANCELLED

    code = status_of(exc)
    if code:
        if code in _RETRYABLE_STATUS:
            return TRANSIENT
        if code in _PERMANENT_STATUS or 400 <= code < 500:
            return PERMANENT
        if code >= 500:
            return TRANSIENT          # an unlisted 5xx is still a server fault
        return PERMANENT

    name = type(exc).__name__
    if name in _TRANSIENT_EXC_NAMES:
        return TRANSIENT
    # An OSError that is not obviously a connection fault (ENOSPC, EACCES) is
    # not worth a retry; a bare OSError from a socket layer usually is. The
    # errno is the only thing that separates them.
    if isinstance(exc, OSError) and getattr(exc, "errno", None) in (
            104, 110, 111, 113, 32, 54, 10054, 10060, 10061):
        return TRANSIENT
    return PERMANENT


# ---------------------------------------------------------------------------
# Backoff
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Policy:
    """How hard to try, and for how long.

    `attempts` counts the FIRST try, so `attempts=3` means two retries.
    `budget` is the ceiling on the SUM of the sleeps — not on the request time,
    which the transport's own timeout owns — so a wedged endpoint costs a bounded
    delay rather than an unbounded one.
    """
    attempts: int = 3
    base_delay: float = 0.5
    max_delay: float = 8.0
    factor: float = 2.0
    jitter: float = 0.25          # fraction of the delay, +/- 
    budget: float = 20.0

    def delay_for(self, retry_index: int, *, rng: random.Random | None = None) -> float:
        """Seconds to wait before retry number `retry_index` (0-based).

        Exponential with a cap, plus symmetric jitter. The jitter is not
        decoration: several turns can be in flight at once (two slots, an aux
        call, a delegate), and without it they retry in lockstep and hammer an
        engine that is trying to recover.
        """
        raw = self.base_delay * (self.factor ** max(0, retry_index))
        capped = min(raw, self.max_delay)
        if self.jitter <= 0:
            return capped
        r = rng or random
        return max(0.0, capped * (1.0 + r.uniform(-self.jitter, self.jitter)))


DEFAULT = Policy()

# A single chat turn should not spend half a minute discovering that the engine
# is down; the circuit breaker is what makes a sustained outage fast.
TURN = Policy(attempts=3, base_delay=0.4, max_delay=4.0, budget=8.0)
# A background/aux call (auto-title, compaction, a delegate) can afford more.
AUX = Policy(attempts=4, base_delay=0.5, max_delay=8.0, budget=20.0)


# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

@dataclass
class CircuitBreaker:
    """Stop hammering a dead endpoint, and notice when it comes back.

    WHY. Without this, N concurrent turns each spend their full retry budget on
    an engine that is gone, and the retries themselves are load on the thing
    trying to recover. The breaker turns "everything is failing" from N×budget
    seconds into one immediate, honest error.

    It is per-instance and holds no global state on purpose: a breaker shared
    across endpoints would let one dead port close the circuit for a live one.

    `half_open` admits a single probe after the cooldown. A probe that succeeds
    closes the circuit; a probe that fails re-opens it and doubles the cooldown
    up to `max_cooldown`, so a flapping endpoint settles into a slow poll rather
    than a tight loop.
    """
    threshold: int = 3
    cooldown: float = 15.0
    max_cooldown: float = 120.0
    failures: int = 0
    opened_at: float = 0.0
    _cooldown: float = 0.0
    _probing: bool = False

    def __post_init__(self) -> None:
        self._cooldown = self.cooldown

    def allow(self, now: float | None = None) -> bool:
        """May a request be attempted right now?"""
        if self.failures < self.threshold:
            return True
        now = time.monotonic() if now is None else now
        if now - self.opened_at < self._cooldown:
            return False
        # Cooldown elapsed: admit exactly one probe. `_probing` is what stops a
        # burst of callers from all deciding they are the probe.
        if self._probing:
            return False
        self._probing = True
        return True

    def record_success(self) -> None:
        self.failures = 0
        self.opened_at = 0.0
        self._cooldown = self.cooldown
        self._probing = False

    def record_failure(self, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        self.failures += 1
        self._probing = False
        if self.failures >= self.threshold:
            self.opened_at = now
            if self.failures > self.threshold:
                # It failed again after a probe: back off further.
                self._cooldown = min(self._cooldown * 2, self.max_cooldown)

    @property
    def open(self) -> bool:
        return self.failures >= self.threshold

    def retry_after(self, now: float | None = None) -> float:
        """Seconds until the next probe is allowed, or 0."""
        if not self.open:
            return 0.0
        now = time.monotonic() if now is None else now
        return max(0.0, self._cooldown - (now - self.opened_at))


# ---------------------------------------------------------------------------
# The retry wrapper
# ---------------------------------------------------------------------------

class RetryExhausted(Exception):
    """Every attempt failed. Carries the last real failure, not a summary.

    The original exception is chained (`raise ... from last`) so a traceback still
    points at the true cause, and `attempts` is carried so a caller can say
    "after 3 tries" rather than implying it gave up immediately.
    """

    def __init__(self, message: str, *, attempts: int, last: BaseException):
        super().__init__(message)
        self.attempts = attempts
        self.last = last


def describe(exc: BaseException) -> str:
    """A failure in words a user can act on.

    "the engine is unreachable" is not actionable; naming the transport fault and
    whether we tried again is. An empty `str(exc)` (which several exception types
    have) must not produce a sentence with a hole in it.
    """
    text = str(exc).strip()
    if not text:
        text = type(exc).__name__
    code = status_of(exc)
    if code:
        return f"HTTP {code}: {text}"
    return text


async def retry_async(
    op: Callable[[], Awaitable[Any]],
    *,
    policy: Policy = DEFAULT,
    breaker: CircuitBreaker | None = None,
    what: str = "the request",
    sleep: Callable[[float], Awaitable[None]] | None = None,
    clock: Callable[[], float] | None = None,
    on_retry: Callable[[int, float, BaseException], None] | None = None,
) -> Any:
    """Run `op`, retrying TRANSIENT failures within `policy`.

    `op` must be safe to call more than once. That is the caller's contract and
    it is the reason this is not applied blindly: a non-idempotent operation
    retried is a duplicate write, not a recovered request.

    Raises the original exception when it is PERMANENT or CANCELLED (retrying
    either is wrong), and `RetryExhausted` when the budget runs out.
    """
    _sleep = sleep or asyncio.sleep
    _clock = clock or time.monotonic
    attempts = max(1, int(policy.attempts))
    started = _clock()
    last: BaseException | None = None

    for i in range(attempts):
        if breaker is not None and not breaker.allow(_clock()):
            wait = breaker.retry_after(_clock())
            raise RetryExhausted(
                f"{what} is not being attempted: it has failed "
                f"{breaker.failures} times in a row and is being given "
                f"{wait:.0f}s to recover",
                attempts=0, last=last or RuntimeError("circuit open"))
        try:
            result = await op()
        except BaseException as e:            # noqa: BLE001 — classified below
            kind = classify(e)
            if kind == CANCELLED:
                raise
            if kind == PERMANENT:
                if breaker is not None:
                    breaker.record_failure(_clock())
                raise
            last = e
            if breaker is not None:
                breaker.record_failure(_clock())
            if i + 1 >= attempts:
                break
            delay = policy.delay_for(i)
            if (_clock() - started) + delay > policy.budget:
                break                          # the budget, not the count, binds
            if on_retry is not None:
                on_retry(i + 1, delay, e)
            await _sleep(delay)
            continue
        else:
            if breaker is not None:
                breaker.record_success()
            return result

    raise RetryExhausted(
        f"{what} failed after {min(attempts, i + 1)} "
        f"{'attempt' if min(attempts, i + 1) == 1 else 'attempts'}: "
        f"{describe(last) if last else 'unknown error'}",
        attempts=min(attempts, i + 1),
        last=last or RuntimeError("unknown error"))


# ---------------------------------------------------------------------------
# Truncation / thinking-loop detection
# ---------------------------------------------------------------------------

@dataclass
class LoopGuard:
    """Notice a turn that is producing nothing, or the same thing, forever.

    WHY IT IS NEEDED ON TOP OF THE RUN LOOP'S GUARDS. The run loop's streaks
    count ROUNDS. A single round can still burn the whole output budget emitting
    reasoning and no answer, and a model that has collapsed into a repetition
    loop can emit the same paragraph until it hits `max_tokens` — both of which
    look like a healthy turn that produced text. This counts the SHAPE of the
    text, which is the thing the streaks cannot see.

    Three distinct failures, deliberately kept separate because they need
    different messages:
      * `empty`    — nothing at all came back (the reply was all whitespace, or
                     only reasoning with no content).
      * `repeat`   — the same normalised text `repeats` times in a row.
      * `truncated`— the engine stopped on the token ceiling mid-answer.

    It is a dataclass with explicit state rather than a bag of locals because the
    caller's loop is already the most complicated code in the project; keeping
    the bookkeeping out of it is the point.

    STATUS: TESTED PRIMITIVE, NOT WIRED IN. It was briefly integrated into the
    chat turn loop and REVERTED, because the loop already answers both cases
    better than this could. `test_thinking_only_turn_is_told_not_swallowed`
    asserts the user is told the model "spent this turn reasoning", and
    `test_silent_stop_is_not_reported_as_a_limit` asserts a silent round after
    tool calls is reported as "stopped after its tool calls" rather than as a
    limit. Both are more specific and more useful than "the model returned no
    answer (reasoning only, or empty)", which is what this module would have
    replaced them with — so wiring it in was a REGRESSION in message quality,
    caught by those two tests. The run loop additionally has `frozen_streak`,
    `error_streak` and `lazy_streak`. What remains genuinely uncovered is
    repeated text inside a single long round; if you wire this in, wire it in for
    THAT only, and keep the two existing messages.
    """
    window: int = 3
    min_chars: int = 1
    _recent: list[str] = field(default_factory=list)

    @staticmethod
    def normalize(text: str) -> str:
        """Whitespace-collapsed and lowercased.

        A model that repeats itself rarely emits byte-identical text; it emits
        the same words with different spacing or capitalisation. Comparing raw
        strings misses exactly the loop worth catching.
        """
        return " ".join(str(text or "").split()).lower()

    def observe(self, text: str, *, reasoning: str = "",
                finish_reason: str | None = None) -> str:
        """Record one turn's output. Returns "", "empty", "repeat" or "truncated".

        "truncated" is checked FIRST and independently: a reply cut off by the
        ceiling is worth reporting even when it is unique and non-empty, because
        the user is looking at half an answer.
        """
        answer = str(text or "")
        if finish_reason == "length":
            return "truncated"

        norm = self.normalize(answer)
        # A reply that is only reasoning has no answer in it. Counting the
        # reasoning as content is how "the model thought for 4000 tokens and said
        # nothing" reads as success.
        if len(norm) < self.min_chars:
            if self.normalize(reasoning):
                return "empty"
            return "empty" if not norm else ""

        self._recent.append(norm)
        if len(self._recent) > self.window:
            del self._recent[:-self.window]
        if len(self._recent) >= self.window and len(set(self._recent)) == 1:
            return "repeat"
        return ""

    def reset(self) -> None:
        self._recent.clear()

    @property
    def streak(self) -> int:
        """How many times the newest text has repeated consecutively."""
        if not self._recent:
            return 0
        newest = self._recent[-1]
        n = 0
        for item in reversed(self._recent):
            if item != newest:
                break
            n += 1
        return n


def loop_message(kind: str, guard: LoopGuard) -> str:
    """What to tell the user. Naming the shape is what makes it fixable."""
    if kind == "truncated":
        return ("the reply hit the output limit before it finished — raise the "
                "max-tokens setting, or ask for a shorter answer")
    if kind == "repeat":
        return (f"the model repeated itself {guard.streak} times without "
                f"progress and was stopped")
    if kind == "empty":
        return ("the model returned no answer (reasoning only, or empty) — "
                "try again, or turn thinking off for this chat")
    return "the turn made no progress and was stopped"
