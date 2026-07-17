"""Gemini 4-agent review 2026-07-18: DoS/SSRF/robustness fixes."""
import time

from rigma import tools


def test_calculator_pow_chain_fails_fast():
    t0 = time.monotonic()
    out = tools.run_tool("calculator", {"expression": "9**9**9**9"})
    assert "too large" in out
    assert time.monotonic() - t0 < 1.0        # rejected BEFORE computing


def test_calculator_still_does_normal_powers():
    assert tools.run_tool("calculator", {"expression": "2 ** 10"}) == "1024"
    assert tools.run_tool("calculator", {"expression": "2**64"}) \
        == "18446744073709551616"


def test_ssrf_ipv4_mapped_ipv6_blocked():
    # ::ffff:127.0.0.1 reports is_loopback=False on the raw v6 object
    assert tools._is_public_host("::ffff:127.0.0.1") is False
    assert tools._is_public_host("::ffff:169.254.169.254") is False


def test_run_python_stdin_does_not_hang():
    t0 = time.monotonic()
    out = tools.run_tool("run_python", {"code": "print(input())"},
                         {"allow_code": True})
    # EOF on stdin -> instant EOFError, not a 30s timeout
    assert time.monotonic() - t0 < 10
    assert "timed out" not in out
