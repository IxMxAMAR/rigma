"""R3-ENG-1 — the engine build must be measured, not assumed.

Every `--version` string in here was captured from a REAL binary on the owner's
machine. The four directories below all sit under `~/.rigma/engines/b9867/`, i.e.
all claim to be the manifest's pin, and one of them is a third-party fork about
1,800 builds newer. Rigma's engine resolution checked only `exe.exists()` plus a
`.ready` sentinel, so it could not tell them apart.
"""
from __future__ import annotations

import subprocess

import pytest

from rigma import engine_build

# VERBATIM. The legacy `bNNNN` scheme: the version IS the build number.
LEGACY = (
    "llama-server.exe : version: 9867 (152d337fa)\n"
    "built with Clang 20.1.8 for Windows x86_64\n")

# VERBATIM. The modern scheme, from the hand-installed PrismML fork. Note the
# version field is NOT numeric — a legacy-only parser reads this as no version at
# all, which is why both schemes have to be handled.
MODERN = (
    "llama-server.exe : version: 0.2.0-dev (build 10709, commit 9a9394a89)\n"
    "built with Clang 21.0.0 for Windows AMD64\n")


def test_the_legacy_scheme_puts_the_build_in_the_version_field():
    b = engine_build.parse_version(LEGACY)
    assert b.ok
    assert b.build == 9867
    assert b.version == "9867"
    assert b.commit == "152d337fa"
    assert "Clang 20.1.8" in b.compiler


def test_the_modern_scheme_states_the_build_separately():
    """The fork that can load the owner's ternary model. A parser that only knew
    the legacy scheme would report no version for it — i.e. exactly the build that
    matters most would be the one it could not identify."""
    b = engine_build.parse_version(MODERN)
    assert b.ok
    assert b.build == 10709
    assert b.version == "0.2.0-dev"
    assert b.commit == "9a9394a89"


def test_the_two_schemes_are_comparable_by_build_number():
    """The point of parsing rather than string-matching: these are 1,800 builds
    apart and a string compare cannot say so."""
    assert engine_build.parse_version(LEGACY).build < engine_build.parse_version(MODERN).build


def test_identity_distinguishes_builds_that_share_a_version():
    """Two builds can agree on the version string and differ in what they load.
    Identity therefore prefers build+commit."""
    a = engine_build.parse_version(LEGACY)
    b = engine_build.parse_version(MODERN)
    assert a.identity != b.identity
    assert a.identity == "b9867+152d337fa"
    assert b.identity == "b10709+9a9394a89"


def test_the_oracle_reports_its_build_too():
    """`llama-fit-params` carries the same line, which matters because it is the
    cheap capability probe — a probe from a different build than the server would
    answer a question nobody asked."""
    b = engine_build.parse_version("llama-fit-params.exe : version: 9867 (152d337fa)\n"
                                   "built with Clang 20.1.8 for Windows x86_64\n")
    assert b.ok and b.build == 9867


# --- degrading honestly ------------------------------------------------------

def test_empty_output_is_not_a_version():
    b = engine_build.parse_version("")
    assert b.ok is False and b.reason
    assert b.build is None


def test_unrecognised_output_is_not_a_version():
    """A wrong answer here would be worse than none: it would silently label a
    calibration with a build that was never measured."""
    b = engine_build.parse_version("llama-server.exe : hello\n")
    assert b.ok is False
    assert b.build is None
    assert "version line" in b.reason


def test_a_missing_binary_is_not_a_version(tmp_path):
    b = engine_build.read_build(tmp_path / "nope.exe")
    assert b.ok is False and "does not exist" in b.reason


def test_a_binary_that_cannot_run_is_not_a_version(tmp_path):
    bad = tmp_path / "bad.exe"
    bad.write_bytes(b"not an executable")

    def boom(*a, **k):
        raise OSError("not a valid Win32 application")

    b = engine_build.read_build(bad, popen=boom)
    assert b.ok is False
    assert "could not run" in b.reason


def test_read_build_reads_whichever_stream_carries_the_line(tmp_path):
    """The server and the oracle do not agree on the stream, so both are read
    rather than assuming one."""
    exe = tmp_path / "llama-server.exe"
    exe.write_bytes(b"x")

    def on_stderr(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, "", LEGACY)

    def on_stdout(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, LEGACY, "")

    assert engine_build.read_build(exe, popen=on_stderr).build == 9867
    assert engine_build.read_build(exe, popen=on_stdout).build == 9867


# --- comparing against the pin ----------------------------------------------

@pytest.mark.parametrize("measured,expected,ok", [
    (LEGACY, "b9867", True),
    (LEGACY, "9867", True),      # the manifest may spell it either way
    (LEGACY, "b9999", False),
    (MODERN, "b9867", False),    # the fork is NOT the pin — the whole defect
    (MODERN, "10709", True),
])
def test_matches_manifest_compares_build_numbers(measured, expected, ok):
    assert engine_build.matches_manifest(engine_build.parse_version(measured),
                                         expected) is ok


def test_matches_manifest_is_false_when_nothing_was_measured():
    """Unknown is not a match. Treating an unreadable build as matching the pin is
    how a fork goes unnoticed."""
    assert engine_build.matches_manifest(engine_build.parse_version(""), "b9867") is False
    assert engine_build.matches_manifest(engine_build.parse_version(LEGACY), "") is False
