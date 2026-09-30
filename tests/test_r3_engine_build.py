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


# --- lenient fallback --------------------------------------------------------
# The research for this module was explicit that format-matching must not be the
# only path: the format already differs between the two builds in this defect, and
# `llama-bench --version` only gained the flag in PR #28971. A `bNNNN` is
# `git rev-list --count HEAD`, so any plausible build number is better than nothing.

def test_an_unfamiliar_format_still_yields_a_build_number():
    b = engine_build.parse_version("version b4321, commit deadbeefcafe\n")
    assert b.ok is True
    assert b.build == 4321
    assert b.commit == "deadbeefcafe"


def test_a_version_line_with_no_number_is_still_unknown():
    """Leniency must not become invention: text that says 'version' but carries no
    build number is an unknown, not a guess."""
    b = engine_build.parse_version("version: unknown\n")
    assert b.ok is False
    assert b.build is None


def test_text_with_no_version_line_is_never_guessed_at():
    """The guard that keeps the loose path honest. Without it, any stray number in
    a binary's output would be read as its build."""
    b = engine_build.parse_version("built with Clang 20.1.8 for Windows x86_64\n")
    assert b.ok is False
    assert b.build is None


def test_the_exact_patterns_win_over_the_loose_one():
    """Loose matching is a last resort, so a well-formed line must not be misread.
    The commit hash here contains digits that a loose scan could grab."""
    b = engine_build.parse_version(LEGACY)
    assert b.build == 9867
    assert b.commit == "152d337fa"


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


# --- cached verification -----------------------------------------------------

def _exe(tmp_path, name="llama-server.exe"):
    p = tmp_path / name
    p.write_bytes(b"x")
    return p


def test_cached_build_spawns_once(tmp_path):
    """`--version` is a process spawn on a hot-ish path, and the answer cannot
    change unless the file does."""
    exe = _exe(tmp_path)
    calls = {"n": 0}

    def fake(*a, **k):
        calls["n"] += 1
        return subprocess.CompletedProcess(a[0], 0, "", LEGACY)

    engine_build._BUILD_CACHE.clear()
    assert engine_build.cached_build(exe, popen=fake).build == 9867
    assert engine_build.cached_build(exe, popen=fake).build == 9867
    assert calls["n"] == 1


def test_replacing_the_binary_invalidates_the_cache(tmp_path):
    """Exactly how the owner's fork arrived: a different binary written to the same
    path. A path-keyed cache would report the old build forever and hide it."""
    exe = _exe(tmp_path)
    modern = {"v": False}

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, "", MODERN if modern["v"] else LEGACY)

    engine_build._BUILD_CACHE.clear()
    assert engine_build.cached_build(exe, popen=fake).build == 9867
    # same path, different file: change size and mtime
    modern["v"] = True
    exe.write_bytes(b"a much longer replacement binary")
    assert engine_build.cached_build(exe, popen=fake).build == 10709


def test_a_same_size_swap_inside_one_second_is_detected(tmp_path):
    """The memo was keyed on `int(mtime)`, so a same-size replacement inside the
    same whole second served the OLD build's identity: the file was never
    re-read. That identity is what keys the KV cache and the calibration
    provenance, so the stale answer is not cosmetic. The nanosecond part of the
    mtime is what tells the two writes apart."""
    import os
    exe = _exe(tmp_path)
    state = {"modern": False}

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, "",
                                           MODERN if state["modern"] else LEGACY)

    engine_build._BUILD_CACHE.clear()
    base = 1_700_000_000_000_000_000          # whole second 1700000000
    exe.write_bytes(b"x")
    os.utime(exe, ns=(base, base))
    assert engine_build.cached_build(exe, popen=fake).build == 9867
    # same path, same SIZE, same whole second: only nanoseconds differ
    state["modern"] = True
    exe.write_bytes(b"y")
    os.utime(exe, ns=(base + 100_000_000, base + 100_000_000))
    assert exe.stat().st_size == 1, "the test must not change the size"
    assert int(exe.stat().st_mtime) == 1_700_000_000, "nor the whole second"
    assert engine_build.cached_build(exe, popen=fake).build == 10709


def test_an_unreadable_build_is_not_cached(tmp_path):
    """A transient failure must not freeze 'unknown' for the life of the process."""
    exe = _exe(tmp_path)
    state = {"fail": True}

    def fake(*a, **k):
        if state["fail"]:
            raise OSError("transient")
        return subprocess.CompletedProcess(a[0], 0, "", LEGACY)

    engine_build._BUILD_CACHE.clear()
    assert engine_build.cached_build(exe, popen=fake).ok is False
    state["fail"] = False
    assert engine_build.cached_build(exe, popen=fake).build == 9867


def test_verify_engine_flags_the_fork_against_the_pin(tmp_path):
    """The real situation, end to end: a directory claiming to be b9867 that holds
    build 10709."""
    exe = _exe(tmp_path)

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, "", MODERN)

    engine_build._BUILD_CACHE.clear()
    got, ok = engine_build.verify_engine(exe, "b9867", popen=fake)
    assert ok is False
    assert got.build == 10709


def test_verify_engine_accepts_the_real_pin(tmp_path):
    exe = _exe(tmp_path)

    def fake(*a, **k):
        return subprocess.CompletedProcess(a[0], 0, "", LEGACY)

    engine_build._BUILD_CACHE.clear()
    got, ok = engine_build.verify_engine(exe, "b9867", popen=fake)
    assert ok is True and got.build == 9867


# --- comparing engine strings from two producers ------------------------------
# Engine strings are written by two different producers and must compare equal across
# both, or every calibration measured before R3-ENG-6 would be discarded the moment
# R3-ENG-6 started recording the fuller answer.

def test_the_bare_manifest_form_matches_the_measured_identity():
    """`b9867` (what older entries recorded) vs `b9867+152d337fa` (what is recorded
    now). A plain != would call these different and throw away valid calibrations."""
    assert engine_build.same_build("b9867", "b9867+152d337fa") is True
    assert engine_build.same_build("b9867+152d337fa", "b9867") is True


def test_the_fork_does_not_match_the_pin():
    """The whole point. The owner's rocm directory held build 10709 while the
    manifest said b9867, and a calibration measured on the fork must not be replayed
    as if it described the pin."""
    assert engine_build.same_build("b10709+9a9394a89", "b9867+152d337fa") is False
    assert engine_build.same_build("b10709+9a9394a89", "b9867") is False


def test_a_different_commit_on_the_same_build_number_is_a_different_build():
    """A re-tagged release and a fork can share the counter, which is exactly why
    identity carries the commit at all."""
    assert engine_build.same_build("b9867+152d337fa", "b9867+deadbeef0") is False


def test_a_truncated_commit_still_matches_its_full_form():
    """`--version` prints a short hash; a longer recorded one is the same commit."""
    assert engine_build.same_build("b9867+152d337fa", "b9867+152d337fadb93c2a") is True


def test_empty_never_matches():
    """Unknown must not compare equal to a real build, or a failed probe would look
    like a match."""
    assert engine_build.same_build("", "b9867") is False
    assert engine_build.same_build("b9867", "") is False
    assert engine_build.same_build("", "") is True


def test_unparseable_strings_fall_back_to_exact_comparison():
    assert engine_build.same_build("weird", "weird") is True
    assert engine_build.same_build("weird", "other") is False
