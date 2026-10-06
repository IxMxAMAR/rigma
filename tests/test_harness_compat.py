"""The compatibility matrix is the ONE place the three versions are pinned.

WHY THIS TEST EXISTS. The DSH pin sat at `0.1.6-alpha.2` while the checkout on
this machine had moved to `0.2.0-rc.2`, and 0.2.0-rc.2 made the `protocol` key
FATAL at mount. Nothing failed loudly: the pin lived in one adapter file, nothing
tied it to the Rigma that shipped it, and `harness.conformance` can only diff a
pin it is handed. Every turn died at DSH mount instead.

So the matrix is now the source, both adapters read their `VERIFIED` from it, and
this file is what makes a silent divergence impossible: edit one and not the
other and it goes red.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from rigma.data import compat
from rigma import harness_dsh, harness_mcode

ROOT = Path(__file__).resolve().parents[1]


def test_the_shipped_matrix_parses_to_the_built_in_fallback():
    """`data/compat.yaml` must parse, and must agree with `_FALLBACK`.

    The loader never raises and falls back to `_FALLBACK` so a broken matrix
    cannot take down every import of Rigma. That safety net is only honest if
    the real file is known to parse — otherwise a typo in the YAML would silently
    pin the versions to whatever `_FALLBACK` happens to say, which is exactly the
    invisible-stale-pin failure this module exists to end.
    """
    parsed = compat._parse(
        (ROOT / "src" / "rigma" / "data" / "compat.yaml").read_text(encoding="utf-8"))
    assert parsed, "compat.yaml parsed to nothing"
    for key, value in compat._FALLBACK.items():
        assert parsed.get(key) == value, (
            f"compat.yaml {key!r} is {parsed.get(key)!r} but the loader's "
            f"fallback says {value!r} — they must not drift")


def test_every_matrix_key_is_present_and_non_empty_where_it_must_be():
    """The four load-bearing keys, and the one that is empty ON PURPOSE.

    `dsh_max` empty means "no known upper bound", which is the honest value and
    not a missing one, so it is excluded from the non-empty check and named here
    instead. `rigma`, `dsh` and `mcode` empty would each be a real defect: an
    empty `VERIFIED` reads as "unknown" to `harness.conformance` and switches
    drift detection OFF.
    """
    for key in ("rigma", "dsh", "dsh_min", "mcode"):
        assert compat.MATRIX[key].strip(), f"matrix key {key!r} is empty"
    assert compat.MATRIX["dsh_max"] == "", (
        "dsh_max is empty to mean 'no known upper bound'; if a bound is now "
        "known, set it AND update this test's expectation")


def test_both_adapters_read_their_pin_from_the_matrix():
    """The point of the exercise: one edit, both adapters.

    Before this, `harness_dsh.VERIFIED` and `harness_mcode.VERIFIED` were
    literals in their own files and nothing compared them to anything.
    """
    assert harness_dsh.VERIFIED == compat.MATRIX["dsh"]
    assert harness_mcode.VERIFIED == compat.MATRIX["mcode"]


def test_the_matrix_rigma_version_matches_the_package_and_pyproject():
    """Three places claim Rigma's version; they must agree.

    `pyproject.toml`, `rigma.__version__` and the matrix. A matrix that names a
    Rigma that does not exist is worse than no matrix, because it is the value a
    person reads when deciding whether a backend is compatible.
    """
    import rigma

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, re.MULTILINE)
    assert match, "pyproject.toml has no version"
    assert compat.MATRIX["rigma"] == rigma.__version__ == match.group(1)


def test_conformance_still_reports_drift_for_both_backends():
    """`harness.conformance` must keep working, now that the pins moved.

    It returns a LIST of rows and computes `drift` as
    `(have != want) if (have and want) else None`, so the two things worth
    pinning are that `want` is still a real version for each backend — not the
    "" that would switch drift detection off — and that the tri-state logic
    still distinguishes "they agree" from "I could not tell".

    Deliberately NOT asserting `drift is False`: whether this machine's checkout
    matches the pin is an environment fact, and a test that depends on it would
    fail on a machine with a different build for the right reason.
    """
    from rigma import harness

    for name in ("dsh", "mcode"):
        rows = harness.conformance(name)
        assert len(rows) == 1, rows
        row = rows[0]
        assert row["verified"] == compat.MATRIX[name], row
        assert row["verified"], f"conformance has no verified version for {name}"
        if row["version"]:
            assert row["drift"] is (row["version"] != row["verified"]), row
        else:
            # a backend that cannot answer is UNKNOWN, not agreement
            assert row["drift"] is None, row


def test_the_pin_survives_every_import_order():
    """`conformance` must see the DSH pin whichever module is imported first.

    THIS IS A REAL BUG THIS FILE CAUGHT, not a hypothetical. `harness.py` builds
    its BACKENDS table at import time and reads each adapter's `VERIFIED` through
    `_verified_of`, which imports the adapter. When `rigma.harness_dsh` was
    imported FIRST it found itself half-initialised in `sys.modules` and cached
    "" as the pin — so DSH drift detection switched itself OFF, silently, and
    only for DSH, because `harness_mcode` already defined its `VERIFIED` above
    its own `.harness` import. Measured before the fix: order A gave
    "0.2.0-rc.2", order B gave "". `harness_dsh` now hoists the constant for the
    same documented reason mcode does.

    Run in SUBPROCESSES because import order is a property of the process, and
    this test module has already imported both adapters by the time it runs.
    """
    import os
    import subprocess
    import sys

    # The child must import the SAME rigma this test is running against.
    # pytest's `pythonpath` ini (pyproject.toml) puts src/ on THIS process's
    # sys.path and does NOT reach a subprocess, and this machine also has an
    # older rigma in site-packages (0.10.0, which has no `rigma.harness` at all)
    # — so without this the child died with ModuleNotFoundError instead of
    # answering the question. Derived from the loaded package, so it is right
    # from a source tree and from an editable install alike.
    src_root = str(Path(compat.__file__).resolve().parents[2])
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [src_root] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))

    probe = (
        "import rigma.harness as h;"
        "print(h.conformance('dsh')[0]['verified'] or 'EMPTY')"
    )
    for prelude in ("", "import rigma.harness_dsh;", "import rigma.harness_mcode;"):
        out = subprocess.run([sys.executable, "-c", prelude + probe],
                             capture_output=True, text=True, timeout=120,
                             cwd=str(ROOT), env=env)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == compat.MATRIX["dsh"], (
            f"import order {prelude!r} lost the DSH pin: {out.stdout.strip()!r}")


def test_a_pre_release_sorts_below_its_release():
    """The ordering the drift wording depends on.

    A lexical compare gets both of these wrong, and the notice would then tell a
    person their build is "too old" when it is newer — a wrong claim is worse
    than the plain drift sentence it replaces.
    """
    key = harness_dsh._version_key
    assert key("0.2.0-rc.2") < key("0.2.0")
    assert key("0.9.0") < key("0.10.0")
    assert key("0.1.6-alpha.2") < key("0.1.7-alpha.1")
    assert key("not-a-version") == ()


@pytest.mark.parametrize("have,expected", [
    # below dsh_min: the fatal edge, and the notice must SAY so
    ("0.1.6-alpha.2", "OLDER than the oldest build"),
    # ahead of the pin but inside the band: plain drift
    ("0.2.0", "is not the build this adapter was measured against"),
])
def test_the_drift_notice_wording_follows_the_matrix_band(monkeypatch, have, expected):
    """The notice names the fatal edge only when the matrix proves it.

    Exercised by monkeypatching `backend_version`, so no checkout is needed and
    no process is started.
    """
    monkeypatch.setattr(harness_dsh, "_DRIFT_SAID", False)
    monkeypatch.setattr(harness_dsh, "backend_version", lambda: have)
    notice = harness_dsh._drift_notice()
    assert notice is not None
    assert expected in notice.text, notice.text


def test_the_drift_notice_is_silent_when_the_build_matches(monkeypatch):
    """No notice for the pinned build — and it is still once-per-process."""
    monkeypatch.setattr(harness_dsh, "_DRIFT_SAID", False)
    monkeypatch.setattr(harness_dsh, "backend_version", lambda: harness_dsh.VERIFIED)
    assert harness_dsh._drift_notice() is None
    # and once it has spoken, it does not speak again in this process
    monkeypatch.setattr(harness_dsh, "_DRIFT_SAID", False)
    monkeypatch.setattr(harness_dsh, "backend_version", lambda: "9.9.9")
    assert harness_dsh._drift_notice() is not None
    assert harness_dsh._drift_notice() is None
