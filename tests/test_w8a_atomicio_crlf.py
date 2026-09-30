"""A11b: the behaviour `atomic_write_bytes`'s docstring now claims.

The docstring used to say the helper exists because `atomic_write_text` would
rewrite CRLF and alter line endings. The verifier measured that
`atomic_write_text` opens with `newline="\\n"`, which DISABLES translation, so
it round-trips CRLF bytes exactly; the real trap is the READ side. The wording
was corrected in 39792e3, but nothing pinned the behaviour it now asserts, so a
future edit that "helpfully" translates line endings would not fail a test.
These do.

(No before/after: the docstring correction already landed on this branch's base,
so the code change this pins is not in this commit. These are behavioural pins.)
"""
from __future__ import annotations

from pathlib import Path

from rigma import atomicio

CRLF = b"first\r\nsecond\r\n\r\nlast\r\n"


def _read_without_translating(path: Path) -> str:
    """The one read that keeps `\\r\\n` as characters: `newline=""`."""
    with open(path, "r", encoding="utf-8", newline="") as f:
        return f.read()


def test_atomic_write_text_round_trips_crlf_bytes(tmp_path):
    """The docstring's claim: `atomic_write_text` pins `newline="\\n"`, so it
    cannot translate. A CRLF document written through it comes back
    byte-for-byte."""
    p = tmp_path / "m.txt"
    p.write_bytes(CRLF)
    text = _read_without_translating(p)
    assert "\r\n" in text
    atomicio.atomic_write_text(p, text)
    assert p.read_bytes() == CRLF, (
        "atomic_write_text translated line endings; its newline=\"\\n\" says it "
        "must not")


def test_atomic_write_bytes_round_trips_crlf_bytes(tmp_path):
    p = tmp_path / "m.bin"
    atomicio.atomic_write_bytes(p, CRLF)
    assert p.read_bytes() == CRLF


def test_read_text_is_the_translating_half_the_docstring_names(tmp_path):
    """The REAL trap the corrected docstring names: `Path.read_text()` uses
    universal newlines, so a CRLF file read that way and written back with
    `atomic_write_text` comes out LF-only. That is why `atomic_write_bytes`
    takes bytes rather than text."""
    p = tmp_path / "m.txt"
    p.write_bytes(CRLF)
    translated = p.read_text(encoding="utf-8")
    assert "\r\n" not in translated
    atomicio.atomic_write_text(p, translated)
    assert p.read_bytes() == CRLF.replace(b"\r\n", b"\n")
