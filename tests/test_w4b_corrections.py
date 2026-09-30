"""Wave 4b corrections from the independent A7/A11 verifier.

Three follow-ups, none of them new behaviour:

  - A11 nit 1: `atomic_write_bytes`'s docstring claimed a trap that does not
    exist. The verifier measured that `atomic_write_text` opens with
    `newline="\\n"`, which DISABLES translation, so it round-trips CRLF bytes
    exactly; the real trap is the read side (`Path.read_text()` uses universal
    newlines) plus `Path.write_text` for a new write.
  - A7 nit 1: `run_profile=profile if profile in PROFILES else "all"` became a
    dead else-branch once the route started returning 400 for an unknown profile.
  - A11 nit 5: `_rollback_stores` caught only `OSError`, so a non-OSError
    write-back failure escaped and skipped the remaining files.
"""
from __future__ import annotations

from rigma import atomicio, serve


def _serve_source() -> str:
    with open(serve.__file__, encoding="utf-8") as f:
        return f.read()


# --- A11 nit 1: the docstring's stated reason is false ------------------------


def test_atomic_write_bytes_docstring_names_the_real_trap():
    doc = atomicio.atomic_write_bytes.__doc__ or ""
    assert "silently rewrite every line ending" not in doc, (
        "the verifier measured that atomic_write_text pins newline=\"\\n\", which "
        "DISABLES translation — the claim that it rewrites CRLF is false")
    assert "read_text" in doc, (
        "the real trap is the READ side (universal newlines), and it must be named")
    assert "write_text" in doc
    assert "tools._atomic_bytes" in doc, (
        "the existing private bytes writer that could now delegate here must be "
        "noted, per the verifier")


# --- A7 nit 1: the dead else-branch -------------------------------------------


def test_the_run_profile_is_not_re_coerced_after_the_400():
    """The 400 above the assignment makes `profile in PROFILES` always true, so
    the else was dead. A future edit that removes the 400 must not silently
    reintroduce the "all" coercion this finding is about."""
    src = _serve_source()
    assert 'run_profile=profile if profile in _runs.PROFILES else "all"' not in src, (
        "the dead else-branch is still there")
    assert "run_profile=profile)" in src


# --- A11 nit 5: the rollback's catch is too narrow ----------------------------


def test_rollback_keeps_going_after_a_non_oserror_write_back(monkeypatch):
    """A write-back failure that is not an OSError must not skip the files after
    it: the rollback is best-effort over ALL of them, and the failing label is
    reported rather than raised."""
    seen = []

    def _flaky(path, data):
        seen.append(str(path))
        if str(path) == "a.json":
            raise RuntimeError("not an OSError")

    monkeypatch.setattr(serve, "_restore_snapshot", _flaky)
    prior = [("a", "a.json", b"1"), ("b", "b.json", b"2"), ("c", "c.json", b"3")]
    failed = serve._rollback_stores(prior)
    assert failed == ["a"], failed
    assert seen == ["c.json", "b.json", "a.json"], (
        "the rollback stopped at the non-OSError instead of trying every file")
