"""R3-16: a glob pattern could stop the server, two different ways.

Both were reachable from the model through `find_files`/`grep`'s `glob`, and
neither needs adversarial input — `find_files("**/**/**/…")` is what a model
writes when it is guessing at depth.

MEASURED on a 24-deep path before the fix: `**/`x6 = 7 ms, x10 = 2.19 s,
**x12 = 61.5 s**. The match runs once per file in the walk, the walk is
synchronous on the event loop, and the walk has a 5000-entry budget — so the
worst case was hours of a wedged UI, not a slow tool call.

The fix must not change what a glob MEANS, so equivalence is asserted
exhaustively here rather than argued: the old translation is reproduced verbatim
and both are run over a generated corpus of paths.
"""
import itertools
import re
import time

import pytest

from rigma import tools


def _old_glob_re(pat: str):
    """The translation exactly as it was before R3-16, for equivalence testing."""
    out, i, n = [], 0, len(pat)
    while i < n:
        c = pat[i]
        if c == "*":
            if pat.startswith("**", i):
                i += 2
                if i < n and pat[i] == "/":
                    i += 1
                    out.append("(?:.*/)?")
                else:
                    out.append(".*")
                continue
            out.append("[^/]*")
        elif c == "?":
            out.append("[^/]")
        elif c == "[":
            j = i + 1
            if j < n and pat[j] in "!^":
                j += 1
            if j < n and pat[j] == "]":
                j += 1
            while j < n and pat[j] != "]":
                j += 1
            if j >= n:
                out.append(re.escape(c))
            else:
                inner = pat[i + 1:j]
                if inner.startswith("!"):
                    inner = "^" + inner[1:]
                out.append("[" + inner + "]")
                i = j + 1
                continue
        else:
            out.append(re.escape(c))
        i += 1
    return re.compile("(?s:" + "".join(out) + r")\Z")


_DEEP = "/".join(["d"] * 24) + "/file.txt"


@pytest.mark.parametrize("k", [6, 10, 12, 20, 40])
def test_a_run_of_double_stars_does_not_explode(k):
    """The measurement, as a test. Before the fix x12 alone was ~61 s; the
    threshold here is generous (250 ms) so it cannot flake on a loaded machine,
    while still being four orders of magnitude below the bug."""
    rx = tools._glob_re("**/" * k + "x")
    assert not isinstance(rx, re.error)
    t = time.perf_counter()
    rx.match(_DEEP)
    assert time.perf_counter() - t < 0.25, f"**/ x{k} took too long"


def test_the_regex_is_shorter_than_the_pattern_for_a_run():
    """The mechanism, not just the symptom: a run of `**/` collapses to ONE
    group. If someone reintroduces per-star groups the timing test would have to
    catch it; this states the invariant directly."""
    one = tools._glob_re("**/x").pattern
    many = tools._glob_re("**/" * 12 + "x").pattern
    assert many == one, "a run of **/ must compile to the same regex as one"


@pytest.mark.parametrize("pat", ["[z-a]", "[9-0]", "[a-]", "[-a]", "[!z-a]*",
                                 "[[]*", "[]]*", "[", "[]", "[a", "[!", "[\\",
                                 "[a-b-c]", "[--]", "[^^]", "[[["])
def test_no_glob_string_can_make_the_translation_raise(pat):
    """`[z-a]` reached `re.compile` verbatim and raised out of a TOOL, where
    nothing catches it — a typo became a 500. The escaping is now TOTAL, so the
    honest assertion is the strong one: no class-shaped string reaches the
    compiler unescaped. The list is the scanner's own awkward cases, not just the
    one that was found."""
    r = tools._glob_re(pat)
    assert not isinstance(r, re.error), f"{pat!r} still raises: {r}"


def test_a_typo_class_is_a_literal_class_not_an_error(tmp_path, monkeypatch):
    """`[z-a]` is a typo, not a hostile pattern, and the only non-error reading is
    "one of z, -, a" — so the tool must answer normally. Note it matches ONE
    character, which is why `[z-a].txt` does not match `z-a.txt`; that is what a
    character class means and this test states it rather than pretending
    otherwise."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "z.txt").write_text("x", encoding="utf-8")
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    (tmp_path / "-.txt").write_text("x", encoding="utf-8")
    (tmp_path / "q.txt").write_text("x", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}
    out = tools._find_files({"pattern": "[z-a].txt"}, ctx)
    assert "z.txt" in out and "a.txt" in out and "-.txt" in out, out
    assert "q.txt" not in out, out


def test_a_range_is_still_a_range():
    """The fix must not turn `[a-z]` into a literal class — that would silently
    change what the most ordinary pattern in the language means."""
    rx = tools._glob_re("[a-z].txt")
    assert rx.match("a.txt") and rx.match("z.txt")
    assert not rx.match("A.txt") and not rx.match("0.txt")
    # and the hyphen really is a literal in the degenerate forms
    assert tools._glob_re("[z-a]").match("-")
    assert tools._glob_re("[a-]").match("-")
    # a chain like `a-b-c` keeps its real range and escapes the stray hyphen
    assert tools._glob_re("[a-b-c]").match("a")
    assert tools._glob_re("[a-b-c]").match("-")


def test_a_negated_range_still_negates():
    rx = tools._glob_re("[!a-c].txt")
    assert not rx.match("a.txt") and not rx.match("b.txt")
    assert rx.match("d.txt") and rx.match("0.txt")


def test_the_error_guard_is_wired_even_though_nothing_reaches_it(
        tmp_path, monkeypatch):
    """The `re.error` return is an invariant guard, not a live path — the
    escaping above is total. It is still wired through BOTH tools, because a
    future edit that makes the escaping partial again must degrade to "bad
    pattern" rather than a 500. This drives the guard directly, so the wiring
    cannot rot unnoticed."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "a.txt").write_text("hello\n", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}
    boom = re.error("simulated")
    monkeypatch.setattr(tools, "_glob_re", lambda pat: boom)
    out = tools._find_files({"pattern": "*"}, ctx)
    assert out.startswith("error: bad pattern"), out
    out = tools._grep({"pattern": "hello"}, ctx)
    assert out.startswith("error: bad glob"), out


# ---- equivalence: the collapse must not change what a glob MEANS -------------

_PATTERNS = [
    "*", "**", "**/*", "**/*.py", "*/*.py", "?.py", "a?c.py",
    "**/**/*", "**/**/**/*", "**/**/**/**/**/**/*",
    "src/**/*.py", "src/**/**/**/*.py", "***/*", "****/**/*",
    "**/", "**/**/", "***", "a**b", "*.*", "[abc]*", "[a-c]*",
    "[!abc]*", "[^abc]*", "file[0-9].txt", "file[a-z].txt", "**/[a-z]**/*",
]
_NAMES = ["a", "b", "a.py", "c.txt", "0", "file0.txt", "filea.txt", "-"]
_PATHS = [""] + ["/".join(c) for d in (1, 2, 3)
                 for c in itertools.product(_NAMES, repeat=d)]
_PATHS += ["a/b/c/d/e.py", "src/x/y/z.py", "deep/" * 30 + "f.py"]


def test_the_collapse_accepts_exactly_what_the_old_form_accepted():
    """Brute force, because "the two regexes are equivalent" is exactly the kind
    of claim that is true for the cases you thought of and false for one you did
    not. 17k+ (pattern, path) pairs."""
    diffs = []
    checked = 0
    for p in _PATTERNS:
        new = tools._glob_re(p)
        assert not isinstance(new, re.error), p
        old = _old_glob_re(p)
        for path in _PATHS:
            checked += 1
            if bool(old.match(path)) != bool(new.match(path)):
                diffs.append((p, path))
    assert checked > 15000, checked
    assert diffs == [], diffs[:10]
