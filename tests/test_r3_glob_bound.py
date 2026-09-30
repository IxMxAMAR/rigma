"""R3-16 (bound): no glob pattern may make the walk backtrack exponentially.

`test_r3_glob_redos.py` covers the two defects that were fixed first: a RUN of
`**/` collapsing to one group, and a malformed class no longer reaching
`re.compile`. This file covers the residual the collapse does not reach — a
pattern that SEPARATES the groups with literals, so each `**/` still becomes its
own `(?:.*/)?` and the groups backtrack against each other.

MEASURED at the base of this change (24-deep-ish paths, one match):
  `**/a/`x10 -> 0.085 s   x12 -> 1.72 s   x14 -> >25 s (killed)
  `**a`x20   -> >25 s (killed)
A model authors these; `find_files`/`grep` are `safe=True`, so nothing asks
first. The fix routes any translation with more than two ambiguous quantifiers
to a position-set DP (`_GlobMatcher`) instead of the regex engine. These tests
assert the STRUCTURE (which engine, which pattern string) rather than a timing,
because a timing assertion is a flake.
"""
import re

import pytest

from rigma import tools


# ---- structure: what reaches the regex engine ------------------------------

def test_a_run_of_double_stars_is_one_group_at_any_length():
    """The collapse is what makes the reported pattern safe, and it must hold
    for every repetition count, not just the one in the finding."""
    one = tools._glob_re("**/x.py").pattern
    assert one.count("(?:") == 1
    for k in (2, 10, 50, 200):
        assert tools._glob_re("**/" * k + "x.py").pattern == one


def test_an_interleaved_run_never_reaches_the_regex_engine():
    """`**/a/**/a/...` cannot be collapsed (the literals anchor the groups), so
    the only bound is to stop handing it to `re`. This is the structural
    assertion the brief asks for: the object is not a compiled pattern."""
    for k in (4, 12, 40, 200):
        rx = tools._glob_re("**/a/" * k + "x.py")
        assert not isinstance(rx, re.Pattern), (
            f"**/a/ x{k} still compiled to a backtracking regex")
        assert isinstance(rx, tools._GlobMatcher)
        assert rx.pattern                      # still introspectable


def test_interleaved_double_star_without_a_slash_is_also_bounded():
    """`**a` translates to `.*a` and the same argument applies to runs of it."""
    rx = tools._glob_re("**a" * 40 + "b")
    assert not isinstance(rx, re.Pattern)
    assert isinstance(rx, tools._GlobMatcher)


def test_ordinary_globs_still_use_the_fast_engine():
    """The bound must not move the common patterns onto the DP path; they are
    the hot path of the walk and the regex engine is faster for them."""
    for pat in ("**/*.py", "src/**/*.py", "**/*", "*.txt", "a?c.py"):
        assert isinstance(tools._glob_re(pat), re.Pattern), pat


# ---- semantics: the fallback must match exactly what the regex did ----------

_SEMANTIC = [
    # (pattern, path, expected)
    ("*.py", "a.py", True),
    ("*.py", "a.txt", False),
    ("*.py", "d/a.py", False),          # `*` does not cross `/`
    ("?", "a", True),
    ("?", "ab", False),
    ("?", "/", False),                  # `?` is `[^/]`
    # DR5: `?` is one character EACH — a run must not collapse to one.
    ("??.py", "ab.py", True),
    ("??.py", "a.py", False),
    ("??.py", "abc.py", False),
    ("ch??.md", "ch12.md", True),
    ("ch??.md", "ch1.md", False),
    ("???", "abc", True),
    ("???", "ab", False),
    ("a??b", "axyb", True),
    ("a??b", "axb", False),
    ("[abc].txt", "b.txt", True),
    ("[abc].txt", "d.txt", False),
    ("[!abc].txt", "d.txt", True),
    ("[!abc].txt", "a.txt", False),
    ("**/x.py", "x.py", True),          # `**/` matches zero directories
    ("**/x.py", "a/b/x.py", True),
    ("**/x.py", "a/b/x.txt", False),
    ("**", "a/b/c", True),              # bare `**` crosses `/`
    ("src/*.py", "src/a.py", True),
    ("src/*.py", "src/d/a.py", False),
    ("literal", "literal", True),
    ("literal", "literals", False),
    ("dir/", "dir/", True),             # trailing slash
    ("dir/", "dir", False),
    ("", "", True),                     # empty pattern, empty path
    ("", "a", False),
    ("a/b", "a/b", True),               # literal `/` in the middle
    ("a/b", "ab", False),
    ("a\\b", "a\\b", True),             # a backslash is a literal, not a sep
    ("a\\b", "a/b", False),
    ("**/" * 10 + "x.py", "a/b/c/x.py", True),
    ("**/" * 10 + "x.py", "a/b/c/x.txt", False),
    ("**/a/" * 10 + "x.py", "a/" * 10 + "x.py", True),
    ("**/a/" * 10 + "x.py", "a/" * 9 + "b/x.py", False),
    ("**a" * 10 + "b", "a" * 30 + "b", True),
    ("**a" * 10 + "b", "a" * 30 + "c", False),
]


@pytest.mark.parametrize("pat,path,expected", _SEMANTIC)
def test_semantics_are_unchanged(pat, path, expected):
    assert bool(tools._glob_re(pat).match(path)) is expected


def test_the_linear_matcher_agrees_with_its_own_regex():
    """For the patterns that take the DP path, compile the very regex string the
    matcher exposes and brute-force the two against each other. The corpus paths
    are short, so the reference regex is safe to run here."""
    pats = ["**/a/" * k + "x.py" for k in (1, 2, 3, 4, 5)]
    pats += ["**a" * k + "b" for k in (1, 2, 3, 4)]
    pats += ["**/a/**/b/**/c.txt", "***/*", "****/**/*", "**/[a-z]**/*"]
    # `?` tokens go through the DP path too, so the two must agree on them.
    pats += ["**/a/" * 3 + "??", "**/a/" * 3 + "?b?", "**/a/" * 2 + "x?.py",
             "**a" * 2 + "??"]
    paths = [""]
    for a in ("a", "b", "x.py", "c.txt"):
        paths.append(a)
    for a in ("a", "b", "x"):
        for b in ("a", "b", "x.py", "c.txt"):
            paths.append(a + "/" + b)
    for a in ("a", "x"):
        for b in ("a", "b"):
            for c in ("a", "x.py", "c.txt"):
                paths.append("/".join((a, b, c)))
    paths += ["a/" * 6 + "x.py", "a" * 12 + "b", "a/" * 6 + "c.txt"]

    checked = 0
    for pat in pats:
        matcher = tools._glob_re(pat)
        ref = re.compile(matcher.pattern)
        for path in paths:
            checked += 1
            assert bool(ref.match(path)) == bool(matcher.match(path)), (pat, path)
    assert checked > 400, checked


# ---- the bound, on a fixture tree ------------------------------------------

def test_an_absurd_glob_returns_normally(tmp_path, monkeypatch):
    """`**/` x200 collapses, so it is the regex path; `**/a/` x200 does not, so
    it is the DP path. Both must answer over a real walk instead of hanging."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "x.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x.py").write_text("print(2)\n", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}

    out = tools._find_files({"pattern": "**/" * 200 + "x.py"}, ctx)
    assert "x.py" in out and "a/x.py" in out, out

    # 200 required `a/` segments cannot exist here: the honest answer is
    # "no files match", and it must arrive without backtracking.
    out = tools._find_files({"pattern": "**/a/" * 200 + "x.py"}, ctx)
    assert out.startswith("no files match"), out

    out = tools._grep({"pattern": "print", "glob": "**/a/" * 200 + "*.py"}, ctx)
    assert out.startswith("no matches"), out


# ---- DR5: `?` is one character each, through the TOOLS ----------------------

def test_a_two_question_glob_finds_a_two_character_stem(tmp_path, monkeypatch):
    """The model's `??.py` must answer with the files that exist.

    Before DR5 the run collapsed to a single `[^/]`, so this returned "no files
    match ab.py"-style nothing for a tree that plainly had `ab.py` — and the
    model's next move was to conclude the file did not exist.
    """
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "ab.py").write_text("print(1)\n", encoding="utf-8")
    (tmp_path / "a.py").write_text("print(2)\n", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}

    out = tools._find_files({"pattern": "??.py"}, ctx)
    assert out.strip() == "ab.py", out


def test_a_two_question_GLOB_filters_grep_to_two_character_stems(tmp_path,
                                                                 monkeypatch):
    """The same collapse reached `grep`'s `glob` argument (A4 carried it into the
    token list), so a two-character stem was invisible to grep too."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    (tmp_path / "ch12.md").write_text("needle\n", encoding="utf-8")
    (tmp_path / "ch1.md").write_text("needle\n", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}

    out = tools._grep({"pattern": "needle", "glob": "ch??.md"}, ctx)
    assert "ch12.md:1:" in out, out
    assert "ch1.md:1:" not in out, out
