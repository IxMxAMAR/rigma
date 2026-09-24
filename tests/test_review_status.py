"""Tests for the review fan-out ledger (tools/review_status.py).

The ledger exists so that a review which dies half way can be resumed instead of
being mistaken for a finished one, so these tests are about the distinction
between "an agent wrote a file" and "an agent finished its area".
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location("review_status", ROOT / "tools" / "review_status.py")
    mod = importlib.util.module_from_spec(spec)
    # Register before exec: dataclasses looks the module up in sys.modules to
    # resolve the annotations on the classes this file defines.
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load()

PLAN = "01\talpha\tsrc/a.py;src/b.py\n02\tbeta\tsrc/c.py\n"

FULL = """# 01 alpha
## Coverage
coverage: src/a.py L1-L10 (read fully)
coverage: src/b.py (read fully)
commands: none

## Findings
### 01-1 a real bug
- severity: HIGH
- kind: bug
### 01-2 another
- severity: LOW
<!-- END 01 -->
"""


@pytest.fixture()
def tree(tmp_path):
    (tmp_path / "docs" / "review" / "findings").mkdir(parents=True)
    (tmp_path / "docs" / "review" / "plan.tsv").write_text(PLAN, encoding="utf-8")
    return tmp_path


def _write(root: Path, rel: str, text: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")


def test_parse_plan_reads_id_slug_and_files():
    areas = rs.parse_plan(PLAN)
    assert [a.id for a in areas] == ["01", "02"]
    assert areas[0].files == ["src/a.py", "src/b.py"]
    assert areas[0].rel_path == "docs/review/findings/01-alpha.md"


def test_parse_plan_skips_blanks_comments_and_junk():
    areas = rs.parse_plan("\n# comment\nnonsense\n03\tgamma\n")
    assert [a.id for a in areas] == ["03"]
    assert areas[0].files == []


def test_a_missing_file_is_missing_not_clean(tree):
    statuses = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))
    assert [s.state for s in statuses] == [rs.MISSING, rs.MISSING]
    assert not statuses[0].ok


def test_an_empty_file_is_not_a_result(tree):
    _write(tree, "docs/review/findings/01-alpha.md", "   \n")
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.state == rs.EMPTY
    assert not s.ok


def test_a_file_without_the_sentinel_is_truncated(tree):
    # The agent was killed mid-write: the file looks plausible but is not a result.
    _write(tree, "docs/review/findings/01-alpha.md", FULL.replace("<!-- END 01 -->", ""))
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.state == rs.TRUNCATED
    assert not s.ok


def test_a_complete_file_counts_severities_and_findings(tree):
    _write(tree, "docs/review/findings/01-alpha.md", FULL)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.state == rs.COMPLETE
    assert s.findings == 2
    assert s.counts["HIGH"] == 1 and s.counts["LOW"] == 1 and s.counts["CRITICAL"] == 0
    assert s.gaps == []
    assert s.ok


def test_a_scoped_file_absent_from_coverage_is_a_gap(tree):
    # The quiet failure: the agent reports the area clean without opening b.py.
    text = FULL.replace("coverage: src/b.py (read fully)\n", "")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.state == rs.COMPLETE
    assert not s.ok
    assert s.gaps and "src/b.py" in s.gaps[0]


def test_coverage_marked_not_read_is_a_gap(tree):
    text = FULL.replace("coverage: src/b.py (read fully)", "coverage: src/b.py (NOT READ)")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert not s.ok
    assert "NOT READ" in s.gaps[0]


def test_a_path_without_a_line_range_is_not_coverage(tree):
    """The docstring promises "naming the lines it read"; a bare path is not
    evidence the file was opened."""
    text = FULL.replace("coverage: src/a.py L1-L10 (read fully)",
                        "coverage: src/a.py (read it)")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert not s.ok
    assert any("src/a.py" in g and "range" in g for g in s.gaps)


def test_a_longer_sibling_name_does_not_cover_the_scoped_file(tree):
    """src/a.py.bak is a different file; it must not satisfy src/a.py."""
    text = FULL.replace("coverage: src/a.py L1-L10 (read fully)",
                        "coverage: src/a.py.bak L1-L10 (read fully)")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert not s.ok
    assert "src/a.py" in s.gaps[0]


def test_a_directory_scope_needs_a_path_boundary(tree):
    """src/tests_helper.py is not under tests/."""
    (tree / "docs" / "review" / "plan.tsv").write_text("05\tdir\ttests/\n", encoding="utf-8")
    _write(tree, "docs/review/findings/05-dir.md",
           "coverage: src/tests_helper.py L1-L9 (read fully)\n<!-- END 05 -->\n")
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert not s.ok
    assert "tests/" in s.gaps[0]


def test_a_partial_read_note_with_ranges_is_not_a_gap(tree):
    """A NOT READ note that also names the lines read covers the rest of the
    file being out of scope (areas 13 and 15 of the real review)."""
    text = FULL.replace("coverage: src/b.py (read fully)",
                        "coverage: src/b.py L1-L9 (read; L10-L40 not read)")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.ok, s.gaps


def test_coverage_matching_is_case_and_separator_insensitive(tree):
    text = FULL.replace("coverage: src/a.py L1-L10 (read fully)", "coverage: SRC\\A.PY L1-L10")
    _write(tree, "docs/review/findings/01-alpha.md", text)
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.ok


def test_a_directory_scope_matches_by_prefix(tree):
    (tree / "docs" / "review" / "plan.tsv").write_text("05\tdir\ttests/;pyproject.toml\n", encoding="utf-8")
    _write(
        tree,
        "docs/review/findings/05-dir.md",
        "coverage: tests/test_a.py L1-L10 (read fully)\n"
        "coverage: pyproject.toml L1-L5 (read fully)\n<!-- END 05 -->\n",
    )
    s = rs.scan(tree, rs.load_plan(tree / "docs/review/plan.tsv"))[0]
    assert s.ok, s.gaps


def test_missing_lists_only_what_needs_redispatch(tree, capsys):
    _write(tree, "docs/review/findings/01-alpha.md", FULL)
    rc = rs.main(["--root", str(tree), "--missing"])
    out = capsys.readouterr().out
    assert rc == 0
    assert out.startswith("02\tbeta\tMISSING")
    assert "01" not in out


def test_check_fails_until_every_area_is_complete_and_covered(tree):
    _write(tree, "docs/review/findings/01-alpha.md", FULL)
    assert rs.main(["--root", str(tree), "--check"]) == 1
    _write(tree, "docs/review/findings/02-beta.md", "coverage: src/c.py L1-L5 (read fully)\n<!-- END 02 -->\n")
    assert rs.main(["--root", str(tree), "--check"]) == 0


def test_json_output_is_parseable(tree, capsys):
    import json

    _write(tree, "docs/review/findings/01-alpha.md", FULL)
    rs.main(["--root", str(tree), "--json"])
    data = json.loads(capsys.readouterr().out)
    assert {d["id"] for d in data} == {"01", "02"}
    assert data[0]["counts"]["HIGH"] == 1


def test_a_missing_plan_is_a_usage_error(tmp_path, capsys):
    assert rs.main(["--root", str(tmp_path)]) == 2
    assert "no plan" in capsys.readouterr().err
