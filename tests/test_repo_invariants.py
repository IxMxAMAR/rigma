"""Properties of the repository itself, checked the way CI checks them.

These exist because CI ran `ruff` before `pytest` and the lint step failed for
weeks, so the suite never ran there at all — and a test-only import bug sat
undetected the whole time, passing locally and failing on every push.
"""
from pathlib import Path

import pytest

TESTS = Path(__file__).parent
ROOT = TESTS.parent


def test_no_test_imports_the_tests_package():
    """A helper is imported by its flat module name, never as `tests.<name>`.

    `python -m pytest` prepends the working directory to sys.path, so the repo
    root is importable and `from tests.x import y` resolves. The bare `pytest`
    that CI runs does not, so the same line is a ModuleNotFoundError there. Both
    invocations put THIS directory on the path, so the flat name always works.

    Local verification must therefore use bare `pytest`, or it is not testing
    what CI tests.
    """
    offenders = []
    for path in sorted(TESTS.glob("test_*.py")):
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from tests.", "import tests.")) or \
                    stripped.startswith("from tests import"):
                offenders.append(f"{path.name}:{n}: {stripped}")
    assert not offenders, (
        "these imports resolve under `python -m pytest` and fail under the bare "
        "`pytest` CI runs — drop the `tests.` prefix:\n  "
        + "\n  ".join(offenders))


def test_the_shipped_ui_bundle_is_present():
    """The wheel serves data/ui_v2; a source tree missing it ships a 404 UI.

    Nothing else notices it is gone: it is a build artifact, so no Python
    imports it and no vitest test loads it.
    """
    bundle = ROOT / "src" / "rigma" / "data" / "ui_v2"
    assert (bundle / "index.html").is_file(), f"{bundle} has no index.html"
    assets = list((bundle / "assets").glob("*.js"))
    assert assets, f"{bundle / 'assets'} contains no javascript bundle"


def test_the_bundle_guard_hook_is_tracked_executable():
    """Git ignores a non-executable hook on POSIX, so `core.hooksPath .githooks`
    alone does not arm it. The mode must live in the index (100755)."""
    import subprocess
    out = subprocess.run(["git", "ls-files", "-s", ".githooks/pre-commit"],
                         cwd=ROOT, capture_output=True, text=True)
    if out.returncode != 0 or not out.stdout.strip():
        pytest.skip("not a git checkout")
    mode = out.stdout.split()[0]
    assert mode == "100755", (
        f".githooks/pre-commit is tracked {mode}; on POSIX git refuses to run "
        "it, so the bundle guard is silent. Fix: "
        "git update-index --chmod=+x .githooks/pre-commit")


def test_ci_bundle_staleness_check_counts_untracked_output():
    """`git diff` compares tracked files only, so a build that emits a new
    untracked asset looks clean. `git status --porcelain` includes `??`."""
    ci = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    assert "git status --porcelain" in ci
    assert "git diff --quiet -- src/rigma/data/ui_v2" not in ci


def test_publish_is_gated_on_tests_and_a_matching_tag():
    """A GitHub Release can be published from any commit, and `release:
    published` fires without waiting for CI, so without this gate the wheel
    reaches PyPI having never run pytest (AUDIT 12-6)."""
    import yaml
    text = (ROOT / ".github" / "workflows" / "publish.yml").read_text(
        encoding="utf-8")
    jobs = yaml.safe_load(text)["jobs"]
    seen, frontier, runs_pytest = set(), ["publish"], False
    while frontier:
        name = frontier.pop()
        if name in seen:
            continue
        seen.add(name)
        job = jobs.get(name) or {}
        if any("pytest" in str(s.get("run", "")) for s in job.get("steps", [])):
            runs_pytest = True
        needs = job.get("needs", [])
        frontier.extend([needs] if isinstance(needs, str) else needs)
    assert runs_pytest, "publish does not depend on any job that runs pytest"
    assert "github.event.release.tag_name" in text


def test_the_browser_smokes_target_the_right_ui():
    """`/` is the v2 React shell; the legacy DOM the old smoke drives moved to
    /rizz at the cutover, so a root-relative BASE tested the wrong page."""
    import re
    ui = ROOT / "tools" / "ui-smoke"
    legacy = (ui / "smoke_browser.py").read_text(encoding="utf-8")
    assert re.search(r'BASE\s*=\s*"http://127\.0\.0\.1:18500/rizz"', legacy)
    v2_path = ui / "smoke_browser_v2.py"
    assert v2_path.is_file(), "no browser smoke for the v2 shell at /"
    assert re.search(r'BASE\s*=\s*"http://127\.0\.0\.1:18500"',
                     v2_path.read_text(encoding="utf-8"))
