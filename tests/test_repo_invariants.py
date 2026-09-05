"""Properties of the repository itself, checked the way CI checks them.

These exist because CI ran `ruff` before `pytest` and the lint step failed for
weeks, so the suite never ran there at all — and a test-only import bug sat
undetected the whole time, passing locally and failing on every push.
"""
from pathlib import Path

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
