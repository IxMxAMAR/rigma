import tomllib
from pathlib import Path

import rigma


def test_version_present():
    assert rigma.__version__.startswith("0.")


def test_version_matches_pyproject():
    pyproject = Path(__file__).parent.parent / "pyproject.toml"
    meta = tomllib.loads(pyproject.read_text(encoding="utf-8"))
    assert rigma.__version__ == meta["project"]["version"]
