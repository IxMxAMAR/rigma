"""Read `data/compat.yaml`, the one place the three pinned versions live.

WHY A HAND-ROLLED PARSER AND NOT PyYAML. Rigma does not declare PyYAML as a
dependency and does not import it anywhere at runtime (`tests/test_harness_dsh.py`
even guards its own `import yaml` with `try/except ImportError`). A loader that
needed it would either add a dependency for one flat file or make the version pin
itself optional — and an OPTIONAL version pin is the failure this module exists to
end. So the matrix is restricted to flat `key: value` scalars and parsed here in
about twenty lines. `data/compat.yaml` says the same thing to whoever edits it.

NEVER RAISES. `harness_dsh` and `harness_mcode` read their `VERIFIED` constant
from here at import time. A missing or malformed matrix must degrade to the
built-in defaults rather than take down every import of Rigma — but it must NOT
degrade to an empty string, because `harness.conformance` reads an empty pin as
"unknown" and would silently stop reporting drift
(`harness.py`, `drift: (have != want) if (have and want) else None`). So the
fallback carries real versions, and `tests/test_harness_compat.py` asserts the
shipped file parses to the same values.
"""
from __future__ import annotations

import importlib.resources as resources
from pathlib import Path

# WHAT A BROKEN MATRIX FALLS BACK TO. Real versions on purpose — see the module
# docstring. Kept in step with `data/compat.yaml` by
# `tests/test_harness_compat.py`, which fails if the two disagree.
_FALLBACK: dict[str, str] = {
    "rigma": "0.12.1",
    "dsh": "0.2.0-rc.2",
    "dsh_min": "0.1.7-alpha.1",
    "dsh_max": "",
    "mcode": "0.5.4",
}

# The keys `load` is willing to return, in the order the file documents them.
KEYS = ("rigma", "dsh", "dsh_min", "dsh_max", "mcode")

_REL = "data/compat.yaml"


def _parse(text: str) -> dict[str, str]:
    """Read the flat `key: value` matrix, ignoring comments and blank lines.

    Deliberately not a YAML parser: it accepts `key: value`, strips one layer of
    matching quotes, and stops. A value containing a `#` is kept whole (the file
    never needs one), and a key it does not know is dropped rather than trusted.
    """
    out: dict[str, str] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, _, value = stripped.partition(":")
        key = key.strip()
        if key not in KEYS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        out[key] = value
    return out


def load() -> dict[str, str]:
    """The matrix, every key present, never raising.

    A key the file omits falls back to `_FALLBACK`, so a partially edited matrix
    degrades one field at a time instead of blanking the whole thing.
    """
    parsed: dict[str, str] = {}
    try:
        path = Path(str(resources.files("rigma").joinpath(_REL)))
        parsed = _parse(path.read_text(encoding="utf-8"))
    except Exception:
        parsed = {}
    return {key: parsed.get(key, _FALLBACK[key]) for key in KEYS}


MATRIX: dict[str, str] = load()


def version(key: str) -> str:
    """One pinned version, or "" for a key the matrix does not carry.

    "" is the honest answer for an unknown key and matches the empty-`VERIFIED`
    convention the adapters already use for "not measured".
    """
    return MATRIX.get(key, "")


def supported_band() -> tuple[str, str]:
    """`(dsh_min, dsh_max)` — either may be "", meaning "no known bound"."""
    return MATRIX["dsh_min"], MATRIX["dsh_max"]
