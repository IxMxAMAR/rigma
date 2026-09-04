from __future__ import annotations

import json
import re
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

MUTABLE_FIELDS = ("name", "system_prompt", "greeting", "params")
_BUILTIN_PREFIX = "usecase:"

# A preset id becomes a FILENAME and arrives from a URL path parameter.
# Starlette's {param} converter is [^/]+, so it stops %2f but not %5c, and
# uvicorn has decoded the path before routing -- which made
# DELETE /api/presets/..%5C..%5CDocuments%5Cbudget unlink a real file on
# Windows. Same shape as skills._SAFE_NAME, minus the space: ids here are
# generated (token_hex), never typed.  # AUDIT F40: docs/audit-2026-09-04-full.md
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# CON, PRN, AUX, NUL, COM1-9, LPT1-9 are unopenable as files on Windows
_WIN_DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


class PresetIdError(ValueError):
    """The requested id can't name a preset file."""


def presets_dir() -> Path:
    d = rigma_home() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(preset_id: str) -> Path:
    """The file for a preset id, or raise. Rejected, never sanitised: turning
    "..\\..\\budget" into "budget" writes over a file the user never named."""
    pid = str(preset_id or "")
    if not _SAFE_ID.match(pid) or _WIN_DEVICE.match(pid):
        raise PresetIdError(f"'{pid[:40]}' is not a valid preset id")
    p = (presets_dir() / f"{pid}.json").resolve()
    # belt and braces: whatever the pattern let through must land directly
    # inside the presets folder
    if p.parent != presets_dir().resolve():
        raise PresetIdError("that id would write outside the presets folder")
    return p


def is_builtin(preset_id: str) -> bool:
    return preset_id.startswith(_BUILTIN_PREFIX)


def create(name: str, system_prompt: str, greeting: str = "",
           params: dict | None = None) -> dict:
    now = time.time()
    preset = {"id": secrets.token_hex(6), "name": name,
              "system_prompt": system_prompt, "greeting": greeting,
              "params": params or {}, "builtin": False,
              "created_at": now, "updated_at": now}
    save(preset)
    return preset


def save(preset: dict) -> None:
    if is_builtin(preset["id"]):
        raise ValueError("builtin presets are read-only")
    preset["updated_at"] = time.time()
    p = _path(preset["id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(preset, indent=2), encoding="utf-8")
    tmp.replace(p)


def load(preset_id: str) -> dict | None:
    if is_builtin(preset_id):
        return None  # built-ins resolve via resolve(), not files
    try:
        return json.loads(_path(preset_id).read_text(encoding="utf-8"))
    except Exception:
        return None


def delete(preset_id: str) -> bool:
    if is_builtin(preset_id):
        return False        # checked first: "usecase:" is not a legal filename
    try:
        p = _path(preset_id)
        if p.exists():
            p.unlink()
            return True
    except PresetIdError:
        pass
    # AUDIT F40: docs/audit-2026-09-04-full.md — `list_presets` deliberately
    # shows a hand-placed file whose stem `_path` would refuse, so refusing to
    # delete one leaves a row in the panel that nothing can ever remove. Match
    # on the id INSIDE each file instead. Every candidate here was enumerated
    # from presets_dir(), never built from the caller's string, so this cannot
    # become the traversal the id guard exists to stop.
    for f in presets_dir().glob("*.json"):
        try:
            body = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(body, dict) and body.get("id") == preset_id:
            f.unlink()
            return True
    return False


def _builtins(registry=None) -> list[dict]:
    from .registry import Registry
    reg = registry if registry is not None else Registry.load()
    out = []
    for name, uc in sorted(reg.use_cases.items()):
        out.append({"id": _BUILTIN_PREFIX + name,
                    "name": name.capitalize() + " (built-in)",
                    "system_prompt": uc.system_prompt, "greeting": "",
                    "params": {}, "builtin": True,
                    "created_at": 0.0, "updated_at": 0.0})
    return out


def list_presets(registry=None) -> list[dict]:
    files = []
    for f in presets_dir().glob("*.json"):
        # read the file we already have in hand rather than round-tripping
        # its stem through load(): _path now REJECTS a name that could not
        # have been generated here, and a hand-placed "my preset.json" should
        # still list even though nothing may ask for it by that id
        try:
            p = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue  # corrupt file: skip, never fatal
        if not isinstance(p, dict):
            continue
        files.append(p)
    files.sort(key=lambda p: p.get("name", "").lower())
    return _builtins(registry) + files


def resolve(preset_id: str, registry=None) -> dict | None:
    """A preset by id — file preset or usecase: built-in. None if absent."""
    if not preset_id:
        return None
    if is_builtin(preset_id):
        for b in _builtins(registry):
            if b["id"] == preset_id:
                return b
        return None
    return load(preset_id)
