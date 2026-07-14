from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

MUTABLE_FIELDS = ("name", "system_prompt", "greeting", "params")
_BUILTIN_PREFIX = "usecase:"


def presets_dir() -> Path:
    d = rigma_home() / "presets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(preset_id: str) -> Path:
    return presets_dir() / f"{preset_id}.json"


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
    p = _path(preset_id)
    if is_builtin(preset_id) or not p.exists():
        return False
    p.unlink()
    return True


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
        p = load(f.stem)
        if p is None:  # corrupt file: skip, never fatal
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
