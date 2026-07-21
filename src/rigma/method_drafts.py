"""Method drafts: the same document as a method, at an earlier stage.

A draft lives in ~/.rigma/method_drafts/ and is deliberately NOT in
methods.catalog() -- a half-built method must not show up in the Methods
panel or be applicable to a chat. Promoting validates it exactly the way
methods.save_user does, because a draft that would not survive validation
was never a method in the first place.

Spec §6: docs/superpowers/specs/2026-07-21-custom-methods-design.md
"""
from __future__ import annotations

import json
import secrets
from pathlib import Path

from . import method_schema as ms
from .runtime import rigma_home


def drafts_dir() -> Path:
    d = rigma_home() / "method_drafts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(did: str) -> Path:
    return drafts_dir() / f"{did}.json"


def new_draft(name: str = "", tagline: str = "") -> dict:
    doc = ms.normalize({
        "id": "draft_" + secrets.token_hex(4),
        "name": name, "tagline": tagline,
        "apply": {"system_prompt": "", "params": {}, "effort": "auto",
                  "use_tools": True, "allow_code": True,
                  "notes_template": ""},
    })
    return save(doc)


def save(doc: dict) -> dict:
    """Persist WITHOUT validating -- a draft is allowed to be incomplete;
    that is the whole point of it being a draft. promote() is the gate."""
    full = ms.normalize({**doc, "builtin": False})
    _path(full["id"]).write_text(json.dumps(full, indent=2), encoding="utf-8")
    return full


def load(did: str) -> dict | None:
    f = _path(did)
    if not f.is_file():
        return None
    try:
        raw = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return None            # a corrupt draft loses one draft, not the app
    if not isinstance(raw, dict):
        return None
    return ms.normalize({**raw, "builtin": False})


def delete(did: str) -> bool:
    f = _path(did)
    if not f.is_file():
        return False
    f.unlink()
    return True


def promote(did: str) -> tuple[dict | None, list[str]]:
    """Validate the draft and write it as a real user method. On failure the
    draft is left exactly as it was, so the creation chat can keep fixing it.

    The method takes a slug of its NAME, not the draft's throwaway id: the
    user named the thing, and `draft_9f2a1c` is nobody's idea of an id.
    """
    from . import methods as _methods
    doc = load(did)
    if doc is None:
        return None, [f"no such draft '{did}'"]
    taken = {m["id"] for m in _methods.catalog()}
    final = dict(doc)
    final["id"] = ms.slugify(doc.get("name") or "method", taken)
    saved, errs = _methods.save_user(final)
    if errs:
        return None, errs
    delete(did)
    return saved, []
