"""Method drafts: the same document as a method, at an earlier stage.

A draft lives in ~/.rigma/method_drafts/ and is deliberately NOT in
methods.catalog() -- a half-built method must not show up in the Methods
panel or be applicable to a chat. Promoting validates it exactly the way
methods.save_user does, because a draft that would not survive validation
was never a method in the first place.

Spec §6: the custom-methods design note (local)
"""
from __future__ import annotations

import json
import re
import secrets
from pathlib import Path

from . import method_schema as ms
from .runtime import rigma_home

# A draft id becomes a FILENAME and arrives from a URL path parameter
# (GET /api/methods/draft/{did}). Starlette's {param} converter is [^/]+, so
# it stops %2f but not %5c, and uvicorn decodes before routing -- so on Windows
# "..%5C..%5Cfoo" read and unlinked outside ~/.rigma. Same guard as
# skills._path_for.  # AUDIT F40: docs/audit-2026-09-04-full.md
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
# CON, PRN, AUX, NUL, COM1-9, LPT1-9 are unopenable as files on Windows
_WIN_DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


class DraftIdError(ValueError):
    """The requested id can't name a draft file."""


def drafts_dir() -> Path:
    d = rigma_home() / "method_drafts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(did: str) -> Path:
    """The file for a draft id, or raise. Rejected, never sanitised: a name
    quietly rewritten into something legal overwrites a file nobody named."""
    d = str(did or "")
    if not _SAFE_ID.match(d) or _WIN_DEVICE.match(d):
        raise DraftIdError(f"'{d[:40]}' is not a valid draft id")
    p = (drafts_dir() / f"{d}.json").resolve()
    if p.parent != drafts_dir().resolve():
        raise DraftIdError("that id would write outside the drafts folder")
    return p


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
    try:
        f = _path(did)
        if not f.is_file():
            return None
        raw = json.loads(f.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            return None
        # normalize() is inside the guard too: it does bare int()/list()/
        # dict()/.items() on whatever is in the file, so a hand-edited draft
        # with one wrong-typed key used to raise out of GET /api/methods/draft
        # instead of 404-ing.  # AUDIT F43: docs/audit-2026-09-04-full.md
        return ms.normalize({**raw, "builtin": False})
    except Exception:
        return None            # a corrupt draft loses one draft, not the app


def delete(did: str) -> bool:
    try:
        f = _path(did)
    except DraftIdError:
        return False           # same answer as "no such draft"
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
