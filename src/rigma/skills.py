"""Global skills: reusable instruction blocks the user writes once and pulls
into any chat with a leading /name.

A skill is a plain .md file in ~/.rigma/skills. Files, not a database, so the
user can write them in their own editor, keep them in a git repo, or paste one
in from somewhere else — the same reason methods live on disk.
"""
from __future__ import annotations

import re
from pathlib import Path

from .runtime import rigma_home

# A skill name becomes a FILENAME, and the name arrives over HTTP from the
# browser. Anything outside this set is rejected rather than sanitised away:
# quietly turning "../../etc/passwd" into "etcpasswd" writes a file the user
# never asked for, under a name they will never find again. Refusing says what
# happened.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")


class SkillNameError(ValueError):
    """The requested name can't be a skill file."""


def skills_dir() -> Path:
    d = rigma_home() / "skills"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _clean(name: str) -> str:
    """The bare skill name, or raise. Rejects path separators, traversal,
    drive letters and reserved Windows device names."""
    n = str(name or "").strip()
    if n.lower().endswith(".md"):
        n = n[:-3]
    n = n.strip()
    if not _SAFE_NAME.match(n):
        raise SkillNameError(
            "a skill name may only contain letters, numbers, spaces, "
            "hyphens and underscores (max 64)")
    # CON, PRN, AUX, NUL, COM1-9, LPT1-9 are unopenable as files on Windows
    if re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", n, re.I):
        raise SkillNameError(f"'{n}' is a reserved Windows device name")
    return n


def _path_for(name: str) -> Path:
    p = (skills_dir() / f"{_clean(name)}.md").resolve()
    # belt and braces: even with the pattern above, the file that comes out
    # must be directly inside the skills folder
    if p.parent != skills_dir().resolve():
        raise SkillNameError("that name would write outside the skills folder")
    return p


def list_skills() -> list[dict]:
    out = []
    for f in sorted(skills_dir().glob("*.md")):
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        title = f.stem
        for line in content.splitlines():
            if line.startswith("# "):
                title = line[2:].strip() or f.stem
                break
        out.append({"id": f.stem, "name": f.stem, "title": title,
                    "filename": f.name, "content": content})
    return out


def get_skill(name: str) -> str | None:
    """The skill's text, matched case-insensitively, or None."""
    try:
        wanted = _clean(name).lower()
    except SkillNameError:
        return None
    for f in skills_dir().glob("*.md"):
        if f.stem.lower() == wanted:
            try:
                return f.read_text(encoding="utf-8", errors="replace")
            except OSError:
                return None
    return None


def save_skill(name: str, content: str) -> dict:
    p = _path_for(name)
    p.write_text(str(content or ""), encoding="utf-8")
    return {"id": p.stem, "name": p.stem, "title": p.stem,
            "filename": p.name, "content": content}


def delete_skill(name: str) -> bool:
    try:
        wanted = _clean(name).lower()
    except SkillNameError:
        return False
    for f in skills_dir().glob("*.md"):
        if f.stem.lower() == wanted:
            try:
                f.unlink()
                return True
            except OSError:
                return False
    return False
