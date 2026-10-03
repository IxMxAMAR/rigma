"""Global skills: reusable instruction blocks the user writes once and pulls
into any chat with a leading /name.

A skill is a plain .md file in ~/.rigma/skills. Files, not a database, so the
user can write them in their own editor, keep them in a git repo, or paste one
in from somewhere else — the same reason methods live on disk.

R4-SKILL-2: THE FRONTMATTER IS NOT DECORATION. Rigma mounts DeepSeek Harness's
`skill` tool, and DSH's filesystem skill provider reads a skill by parsing YAML
frontmatter into a catalog entry — `parseSkillFile` warns "skill file ignored:
missing YAML frontmatter" and then "frontmatter requires name and description",
and SKIPS the file. A skill saved as bare prose was therefore invisible to the
agent even after the provider was pointed at this directory: the tool existed,
found the file, and discarded it. Both fields are written below so one file is
valid for Rigma's own `/name` expansion AND for the harness.
"""
from __future__ import annotations

import re
from pathlib import Path

from .runtime import rigma_home
from .atomicio import atomic_write_text

# A skill name becomes a FILENAME, and the name arrives over HTTP from the
# browser. Anything outside this set is rejected rather than sanitised away:
# quietly turning "../../etc/passwd" into "etcpasswd" writes a file the user
# never asked for, under a name they will never find again. Refusing says what
# happened.
_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$")

# A leading `---` line, which is how both Rigma's own files and DSH's provider
# recognise frontmatter.
_FRONTMATTER = re.compile(r"^---\s*\r?\n")


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


def _describe(name: str, content: str) -> str:
    """A one-line description for a skill, for the frontmatter DSH requires.

    Derived rather than asked for, because Rigma's own writer has never taken a
    description and adding a required field to the API would break every existing
    caller to satisfy a field the user never had to think about.

    The first line of prose is the best available answer: it is what the author
    wrote first, it is what the model needs to judge relevance, and it is already
    there. Headings are skipped (they are titles, not summaries), as is a second
    frontmatter block. Falls back to the name, so the field is never empty —
    an empty `description` is exactly as unusable to the provider as a missing
    one.
    """
    for line in str(content or "").splitlines():
        s = line.strip()
        if not s or s == "---" or s.startswith("#"):
            continue
        if s.startswith(("-", "*", ">", "|")) or s.endswith(":"):
            continue        # a list item, a quote, or a bare YAML key
        # Long enough to mean something, short enough to be a summary.
        return (s[:157] + "...") if len(s) > 160 else s
    return f"{name} skill"


def _with_frontmatter(name: str, content: str) -> str:
    """The content as saved: unchanged if it already declares frontmatter.

    Idempotent on purpose. A user who writes their own frontmatter — with a
    `disable-model-invocation` flag, say, which the provider honours — must not
    have it replaced by a generated one, and re-saving a skill must not stack a
    second block on top of the first.
    """
    text = str(content or "")
    if _FRONTMATTER.match(text):
        return text
    body = text.lstrip("\ufeff")
    desc = _describe(name, body).replace("\\", "\\\\").replace('"', '\\"')
    safe = name.replace("\\", "\\\\").replace('"', '\\"')
    return f'---\nname: "{safe}"\ndescription: "{desc}"\n---\n\n{body}'


def save_skill(name: str, content: str) -> dict:
    p = _path_for(name)
    text = _with_frontmatter(_clean(name), content)
    atomic_write_text(p, text)
    return {"id": p.stem, "name": p.stem, "title": p.stem,
            "filename": p.name, "content": text}


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
