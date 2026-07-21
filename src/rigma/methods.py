"""Methods: one-click workflow setups for the things people actually do.

A preset sets a system prompt and sliders. A METHOD configures the whole
activity — prompt, sampler profile, thinking effort, tool posture, a Notes
template (the durable 'bible' that survives compaction), and a short guide
for how to work — because each activity has a right way to use a small
local model, and the user shouldn't have to rediscover it per chat.

Owner request 2026-07-21 ("there are only so many things, and there's a
Method for them"), born from the book-writing question: 6 chapters had eaten
21% of context because the manuscript lived in the conversation instead of
on disk with a bible in Notes.

Applying a method NEVER overwrites Notes the user already wrote — the
template lands only in an empty Notes field.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import method_schema as ms
from .runtime import rigma_home

METHODS: list[dict] = [
    {
        "id": "coding",
        "name": "Coding",
        "tagline": "build an app, tool, or script — plan, write, run, verify",
        "apply": {
            "system_prompt": (
                "You are a senior software engineer pair-programming in the "
                "user's workspace. Work in small verified steps: understand "
                "before changing, prefer the smallest edit_file diff, and "
                "run code (run_python / run_shell / start_job for anything "
                "slow) to PROVE a change works before calling it done. Keep "
                "files small and focused. When starting a project, write a "
                "short PLAN file first and keep it current."),
            "params": {"temperature": 0.6, "top_p": 0.95,
                       "dry_multiplier": 0.0, "repeat_penalty": 1.0},
            "effort": "on",
            "use_tools": True, "allow_code": True,
            "notes_template": (
                "PROJECT BRIEF (edit me — this is pinned every turn):\n"
                "- goal: \n- stack: \n- constraints: \n"
                "- decisions so far: "),
        },
        "guide": [
            "Set the workspace to the project folder first",
            "Ask for a plan file before any code on bigger builds",
            "Tests and builds run via tools — ask it to verify, not claim",
            "Long installs/builds: it can use start_job and keep working",
            "Keep the brief in Notes current — it survives compaction",
        ],
    },
    {
        "id": "book",
        "name": "Writing a book",
        "tagline": "long fiction that never outgrows the context window",
        "apply": {
            "system_prompt": (
                "You are a novelist collaborating on the user's book. The "
                "MANUSCRIPT LIVES ON DISK, not in this conversation: draft "
                "chapters with write_file (append=true for long parts), and "
                "never re-emit a whole existing chapter from memory. "
                "ADDING new text — a new bible entry, a new scene, the next "
                "part of a chapter — is an APPEND: use write_file with "
                "append=true. Use edit_file only to CHANGE words that "
                "already exist. Never stall weighing the two: if you are "
                "adding rather than rewriting, append and move on. The "
                "story "
                "bible in the notes is authoritative for cast, world, voice "
                "and what has happened. When a chapter is finished, update "
                "the bible's STORY SO FAR with 2-3 sentences for it."),
            "params": {"temperature": 0.85, "top_p": 0.95, "min_p": 0.05,
                       "dry_multiplier": 0.8, "dry_allowed_length": 2,
                       "repeat_penalty": 1.05},
            "effort": "auto",
            "use_tools": True, "allow_code": True,
            "notes_template": (
                "STORY BIBLE (edit me — pinned every turn, survives "
                "compaction):\nCAST:\n- \nWORLD RULES:\n- \n"
                "VOICE / STYLE:\n- \nSTORY SO FAR:\n- "),
        },
        "guide": [
            "One chat per chapter — the bible in Notes carries everything",
            "Set the workspace to the manuscript folder",
            "Chapters live in files; the chat is scratch space",
            "Continuity questions: ask it to delegate, not re-read chapters",
            "Done with a chapter? Hit 'Finish chapter' below — the bible "
            "updates itself and a fresh chat opens for the next one",
        ],
        # the ritual is what makes this a METHOD, not a preset: the workflow
        # move itself is a button (owner critique 2026-07-21: "this is
        # simply a system preset")
        "ritual": {"kind": "book_next_chapter",
                   "label": "Finish chapter — update bible, start next"},
    },
    {
        "id": "roleplay",
        "name": "Roleplay",
        "tagline": "immersive character play with stable long-scene memory",
        "apply": {
            "system_prompt": (
                "Stay fully in character. Write vivid, sensory scene prose; "
                "advance the scene rather than summarising it; NEVER speak "
                "or act for the user's character. Keep replies a few "
                "paragraphs unless asked otherwise. The notes define the "
                "characters and scenario and are authoritative."),
            "params": {"temperature": 1.0, "top_p": 0.95, "min_p": 0.05,
                       "dry_multiplier": 0.9, "dry_allowed_length": 2,
                       "repeat_penalty": 1.02, "xtc_probability": 0.3,
                       "xtc_threshold": 0.1},
            "effort": "off",
            "use_tools": True, "allow_code": False,
            "notes_template": (
                "SCENARIO (edit me — pinned every turn):\n"
                "YOUR CHARACTER:\n- \nMY CHARACTER:\n- \n"
                "SETTING:\n- \nHARD RULES:\n- "),
        },
        "guide": [
            "Define both characters and the scenario in Notes before "
            "starting",
            "Use the author's note for in-the-moment steering (pacing, "
            "mood) — it injects near the end where it's strongest",
            "Thinking is off for immersion and speed; flip effort back on "
            "if a scene needs plotting",
            "A real request mid-scene (save this, remember that) still "
            "gets done — the fiction never cancels tools",
        ],
    },
    {
        "id": "research",
        "name": "Research & analysis",
        "tagline": "grounded answers with sources, not vibes",
        "apply": {
            "system_prompt": (
                "You are a careful research analyst. Search the web for "
                "anything you are not certain of, read the sources "
                "(fetch_url pages through long articles), and cite where "
                "each claim came from. Separate what the sources SAY from "
                "what you infer. Write findings to files as you go so "
                "nothing is lost; end with a short sourced summary."),
            "params": {"temperature": 0.4, "top_p": 0.9},
            "effort": "on",
            "use_tools": True, "allow_code": True,
            "notes_template": (
                "RESEARCH QUESTION (edit me):\n- \n"
                "WHAT COUNTS AS AN ANSWER:\n- "),
        },
        "guide": [
            "Put the actual question in Notes — it stays pinned",
            "Ask for findings in a file, so long research survives",
            "Grounded chat + an indexed folder = it cites YOUR documents",
            "Big side-questions: it can delegate them to a helper",
        ],
    },
    {
        "id": "tutor",
        "name": "Learning & study",
        "tagline": "a tutor that quizzes you instead of lecturing",
        "apply": {
            "system_prompt": (
                "You are a patient tutor. Explain one concept at a time, "
                "with a concrete example, then CHECK understanding with a "
                "short question before moving on. Adapt to wrong answers "
                "by re-explaining differently, not repeating. Track the "
                "syllabus and progress in the notes."),
            "params": {"temperature": 0.5, "top_p": 0.9},
            "effort": "auto",
            "use_tools": True, "allow_code": True,
            "notes_template": (
                "STUDYING (edit me):\nTOPIC:\n- \nMY LEVEL:\n- \n"
                "SYLLABUS / PROGRESS:\n- "),
        },
        "guide": [
            "Tell it your level in Notes — it pitches everything to that",
            "Ground the chat on your course materials for cited answers",
            "Ask for a quiz at the end of each session",
        ],
    },
    {
        "id": "organize",
        "name": "Organizing files",
        "tagline": "sort, rename, and tidy folders without retyping paths",
        "apply": {
            "system_prompt": (
                "You organise the user's files carefully. ALWAYS explore "
                "first (list_directory summaries, sample_files) and act BY "
                "REFERENCE — move_files/copy_files on the sample, never "
                "retyped paths. Propose the plan (what moves where) before "
                "moving anything, prefer copy over move when unsure, and "
                "never delete without being asked twice."),
            "params": {"temperature": 0.3, "top_p": 0.9},
            "effort": "auto",
            "use_tools": True, "allow_code": True,
            "notes_template": (
                "ORGANIZING (edit me):\nFOLDER(S):\n- \n"
                "WHAT GOOD LOOKS LIKE:\n- \nNEVER TOUCH:\n- "),
        },
        "guide": [
            "Point the workspace at (or above) the folder to organise",
            "sample_files → view_sample → move by reference — no paths "
            "get retyped, so nothing gets lost",
            "Big jobs (thousands of files): use Autonomous mode instead "
            "and it works unattended",
        ],
    },
]


def methods_dir() -> Path:
    d = rigma_home() / "methods"
    d.mkdir(parents=True, exist_ok=True)
    return d


def builtins() -> list[dict]:
    """Normalized COPIES of the module literals. normalize() never mutates
    its input, so METHODS stays the pristine source and a bad save can't
    poison the catalog for the life of the process."""
    return [ms.normalize({**m, "builtin": True}) for m in METHODS]


def user_methods() -> list[dict]:
    """Every readable user method. A corrupt file is skipped, never fatal --
    the user should lose one broken method, not the whole Methods panel."""
    out = []
    for f in sorted(methods_dir().glob("*.json")):
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(raw, dict) and raw.get("id"):
            out.append(ms.normalize({**raw, "builtin": False}))
    return out


def catalog() -> list[dict]:
    """Built-ins then user methods, UI-ready. A user method that reuses a
    built-in's id REPLACES it in place -- that is how 'clone and edit a
    built-in' presents as editing the thing you cloned rather than as a
    confusing duplicate row."""
    merged = builtins()
    by_id = {m["id"]: i for i, m in enumerate(merged)}
    for u in user_methods():
        if u["id"] in by_id:
            merged[by_id[u["id"]]] = u
        else:
            merged.append(u)
    return merged


def get(method_id: str) -> dict | None:
    return next((m for m in catalog() if m["id"] == method_id), None)


def tool_names() -> set[str]:
    """Every tool a step may name. Imported lazily: tools.py pulls in the
    whole tool stack and methods.py is imported by the CLI."""
    from . import tools as toolkit
    return set(toolkit._REGISTRY)


def save_user(doc: dict) -> tuple[dict | None, list[str]]:
    """Validate and persist a user method. Returns (saved, []) or
    (None, errors) -- errors are phrased for a model to self-correct."""
    full = ms.normalize({**doc, "builtin": False})
    errs = ms.validate(full, tool_names())
    if errs:
        return None, errs
    (methods_dir() / f"{full['id']}.json").write_text(
        json.dumps(full, indent=2), encoding="utf-8")
    return full, []


def delete_user(method_id: str) -> bool:
    f = methods_dir() / f"{method_id}.json"
    if not f.is_file():
        return False
    f.unlink()
    return True


def apply_to_session(session: dict, method_id: str) -> dict | None:
    """Merge a method into a session dict (caller saves). Returns the
    session, or None for an unknown id. User-authored notes are never
    overwritten — the template lands only in an empty Notes field."""
    m = get(method_id)
    if m is None:
        return None
    a = m["apply"]
    session["system_prompt"] = a["system_prompt"]
    session["params"] = {**(session.get("params") or {}), **a["params"]}
    session["effort"] = a["effort"]
    session["use_tools"] = a["use_tools"]
    session["allow_code"] = a["allow_code"]
    if not str(session.get("notes") or "").strip():
        session["notes"] = a["notes_template"]
    session["method"] = method_id
    return session
