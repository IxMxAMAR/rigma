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
import re
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
        "rules": [
            {"id": "verify_first", "kind": "standing",
             "text": "Prove a change works with a tool before calling it "
                     "done."}],
        "macros": [
            {"id": "run_tests", "label": "Run tests",
             "hint": "run the project's test command and report",
             "steps": [{"kind": "prompt",
                        "text": "Run this project's test suite with a tool "
                                "and report the result. Do not guess the "
                                "command — look for it first."}]},
            {"id": "explain_file", "label": "Explain this file",
             "hint": "read a file and summarise what it does",
             "steps": [
                 {"kind": "tool", "name": "read_file",
                  "args": {"path": "{{ask:Which file}}"}},
                 {"kind": "prompt", "to": "aux",
                  "text": "Explain in 5 sentences what this file does and "
                          "what depends on it:\n\n{{step:0}}"}]},
            {"id": "write_plan", "label": "Write PLAN file",
             "hint": "capture the current plan on disk",
             "steps": [{"kind": "prompt",
                        "text": "Write the current plan to PLAN.md with "
                                "write_file: goal, steps, and what is done "
                                "so far. Keep it under 40 lines."}]},
            {"id": "review_diff", "label": "Review my diff",
             "hint": "read the working diff and critique it",
             "steps": [{"kind": "prompt",
                        "text": "Show the working diff with a tool, then "
                                "review it for bugs and missed cases. Be "
                                "specific about file and line."}]}],
        "workflows": [
            {"id": "feature", "label": "Feature: plan, build, test, log",
             "resumable": True,
             "steps": [
                 {"kind": "prompt",
                  "text": "Write a short plan for: {{ask:What feature}}. "
                          "Save it to PLAN.md."},
                 {"kind": "prompt",
                  "text": "Implement the first unchecked step of PLAN.md."},
                 {"kind": "prompt",
                  "text": "Run the tests and fix what fails."},
                 {"kind": "prompt",
                  "text": "Mark the finished step done in PLAN.md."}]}],
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
        # the components are what make this a METHOD, not a preset: the
        # workflow moves themselves are buttons (owner critique 2026-07-21:
        # "this is simply a system preset"). `finish_chapter` is the old
        # hardcoded book_next_chapter ritual, now an ordinary step list.
        "vars": {
            "bible": {"label": "Story bible file", "default": "story_bible.md",
                      "kind": "path"}},
        "rules": [
            {"id": "bible_is_law", "kind": "standing",
             "text": "The story bible in the notes is authoritative for cast, "
                     "world, voice and what has happened."},
            {"id": "chapter_written", "kind": "trigger",
             "on": {"event": "tool_ran", "tool": "write_file",
                    "path_glob": "chapters/*"},
             "do": {"mode": "nudge",
                    "text": "A chapter file just changed — update the "
                            "bible's STORY SO FAR."}}],
        "macros": [
            {"id": "finish_chapter", "label": "Finish chapter",
             "hint": "summarise into the bible, then open the next chapter",
             "steps": [
                 {"kind": "prompt", "to": "aux",
                  "text": "A chapter of a novel was just finished in the "
                          "excerpts below. Write 2-3 short sentences for the "
                          "story bible's STORY SO FAR: what HAPPENED (events "
                          "and changes only, no praise, no analysis). Reply "
                          "with the sentences only.\n\n{{transcript}}"},
                 {"kind": "note", "op": "append", "text": "- {{step:0}}"},
                 {"kind": "new_chat", "title": "{{title_next}}",
                  "carry": ["notes", "method", "workspace", "params",
                            "use_rag"]}]},
            {"id": "recap_so_far", "label": "Recap so far",
             "hint": "a private recap that does not enter the chat",
             "steps": [{"kind": "prompt", "to": "aux",
                        "text": "Recap what has happened in these excerpts "
                                "in 5 bullet points:\n\n{{transcript}}"}]},
            {"id": "continue_drafting", "label": "Continue drafting",
             "hint": "keep writing from where the prose stopped",
             "steps": [{"kind": "prompt",
                        "text": "Continue the chapter from where it stopped. "
                                "Append to the chapter file rather than "
                                "re-emitting what is already written."}]},
            {"id": "check_continuity", "label": "Check continuity",
             "hint": "read the bible and flag contradictions",
             "steps": [
                 {"kind": "tool", "name": "read_file",
                  "args": {"path": "{{bible}}"}},
                 {"kind": "prompt",
                  "text": "Check the recent prose against this bible and "
                          "list any contradictions in cast, world or "
                          "timeline. If there are none, say so.\n\n"
                          "{{step:0}}"}]}],
        "workflows": [
            {"id": "new_chapter", "label": "New chapter", "resumable": True,
             "steps": [
                 {"kind": "prompt",
                  "text": "Outline the next chapter in 5 beats, using the "
                          "bible in the notes."},
                 {"kind": "prompt",
                  "text": "Draft beat one as prose and append it to the "
                          "chapter file."},
                 {"kind": "prompt",
                  "text": "Draft the remaining beats, appending as you go."},
                 {"kind": "prompt", "to": "aux",
                  "text": "Summarise this chapter in 2-3 sentences for STORY "
                          "SO FAR:\n\n{{transcript}}"},
                 {"kind": "note", "op": "append", "text": "- {{step:3}}"}]}],
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
        "rules": [
            {"id": "never_puppet", "kind": "standing",
             "text": "Never speak or act for the user's character."}],
        "macros": [
            {"id": "advance_scene", "label": "Advance the scene",
             "hint": "move time forward",
             "steps": [{"kind": "prompt",
                        "text": "Advance the scene. Something changes; do "
                                "not summarise what already happened."}]},
            {"id": "describe_surroundings", "label": "Describe surroundings",
             "hint": "sensory detail, in character",
             "steps": [{"kind": "prompt",
                        "text": "Describe the surroundings in sensory "
                                "detail, staying in character."}]},
            {"id": "ooc_note", "label": "OOC note",
             "hint": "step out of character with thinking on",
             "steps": [
                 {"kind": "settings", "set": {"effort": "on"}},
                 {"kind": "prompt",
                  "text": "Out of character for one message: "
                          "{{ask:What do you want to say}}"}]},
            {"id": "save_to_notes", "label": "Save this to notes",
             "hint": "pin what just happened",
             "steps": [
                 {"kind": "prompt", "to": "aux",
                  "text": "In one line, what changed in the scenario in "
                          "these excerpts?\n\n{{transcript}}"},
                 {"kind": "note", "op": "append", "text": "- {{step:0}}"}]}],
        "workflows": [],
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
        "rules": [
            {"id": "say_vs_infer", "kind": "standing",
             "text": "Separate what the sources say from what you infer."}],
        "macros": [
            {"id": "find_sources", "label": "Find sources",
             "hint": "search the web for the question in notes",
             "steps": [{"kind": "prompt",
                        "text": "Search the web for the research question in "
                                "the notes and list the best sources with "
                                "one line each on why."}]},
            {"id": "summarise_cited", "label": "Summarise with citations",
             "hint": "answer so far, every claim sourced",
             "steps": [{"kind": "prompt",
                        "text": "Summarise what we know so far. Every claim "
                                "carries the source it came from."}]},
            {"id": "save_findings", "label": "Save findings",
             "hint": "write findings to a file",
             "steps": [{"kind": "prompt",
                        "text": "Write the findings so far to findings.md "
                                "with write_file, sources included."}]}],
        "workflows": [
            {"id": "research_question", "label": "Research a question",
             "resumable": True,
             "steps": [
                 {"kind": "prompt",
                  "text": "Search the web for: {{ask:What question}}"},
                 {"kind": "prompt",
                  "text": "Read the most promising sources with fetch_url "
                          "and note what each actually claims."},
                 {"kind": "prompt",
                  "text": "Write findings.md: the answer, the evidence, and "
                          "what is still uncertain."}]}],
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
        "rules": [
            {"id": "ask_first", "kind": "standing",
             "text": "Ask a checking question before revealing an answer."}],
        "macros": [
            {"id": "quiz_me", "label": "Quiz me",
             "hint": "five questions on the current topic",
             "steps": [{"kind": "prompt",
                        "text": "Quiz me with 5 questions on what we have "
                                "covered. Ask them all, then wait."}]},
            {"id": "explain_simpler", "label": "Explain simpler",
             "hint": "re-explain the last answer differently",
             "steps": [{"kind": "prompt",
                        "text": "Explain that again, differently and more "
                                "simply. Do not repeat the same words."}]},
            {"id": "worked_example", "label": "Worked example",
             "hint": "one concrete example, step by step",
             "steps": [{"kind": "prompt",
                        "text": "Give one concrete worked example, step by "
                                "step, with the reasoning shown."}]},
            {"id": "track_progress", "label": "Track progress",
             "hint": "append what was covered to the notes",
             "steps": [
                 {"kind": "prompt", "to": "aux",
                  "text": "In one line, what did the student cover in these "
                          "excerpts?\n\n{{transcript}}"},
                 {"kind": "note", "op": "append", "text": "- {{step:0}}"}]}],
        "workflows": [],
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
        "rules": [
            {"id": "propose_first", "kind": "standing",
             "text": "Propose the plan before moving anything, and never "
                     "delete."}],
        "macros": [
            {"id": "preview_plan", "label": "Preview plan",
             "hint": "explore and propose, changing nothing",
             "steps": [
                 {"kind": "tool", "name": "list_directory",
                  "args": {"path": "{{ask:Which folder}}"}},
                 {"kind": "prompt",
                  "text": "Propose how to organise this folder. Say what "
                          "moves where. Change nothing yet.\n\n"
                          "{{step:0}}"}]},
            {"id": "execute_plan", "label": "Execute plan",
             "hint": "carry out the proposed moves",
             "steps": [{"kind": "prompt",
                        "text": "Carry out the plan you just proposed using "
                                "move_files by reference. Never retype "
                                "paths."}]},
            {"id": "undo_last", "label": "Undo last",
             "hint": "reverse the most recent change",
             "steps": [{"kind": "prompt",
                        "text": "Undo the most recent change you made, using "
                                "undo_last_change."}]}],
        "workflows": [],
    },
]


def methods_dir() -> Path:
    d = rigma_home() / "methods"
    d.mkdir(parents=True, exist_ok=True)
    return d


class MethodIdError(ValueError):
    """The requested id can't name a method file."""


# CON, PRN, AUX, NUL, COM1-9, LPT1-9 are unopenable as files on Windows
_WIN_DEVICE = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.I)


def _method_file(method_id: str) -> Path:
    """The file for a method id, or raise. The id reaches this line straight
    from a URL path parameter, and Starlette's {param} converter is [^/]+ --
    it stops %2f but not %5c, which uvicorn has already decoded by the time
    routing happens, so on Windows "..%5C..%5Cfoo" was a real traversal out of
    ~/.rigma. Reject rather than sanitise, the way skills._path_for does: a
    silently rewritten name unlinks a file nobody asked about. The pattern is
    ms._ID_RE because that is what the save path already validates against.
    # AUDIT F40: docs/audit-2026-09-04-full.md
    """
    mid = str(method_id or "")
    if not ms._ID_RE.match(mid) or _WIN_DEVICE.match(mid):
        raise MethodIdError(
            "a method id must be 1-40 characters of a-z, 0-9 and underscore")
    p = (methods_dir() / f"{mid}.json").resolve()
    if p.parent != methods_dir().resolve():
        raise MethodIdError("that id would write outside the methods folder")
    return p


# Every key ms.normalize() reaches for with a bare int(), list(), dict() or
# .items(). Checking them here is what turns a type-confused document into the
# documented 400 with a self-correcting `errors` array instead of a traceback:
# validate() is itself unsafe on the same inputs, so ordering alone would not
# have been enough.  # AUDIT F43: docs/audit-2026-09-04-full.md
_SHAPES = {"apply": dict, "vars": dict, "guide": list, "rules": list,
           "macros": list, "workflows": list}


def _shape_errors(doc) -> list[str]:
    """Type problems, phrased like every other validator string so a model can
    fix them in one turn. Empty list means normalize() will survive `doc`."""
    if not isinstance(doc, dict):
        return ["a method must be a JSON object"]
    errs: list[str] = []
    for key, want in _SHAPES.items():
        v = doc.get(key)
        if v is not None and not isinstance(v, want):
            errs.append(f"{key} must be "
                        + ("an object" if want is dict else "a list"))
    if isinstance(doc.get("vars"), dict):
        for k, v in doc["vars"].items():
            if not isinstance(v, dict):
                errs.append(f"var '{k}': must be an object with a 'kind'")
    for key in ("rules", "macros", "workflows"):
        if isinstance(doc.get(key), list):
            for i, c in enumerate(doc[key]):
                if not isinstance(c, dict):
                    errs.append(f"{key}[{i}]: must be an object")
    try:
        int(doc.get("version", 1) or 1)      # mirrors normalize() exactly
    except (TypeError, ValueError):
        errs.append("version must be a whole number")
    return errs


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
        # ms.normalize() is INSIDE the try, and _shape_errors runs before it.
        # It used to sit one line below, outside the guard the docstring above
        # promises: normalize does bare int()/list()/dict()/.items() on
        # caller-supplied fields, so a file that was valid JSON with one
        # wrong-typed key ("version": "two") raised straight out of catalog()
        # and took down GET /api/methods -- including the import endpoint you
        # would have used to undo it.  # AUDIT F43: docs/audit-2026-09-04-full.md
        try:
            raw = json.loads(f.read_text(encoding="utf-8"))
            if not (isinstance(raw, dict) and raw.get("id")):
                continue
            if _shape_errors(raw):
                continue      # anything save_user would refuse is not a method
            out.append(ms.normalize({**raw, "builtin": False}))
        except Exception:
            continue
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
    errs = _shape_errors(doc)
    if errs:
        return None, errs           # before normalize(), which would raise
    full = ms.normalize({**doc, "builtin": False})
    errs = ms.validate(full, tool_names())
    if errs:
        return None, errs
    try:
        f = _method_file(full["id"])
    except MethodIdError as e:
        return None, [str(e)]       # an id error is a 400, never a traceback
    f.write_text(json.dumps(full, indent=2), encoding="utf-8")
    return full, []


def delete_user(method_id: str) -> bool:
    try:
        f = _method_file(method_id)
    except MethodIdError:
        return False                # same answer as "no such user method"
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
