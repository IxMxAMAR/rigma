"""Chat sessions: the store, the prompt builder and the sampler whitelist.

CONCURRENT WRITES — the API every other module must use
-------------------------------------------------------
Every session row carries a monotonic integer revision, bumped by each write.

  * `sessions.REV_KEY` ("_rev") is the key `load()` puts the row's current
    revision under on the session dict. It is bookkeeping, never part of the
    document: `save()` strips it before serialising, so it can never reach the
    stored body.
  * `sessions.StaleWriteError` is raised by `save()` when the guard loses.
  * `save(session, *, base_rev=None)` writes unconditionally when `base_rev`
    is None — exactly what every caller written before the guard does. Given
    a `base_rev`, it writes ONLY while the stored revision is still exactly
    that; otherwise it writes nothing and raises. The test and the write are
    one SQL statement, so nothing can land between them. It returns the new
    revision and also refreshes `session[REV_KEY]`, so a caller saving several
    times in one turn can keep passing `session[REV_KEY]` without reloading.

      s = sessions.load(sid)
      s["messages"].append(...)
      sessions.save(s, base_rev=s[sessions.REV_KEY])   # may raise

  * `reload_and_extend(sid, messages, since, *, amend_from=None)` is the merge
    for the reload-then-write pattern: it re-reads the session and grafts this
    turn's messages (`messages[since:]`) onto whatever is stored NOW, so a
    rename, a compaction, or another turn that landed during a long generation
    all survive alongside it. It returns the merged session carrying the
    CURRENT revision under REV_KEY — ready to hand straight to `save()` as
    `base_rev` — or None if the session was deleted meanwhile.

Why (audit 2026-09-04, F1/F2/F4/F15): `save()` writes the whole row, so any
writer holding a snapshot taken before someone else's write silently destroyed
everything that landed in between.
"""
from __future__ import annotations

import json
import logging
import os
import re
import secrets
import sqlite3
import time
from pathlib import Path

from .db import REV_KEY          # re-exported: half the load()/save() contract
from .runtime import rigma_home

_log = logging.getLogger(__name__)


class StaleWriteError(RuntimeError):
    """The stored session moved past the revision this write was based on (or
    was deleted). Reload — see reload_and_extend — and write again; never
    overwrite work this caller has not seen."""


class SessionIdError(ValueError):
    """The requested id can't name a session file."""


MUTABLE_FIELDS = ("title", "system_prompt", "use_rag", "messages",
                  "preset_id", "params", "notes", "digest", "effort",
                  "authors_note", "authors_note_depth", "prefill",
                  "use_tools", "allow_code", "workspace", "auto_compact",
                  "max_tool_rounds", "one_action", "method", "carry_reasoning",
                  # which agent backend owns this session's turns. It belongs
                  # here because `update_session` offers exactly this tuple —
                  # and without it the harness seam was UNREACHABLE from the
                  # product: a session could only be pointed at DSH by editing
                  # its JSON by hand, so the whole integration was library-only.
                  # `serve.update_session` validates the name through
                  # `harness.resolve`, so an unknown or unusable backend is
                  # refused at the write instead of 400-ing every turn after it.
                  # Deriving `_MERGE_FROM_STORE` from this tuple also means a
                  # harness switched mid-turn is not undone by the turn's own
                  # end-of-turn write.
                  "harness",
                  # How much the agent backend may do without being asked.
                  # A SESSION field rather than a constant because the honest
                  # default is a trade, not a fact: headless has nobody to ask,
                  # so a mode that wants to ask FAILS THE RUN. Rigma therefore
                  # defaults to `full` (do not ask) and lets the owner choose
                  # otherwise with the cost stated — rather than hardcoding the
                  # choice and calling it the only option.
                  "permission",
                  # AUDIT 13-3: spawning a process is its own grant. The
                  # destructive-command regex is a literal-text advisory and
                  # cannot be the boundary, so run_shell/start_job/run_python
                  # need this explicit confirmation as well as allow_code.
                  "confirm_exec",
                  # AUDIT 13-2: reads default to the workspace and outbound
                  # POSTs carrying a body are refused. These two grants restore
                  # the pre-fix behaviour explicitly rather than by default.
                  "allow_absolute_reads", "allow_outbound_post",
                  # R3-TOOL-4: writing OUTSIDE the workspace is its own grant.
                  # `write_file`/`edit_file` have always been confined to the
                  # workspace, but `move_files`/`copy_files` reached any absolute
                  # destination on disk with only the default-on `allow_code` —
                  # so the WEAKER tool (write_file) was strictly more confined
                  # than its sibling, and a planted file in the Startup folder or
                  # an extension directory was one move away. Deliberately NOT
                  # reusing `allow_absolute_reads`: reading a file and replacing
                  # one are different risks, and a user who granted one did not
                  # grant the other.
                  "allow_absolute_writes",
                  # OD-2 (R3-10): the ABSOLUTE WRITE ROOTS a `copy_files`/
                  # `move_files` destination may land in without the blanket
                  # grant above. The owner accepted option 1 of OD-2 — restrict
                  # the destination to the workspace plus an allowlist seeded
                  # from the folders they already work in — because a single
                  # blanket switch cannot express "these folders are fine, the
                  # Startup folder is not". A LIST, so it is typed `list` below
                  # and never read through the boolean coercion table.
                  "write_allowlist",
                  # R5-PERSIST: the agent's own durable state — its goal, its todo
                  # list, whether it is in plan mode. The backend reports these as
                  # SSE events, and the UI drew them from the LIVE TURN only, so
                  # the whole panel vanished on reload even though the agent's
                  # state had not. A SESSION field because that is what it is: it
                  # belongs to the conversation, not to the turn that happened to
                  # report it.
                  #
                  # Deliberately does NOT include subagents. A subagent row names a
                  # child process that belongs to the turn that spawned it; one
                  # restored after a reload would claim a child that is long gone,
                  # and a panel that lies is worse than a panel that is empty.
                  "agent_state",
                  # R6-ACP-TURN: WHICH of mcode's two wires drives this chat.
                  #
                  # mcode has two transports and they are not equivalent. `exec` is a
                  # projection of one turn; ACP is a session, so it can report a queue,
                  # a delegation tree, a plan review and the live model/permission
                  # selects, and it can ASK a question and wait for the answer. The
                  # permission prompt is the one that matters most: on `exec` nobody
                  # can answer, so `smart` deciding to ask blocks the chat PERMANENTLY.
                  #
                  # A SESSION field because it is a property of the conversation, not
                  # of a turn — and because switching mid-conversation would strand the
                  # session id, which belongs to one wire.
                  "mcode_transport")
# Which wire mcode is driven over. `exec` is the DEFAULT and stays the default:
# it is the transport that has been exercised against a real engine, and ACP's
# `drive_turn_acp` has not. `acp` is offered because it is the only one of the two
# that can answer a question at all.
MCODE_TRANSPORTS = ("exec", "acp")
# What each agent backend understands. `smart` classifies and asks when it
# judges risk high; `full` does not ask; `off` disarms the agent's tools. `ask`
# is refused headlessly by mcode itself ("requires an interactive host"), so it
# is not offered. Measured, with the caveat that `full` is NOT "no safety": a
# hard, workspace-scoped policy bounds recursive writes and deletes under every
# one of these modes — see docs/mcode-permission-modes.md.
PERMISSION_MODES = ("full", "smart", "off")
# "" / off / auto / on are Rigma's own binary thinking switch. The four
# named levels are Qwen3.8's published reasoning efforts, which its chat
# template reads from `reasoning_effort` — verified against a live engine
# 2026-08-19 (a plain turn renders 79 prompt chars, xhigh renders 316).
# A template that has never heard of the kwarg simply ignores it.
QWEN_EFFORTS = ("low", "medium", "high", "xhigh")
EFFORT_LEVELS = ("", "off", "auto", "on") + QWEN_EFFORTS

# AUDIT 01-3: the PATCH body is JSON of any shape and `update_session` copied
# every MUTABLE_FIELD straight through. A wrong type did not fail the write —
# it failed LATER, in a reader: `export_markdown` joins message content into a
# str-only list, `build_messages` joins system_prompt into the system block,
# and `validate_params` calls `.items()` on `params`. Refusing at the write is
# the only place the client can still be told which field is wrong. Fields the
# PATCH body does not offer today (`archive`, `pending_nudges`,
# `trigger_state`) are listed too, so adding one to MUTABLE_FIELDS cannot
# inherit the hole.
_FIELD_TYPES: dict[str, type] = {
    "title": str, "system_prompt": str, "preset_id": str, "notes": str,
    "digest": str, "effort": str, "authors_note": str, "prefill": str,
    "workspace": str, "method": str, "harness": str, "permission": str,
    # R6-ACP-TURN: a str, and validated against MCODE_TRANSPORTS at the write like
    # `permission` is. Typing it here is not enough on its own — the type only stops
    # a list reaching a reader; the VALUE check is what stops an unknown transport
    # name reaching the adapter and being treated as "not acp".
    "mcode_transport": str,
    "messages": list, "archive": list, "pending_nudges": list,
    "params": dict, "trigger_state": dict,
    # R5-PERSIST: a dict like `params`/`trigger_state`, so a client cannot store a
    # string where the reader subscripts. Nothing on the PATCH surface sets it
    # today — the SERVER writes it mid-turn — but it is typed here because
    # `test_r3_session_field_types.py` asserts every MUTABLE_FIELD has an entry,
    # and that guard is what stops a future field inheriting the
    # `bool("false") is True` hole.
    "agent_state": dict,
    # AUDIT 13-3 regression: this is a GRANT, so it is type-checked where the
    # other booleans are not. `bool("false")` is True, and a client that sent
    # the string form of "no" would otherwise have been read as "yes" — the one
    # mistake here that hands a chat process spawning.
    "confirm_exec": bool,
    # R3-1: the hole the comment above admitted. `bool("false")` is True for
    # EVERY switch below too, and two of them are grants, so a client that sent
    # the quoted form of "no" turned the capability ON — the exact mistake
    # 09-6 fixed for methods/macros (method_schema.SETTINGS_FIELDS) and 13-3
    # fixed for `confirm_exec` alone. Measured before this: PATCH
    # {"allow_absolute_reads": "false"} answered 200, stored the string, and
    # `tools._absolute_reads_allowed` read it as a GRANT; {"use_tools": "false"}
    # left tools on. The UI's `grants.readGrants` only honours a literal `true`,
    # so it also DISPLAYED the grant as off while the server honoured it.
    "use_tools": bool, "use_rag": bool, "allow_code": bool,
    "auto_compact": bool, "one_action": bool, "carry_reasoning": bool,
    "allow_absolute_reads": bool, "allow_outbound_post": bool,
    # R3-TOOL-4: bool like its siblings, so the quoted "false" cannot grant it.
    "allow_absolute_writes": bool,
    # OD-2: a LIST of path strings, like `messages`/`archive`/`pending_nudges`
    # — NOT a bool, so it must not join the coercion table above. A string
    # where a list belongs is refused by `validate_field_types`, and
    # `_write_path` ignores any entry that is not an absolute path.
    "write_allowlist": list,
    # int, not bool: `build_messages` does int(...) on the depth and
    # `_round_cap` does int(...) on the cap, so a string that reached either
    # raised. True is deliberately NOT accepted (isinstance(True, int) is True,
    # so it is excluded explicitly) — a boolean where a count belongs is a
    # client bug worth naming.
    "authors_note_depth": int, "max_tool_rounds": int,
}


def validate_field_types(body: dict) -> None:
    """Raise ValueError('<field>: must be a <type>') for a mistyped write.

    `null` is COERCED to the field's empty value rather than refused: every
    reader already treats a missing value as empty, so a client that sends
    null for an unset box gets the same result as omitting the key. Any other
    value of the wrong type is refused by name.

    Every name in MUTABLE_FIELDS must appear in _FIELD_TYPES (test_r3_http.py
    asserts it), so a field added to the PATCH surface cannot inherit the
    `bool("false") is True` hole.
    """
    for key, want in _FIELD_TYPES.items():
        if key not in body:
            continue
        value = body[key]
        if value is None:
            body[key] = want()      # str() / list() / dict() = the empty value
        elif want is bool and not isinstance(value, bool):
            # bool first: isinstance(1, int) is True, so the int branch below
            # would have accepted 1 for a switch — and `_exec_confirmed` would
            # then read it as a grant. Only a real boolean is a boolean (09-6).
            raise ValueError(f"{key}: must be true or false")
        elif want is int and (isinstance(value, bool)
                              or not isinstance(value, int)):
            raise ValueError(f"{key}: must be a whole number")
        elif want is not int and not isinstance(value, want):
            raise ValueError(f"{key}: must be a {want.__name__}")

PARAM_RANGES = {"temperature": (0.0, 4.0), "top_p": (0.0, 1.0),
                "min_p": (0.0, 1.0), "repeat_penalty": (0.5, 2.0),
                "max_tokens": (1, 262144), "top_k": (0, 200),
                "seed": (-1, 2**31 - 1),
                "frequency_penalty": (-2.0, 2.0),
                "presence_penalty": (-2.0, 2.0),
                # modern anti-repetition samplers (llama-server per-request)
                "dry_multiplier": (0.0, 2.0), "dry_base": (1.0, 4.0),
                # AUDIT F3: docs/audit-2026-09-04-full.md — the cap was 10,
                # and hangar._DRY_BASELINE deliberately ships 16 (a filename
                # is 8-12 tokens and DRY must not punish re-typing one).
                # _safe_params dropped it key by key, so chat turns ran on
                # llama.cpp's allowance of 2 — documented above as harmful.
                "dry_allowed_length": (1, 64),
                # was MISSING here, so validation silently stripped it and
                # DRY scanned the whole context — RUN_PARAMS' cap (serve.py)
                # never reached the engine on chat turns. Found by the
                # 2026-07-21 Gemini code audit.
                "dry_penalty_last_n": (-1, 262144),
                "xtc_probability": (0.0, 1.0), "xtc_threshold": (0.0, 0.5),
                "top_n_sigma": (-1.0, 5.0)}
_INT_PARAMS = ("max_tokens", "dry_allowed_length", "seed", "top_k",
               "dry_penalty_last_n")
_MAX_STOPS = 4
# control bytes minus \t \n \r — a run of these in model-visible text is file
# corruption (NUL floods from crash-interrupted writes) and reads to the model
# as end-of-document: it answers with instant EOS. See tools._defuse_control_bytes.
_CTRL_RUN = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]+")

# AUDIT 10-9: the stored document's schema version, stamped by save(). It exists
# so the 50->1000 lift below runs on a body written by an old build and NOT on
# every load, which is what made a deliberate 50 revert forever.
_SESSION_SCHEMA = 1

# every field a session is guaranteed to carry — load() backfills these so
# v0.5.x session files survive an upgrade instead of KeyError-ing the app
_SESSION_DEFAULTS = {"title": "New chat", "system_prompt": "",
                     "use_rag": False, "preset_id": "", "params": {},
                     "notes": "", "digest": "", "effort": "", "archive": [],
                     "authors_note": "", "authors_note_depth": 3,
                     # tools on by default, full power (owner's call, their
                     # own local machine) — empty workspace resolves to home
                     "prefill": "", "use_tools": True, "allow_code": True,
                     "workspace": "", "auto_compact": True,
                     # per-turn tool-round backstop. 1000 = effectively
                     # unlimited (owner call 2026-07-21: a whole-story verify
                     # legitimately needs more than 50, and in chat the user
                     # is present with a stop button); it exists only to stop
                     # a true runaway loop.
                     "max_tool_rounds": 1000, "one_action": False,
                     "method": "",       # applied workflow method (methods.py)
                     # set only in a method-creation chat: binds the session
                     # to a draft AND withholds every non-builder tool
                     "method_draft_id": "",
                     # trigger rules: one-line reminders queued for the next
                     # turn, and the loop-guard bookkeeping (see triggers.py)
                     "pending_nudges": [], "trigger_state": {},
                     # R5-PERSIST: the agent's durable state, empty until a backend
                     # reports one. Named here rather than left absent for the same
                     # reason the grants above are: a reader that subscripts a chat
                     # that predates the field would otherwise KeyError, and the UI
                     # cannot offer to show what it cannot see.
                     "agent_state": {},
                     # which agent backend runs this session's turns: the
                     # built-in loop, or an external one wired behind the seam.
                     # `harness.resolve` refuses an unknown or unusable backend
                     # rather than quietly substituting the built-in.
                     "harness": "native",
                     # AUDIT 13-3 regression: execution is its own grant and
                     # defaults OFF. Naming it here (rather than leaving it
                     # absent) is what lets a reader — the UI toggle, the CLI,
                     # an API client — SEE the current value and offer to
                     # change it. `serve.py` and `mcp_server.py` read it.
                     "confirm_exec": False,
                     # R3-1: the other two grants 13-2 added were left OUT of
                     # the defaults while `confirm_exec` was named, so a chat
                     # that predates them loads without the keys at all and any
                     # reader that subscripts them gets a KeyError. The product
                     # only ever used .get(), so this was latent — but the whole
                     # point of naming `confirm_exec` above is that a grant a
                     # surface cannot SEE is one it cannot offer to change.
                     "allow_absolute_reads": False,
                     "allow_outbound_post": False,
                     # R3-TOOL-4: named here for the same reason as the two
                     # above — a grant a surface cannot see is one it cannot
                     # offer to change.
                     "allow_absolute_writes": False,
                     # OD-2: the configured absolute write roots for
                     # copy_files/move_files. `[]` is today's behaviour — an
                     # absolute destination then needs the blanket grant.
                     # `create()` seeds it from the folders the owner already
                     # works in; a chat that predates the field loads with `[]`
                     # rather than KeyError-ing a subscripting reader.
                     "write_allowlist": [],
                     # AUDIT 10-9: the schema version this document was written
                     # at. Stamped by save(); a body that predates the key is
                     # what the max_tool_rounds migration below keys on.
                     "schema": _SESSION_SCHEMA,
                     "messages": []}


def chats_dir() -> Path:
    d = rigma_home() / "sessions" / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


# AUDIT F40: docs/audit-2026-09-04-full.md
# A session id becomes a FILENAME here, and the id arrives over HTTP:
# Starlette's {param} converter is [^/]+, which blocks %2f but not %5c, and
# uvicorn percent-decodes before routing — so on Windows a DELETE of
# "..%5C..%5CDocuments%5Cbudget" reached this function as a traversal and
# unlinked a file outside ~/.rigma. Same shape as skills._path_for: reject
# rather than sanitise (quietly rewriting an id writes a file the user can
# never find again), then assert the resolved parent anyway.
_SAFE_SID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")


def _path(session_id: str) -> Path:
    sid = str(session_id or "")
    if not _SAFE_SID.match(sid):
        raise SessionIdError(
            "a session id may only contain letters, numbers, hyphens and "
            "underscores (max 64)")
    # CON, PRN, AUX, NUL, COM1-9, LPT1-9 are devices, not files, on Windows
    if re.match(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", sid, re.I):
        raise SessionIdError(f"'{sid}' is a reserved Windows device name")
    p = (chats_dir() / f"{sid}.json").resolve()
    if p.parent != chats_dir().resolve():
        raise SessionIdError("that id would resolve outside the chats folder")
    return p


# one-time legacy import, tracked per db path (tests re-home per test)
_imported: set[str] = set()


def _import_legacy() -> None:
    """Pull pre-SQLite chat files into the database ONCE, leaving the files
    in place as a backup (never written again). Idempotent and cheap: one
    id-set query, then only unknown files are parsed."""
    from . import db
    key = str(db.db_path())
    if key in _imported:
        return
    _imported.add(key)
    try:
        d = rigma_home() / "sessions" / "chats"
        if not d.is_dir():
            return
        known = db.known_ids()
        for f in d.glob("*.json"):
            if f.stem in known:
                continue
            try:
                raw = json.loads(f.read_text(encoding="utf-8"))
            except Exception:
                continue          # corrupt file: skip, never fatal
            if raw.get("id"):
                db.upsert_session(raw)
    except Exception:
        pass                      # import is a nicety; the db still works


def _store_workspaces(base: Path) -> list[str]:
    """The `workspace` of every session in the store under `base`.

    OD-2: read through SQLite's JSON1 extractor so the message BODIES never
    leave the database — the seed is PATHS, never prose (standing rule 3). A
    store that does not exist, has no JSON1, or holds a body that is not JSON
    yields nothing: the seed is an optimisation, and failing to build it must
    never fail a create.
    """
    db_file = base / "rigma.db"
    if not db_file.is_file():
        return []
    try:
        con = sqlite3.connect(f"{db_file.resolve().as_uri()}?mode=ro", uri=True)
    except (sqlite3.Error, ValueError):
        return []
    try:
        rows = con.execute(
            "SELECT json_extract(body, '$.workspace') FROM sessions "
            "ORDER BY rowid").fetchall()
    except sqlite3.Error:
        return []          # no JSON1 / no table: no extra roots (fail closed)
    finally:
        con.close()
    return [str(ws).strip() for (ws,) in rows if str(ws or "").strip()]


def _rag_source_folders(base: Path) -> list[str]:
    """The configured RAG source FOLDERS under `base`.

    The same `sources.json` `rag.load_sources()` reads, with the same "not a
    list of strings = no sources" guard. Only an entry that resolves to a
    DIRECTORY can be a write root, so a file or glob source is skipped — the
    allowlist is a set of folders, not a set of patterns.
    """
    try:
        data = json.loads((base / "rag" / "sources.json").read_text("utf-8"))
    except Exception:
        return []
    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        return []
    out = []
    for s in data:
        try:
            if Path(s).is_dir():
                out.append(s)
        except (OSError, ValueError):
            continue
    return out


def default_write_allowlist(home=None) -> list[str]:
    """OD-2: the absolute write roots a NEW session starts with.

    The owner's EXISTING working folders, used as metadata only: the
    `workspace` of every session already in the store, plus the folders
    registered as RAG sources (what `rag.load_sources()` returns for the same
    home). Nothing is opened or read — only path strings.

    Seeding from what the owner already uses is the point of the accepted
    option: an allowlist that defaulted to `[]` would refuse every absolute
    destination the moment `allow_absolute_writes` is off, which is a
    regression rather than a confinement. An EMPTY store and no RAG sources
    therefore returns `[]` — exactly today's behaviour.

    Entries are resolved and deduped (case-insensitively on Windows); a
    non-absolute or unparseable entry is ignored, never raised.
    """
    base = Path(home).expanduser() if home is not None else rigma_home()
    roots: list[str] = []
    seen: set[str] = set()
    for raw in _store_workspaces(base) + _rag_source_folders(base):
        try:
            p = Path(str(raw)).resolve()
        except (OSError, ValueError):
            continue
        if not p.is_absolute():
            continue
        key = os.path.normcase(str(p))
        if key in seen:
            continue
        seen.add(key)
        roots.append(str(p))
    return roots


def create(title: str = "New chat", system_prompt: str = "",
           *, write_allowlist: list[str] | None = None) -> dict:
    now = time.time()
    session = {**json.loads(json.dumps(_SESSION_DEFAULTS)),
               "id": secrets.token_hex(6), "title": title,
               "system_prompt": system_prompt,
               "created_at": now, "updated_at": now}
    # OD-2: a session created WITHOUT an explicit allowlist gets the seed —
    # the folders the owner already works in. An explicit list, including an
    # empty one, is honoured as given; an empty store seeds `[]`.
    session["write_allowlist"] = (default_write_allowlist()
                                  if write_allowlist is None
                                  else [str(x) for x in write_allowlist])
    save(session)
    return session


def save(session: dict, *, base_rev: int | None = None) -> int:
    """Persist the session; return the revision it now sits at.

    AUDIT F1: docs/audit-2026-09-04-full.md — pass `base_rev` (from
    `session[REV_KEY]`, which `load()` set) and the write happens only while
    the stored row is still at that revision; anything else raises
    StaleWriteError and writes NOTHING. Omit it and this is the unconditional
    whole-row write it has always been, unchanged for existing callers.
    """
    from . import db
    _import_legacy()
    session["updated_at"] = time.time()
    # AUDIT 10-9: stamp the schema version so the one-time migrations in load()
    # never run against a document this build wrote.
    session["schema"] = _SESSION_SCHEMA
    rev = db.upsert_session(session, base_rev=base_rev)
    if rev is None:
        raise StaleWriteError(
            f"session {session.get('id', '?')} has moved past revision "
            f"{base_rev} (or was deleted) — reload before writing")
    # keep the snapshot's revision honest so a second guarded save in the same
    # turn can pass session[REV_KEY] again without a reload
    session[REV_KEY] = rev
    return rev


def load(session_id: str) -> dict | None:
    from . import db
    _import_legacy()
    row = db.get_session_row(session_id)
    if row is None:
        return None
    body, rev = row
    try:
        raw = json.loads(body)
    except Exception:
        return None
    # AUDIT 10-9: this heuristic used to run on EVERY load, so a user who
    # deliberately lowered max_tool_rounds to 50 had it rewritten to 1000 the
    # next time the chat was opened. Gate it on the stored schema version,
    # checked on the RAW body before the defaults backfill below: only a
    # document written before the field existed is migrated. The old default
    # WAS 50, so a legacy body at 50 is lifted once; save() then stamps the
    # current version and a later deliberate 50 survives.
    if not raw.get("schema") and raw.get("max_tool_rounds") == 50:
        raw["max_tool_rounds"] = 1000
    # migration: sessions written by older Rigma versions lack newer fields
    for k, v in _SESSION_DEFAULTS.items():
        raw.setdefault(k, json.loads(json.dumps(v)))
    # the revision THIS snapshot was taken at, set last so a body that somehow
    # carries the key cannot lie about it. Never persisted: save() strips it.
    raw[REV_KEY] = rev
    return raw


def reload_and_extend(sid: str, messages: list, since: int,
                      *, amend_from: int | None = None) -> dict | None:
    """Re-read `sid` and graft this turn's messages onto what is stored NOW.

    AUDIT F1: docs/audit-2026-09-04-full.md — the merge half of the
    reload-then-write pattern. `since` is how many messages the caller's list
    held when it took its snapshot: everything from that index on is this
    turn's own work and goes last, after whatever the store has gained
    meanwhile (another turn's messages, a compaction that moved older ones
    into `archive`). Fields outside `messages` come from the reloaded session,
    so a rename or a params edit made during a long generation survives.

    `amend_from` covers the one case that is not an append: a `continue` that
    extends an assistant message ALREADY stored. Pass the index of the first
    such message and `messages[amend_from:since]` replaces the stored copies
    in place, keeping anything stored after them. If a compaction has already
    moved those messages into `archive` there is nothing left to replace, so
    they are appended instead — a duplicate in the archive tail is cheap next
    to losing text the model just generated.

    Returns the merged session — carrying the CURRENT revision under REV_KEY,
    ready to pass to `save(..., base_rev=...)` — or None if the session was
    deleted while this turn ran (never resurrect a chat the user removed).
    """
    fresh = load(sid)
    if fresh is None:
        return None
    stored = list(fresh.get("messages", []))
    since = max(0, min(int(since), len(messages)))
    head, tail_from = stored, since
    if amend_from is not None and 0 <= amend_from < since:
        if since <= len(stored):
            head = (stored[:amend_from] + list(messages[amend_from:since])
                    + stored[since:])
        else:
            tail_from = amend_from      # compacted away: append, never drop
    fresh["messages"] = head + list(messages[tail_from:])
    return fresh


def delete(session_id: str) -> bool:
    from . import db
    _import_legacy()
    hit = db.delete_session(session_id)
    # a legacy file left in place would resurrect the chat at next import
    try:
        _path(session_id).unlink(missing_ok=True)
    except (OSError, SessionIdError):
        # an id that cannot name a file cannot have left one behind either —
        # the row delete above is parameterised SQL and already ran (F40)
        pass
    return hit


def list_sessions() -> list[dict]:
    from . import db
    _import_legacy()
    return db.list_summaries()


# The tool doctrine: rules of engagement appended to every tool-enabled
# session's system block. Written from observed failures of the models this
# harness runs (2026-07-19..21), not generic advice. ~200 tokens, cached
# with the prefix, gated on use_tools so toolless chats never pay for it.
# Kept SHORT on purpose. The first doctrine was nine imperative rules, and a
# live A/B on the real 35B (2026-07-21) showed what that does to a small
# thinking model: same task, same tools — without the doctrine it made a clean
# edit_file call after ~3K chars of thinking; with it, 15K+ chars of spiraling
# deliberation (ending in it literally enumerating the file's words) and NO
# answer at all. Every meta-rule is something to reason ABOUT before acting;
# either/or framings ("edit_file vs write_file") become deliberation traps.
# So: few rules, each decisive, and an explicit order to end with a reply.
TOOL_DOCTRINE = """TOOL RULES — keep them light: act, don't deliberate.
1. Use filenames and paths exactly as tool results list them — never retype
   one from memory.
2. Read a file before changing it. Adding new text? write_file with
   append=true. Rewording existing text? edit_file, smallest possible edit.
   Decide in one breath and act — every change is undoable (undo_last_change).
3. When the tools have answered, ANSWER THE USER: every turn ends with a
   short reply saying what you found or did. Never end in silence.
4. An error result means change your approach — never repeat the identical
   call. What tools returned is the truth about the user's files; your
   memory of them is not.
5. STRICT TOOL CALLING: You must use the exact tool names provided (e.g. 'write_file', not 'WriteFile').
   Output the tool call strictly following the required format. Do NOT invent tools.
6. A real request mid-roleplay ("save this", "remember that") still gets its
   tool call — then continue in character."""


def build_messages(session: dict, default_prompt: str = "",
                   preset: dict | None = None) -> list[dict]:
    prompt = (session.get("system_prompt")
              or (preset or {}).get("system_prompt", "")
              or default_prompt)
    # Coalesce ALL system-role context into ONE leading system message. Many
    # chat templates (Qwen3 among them) `raise_exception` on any second system
    # message anywhere in the conversation — emitting separate system blocks
    # 400s those models ("Unable to generate parser for this template"). This
    # bit autonomous runs (mission + agent prompt = two blocks) and would bite
    # any chat post-compaction (a `digest` block would be the second system).
    sections = []
    # Autonomous Mode: pin the mission FIRST so compaction (which only
    # summarizes the message history) can never summarize away the objective.
    mission = session.get("mission", "")
    if mission:
        sections.append(
            "CORE DIRECTIVE — never lose sight of this:\n" + mission +
            "\n\nYour context is periodically compacted; your plan and progress "
            "live on disk and are shown to you each step. Do NOT read "
            "progress.md yourself — the latest steps are provided for you.")
    if prompt:
        sections.append(prompt)
    # Standing rules from the applied method. They join the ONE leading
    # system block -- a second system message anywhere makes strict Qwen3
    # templates raise, which llama-server returns as HTTP 400. Kept SHORT
    # and imperative on purpose: a 9-rule doctrine measurably sent this
    # model into 15.7K-char deliberation spirals with no reply at all
    # (2026-07-21 live A/B).
    if session.get("method"):
        try:
            from . import methods as _methods
            _m = _methods.get(str(session["method"])) or {}
            standing = [str(r.get("text") or "").strip()
                        for r in _m.get("rules") or []
                        if r.get("kind") == "standing"
                        and str(r.get("text") or "").strip()]
        except Exception:
            standing = []      # a deleted or corrupt method never breaks a chat
        if standing:
            sections.append("METHOD RULES:\n"
                            + "\n".join(f"- {t}" for t in standing))
    notes = session.get("notes", "")
    if notes:
        sections.append("Story notes (authoritative):\n" + notes)
    if session.get("use_tools"):
        sections.append(TOOL_DOCTRINE)
    if session.get("use_rag"):
        # grounded search: the tool does the grounding, but a weak model needs
        # TELLING that the tool is the point of this conversation — without
        # the nudge it answers from its own weights and never reaches for it
        sections.append(
            "GROUNDED CHAT is ON. The user has indexed their own documents. "
            "For any question their documents could answer, call "
            "search_my_documents FIRST and base your answer on what it "
            "returns, citing the source files. Only answer purely from your "
            "own knowledge when the documents have nothing relevant.")
    digest = session.get("digest", "")
    if digest:
        # Frame the summary as REFERENCE, not as instructions. Without this a
        # model treats a compacted summary as a fresh task and restarts work it
        # already finished — and the "reference only" framing alone makes some
        # models stop calling tools and just narrate, so say that too.
        sections.append(
            "EARLIER CONVERSATION (compacted) — REFERENCE ONLY.\n"
            "This is history, not a new instruction. Topic overlap does NOT mean "
            "resume it; do not redo anything recorded here as finished. Your "
            "tools remain fully active — keep calling them for the CURRENT step "
            "rather than describing what you would do.\n"
            + digest
            + "\n--- END OF SUMMARY — act on the messages BELOW, not on this ---")
    head = ([{"role": "system", "content": "\n\n".join(sections)}]
            if sections else [])
    # sanitize: variants/metadata must never reach the model.
    # Also DROP empty assistant turns: a tool-only action persists an assistant
    # message with no text, and some chat templates (this Qwen3 build) 400 on
    # them — "Unable to generate parser for this template". They carry no
    # information for the model either way.
    # Autonomous runs CARRY recent reasoning back in (reasoning_content):
    # Qwen3.6's agent guidance — with preserve_thinking the model stops
    # re-deriving its plan every turn. Only the last few turns' worth, capped,
    # so carried thinking never bloats the window.
    # Back to runs only, after shipping it for chats cost the owner most of
    # their speed (2026-08-28).
    #
    # The four-turn window SLIDES. Every turn drops the oldest carried block and
    # adds a new one, which rewrites the prompt at that position — and on a
    # DeltaNet hybrid there is no KV shifting to recover from a mid-prompt edit
    # (the engine disables --cache-reuse outright). So the cache is invalidated
    # from four turns back on EVERY turn, and the reprefill that follows costs
    # far more than the re-derivation it was meant to save.
    #
    # It stays on for runs because a run's trajectory is append-only in the way
    # that matters, and Qwen3.6's agent guidance asks for it. `carry_reasoning`
    # opts a chat in for anyone who wants it with eyes open.
    carry_think = bool(session.get("one_action")
                       or session.get("carry_reasoning")) \
        and session.get("effort", "") != "off"
    msgs = []
    for m in session.get("messages", []):
        role = m.get("role", "user")
        content = m.get("content", "")
        if role == "assistant" and not str(content or "").strip():
            continue
        if isinstance(content, str) and _CTRL_RUN.search(content):
            # heal sessions that persisted control bytes BEFORE the tool-side
            # defusal existed: a stored NUL run re-poisons every later turn
            # of that chat otherwise (the 2026-07-21 instant-EOS, which came
            # from crash-corrupted files read into a tool-result carrier)
            content = _CTRL_RUN.sub(
                lambda mo: f"[{len(mo.group())} unreadable control byte(s)]",
                content)
        if (msgs and role == "user" and msgs[-1]["role"] == "user"
                and isinstance(content, str)
                and isinstance(msgs[-1]["content"], str)):
            # merge consecutive user messages: strict templates concatenate
            # them into ONE block anyway, so each extra "continue" rewrote
            # the block and invalidated the prompt cache from the carrier
            # onward — measured live: 3816 tokens re-prefilled (~9 s) per
            # turn. One block, stable prefix, cache survives.
            msgs[-1]["content"] = (str(msgs[-1]["content"]) + "\n\n"
                                   + content)
            continue
        entry = {"role": role, "content": content}
        if carry_think and role == "assistant" and m.get("thinking"):
            entry["reasoning_content"] = str(m["thinking"])[:4000]
        msgs.append(entry)
    if carry_think:
        # strip reasoning from all but the last 4 assistant turns
        kept = 0
        for m in reversed(msgs):
            if m["role"] != "assistant" or "reasoning_content" not in m:
                continue
            kept += 1
            if kept > 4:
                del m["reasoning_content"]
    an = session.get("authors_note", "")
    if an:
        # depth-targeted injection: N messages from the end beats the system
        # prompt for steering prose (recency bias is real)
        try:
            depth = max(0, int(session.get("authors_note_depth", 3)))
        except (TypeError, ValueError):
            depth = 3
        # role MUST be user: a system message anywhere but position 0 makes
        # strict templates raise, which llama-server reports as HTTP 400
        # "Unable to generate parser for this template"
        msgs.insert(max(0, len(msgs) - depth),
                    {"role": "user", "content": f"[Author's note: {an}]"})
    # Trigger nudges: one short line each, appended as ONE user message at
    # the end where recency makes them actually land. role MUST be user for
    # the same reason as the author's note above. The caller clears them
    # after the turn — build_messages is read-only on the session.
    nudges = [str(n).strip() for n in session.get("pending_nudges") or []
              if str(n).strip()]
    if nudges:
        msgs.append({"role": "user",
                     "content": "\n".join(f"[{n}]" for n in nudges)})
    return head + msgs


def default_prompt(registry=None) -> str:
    """Registry default system prompt for the running use-case ('' if none)."""
    from . import state as st
    from .registry import Registry
    s = st.read_state() or {}
    reg = registry if registry is not None else Registry.load()
    uc = reg.use_cases.get(s.get("use_case", "general"))
    return uc.system_prompt if uc else ""


def validate_params(raw: dict) -> dict:
    """Whitelisted, range-checked sampler params. Raises ValueError('<field>: ...')."""
    out = {}
    for key, value in (raw or {}).items():
        if key == "stop":
            if not isinstance(value, list) or len(value) > _MAX_STOPS or \
                    not all(isinstance(x, str) and 0 < len(x) <= 64
                            for x in value):
                raise ValueError(
                    f"stop: up to {_MAX_STOPS} non-empty strings, 64 chars max")
            if value:
                out["stop"] = value
            continue
        if key not in PARAM_RANGES:
            continue
        lo, hi = PARAM_RANGES[key]
        if isinstance(value, bool):
            raise ValueError(f"{key}: not a number")
        try:
            value = int(value) if key in _INT_PARAMS else float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: not a number") from None
        if not lo <= value <= hi:
            raise ValueError(f"{key}: must be between {lo} and {hi}")
        out[key] = value
    return out


def _safe_params(raw: dict, source: str = "") -> dict:
    out = {}
    for key, value in (raw or {}).items():
        try:
            out.update(validate_params({key: value}))
        except ValueError as e:
            # AUDIT F3: docs/audit-2026-09-04-full.md
            # Stored junk still never blocks a chat turn — but a model card's
            # default is deliberately authored, and dropping one in silence is
            # what hid dry_allowed_length=16 for weeks while DRY ran at the
            # engine's own harmful allowance of 2.
            if source:
                _log.warning("%s: dropping %s=%r (%s)", source, key, value, e)
            continue
    return out


def effective_params(session: dict, preset: dict | None = None,
                     model_defaults: dict | None = None) -> dict:
    """Weakest to strongest: model-card defaults < preset < session."""
    merged = _safe_params(model_defaults or {}, "model_defaults")
    merged.update(_safe_params((preset or {}).get("params", {})))
    merged.update(_safe_params(session.get("params", {})))
    return merged


def search(query: str) -> list[dict]:
    """Summaries (+ matching snippet) for sessions whose title or message
    bodies contain the query. One indexed FTS/LIKE query instead of the old
    parse-every-file-per-keystroke scan."""
    from . import db
    _import_legacy()
    hits = db.search_sessions(query)
    if not hits:
        return []
    by_id = {s["id"]: s for s in list_sessions()}
    out = []
    for sid, snippet in hits:
        summary = by_id.get(sid)
        if summary:
            out.append({**summary, "snippet": snippet})
    return out


def duplicate(session_id: str) -> dict | None:
    src = load(session_id)
    if src is None:
        return None
    now = time.time()
    dup = json.loads(json.dumps(src))  # deep copy - variants etc. detached
    dup["id"] = secrets.token_hex(6)
    dup["title"] = (src.get("title") or "chat") + " (copy)"
    dup["created_at"] = now
    save(dup)
    return dup


def _as_text(value) -> str:
    """The str `export_markdown`'s join needs, whatever the stored JSON holds.

    AUDIT 01-3: a session written before the write guard can hold a null, a
    dict or an int where the schema says string — the export used to raise
    `TypeError` on exactly that, so a chat became permanently unexportable.
    Coerce instead of raising; the guard stops new ones at the write.
    """
    if isinstance(value, str):
        return value
    if value is None:
        return ""
    return str(value)


def export_markdown(session: dict) -> str:
    lines = ["# " + _as_text(session.get("title") or "chat"), ""]
    if session.get("system_prompt"):
        lines += ["> " + _as_text(session["system_prompt"]).replace("\n", "\n> "),
                  ""]
    if session.get("notes"):
        lines += ["> Story notes: "
                  + _as_text(session["notes"]).replace("\n", "\n> "), ""]

    def _emit(m: dict) -> None:
        who = "**You:**" if m.get("role") == "user" else "**Model:**"
        content = m.get("content", "")
        if isinstance(content, list):   # vision content-parts
            content = "\n".join(
                _as_text(p.get("text", "")) if p.get("type") == "text"
                else "[image]"
                for p in content if isinstance(p, dict))
        else:
            content = _as_text(content)
        lines.extend([who, ""])
        if m.get("thinking"):
            lines.extend(["<details><summary>thinking</summary>", "",
                          _as_text(m["thinking"]), "", "</details>", ""])
        lines.extend([content, ""])

    # AUDIT F6: docs/audit-2026-09-04-full.md
    # After any compaction `messages` holds only the tail, so exporting it
    # alone handed back the last handful of turns of a 200-turn manuscript
    # with no marker that the rest ever existed. `archive` is chronological
    # and precedes `messages`; the digest stands in for whatever the archive's
    # bounded tail (serve.ARCHIVE_MAX) has already dropped, so it belongs
    # between them.
    archive = [m for m in (session.get("archive") or []) if isinstance(m, dict)]
    for m in archive:
        _emit(m)
    digest = str(session.get("digest") or "").strip()
    if digest:
        lines += ["---", "",
                  "> **Compacted summary of the earlier conversation**", ">",
                  "> " + digest.replace("\n", "\n> "), ""]
    if archive or digest:
        lines += ["---", ""]
    for m in session.get("messages", []):
        _emit(m)
    return "\n".join(lines)
