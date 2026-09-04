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
import re
import secrets
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
                  "max_tool_rounds", "one_action", "method", "carry_reasoning")
# "" / off / auto / on are Rigma's own binary thinking switch. The four
# named levels are Qwen3.8's published reasoning efforts, which its chat
# template reads from `reasoning_effort` — verified against a live engine
# 2026-08-19 (a plain turn renders 79 prompt chars, xhigh renders 316).
# A template that has never heard of the kwarg simply ignores it.
QWEN_EFFORTS = ("low", "medium", "high", "xhigh")
EFFORT_LEVELS = ("", "off", "auto", "on") + QWEN_EFFORTS

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


def create(title: str = "New chat", system_prompt: str = "") -> dict:
    now = time.time()
    session = {**json.loads(json.dumps(_SESSION_DEFAULTS)),
               "id": secrets.token_hex(6), "title": title,
               "system_prompt": system_prompt,
               "created_at": now, "updated_at": now}
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
    # migration: sessions written by older Rigma versions lack newer fields
    for k, v in _SESSION_DEFAULTS.items():
        raw.setdefault(k, json.loads(json.dumps(v)))
    # stored 50 is the OLD default leash, not a choice anyone made - lift it
    # to the new backstop (a deliberately lowered value survives untouched)
    if raw.get("max_tool_rounds") == 50:
        raw["max_tool_rounds"] = 1000
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


def export_markdown(session: dict) -> str:
    lines = ["# " + (session.get("title") or "chat"), ""]
    if session.get("system_prompt"):
        lines += ["> " + session["system_prompt"].replace("\n", "\n> "), ""]
    if session.get("notes"):
        lines += ["> Story notes: " + session["notes"].replace("\n", "\n> "), ""]

    def _emit(m: dict) -> None:
        who = "**You:**" if m.get("role") == "user" else "**Model:**"
        content = m.get("content", "")
        if isinstance(content, list):   # vision content-parts
            content = "\n".join(
                p.get("text", "") if p.get("type") == "text" else "[image]"
                for p in content if isinstance(p, dict))
        lines.extend([who, ""])
        if m.get("thinking"):
            lines.extend(["<details><summary>thinking</summary>", "",
                          m["thinking"], "", "</details>", ""])
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
