from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

MUTABLE_FIELDS = ("title", "system_prompt", "use_rag", "messages",
                  "preset_id", "params", "notes", "digest", "effort",
                  "authors_note", "authors_note_depth", "prefill",
                  "use_tools", "allow_code", "workspace", "auto_compact",
                  "max_tool_rounds")
EFFORT_LEVELS = ("", "off", "auto", "on")

PARAM_RANGES = {"temperature": (0.0, 4.0), "top_p": (0.0, 1.0),
                "min_p": (0.0, 1.0), "repeat_penalty": (0.5, 2.0),
                "max_tokens": (1, 262144), "top_k": (0, 200),
                "seed": (-1, 2**31 - 1),
                "frequency_penalty": (-2.0, 2.0),
                "presence_penalty": (-2.0, 2.0),
                # modern anti-repetition samplers (llama-server per-request)
                "dry_multiplier": (0.0, 2.0), "dry_base": (1.0, 4.0),
                "dry_allowed_length": (1, 10),
                "xtc_probability": (0.0, 1.0), "xtc_threshold": (0.0, 0.5),
                "top_n_sigma": (-1.0, 5.0)}
_INT_PARAMS = ("max_tokens", "dry_allowed_length", "seed", "top_k")
_MAX_STOPS = 4

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
                     # per-turn agentic tool-call ceiling (safety backstop, not a
                     # feature limit) — big tasks (view 20 images, write files)
                     # need many; raise it for even longer autonomous runs
                     "max_tool_rounds": 25, "messages": []}


def chats_dir() -> Path:
    d = rigma_home() / "sessions" / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(session_id: str) -> Path:
    return chats_dir() / f"{session_id}.json"


def create(title: str = "New chat", system_prompt: str = "") -> dict:
    now = time.time()
    session = {**json.loads(json.dumps(_SESSION_DEFAULTS)),
               "id": secrets.token_hex(6), "title": title,
               "system_prompt": system_prompt,
               "created_at": now, "updated_at": now}
    save(session)
    return session


def save(session: dict) -> None:
    session["updated_at"] = time.time()
    p = _path(session["id"])
    # unique tmp name: concurrent saves of the same session (turn finishing
    # while a param edit lands from a thread) must not tear each other's file
    tmp = p.with_suffix(f".{secrets.token_hex(4)}.tmp")
    tmp.write_text(json.dumps(session, indent=2), encoding="utf-8")
    tmp.replace(p)   # atomic on same volume - no torn session files


def load(session_id: str) -> dict | None:
    try:
        raw = json.loads(_path(session_id).read_text(encoding="utf-8"))
    except Exception:
        return None
    # migration: sessions written by older Rigma versions lack newer fields
    for k, v in _SESSION_DEFAULTS.items():
        raw.setdefault(k, json.loads(json.dumps(v)))
    return raw


def delete(session_id: str) -> bool:
    p = _path(session_id)
    if not p.exists():
        return False
    p.unlink()
    return True


def list_sessions() -> list[dict]:
    out = []
    for f in chats_dir().glob("*.json"):
        s = load(f.stem)
        if s is None:  # corrupt file: skip, never fatal
            continue
        out.append({"id": s["id"], "title": s.get("title", ""),
                    "updated_at": s.get("updated_at", 0),
                    "use_rag": bool(s.get("use_rag")),
                    "message_count": len(s.get("messages", []))})
    return sorted(out, key=lambda s: s["updated_at"], reverse=True)


def build_messages(session: dict, default_prompt: str = "",
                   preset: dict | None = None) -> list[dict]:
    prompt = (session.get("system_prompt")
              or (preset or {}).get("system_prompt", "")
              or default_prompt)
    head = [{"role": "system", "content": prompt}] if prompt else []
    notes = session.get("notes", "")
    if notes:
        head.append({"role": "system",
                     "content": "Story notes (authoritative):\n" + notes})
    digest = session.get("digest", "")
    if digest:
        head.append({"role": "system",
                     "content": "Earlier conversation (compacted):\n" + digest})
    # sanitize: variants/metadata must never reach the model
    msgs = [{"role": m.get("role", "user"), "content": m.get("content", "")}
            for m in session.get("messages", [])]
    an = session.get("authors_note", "")
    if an:
        # depth-targeted injection: N messages from the end beats the system
        # prompt for steering prose (recency bias is real)
        try:
            depth = max(0, int(session.get("authors_note_depth", 3)))
        except (TypeError, ValueError):
            depth = 3
        msgs.insert(max(0, len(msgs) - depth),
                    {"role": "system", "content": f"[Author's note: {an}]"})
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


def _safe_params(raw: dict) -> dict:
    out = {}
    for key, value in (raw or {}).items():
        try:
            out.update(validate_params({key: value}))
        except ValueError:
            continue  # stored junk never blocks a chat turn
    return out


def effective_params(session: dict, preset: dict | None = None,
                     model_defaults: dict | None = None) -> dict:
    """Weakest to strongest: model-card defaults < preset < session."""
    merged = _safe_params(model_defaults or {})
    merged.update(_safe_params((preset or {}).get("params", {})))
    merged.update(_safe_params(session.get("params", {})))
    return merged


def search(query: str) -> list[dict]:
    """Summaries (+ first matching snippet) for sessions whose title or
    message bodies contain the query, case-insensitively."""
    q = query.strip().lower()
    if not q:
        return []
    out = []
    for summary in list_sessions():
        s = load(summary["id"])
        if s is None:
            continue
        snippet = ""
        if q in s.get("title", "").lower():
            snippet = s.get("title", "")
        else:
            for m in s.get("messages", []):
                content = m.get("content", "")
                if isinstance(content, list):   # vision parts
                    content = " ".join(p.get("text", "") for p in content
                                       if isinstance(p, dict))
                pos = content.lower().find(q)
                if pos != -1:
                    lo = max(0, pos - 40)
                    snippet = content[lo:pos + len(q) + 60].strip()
                    break
        if snippet:
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
    for m in session.get("messages", []):
        who = "**You:**" if m.get("role") == "user" else "**Model:**"
        content = m.get("content", "")
        if isinstance(content, list):   # vision content-parts
            content = "\n".join(
                p.get("text", "") if p.get("type") == "text" else "[image]"
                for p in content if isinstance(p, dict))
        lines += [who, ""]
        if m.get("thinking"):
            lines += ["<details><summary>thinking</summary>", "",
                      m["thinking"], "", "</details>", ""]
        lines += [content, ""]
    return "\n".join(lines)
