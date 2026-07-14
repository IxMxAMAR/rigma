from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

MUTABLE_FIELDS = ("title", "system_prompt", "use_rag", "messages",
                  "preset_id", "params", "notes")

PARAM_RANGES = {"temperature": (0.0, 4.0), "top_p": (0.0, 1.0),
                "min_p": (0.0, 1.0), "repeat_penalty": (0.5, 2.0),
                "max_tokens": (1, 32768)}


def chats_dir() -> Path:
    d = rigma_home() / "sessions" / "chats"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _path(session_id: str) -> Path:
    return chats_dir() / f"{session_id}.json"


def create(title: str = "New chat", system_prompt: str = "") -> dict:
    now = time.time()
    session = {"id": secrets.token_hex(6), "title": title,
               "system_prompt": system_prompt, "use_rag": False,
               "preset_id": "", "params": {}, "notes": "",
               "created_at": now, "updated_at": now, "messages": []}
    save(session)
    return session


def save(session: dict) -> None:
    session["updated_at"] = time.time()
    p = _path(session["id"])
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(session, indent=2), encoding="utf-8")
    tmp.replace(p)   # atomic on same volume - no torn session files


def load(session_id: str) -> dict | None:
    try:
        return json.loads(_path(session_id).read_text(encoding="utf-8"))
    except Exception:
        return None


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


def build_messages(session: dict, default_prompt: str = "") -> list[dict]:
    prompt = session.get("system_prompt") or default_prompt
    head = [{"role": "system", "content": prompt}] if prompt else []
    return head + list(session.get("messages", []))


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
        if key not in PARAM_RANGES:
            continue
        lo, hi = PARAM_RANGES[key]
        try:
            value = int(value) if key == "max_tokens" else float(value)
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


def effective_params(session: dict, preset: dict | None = None) -> dict:
    merged = _safe_params((preset or {}).get("params", {}))
    merged.update(_safe_params(session.get("params", {})))
    return merged
