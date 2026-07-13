from __future__ import annotations

import json
import secrets
import time
from pathlib import Path

from .runtime import rigma_home

MUTABLE_FIELDS = ("title", "system_prompt", "use_rag", "messages")


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
               "created_at": now, "updated_at": now, "messages": []}
    save(session)
    return session


def save(session: dict) -> None:
    session["updated_at"] = time.time()
    _path(session["id"]).write_text(json.dumps(session, indent=2),
                                    encoding="utf-8")


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
