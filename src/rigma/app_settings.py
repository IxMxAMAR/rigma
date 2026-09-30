"""Server-level settings that survive a restart.

Deliberately tiny: almost all Rigma state is per-chat (sessions) or per-model
(registry), and belongs where it already lives. This file exists for the few
knobs that belong to the SERVER itself — today, the idle auto-unload timeout.

Why its own JSON file and not state.json: state.json is rewritten whole at
every launch and DELETED by `rigma stop`, so a setting stored there would not
survive the next `rigma up`. Settings are user intent; state is runtime fact.

No engine, no network, no side effects on import.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from . import writelock
from .atomicio import atomic_write_json
from .runtime import rigma_home

# Every setting, and the value a caller who does not name it gets.
DEFAULTS: dict = {
    # Minutes of inactivity after which the loaded model is unloaded to free
    # VRAM. 0 = NEVER, which is the current default and the safe direction: an
    # unload out from under a slow prefill looks like a bug to the user, and
    # only the owner knows whether this machine has VRAM to spare.
    "idle_unload_minutes": 0.0,
}

# A day. Past this it is indistinguishable from "off" and more likely a typo.
MAX_IDLE_MINUTES = 24 * 60.0


def settings_path() -> Path:
    return rigma_home() / "settings.json"


def _coerce(raw: dict) -> dict:
    """DEFAULTS, overlaid with whatever of `raw` is present and sane. A value
    of the wrong type falls back to the default rather than raising: a
    hand-edited settings.json must not stop the server from starting."""
    out = dict(DEFAULTS)
    if not isinstance(raw, dict):
        return out
    for key, default in DEFAULTS.items():
        if key not in raw:
            continue
        value = raw[key]
        if isinstance(default, float):
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if value != value:          # NaN
                continue
            value = max(0.0, min(value, MAX_IDLE_MINUTES))
        out[key] = value
    return out


def load() -> dict:
    """The stored settings, defaulted. A missing or corrupt file is defaults."""
    try:
        raw = json.loads(settings_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError):
        return dict(DEFAULTS)
    return _coerce(raw)


def validate(patch: dict) -> tuple[dict, str]:
    """Coerce a settings patch. Returns (clean, error_message)."""
    if not isinstance(patch, dict) or not patch:
        return {}, "send at least one setting"
    clean: dict = {}
    for key, value in patch.items():
        if key not in DEFAULTS:
            return {}, f"unknown setting: {key}"
        if key == "idle_unload_minutes":
            try:
                n = float(value)
            except (TypeError, ValueError):
                return {}, ("idle_unload_minutes: must be a number of minutes "
                            "(0 = never unload)")
            if n != n or n < 0 or n > MAX_IDLE_MINUTES:
                return {}, (f"idle_unload_minutes: must be between 0 and "
                            f"{MAX_IDLE_MINUTES:.0f} (0 = never unload)")
            clean[key] = n
    return clean, ""


def save(patch: dict) -> dict:
    """Apply a validated patch atomically and return the whole settings dict.

    Raises ValueError with a message a client can show.
    """
    clean, err = validate(patch)
    if err:
        raise ValueError(err)
    cur = load()
    cur.update(clean)
    p = settings_path()
    # R3-STORE-10: a fixed `<name>.tmp` beside the target collides under
    # concurrency, and this is the store the settings UI writes while the server
    # may be reading it. Shared writer: unique temp + retried replace.
    # ODR-8: WRITER_LOCK, shared with the method writers and held by the
    # restore for its whole region (see rigma.writelock for the lock order).
    with writelock.WRITER_LOCK:
        atomic_write_json(p, cur, indent=2)
    return cur


def replace(clean: dict) -> dict:
    """Write EXACTLY these settings: every key the caller omits goes back to its
    default. Returns the whole settings dict.

    OD-15 (option 1): `save` is a MERGE and must stay one — the settings UI sends
    one key and means "change this one". A restore sends a whole backup document
    and means "the store IS this", so merging lets a key created after the backup
    survive the restore and the store never comes back to the file's state. This
    is that write. There is no invariant/server-managed key to preserve: the only
    setting today is `idle_unload_minutes`, pure user intent whose default (0.0,
    never unload) is safe to fall back to. If a server-managed key is ever added
    to DEFAULTS it must be re-derived here rather than read from `clean`.

    Raises ValueError with a client-facing message, like `save`.
    """
    if not isinstance(clean, dict):
        raise ValueError("settings: must be an object")
    # An empty patch is legal here and means "all defaults"; validate() would
    # refuse it with "send at least one setting", which is the merge caller's
    # rule, not a restore's.
    clean, err = validate(clean) if clean else ({}, "")
    if err:
        raise ValueError(err)
    out = dict(DEFAULTS)
    out.update(clean)
    with writelock.WRITER_LOCK:
        atomic_write_json(settings_path(), out, indent=2)
    return out


def idle_unload_minutes() -> float:
    """The EFFECTIVE idle timeout in minutes (0 = off).

    `RIGMA_KEEP_ALIVE_MIN` remains an override: it is what builds before this
    setting used, and a launcher that sets it must keep working. Read fresh
    each poll, so a change applies without a restart.
    """
    env = os.environ.get("RIGMA_KEEP_ALIVE_MIN")
    if env not in (None, ""):
        try:
            return max(0.0, float(env))
        except ValueError:
            pass
    return float(load()["idle_unload_minutes"])
