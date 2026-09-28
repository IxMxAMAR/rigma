"""One correct way to replace a small JSON/text store on disk.

R3-STORE-1. Every store in this project is a whole-file rewrite: load the
document, mutate it, write it back. That is fine, but the WRITE has to be atomic
or a crash in the middle leaves a truncated file — and the loaders here mostly
treat an unreadable file as "empty", which turns one torn write into permanent
data loss on the next save.

This machine has already lost data to exactly that: a power cut left a git ref
file empty. A truncated `calibration.json` behaves the same way — `load` returns
`{}`, the next save writes back `{}` plus one entry, and every other measured
model is gone.

Two properties matter, and the copies scattered through the codebase each got one
of them wrong:

1. **Write to a temp file, then `os.replace`.** Atomic on NTFS and POSIX alike.
2. **The temp name must be unique per write.** A fixed `<name>.tmp` means two
   concurrent writers (two finishing turns, a tool thread and the loop) collide:
   one replaces the other's file mid-write, or the rename fails with
   `PermissionError [WinError 32]`. `memory.py` documents fixing this for itself;
   the rest of the codebase still used a fixed name.

Use `atomic_write_text` for text. It fsyncs before the replace so the rename
cannot land ahead of the bytes on a crash.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

# Windows can refuse a rename with a sharing violation while another handle has
# the destination open — including another thread doing its own `os.replace` to
# the same path. It is transient, so it is retried rather than reported.
_WIN_SHARING_ERRNOS = {13, 32}          # EACCES, ESHARING
_REPLACE_ATTEMPTS = 20


def _replace_retrying(tmp: Path, path: Path) -> None:
    """`os.replace`, retrying a transient Windows sharing violation.

    A unique temp name stops two writers clobbering each other's FILE, but not
    two writers renaming onto the same DESTINATION at the same instant: measured
    on NTFS, four threads each writing 40 times produced `PermissionError(13,
    'Access is denied')`. The window is microseconds and the operation is
    idempotent, so a bounded retry is the correct answer — the alternative is a
    store that fails under exactly the concurrency (two finishing turns, a tool
    thread and the loop) that made the temp name unique in the first place.
    """
    for attempt in range(_REPLACE_ATTEMPTS):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            if getattr(e, "errno", None) not in _WIN_SHARING_ERRNOS:
                raise
            if attempt == _REPLACE_ATTEMPTS - 1:
                raise
            time.sleep(0.005 * (attempt + 1))


def atomic_write_text(path: Path, text: str, *, encoding: str = "utf-8") -> None:
    """Replace `path` with `text`, atomically and safely under concurrency.

    The temp file is created BESIDE the target, because `os.replace` is only
    atomic within a filesystem — a temp in the system temp dir can silently
    degrade to a copy across volumes.

    A unique suffix rather than a fixed `.tmp`: see the module docstring. The
    temp is removed on failure so a crash mid-write cannot leave litter that a
    later run mistakes for a store.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        with open(tmp, "w", encoding=encoding, newline="\n") as f:
            f.write(text)
            f.flush()
            # Durability, not decoration: without this the rename can be
            # persisted ahead of the data, which is the torn file we are here to
            # prevent. A power cut is the failure this exists for.
            os.fsync(f.fileno())
        _replace_retrying(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def atomic_write_json(path: Path, obj, *, indent: int = 2) -> None:
    """`atomic_write_text` for a JSON-serialisable object."""
    import json
    atomic_write_text(path, json.dumps(obj, indent=indent))
