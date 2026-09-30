"""ODR-8: ONE lock for the method and settings writers.

`serve._apply_restore` held only the memory store's `_xlock`. `methods.save_user`
/ `delete_user` and `app_settings.save` took no lock at all, so a save inside
the restore's millisecond window could be deleted after its caller was told
"saved", or overwritten by the restore's rollback.

LOCK ORDER (the only order used anywhere):

    WRITER_LOCK  ->  MemoryStore._xlock

`_apply_restore` takes WRITER_LOCK first and holds it for the whole
settings/methods/memory region; it may then take `_xlock` (nested inside
`store.locked()`). No path takes `_xlock` and then WRITER_LOCK, and no memory
writer takes WRITER_LOCK, so the order is acyclic — a deadlock would be worse
than the race it closes. WRITER_LOCK is an RLock because `_apply_restore`
calls the locked writers (`save_user`, `app_settings.replace`) while already
holding it.
"""
from __future__ import annotations

import threading

WRITER_LOCK = threading.RLock()
