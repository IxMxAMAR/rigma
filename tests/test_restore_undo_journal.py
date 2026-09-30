"""ODR-4: the restore undo log was in memory only, and `save_user` was not atomic.

Two failure modes, one journal:

* `methods.save_user` used `Path.write_text`, which truncates in place. A crash
  mid-write left a torn method file that `user_methods()` skips, and the prior
  bytes existed only in RAM.
* `/api/restore`'s undo log lived in RAM too. A kill between the stale-method
  deletes and memory's `os.replace` left a mixed store that nothing repaired,
  and methods created after the backup were gone for good.

The fix journals the prior bytes to `~/.rigma/restore-undo/` before the apply
and replays/clears that directory at boot. It is state, never prose to a model.
"""
import pytest

from rigma import app_settings, atomicio, methods, serve
from rigma.memory import MemoryStore


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)
    return tmp_path


def _doc(mid="mine"):
    return {"id": mid, "name": "Mine", "tagline": "t",
            "apply": {"system_prompt": "p", "params": {"temperature": 0.5},
                      "effort": "auto", "use_tools": True,
                      "allow_code": True, "notes_template": ""},
            "macros": [
                {"id": "peek", "label": "Peek", "steps": [
                    {"kind": "tool", "name": "read_file",
                     "args": {"path": "a.txt"}}]}]}


def test_save_user_leaves_the_old_bytes_when_the_replace_is_interrupted(
        home, monkeypatch):
    """The temp write succeeded but `os.replace` never ran: the OLD file must
    still be the old file, not a truncated half-write of the new one."""
    saved, errs = methods.save_user(_doc("mine"))
    assert saved is not None and not errs, errs
    p = methods._method_file("mine")
    before = p.read_bytes()

    def boom(tmp, path):
        raise OSError("simulated crash before os.replace")

    monkeypatch.setattr(atomicio, "_replace_retrying", boom)
    with pytest.raises(OSError):
        methods.save_user({**_doc("mine"), "name": "Changed"})

    assert p.read_bytes() == before, (
        "the interrupted write changed the existing method file")
    # and no temp litter is left beside it
    assert not [f for f in p.parent.glob("*.tmp")], "temp file left behind"


def test_a_killed_restore_is_rolled_back_at_boot(home, monkeypatch):
    """Kill the restore between the stale-method delete and memory's write.
    The on-disk journal must survive the process and the boot step must put the
    store back and clear the journal."""
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="original")
    saved, errs = methods.save_user(_doc("mine"))
    assert saved is not None and not errs, errs
    undo = serve._restore_undo_dir()

    real_delete = methods.delete_user

    def killed(mid):
        real_delete(mid)                    # the delete lands...
        raise KeyboardInterrupt("power cut")  # ...then the process dies

    monkeypatch.setattr(methods, "delete_user", killed)

    with pytest.raises(KeyboardInterrupt):
        serve._apply_restore(
            store,
            [("settings", app_settings.settings_path()),
             ("memory", store.path)],
            {},                             # settings reset
            [],                             # nothing kept -> "mine" is stale
            [{"id": "m1", "kind": "technique", "text": "restored"}])

    # the mixed store the crash left behind
    assert (undo / "manifest.json").is_file(), "no undo journal was written"
    assert not methods._method_file("mine").exists(), "the delete never ran"
    assert [m["text"] for m in store.all()] == ["original"]

    # a fresh boot rolls the partial restore back and clears the journal
    serve._replay_restore_undo()
    assert methods._method_file("mine").is_file(), (
        "the boot step did not restore the deleted method")
    assert methods.get("mine") is not None
    assert [m["text"] for m in store.all()] == ["original"]
    assert not undo.exists() or not any(undo.iterdir()), (
        "the journal was not cleared after a successful replay")


def test_a_clean_restore_clears_the_journal(home):
    """The normal path must not leave a journal that a later boot would replay
    over a good store."""
    store = MemoryStore(home / "memory" / "memories.jsonl")
    store.add(kind="pitfall", text="original")
    full, errs = methods.validate_user(_doc("mine"))
    assert not errs, errs

    before, after, deleted = serve._apply_restore(
        store,
        [("settings", app_settings.settings_path()),
         ("memory", store.path)],
        {"idle_unload_minutes": 5},
        [full],
        [{"id": "m1", "kind": "technique", "text": "restored"}])

    assert deleted == []
    assert after == 1
    undo = serve._restore_undo_dir()
    assert not undo.exists() or not any(undo.iterdir()), (
        "a clean restore left its undo journal behind")
