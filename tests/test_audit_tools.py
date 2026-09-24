"""Agent-tool safety: the confined profile, job lifetime, and the undo net.

Four defects sat behind these, all of them "the tool told you it was safe and
it was not":

  * `confined` was enforced by an exact two-name tuple in two separate places,
    so `start_job` — which builds the byte-identical PowerShell argv — ran
    happily under the profile that refuses `run_shell`.
  * background jobs were detached on purpose (CREATE_NEW_PROCESS_GROUP /
    start_new_session) and nothing ever walked `_JOBS`, so a stopped run left
    children holding VRAM with their integer ids gone after a restart.
  * pathless `undo_last_change` picked the newest entry in a per-install index
    keyed by absolute path across every workspace, so an undo in one project
    could revert a file in another and report only the basename.
  * `_snapshot_before_write` returned None and swallowed everything while
    three call sites promised "call undo_last_change to restore it".

The exec-profile tests are driven off the registry rather than a list of tool
names, so a NEW execution tool is covered the day it is registered.
"""
import sys
import time

import pytest

from rigma import tools


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return {"workspace": str(ws), "allow_code": True}


def _exec_tools():
    return [t.name for t in tools._REGISTRY.values() if t.kind == "exec"]


# --- F30: the confined profile ------------------------------------------------
def test_every_execution_tool_is_marked_as_one():
    # the point of kind="exec" is that it lives next to the registration, so
    # pin the three that spawn a process today; a fourth must opt in there
    assert set(_exec_tools()) >= {"run_shell", "run_python", "start_job"}


def test_confined_profile_advertises_no_execution_tool():
    names = {t["function"]["name"]
             for t in tools.tool_specs(allow_code=True, profile="confined")}
    offered = {t["function"]["name"]
               for t in tools.tool_specs(allow_code=True, profile="all")}
    for name in _exec_tools():
        assert name in offered, f"{name} should exist outside confined"
        assert name not in names, f"confined still advertises {name}"


def test_confined_profile_refuses_every_execution_tool_at_runtime():
    c = {"allow_code": True, "profile": "confined"}
    for name in _exec_tools():
        out = tools.run_tool(name, {"command": "echo hi", "code": "pass"}, c)
        assert out.startswith("error") and "confined" in out, (
            f"{name} ran under confined: {out[:120]}")


def test_start_job_still_runs_outside_confined(ctx):
    # the guard must be the profile, not a blanket refusal
    out = tools.run_tool("start_job", {"command": ""}, ctx)
    assert "confined" not in out


# --- F31: background jobs must not outlive the run ----------------------------
def _sleep_command() -> str:
    return "Start-Sleep -Seconds 20" if sys.platform == "win32" else "sleep 20"


def test_kill_all_jobs_kills_a_live_background_job(ctx):
    started = tools.run_tool("start_job", {"command": _sleep_command()}, ctx)
    assert started.startswith("started job"), started
    jid = max(tools._JOBS)
    proc = tools._JOBS[jid]["proc"]
    try:
        assert proc.poll() is None                    # really running
        assert tools.kill_all_jobs() >= 1             # reports what it killed
        deadline = time.monotonic() + 15
        while proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert proc.poll() is not None, "the job survived kill_all_jobs"
        assert tools.kill_all_jobs() == 0             # nothing left to kill
    finally:
        tools._kill_tree(proc.pid)
        tools._JOBS.pop(jid, None)


# --- F8: undo stays inside the workspace it was called from -------------------
def _two_workspaces(tmp_path):
    a, b = tmp_path / "novel", tmp_path / "code"
    a.mkdir()
    b.mkdir()
    return ({"workspace": str(a), "allow_code": True}, a,
            {"workspace": str(b), "allow_code": True}, b)


def test_pathless_undo_never_reverts_a_file_in_another_workspace(tmp_path):
    ca, a, cb, b = _two_workspaces(tmp_path)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v1"}, ca)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v2"}, ca)
    tools.run_tool("write_file", {"path": "m.py", "content": "one"}, cb)
    tools.run_tool("write_file", {"path": "m.py", "content": "two"}, cb)
    # the novel entry is older; undoing from the novel workspace must reach it
    # and NOT the code file, which is the newest entry in the shared index
    out = tools.run_tool("undo_last_change", {}, ca)
    assert out.startswith("restored"), out
    assert (a / "ch.txt").read_text(encoding="utf-8") == "v1"
    assert (b / "m.py").read_text(encoding="utf-8") == "two"   # untouched


def test_pathless_undo_reports_the_full_path_not_the_basename(tmp_path):
    ca, a, _, _ = _two_workspaces(tmp_path)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v1"}, ca)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v2"}, ca)
    out = tools.run_tool("undo_last_change", {}, ca)
    assert str(a) in out, f"only the basename was reported: {out}"


def test_pathless_undo_says_so_when_this_workspace_has_no_entry(tmp_path):
    ca, _, cb, _ = _two_workspaces(tmp_path)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v1"}, ca)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v2"}, ca)
    out = tools.run_tool("undo_last_change", {}, cb)
    assert out.startswith("error") and "workspace" in out


# --- F9: the undo promise must be honest --------------------------------------
def test_snapshot_reports_whether_it_landed(tmp_path, monkeypatch):
    f = tmp_path / "a.txt"
    f.write_text("body", encoding="utf-8")
    assert tools._snapshot_before_write(f) is True
    assert tools._snapshot_before_write(tmp_path / "missing.txt") is False
    monkeypatch.setattr(tools, "_undo_dir",
                        lambda: (_ for _ in ()).throw(OSError("read-only")))
    assert tools._snapshot_before_write(f) is False


def test_write_file_does_not_promise_undo_when_the_snapshot_failed(ctx,
                                                                   monkeypatch):
    tools.run_tool("write_file", {"path": "ch.txt", "content": "keep"}, ctx)
    monkeypatch.setattr(tools, "_undo_dir",
                        lambda: (_ for _ in ()).throw(OSError("read-only")))
    out = tools.run_tool("write_file", {"path": "ch.txt",
                                        "content": "clobber"}, ctx)
    assert "REPLACED" in out                     # the loud warning survives
    assert "undo_last_change" not in out, (
        "promised an undo that was never written: " + out)
    assert "could not" in out.lower()


def test_edit_file_line_range_promise_is_conditional(ctx, monkeypatch):
    ws = tools.Path(ctx["workspace"])
    (ws / "a.txt").write_text("alpha\nbravo\n", encoding="utf-8")
    monkeypatch.setattr(tools, "_undo_dir",
                        lambda: (_ for _ in ()).throw(OSError("read-only")))
    out = tools.run_tool("edit_file", {"path": "a.txt", "start_line": 1,
                                       "end_line": 1, "new": "ALPHA"}, ctx)
    assert out.startswith("edited")
    assert "undo_last_change" not in out
    assert "ALPHA" in (ws / "a.txt").read_text(encoding="utf-8")


def test_a_failed_snapshot_cannot_destroy_the_previous_one(ctx):
    ws = tools.Path(ctx["workspace"])
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v1"}, ctx)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "v2"}, ctx)
    d = tools._undo_dir()
    entry = tools._undo_index(d)[str(ws / "ch.txt")]
    snap = d / entry["snap"]
    assert snap.read_bytes() == b"v1"
    # the snapshot slot is a fixed filename: writing straight into it truncates
    # the recoverable version before the new bytes land. Fail at the rename and
    # v1 must still be there, whole.
    real_replace = tools.os.replace

    def _boom(src, dst):
        raise OSError("antivirus holds the snapshot")

    tools.os.replace = _boom
    try:
        out = tools.run_tool("write_file", {"path": "ch.txt",
                                            "content": "v3"}, ctx)
    finally:
        tools.os.replace = real_replace
    assert "undo_last_change" not in out
    assert snap.read_bytes() == b"v1", "the previous snapshot was destroyed"
    assert tools.run_tool("undo_last_change", {}, ctx).startswith("restored")
    assert (ws / "ch.txt").read_text(encoding="utf-8") == "v1"


def test_undo_refuses_a_snapshot_whose_size_no_longer_matches(ctx):
    ws = tools.Path(ctx["workspace"])
    tools.run_tool("write_file", {"path": "ch.txt", "content": "the draft"},
                   ctx)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "oops"}, ctx)
    d = tools._undo_dir()
    snap = d / tools._undo_index(d)[str(ws / "ch.txt")]["snap"]
    snap.write_bytes(b"half a dra")            # a torn/partial snapshot
    out = tools.run_tool("undo_last_change", {"path": "ch.txt"}, ctx)
    assert out.startswith("error") and "bytes" in out
    assert (ws / "ch.txt").read_text(encoding="utf-8") == "oops"   # untouched


def test_an_index_written_before_sizes_were_recorded_still_undoes(ctx):
    ws = tools.Path(ctx["workspace"])
    tools.run_tool("write_file", {"path": "ch.txt", "content": "the draft"},
                   ctx)
    tools.run_tool("write_file", {"path": "ch.txt", "content": "oops"}, ctx)
    d = tools._undo_dir()
    idx = tools._undo_index(d)
    idx[str(ws / "ch.txt")].pop("size", None)      # an index from before F9
    (d / "index.json").write_text(tools.json.dumps(idx), encoding="utf-8")
    out = tools.run_tool("undo_last_change", {"path": "ch.txt"}, ctx)
    assert out.startswith("restored"), out
    assert (ws / "ch.txt").read_text(encoding="utf-8") == "the draft"


# --- F38: a note channel shared between concurrent tool calls -----------------
def _two_image_folders(tmp_path):
    from PIL import Image
    out = {}
    for key, stem in (("a", "Alpha_001"), ("b", "Bravo_001")):
        d = tmp_path / key
        d.mkdir()
        Image.new("RGB", (4, 4), (1, 2, 3)).save(d / f"{stem}.png")
        out[key] = (d, f"{stem[:-3]}_{stem[-3:]}.png")   # 001 -> 1: a near miss
    return out


def test_two_parallel_image_calls_do_not_swap_their_notes(tmp_path, monkeypatch):
    """`_FUZZY_NOTES` was a module-level out-parameter.

    serve.py starts every parsed tool call as its own task under a semaphore of
    8, and the system prompt tells the model to view images "in batches" — so
    two concurrent view_images calls cleared and drained each other's correction
    notes (AUDIT F38). The note is a return value now; this pins it by holding
    both calls inside the resolver until each has its own answer.
    """
    import threading
    folders = _two_image_folders(tmp_path)
    ctx = {"workspace": str(tmp_path), "has_vision": True}
    real = tools._fuzzy_file
    both_resolved = threading.Barrier(2, timeout=20)

    def synced(p, ctx=None):
        got = real(p, ctx)
        both_resolved.wait()      # neither call may drain a shared list yet
        return got

    monkeypatch.setattr(tools, "_fuzzy_file", synced)
    results: dict = {}

    def one(key, mangled):
        results[key] = tools.run_tool("view_images",
                                      {"paths": [str(folders[key][0] / mangled)]},
                                      ctx)

    threads = [threading.Thread(target=one, args=(k, folders[k][1]))
               for k in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert set(results) == {"a", "b"}, results
    for key, other in (("a", "b"), ("b", "a")):
        mine, theirs = folders[key][1], folders[other][1]
        assert mine in results[key], results[key]
        assert theirs not in results[key], (
            f"{key} was handed {other}'s correction: {results[key]}")


def test_view_image_alone_reports_its_own_correction(tmp_path):
    """view_image (singular) had no note channel at all, so its correction was
    written to the shared list and read by nobody (AUDIT F38)."""
    from PIL import Image
    Image.new("RGB", (4, 4), (1, 2, 3)).save(tmp_path / "Alpha_001.png")
    out = tools.run_tool("view_image",
                         {"path": str(tmp_path / "Alpha_1.png")},
                         {"workspace": str(tmp_path), "has_vision": True})
    assert out.startswith(tools.IMAGE_SENTINEL)
    assert "Alpha_001.png" in out and "Alpha_1.png" in out, out


# --- F34: a move must say which file it actually moved ------------------------
def test_move_files_reports_the_fuzzy_correction(tmp_path):
    """`_transfer_sources` computed the "asked X / used Y" note and dropped it.

    The reply listed the real post-fuzzy filenames but never as a pairing, and
    only for the first 10 of up to 50 — so a misfiled chapter arrived with no
    explanation (AUDIT F34)."""
    ws = tmp_path / "ws"
    ws.mkdir()
    (ws / "IMG_0013.png").write_bytes(b"pic")
    dest = tmp_path / "sorted"
    out = tools.run_tool("move_files",
                         {"paths": [str(ws / "IMG_13.png")],
                          "dest": str(dest)},
                         {"workspace": str(ws), "allow_code": True})
    assert out.startswith("moved 1 file(s)"), out
    assert "IMG_13.png" in out and "IMG_0013.png" in out, out
    assert (dest / "IMG_0013.png").is_file()


# --- F37: sentinel tools need the loop, not a handler -------------------------
def test_loop_only_tools_are_refused_outside_the_loop(ctx):
    """`delegate` and `use_tools` are placeholders whose only real
    implementation is serve.py's loop. Any other caller used to receive the raw
    control bytes, which the defuser turned into "[1 unreadable control
    byte(s) — file corruption?]…" and a macro substituted into a note or a
    written file (AUDIT F37)."""
    assert tools._LOOP_ONLY_TOOLS == {"delegate", "use_tools"}
    for name, args in (("delegate", {"question": "q"}),
                       ("use_tools", {"names": ["read_file"]})):
        out = tools.run_tool(name, args, ctx)
        assert out.startswith("error"), f"{name} leaked a sentinel: {out[:80]}"
        assert "chat turn" in out


def test_the_loop_host_flag_is_what_lets_the_sentinel_through(ctx):
    out = tools.run_tool("delegate", {"question": "q"},
                         {**ctx, "can_host_loop_tools": True})
    assert out.startswith(tools.DELEGATE_SENTINEL), out


def test_the_loop_only_tools_are_kept_out_of_the_macro_validator():
    from rigma import method_schema as ms
    names = set(tools._REGISTRY) - tools._LOOP_ONLY_TOOLS
    doc = {"name": "M", "apply": {"system_prompt": "s", "effort": "medium"},
           "macros": [{"id": "m1", "label": "L", "steps": [
               {"kind": "tool", "name": "delegate", "args": {}}]}]}
    errs = ms.validate(doc, names)
    assert any("delegate" in e for e in errs), errs


# --- F33: a runaway printer must not be buffered whole ------------------------
def test_runaway_output_is_stopped_at_the_cap(ctx, monkeypatch):
    """`_run_subprocess` piped both streams and called communicate(), holding
    the ENTIRE output in memory before truncating — so a print loop accumulated
    gigabytes inside the server, on a box whose RAM is committed to the model
    (AUDIT F33)."""
    monkeypatch.setattr(tools, "_SHELL_OUT_CAP", 20_000)
    code = "import sys\nwhile True: sys.stdout.write('x' * 4096)"
    started = time.monotonic()
    out = tools._run_subprocess([sys.executable, "-c", code], ctx,
                                python_src=code, timeout=120)
    elapsed = time.monotonic() - started
    assert elapsed < 60, f"the cap took {elapsed:.1f}s to stop a runaway printer"
    assert out.startswith("error") and "output exceeded" in out, out


def test_ordinary_output_still_comes_back_whole(ctx):
    code = "print('hello'); print('world')"
    out = tools._run_subprocess([sys.executable, "-c", code], ctx,
                                python_src=code, timeout=60)
    assert out.startswith("exit 0 (ok)"), out
    assert "hello" in out and "world" in out


# --- F35: a kill that was never confirmed must not be reported as one ---------
def test_kill_tree_reports_whether_the_process_is_actually_gone(ctx):
    proc = tools._launch_killable(
        [sys.executable, "-c", "import time; time.sleep(30)"], False, None)
    try:
        assert proc.poll() is None
        assert tools._kill_tree(proc.pid, proc) is True
        assert proc.poll() is not None
    finally:
        if proc.poll() is None:
            tools._kill_tree(proc.pid, proc)


def test_kill_job_admits_a_kill_it_could_not_confirm(ctx, monkeypatch):
    """`_kill_tree` swallowed every failure and `kill_job` claimed success with
    no re-poll, so the model rebound the port it was told was free (F35)."""
    started = tools.run_tool("start_job", {"command": _sleep_command()}, ctx)
    assert started.startswith("started job"), started
    jid = max(tools._JOBS)
    proc = tools._JOBS[jid]["proc"]
    real = tools._kill_tree
    monkeypatch.setattr(tools, "_kill_tree", lambda *a, **k: False)
    try:
        out = tools.run_tool("kill_job", {"id": jid}, ctx)
        assert out.startswith("error") and "could not confirm" in out, out
        assert str(proc.pid) in out
    finally:
        monkeypatch.setattr(tools, "_kill_tree", real)
        real(proc.pid, proc)
        tools._JOBS.pop(jid, None)

