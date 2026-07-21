"""Phase 3 of the field-parity audit: the missing tools.

Background jobs (the 30s ceiling is gone), first-class move/copy with sample
references, delegate available in ordinary chats, ask_user pausing a run.
"""
import time

import pytest

from rigma import runs, tools


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rhome"))
    return tmp_path


CODE = {"allow_code": True}


# --- background jobs ----------------------------------------------------------
def test_job_lifecycle():
    out = tools.run_tool(
        "start_job",
        {"command": "python -c \"import time; print('begun', flush=True); "
                    "time.sleep(2); print('finished')\""}, CODE)
    assert out.startswith("started job")
    jid = int(out.split()[2])
    # running state visible immediately
    status = tools.run_tool("job_output", {"id": jid}, CODE)
    assert "running" in status or "exited" in status
    # wait for it and see the full output + exit code
    for _ in range(40):
        status = tools.run_tool("job_output", {"id": jid}, CODE)
        if "exited" in status:
            break
        time.sleep(0.25)
    assert "exited 0 (ok)" in status
    assert "begun" in status and "finished" in status


def test_job_kill():
    out = tools.run_tool(
        "start_job",
        {"command": "python -c \"import time; time.sleep(60)\""}, CODE)
    jid = int(out.split()[2])
    killed = tools.run_tool("kill_job", {"id": jid}, CODE)
    assert killed.startswith(f"job {jid} killed")
    for _ in range(20):
        status = tools.run_tool("job_output", {"id": jid}, CODE)
        if "exited" in status:
            break
        time.sleep(0.25)
    assert "exited" in status


def test_job_listing_and_bad_id():
    assert "no such job" in tools.run_tool("job_output", {"id": 9999}, CODE)
    listing = tools.run_tool("job_output", {}, CODE)
    assert "job" in listing or "no jobs" in listing


def test_start_job_respects_blocklist():
    out = tools.run_tool("start_job", {"command": "format C:"}, CODE)
    assert out.startswith("error") and "blocked" in out


def test_run_shell_timeout_param(tmp_path):
    t0 = time.monotonic()
    out = tools.run_tool(
        "run_shell",
        {"command": "python -c \"import time; time.sleep(30)\"",
         "timeout": 2}, CODE)
    assert time.monotonic() - t0 < 25
    assert "timed out after 2s" in out
    assert "start_job" in out           # points at the right tool for slow work


# --- move/copy ----------------------------------------------------------------
def _ws(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir(exist_ok=True)
    return {"workspace": str(ws), "allow_code": True}, ws


def test_move_files_by_paths(tmp_path):
    ctx, ws = _ws(tmp_path)
    (ws / "a.png").write_text("a", encoding="utf-8")
    (ws / "b.png").write_text("b", encoding="utf-8")
    out = tools.run_tool("move_files", {
        "dest": "sorted", "paths": ["a.png", "b.png"]}, ctx)
    assert out.startswith("moved 2")
    assert (ws / "sorted" / "a.png").is_file()
    assert not (ws / "a.png").exists()


def test_copy_never_overwrites(tmp_path):
    ctx, ws = _ws(tmp_path)
    (ws / "x.txt").write_text("new", encoding="utf-8")
    dest = ws / "out"
    dest.mkdir()
    (dest / "x.txt").write_text("precious", encoding="utf-8")
    out = tools.run_tool("copy_files", {"dest": "out",
                                        "paths": ["x.txt"]}, ctx)
    assert out.startswith("copied 1")
    assert (dest / "x.txt").read_text(encoding="utf-8") == "precious"
    assert (dest / "x (2).txt").read_text(encoding="utf-8") == "new"


def test_move_fuzzy_recovers_mistyped_name(tmp_path):
    ctx, ws = _ws(tmp_path)
    (ws / "ComfyUI_00428_.png").write_text("img", encoding="utf-8")
    out = tools.run_tool("move_files", {
        "dest": "keep", "paths": ["Comfy_UI_428.png"]}, ctx)
    assert out.startswith("moved 1")
    assert (ws / "keep" / "ComfyUI_00428_.png").is_file()


def test_move_blocked_under_no_delete(tmp_path):
    ctx, ws = _ws(tmp_path)
    ctx["profile"] = "no-delete"
    (ws / "f.txt").write_text("x", encoding="utf-8")
    out = tools.run_tool("move_files", {"dest": "d", "paths": ["f.txt"]}, ctx)
    assert out.startswith("error") and "copy_files" in out
    # copy is fine under no-delete
    out2 = tools.run_tool("copy_files", {"dest": "d", "paths": ["f.txt"]}, ctx)
    assert not out2.startswith("error")


def test_move_from_sample_reference(tmp_path, monkeypatch):
    ctx, ws = _ws(tmp_path)
    imgs = []
    for i in range(3):
        p = ws / f"IMG_{i:05d}_final_render.png"
        p.write_text("x", encoding="utf-8")
        imgs.append(str(p))
    run = runs.create("m", "sid", workspace=str(ws))
    runs.set_last_sample(run["id"], imgs)
    ctx["run_id"] = run["id"]
    out = tools.run_tool("move_files", {"dest": "picked", "first": 1,
                                        "count": 2}, ctx)
    assert out.startswith("moved 2")
    assert (ws / "picked").is_dir()


# --- gating -------------------------------------------------------------------
def test_delegate_available_in_chat_with_workspace(tmp_path):
    ws_names = {t["function"]["name"]
                for t in tools.tool_specs(workspace=str(tmp_path))}
    assert "delegate" in ws_names          # chat + workspace: yes
    bare = {t["function"]["name"] for t in tools.tool_specs()}
    assert "delegate" not in bare          # no workspace: no


def test_new_tools_gated_behind_code():
    base = {t["function"]["name"] for t in tools.tool_specs()}
    for name in ("start_job", "job_output", "kill_job",
                 "move_files", "copy_files"):
        assert name not in base, name
    with_code = {t["function"]["name"]
                 for t in tools.tool_specs(allow_code=True)}
    for name in ("start_job", "job_output", "kill_job",
                 "move_files", "copy_files"):
        assert name in with_code, name


# --- ask_user -----------------------------------------------------------------
def test_ask_user_pauses_run_and_stores_question(tmp_path):
    run = runs.create("mission", "sid", workspace=str(tmp_path))
    ctx = {"run_id": run["id"]}
    out = tools.run_tool("ask_user", {
        "question": "Should chapter names use roman numerals?"}, ctx)
    assert "PAUSED" in out
    r = runs.load(run["id"])
    assert r["paused"] is True
    assert "roman numerals" in r["pending_question"]["q"]


def test_ask_user_needs_run_and_question(tmp_path):
    assert "only available" in tools.run_tool(
        "ask_user", {"question": "x"}, {})
    run = runs.create("m", "sid", workspace=str(tmp_path))
    out = tools.run_tool("ask_user", {"question": "  "},
                         {"run_id": run["id"]})
    assert out.startswith("error")
