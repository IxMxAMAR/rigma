"""Phase 0 of the 2026-07-21 field-parity audit: tool-contract bug fixes.

Each test pins a fix from the audit (docs/audit-2026-07-21-field-parity.md):
exit codes always visible, blocklist false-positives gone, schema
contradictions resolved, safe-tier honesty, and the fit table covering every
offered KV type.
"""
import pytest

from rigma import bench, tools
from rigma.models import ComboFlags, GgufFile, RunPlan


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


CODE = {"allow_code": True}


# --- exit codes always visible ------------------------------------------------
def test_run_python_success_leads_with_exit_zero():
    out = tools.run_tool("run_python", {"code": "print('hi')"}, CODE)
    assert out.splitlines()[0].startswith("exit 0")
    assert "hi" in out


def test_run_python_failure_with_output_is_marked_failed():
    # a failing script that PRINTS used to look identical to success — the
    # exit code appeared only when output was empty
    out = tools.run_tool(
        "run_python",
        {"code": "print('partial work'); raise SystemExit(3)"}, CODE)
    assert "exit 3 (FAILED)" in out.splitlines()[0]
    assert "partial work" in out


def test_run_python_no_output_still_reports_exit():
    out = tools.run_tool("run_python", {"code": "pass"}, CODE)
    assert out.startswith("exit 0") and "no output" in out


# --- blocklist: `format` matches the drive-wipe form only ---------------------
def test_format_flags_and_cmdlets_not_blocked():
    # `git log --format=…` and PowerShell Format-Table were refused as
    # "destructive"; a small model told that abandons the correct approach
    for cmd in ("git log --format=%H",
                "Get-Process | Format-Table -AutoSize",
                "Get-ChildItem | Format-List",
                "aria2c --format=json"):
        assert tools._BLOCKED_CMD.search(cmd) is None, cmd


def test_format_drive_still_blocked():
    for cmd in ("format C:", "FORMAT d: /q", "echo y | format e:"):
        assert tools._BLOCKED_CMD.search(cmd), cmd
    out = tools.run_tool("run_shell", {"command": "format C:"}, CODE)
    assert out.startswith("error") and "blocked" in out


# --- AUDIT F04-1/04-3: the cmdlets PowerShell actually runs -------------------
#
# These assert the guard REGEX on the command TEXT, per safety rule 7. Handing
# any of these strings to run_shell would be handing a dangerous command to
# something that executes it, so the end-to-end path is not exercised: a regex
# that stopped matching would mean the command ran.
def test_powershell_destructive_cmdlets_are_blocked():
    for cmd in ("Remove-Item -Recurse -Force C:\\Users\\Bob",
                "ri -Recurse -Force C:\\Users\\Bob",
                "Remove-Item C:\\build -Recurse",
                "Stop-Computer -Force",
                "Restart-Computer",
                "Format-Volume -DriveLetter D",
                "Clear-Disk -Number 1 -RemoveData",
                "Initialize-Disk -Number 1",
                "Remove-Partition -DriveLetter D",
                "Set-Acl C:\\ x",
                "logoff",
                "shutdown /s /t 0",
                "diskpart /s x.txt"):
        assert tools._BLOCKED_CMD.search(cmd), cmd


def test_ordinary_commands_are_not_destructive():
    # A plain (non-recursive) Remove-Item is an ordinary delete: it is the run
    # profile's business, not the destructive list's. Same for a relative rmtree.
    for cmd in ("Remove-Item foo.txt", "ri foo.txt",
                "git rm -r --cached .", "git log --format=%H",
                "Get-Process | Format-Table -AutoSize",
                "python -c \"import shutil; shutil.rmtree('build')\""):
        assert tools._BLOCKED_CMD.search(cmd) is None, cmd
    assert tools._BLOCKED_PY.search("import shutil; shutil.rmtree('build')") is None


# --- AUDIT F04-4: rm flags in any order, long form, and $HOME -----------------
def test_rm_recursive_force_against_a_root_is_blocked():
    for cmd in ("rm -rf /", "rm -r -f /", "rm -fr /", "rm --recursive --force /",
                "rm -rf ~/", "rm -rf $HOME", "rm -rf ${HOME}",
                "sudo rm -r -f /"):
        assert tools._BLOCKED_CMD.search(cmd), cmd


# --- AUDIT F04-2: a rooted or relative rmtree, and shelled-out delete verbs ---
def test_python_blocklist_covers_rooted_and_relative_rmtree():
    for code in ("import shutil; shutil.rmtree('/')",
                 "import shutil; shutil.rmtree('.')",
                 "import shutil; shutil.rmtree('..')",
                 "import shutil; shutil.rmtree('D:/')",
                 "import shutil; shutil.rmtree(os.path.expanduser('~'))",
                 "import os; os.system('rd /s /q D:\\\\')",
                 "import os; os.system('rm -rf /')",
                 "import os; os.system('Stop-Computer -Force')"):
        assert tools._BLOCKED_PY.search(code), code


def test_no_delete_profile_knows_the_powershell_alias():
    assert tools._DELETE_CMD.search("ri -Recurse -Force C:\\Users\\Bob")
    assert tools._DELETE_CMD.search("Remove-Item -Recurse -Force C:\\Users\\Bob")
    # ...but not the letters inside a filename
    assert tools._DELETE_CMD.search("tar -xzf ri.tar") is None


# --- view_images: the two modes no longer fight the schema --------------------
def test_view_images_requires_nothing():
    spec = next(t for t in tools.tool_specs(has_vision=True)
                if t["function"]["name"] == "view_images")
    # `paths` was required while folder mode needed it absent — under grammar
    # enforcement the model could never invoke folder mode correctly
    assert spec["function"]["parameters"]["required"] == []


# --- http_request: safe tier means no side effects ----------------------------
def test_http_request_refuses_state_changing_verbs():
    for m in ("DELETE", "PUT", "PATCH"):
        out = tools.run_tool(
            "http_request", {"url": "http://example.com/x", "method": m}, {})
        assert out.startswith("error") and "not allowed" in out, m


def test_http_request_method_is_enum():
    spec = next(t for t in tools.tool_specs()
                if t["function"]["name"] == "http_request")
    method = spec["function"]["parameters"]["properties"]["method"]
    assert method["enum"] == ["GET", "POST"]


# --- manage_plan: the verb is grammar-enforceable -----------------------------
def test_manage_plan_action_is_enum():
    spec = next(t for t in tools.tool_specs(has_run=True)
                if t["function"]["name"] == "manage_plan")
    action = spec["function"]["parameters"]["properties"]["action"]
    assert action["enum"] == ["add", "complete", "update", "list"]


# --- grep: case-insensitive search + actionable cap ---------------------------
def test_grep_ignore_case(tmp_path):
    (tmp_path / "a.txt").write_text("Hello World\n", encoding="utf-8")
    ctx = {"workspace": str(tmp_path)}
    assert "no matches" in tools.run_tool("grep", {"pattern": "hello"}, ctx)
    out = tools.run_tool(
        "grep", {"pattern": "hello", "ignore_case": True}, ctx)
    assert "a.txt:1" in out


def test_grep_cap_gives_narrowing_advice(tmp_path):
    (tmp_path / "big.txt").write_text("match\n" * 150, encoding="utf-8")
    out = tools.run_tool("grep", {"pattern": "match"},
                         {"workspace": str(tmp_path)})
    assert "narrow the pattern" in out


# --- remember/recall: no silent replacement, no unbounded dump ----------------
def test_remember_reports_replacement():
    assert tools.run_tool(
        "remember", {"key": "k", "value": "one"}, {}) == "remembered 'k'"
    out = tools.run_tool("remember", {"key": "k", "value": "two"}, {})
    assert "REPLACED" in out and "one" in out
    # unchanged value: no scary warning
    out2 = tools.run_tool("remember", {"key": "k", "value": "two"}, {})
    assert "REPLACED" not in out2


def test_recall_clips_unbounded_dump():
    for i in range(120):
        tools.run_tool("remember", {"key": f"k{i}", "value": "v" * 80}, {})
    out = tools.run_tool("recall", {}, {})
    assert len(out) < 6000
    assert "clipped" in out and "key" in out    # names the recovery path


# --- read_file: big files get a recovery path, not a dead end -----------------
def test_read_file_large_offers_paging(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("line\n" * 120_000, encoding="utf-8")     # ~600 KB
    ctx = {"workspace": str(tmp_path)}
    out = tools.run_tool("read_file", {"path": "big.txt"}, ctx)
    assert out.startswith("error") and "offset" in out       # tells it HOW
    paged = tools.run_tool(
        "read_file", {"path": "big.txt", "offset": 1, "limit": 5}, ctx)
    assert not paged.startswith("error") and "line" in paged


# --- fit math covers every KV type the UI offers ------------------------------
def test_cache_bytes_covers_all_offered_kv_types():
    from rigma.resolve import CACHE_BYTES
    from rigma.server_ops import KV_CACHE_TYPES
    for t in KV_CACHE_TYPES:
        assert t in CACHE_BYTES, f"{t} offered but never re-fitted"


# --- calibration never crowns q4_0 KV on a tools-capable model ----------------
def test_sweep_filters_q4_kv_for_tools_models(monkeypatch, tmp_path):
    class _FakeSrv:
        def stop(self):
            pass

    monkeypatch.setattr(bench, "launch_server", lambda *a, **k: _FakeSrv())
    monkeypatch.setattr(bench, "run_bench", lambda port, **k: bench.BenchResult(
        pp_tps=10, tg_tps=50, prompt_tokens=8, gen_tokens=8))
    monkeypatch.setattr(bench, "sweep_configs", lambda base, moe, caps=(): [
        ("baseline", {}),
        ("kv-q4", {"cache_type_k": "q4_0", "cache_type_v": "q4_0"})])
    # unknown slug -> capabilities unknown -> assume tools, protect quality
    plan = RunPlan(model_slug="unknown-model",
                   gguf=GgufFile(repo="r", file="f", bytes=1, quant="Q4"),
                   backend="vulkan", flags=ComboFlags(ctx=8192),
                   origin="calculator")
    rows = bench.run_sweep(plan, tmp_path / "srv.exe", tmp_path / "m.gguf",
                           port=11601)
    assert all(r["label"] != "kv-q4" for r in rows)


def test_tools_capable_defaults_true_for_unknown():
    assert bench._tools_capable("definitely-not-a-real-slug") is True
