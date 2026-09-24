"""R3-4: `run_shell` carries the payload `run_python` refuses.

`exec_decision` picks its wordlist from WHICH TOOL called it, so a Python
payload handed to the shell was judged by the cmd/PowerShell list only. On
win32 `run_shell` builds `powershell -Command <text>`, and the text can be
`python -c "import shutil; shutil.rmtree('/')"` — so the Python-specific rules
`_BLOCKED_PY`/`_DELETE_PY` (added because the shell wordlist cannot be applied
to Python) were bypassed by wrapping the same payload in a shell call, and
under `no-delete` an ordinary `python -c "import os; os.remove('x')"` passed a
profile whose promise is that deletion is disabled.

Only TEXT is judged here; nothing is executed (safety rule 7).
"""
import pytest

from rigma import tools

ALL = {"allow_code": True, "confirm_exec": True, "profile": "all"}
NO_DELETE = {**ALL, "profile": "no-delete"}

_RMTREE = "python -c \"import shutil; shutil.rmtree('/')\""
_RM_D = "python -c \"import shutil; shutil.rmtree('D:/')\""
_OS_REMOVE = "python -c \"import os; os.remove('x')\""
_SUBPROCESS = "python -c \"import subprocess; subprocess.run(['rm', '-rf', '/'])\""


@pytest.mark.parametrize("cmd", [_RMTREE, _RM_D, _SUBPROCESS])
def test_the_shell_cannot_carry_a_blocked_python_payload(cmd):
    allowed, why = tools.exec_decision(cmd, ALL)
    assert allowed is False, cmd
    assert "blocked" in why


@pytest.mark.parametrize("cmd", [_OS_REMOVE, _RMTREE,
                                 "py -c \"import shutil; shutil.rmtree('build')\""])
def test_no_delete_covers_the_python_payload_in_a_shell_call(cmd):
    allowed, why = tools.exec_decision(cmd, NO_DELETE)
    assert allowed is False, cmd
    # `rmtree('/')` is refused a step earlier, as destructive; either refusal
    # is a refusal, and neither is "allowed"
    assert "blocked" in why


def test_ordinary_shell_text_is_untouched():
    for cmd in ("echo hi", "python -m pytest tests -q",
                "git status --short", "ls -la",
                "python -c \"print('hello')\""):
        allowed, why = tools.exec_decision(cmd, ALL)
        assert allowed is True, (cmd, why)


def test_ordinary_python_is_untouched():
    for src in ("print(1)", "import os\nos.path.join('a', 'b')",
                "d = {}\nprint('{}'.format(1))"):
        allowed, why = tools.exec_decision(src, ALL, python_src=src)
        assert allowed is True, (src, why)


def test_the_spawn_path_applies_the_same_rules(monkeypatch):
    """The last line before a spawn must not disagree with the gate."""
    called = []
    monkeypatch.setattr(tools, "_launch_killable",
                        lambda *a, **k: called.append(a))
    out = tools._run_subprocess(["python", "-c", "1"], ALL, python_src=_RMTREE)
    assert out.startswith("error") and "blocked" in out
    assert called == []


def test_run_shell_refuses_it_without_spawning(monkeypatch):
    called = []
    monkeypatch.setattr(tools, "_launch_killable",
                        lambda *a, **k: called.append(a))
    out = tools.run_tool("run_shell", {"command": _OS_REMOVE}, NO_DELETE)
    assert out.startswith("error") and "deletion" in out, out
    assert called == []
