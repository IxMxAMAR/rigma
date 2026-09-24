"""R3-8: the run-progress exemption outranked every credential rule.

`_credential_path_reason` returned "" for ANY file named `progress.md`/
`progress.txt` before it looked at anything else. The exemption exists because
the run loop hands the model its own progress log BY NAME and that log lives
under Rigma's state dir — so it needs to punch through the STATE-DIR rule, and
nothing else. As written it also punched through the credential-file,
credential-directory and browser-profile rules, making `~/.ssh/progress.md` (or
a file of that name in `.aws`, or in a Chrome profile) readable.
"""
import pytest

from rigma import tools


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    return tmp_path


@pytest.mark.parametrize("d", [".ssh", ".aws", ".gnupg"])
def test_a_progress_named_file_in_a_credential_dir_is_refused(home, d):
    cred = home / d
    cred.mkdir(parents=True)
    (cred / "progress.md").write_text("PRIVATE KEY MATERIAL\n", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": str(cred / "progress.md")},
                         {"workspace": str(home), "allow_absolute_reads": True})
    assert out.startswith("error"), out
    assert "PRIVATE KEY" not in out


def test_the_run_log_exemption_still_works_inside_the_state_dir(home):
    """The reason the exemption exists: a workspace that IS the state dir."""
    from rigma.runtime import rigma_home
    ws = rigma_home() / "runs" / "r1"
    ws.mkdir(parents=True)
    (ws / "progress.md").write_text("step 1 done\n", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": "progress.md"},
                         {"workspace": str(ws)})
    assert out.strip() == "step 1 done"


def test_an_ordinary_workspace_progress_log_still_reads(home):
    ws = home / "project"
    ws.mkdir()
    (ws / "progress.md").write_text("hello\n", encoding="utf-8")
    out = tools.run_tool("read_file", {"path": "progress.md"},
                         {"workspace": str(ws)})
    assert out.strip() == "hello"
