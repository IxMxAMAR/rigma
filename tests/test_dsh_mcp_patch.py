"""R6-MCP-INSERT: the generated MCP patch was silently rejected by DSH.

THE DEFECT, and it is the worst shape a defect can take. `mcp_patch_file` wrote a BARE
row. A DSH patch list is a list of OPERATIONS, and a bare row is read as a REPLACE — so
DSH answered

    dsh: [<tmp>\\rigma-dsh-mcp.yaml] patch: entry "mcp-rigma" not found

dropped the whole overlay, and **exited 0**. Rigma's own MCP tools therefore never
reached a DSH turn, while `harness.py`'s capability menu said they were mounted. Nothing
failed; nothing was logged above debug; the tools were simply absent.

WHY IT SURVIVED. The file had been checked as YAML (it parsed) and as a string (the rows
were there), and the MCP server's own handshake was verified by hand — none of which is
the claim that DSH ACCEPTS the overlay. The check that settles it needs no model:

    dsh --profile sdk-minimal --dump-config --patch <file>

which composes the profile tree and exits. `--dump-config` is what found this, and these
tests pin the two things it established: the `insert:` key is present, and the generated
YAML actually parses as an insert operation rather than as a replace.
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import tempfile

import yaml

from rigma import harness_dsh

# `mcp_server.offered` gates the file, so a machine with nothing to offer gets "". These
# tests exercise the SHAPE, which is what was wrong, and skip honestly when the gate is
# closed rather than asserting on an empty string.
def _generated(tmp_path) -> str:
    return harness_dsh.mcp_patch_file(tmp_path, cwd=str(tmp_path))


def test_the_generated_patch_is_an_INSERT_not_a_bare_row(tmp_path):
    """The defect in one assertion. A bare row is read as a REPLACE, and DSH then drops
    the overlay with a warning and exit 0."""
    path = _generated(tmp_path)
    if not path:
        return  # nothing to offer on this machine; the shape tests below still apply
    text = pathlib.Path(path).read_text(encoding="utf-8")
    assert "- insert:" in text, (
        "without `insert:` DSH reads the row as a REPLACE, cannot find `mcp-rigma`, and "
        "silently drops the overlay — the tools never reach a DSH turn")


def test_it_parses_as_one_insert_operation(tmp_path):
    """Structural rather than textual: the top level must be a list of operations whose
    first key is `insert`, which is what the shipped capability patch does."""
    path = _generated(tmp_path)
    if not path:
        return
    doc = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
    assert isinstance(doc, list), doc
    assert len(doc) == 1, doc
    assert list(doc[0]) == ["insert"], doc
    rows = doc[0]["insert"]
    assert [r["id"] for r in rows] == ["mcp-rigma"], rows


def test_the_row_names_the_client_and_the_server_prefix(tmp_path):
    """`serverName` is what becomes the `mcp__<server>__<tool>` prefix, so it is part of
    the tool names the model sees rather than a label."""
    path = _generated(tmp_path)
    if not path:
        return
    row = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))[0]["insert"][0]
    assert row["name"] == "@deepseek-ai/dsh-mcp-client"
    assert row["config"]["serverName"] == "rigma"
    assert row["config"]["transport"] == "stdio"


def test_the_command_is_an_absolute_interpreter(tmp_path):
    """A bare `python` may not be the interpreter that has Rigma installed, so the path
    has to be absolute and is why this file is generated rather than shipped."""
    path = _generated(tmp_path)
    if not path:
        return
    row = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))[0]["insert"][0]
    cmd = row["config"]["command"]
    assert pathlib.Path(cmd).is_absolute(), cmd
    assert pathlib.Path(cmd).is_file(), cmd
    assert row["config"]["args"] == ["-m", "rigma.mcp_server"]


def test_the_env_carries_both_variables_the_server_needs(tmp_path):
    """`dsh-mcp-client` merges this over a SCRUBBED environment, so relying on the
    ambient one would work on this machine and fail on a clean one.

    `RIGMA_MCP_ALLOW_CODE` is not optional: the server is pessimistic and
    `undo_last_change` does not exist without it. `RIGMA_MCP_WORKSPACE` is passed rather
    than guessed, because a tool silently operating on the wrong directory is worse than
    one that refused.
    """
    path = _generated(tmp_path)
    if not path:
        return
    row = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))[0]["insert"][0]
    env = row["config"]["env"]
    assert env["RIGMA_MCP_ALLOW_CODE"] == "1"
    assert env["RIGMA_MCP_WORKSPACE"] == str(tmp_path)


def test_the_composed_tree_contains_the_row():
    """The end-to-end check, run only where the DSH CLI exists.

    This is the assertion that would have caught the defect at the time: it composes the
    real profile with the real patch and asks whether DSH kept the row. It boots NO model
    — `--dump-config` prints the tree and exits — so it does not touch the standing order.
    """
    cli = harness_dsh.dsh_bin()
    if cli is None or not pathlib.Path(cli).is_file():
        return  # no DSH on this machine; the shape tests above still apply
    caps = harness_dsh.capability_patch()
    with tempfile.TemporaryDirectory(prefix="rigma-mcp-test-") as tmp:
        mcp = harness_dsh.mcp_patch_file(tmp, cwd=tmp)
        if not mcp:
            return
        argv = [str(cli), "--profile", "sdk-minimal", "--dump-config",
                "--patch", str(caps), "--patch", str(mcp)]
        env = dict(os.environ)
        env["DSH_HOME"] = str(harness_dsh.data_home())
        r = subprocess.run(argv, capture_output=True, text=True, timeout=300, env=env)
    assert r.returncode == 0, r.stderr
    # A rejected overlay is the failure mode, and DSH reports it on STDERR while still
    # exiting 0 — so the return code alone is not enough to call this a pass.
    assert "not found" not in r.stderr, r.stderr
    assert "mcp-rigma" in r.stdout, (
        "DSH composed the profile without Rigma's MCP row")
    assert "dsh-mcp-client" in r.stdout
