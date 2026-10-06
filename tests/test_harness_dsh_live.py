"""A REAL DeepSeek Harness turn, against a fake OpenAI-compatible server.

Skipped unless the checkout is on this machine. When it IS, this is the only
test that proves the seam is real rather than plausible: Rigma's adapter spawns
the runner, the runner imports the SDK, the SDK drives the `dsh` CLI, the CLI
POSTs to an OpenAI-compatible endpoint, and the reply comes back through the
translation. Every layer is the shipping one — only the model is fake.

It also pins the things that silently break the wire:
  * the request must land on `/v1/chat/completions`, not `/v1/messages`. The
    stock `sdk-minimal` profile mounts a Messages-only DeepSeek adapter, so this
    is what proves the generated patch took. It used to prove that by setting
    `protocol: chat-completions` on that adapter; DSH 0.2.0-rc.2 REMOVED that key
    and now throws on it at mount, so the generated patch instead disables that
    row and declares a pi-ai route with `api: openai-completions`. The assertion
    is unchanged because the FACT is unchanged.
  * `max_tokens` must be Rigma's, not the SDK's 256,000 default, which a 32K
    local server rejects outright.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from rigma import harness_dsh

FAKE = Path(__file__).parent / "fake_oai_server.py"

# MARKED `hardware`, AND THAT IS NOT A FORMALITY. The guard below OPENS this module
# when DSH is installed — and DSH is installed on the development machine — so without a
# marker `pytest -m "not hardware"` (the documented command, and what CI runs) collected
# two tests that spawn the real `dsh` CLI and run a REAL agent turn. Only the model is
# fake; the turn is not. The standing order for this work forbids live harness turns, and
# the guard alone was quietly running them.
#
# `hardware` is this project's existing marker for "needs the real thing", and every
# documented test command already excludes it. So the live turn now happens only when it
# is asked for by name:  pytest tests/test_harness_dsh_live.py -m hardware
pytestmark = [
    pytest.mark.hardware,
    pytest.mark.skipif(
        not harness_dsh.available(),
        reason="no DeepSeek Harness checkout on this machine "
               "(set RIGMA_DSH_HOME)"),
]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def fake_engine(tmp_path):
    """The fake model server, plus the log of what it was asked for.

    Also yields the `--dump` path, because the log line carries the tool COUNT
    and not the tool NAMES: proving the capability patch and Rigma's MCP server
    reached DSH needs the whole body.
    """
    port = _free_port()
    log = tmp_path / "requests.jsonl"
    dump = tmp_path / "bodies.jsonl"
    proc = subprocess.Popen(
        [sys.executable, str(FAKE), "--port", str(port), "--log", str(log),
         "--dump", str(dump)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(80):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        yield port, log, dump
    finally:
        proc.terminate()


def test_a_real_dsh_turn_answers_through_the_adapter(fake_engine, tmp_path):
    port, _log, _dump = fake_engine
    events = list(harness_dsh.drive_turn(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="local-test",
        prompt="Reply with exactly: hello from dsh",
        system_prompt="You are a terse test agent.",
        session_id=f"live-{uuid.uuid4().hex[:12]}",
        cwd=str(tmp_path),
        max_tokens=256, context_window=8192, timeout=240.0))

    assert not [e for e in events if e.kind == "error"], \
        [e.text for e in events if e.kind == "error"]
    said = "".join(e.text for e in events if e.kind == "text")
    assert said == "hello from dsh", events


def test_the_turn_asks_rigmas_endpoint_the_openai_way(fake_engine, tmp_path):
    """The patch, the auth header, the token ceiling and the roster, on the wire."""
    port, log, dump = fake_engine
    # `context_window` IS 32K HERE ON PURPOSE, and it is not decoration. DSH
    # clamps the output cap to the room left in the window, so the ceiling can
    # only be asserted where the prompt FITS: measured, the capability patch's
    # roster makes the body ~27KB (~6.7K tokens), and against the 8K window this
    # test used to pass the harness sends `max_tokens: 1` — honest ("no room
    # left"), but it proves nothing about Rigma's cap. At 32K Rigma's 256 is what
    # reaches the wire, which is the fact worth pinning. The 8K window is still
    # exercised by the turn test above.
    list(harness_dsh.drive_turn(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="local-test", prompt="hi",
        session_id=f"live-{uuid.uuid4().hex[:12]}",
        cwd=str(tmp_path), max_tokens=256, context_window=32768,
        timeout=240.0))

    rows = [json.loads(line) for line in
            log.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows, "DSH never called the model server"
    first = rows[0]
    # `messages` would mean the Anthropic default survived and the patch missed
    assert first["path"] == "/v1/chat/completions", first
    assert first["auth"] == "Bearer local", first
    assert first["stream"] is True, first
    assert first["model"] == "local-test", first
    # the SDK default is 256000; a 32K server would reject that outright
    assert first["max_tokens"] == 256, first
    # THE ROSTER IS THE POINT, NOT ITS SIZE. This used to assert
    # `first["tools"] <= 2` — a COUNT, which the log line carries — reading
    # "sdk-minimal advertises a deliberately tiny roster: one tool on Windows".
    # That bound was never true once `drive_turn` started always applying the
    # capability patch (`harness_dsh.capability_patch`, passed as
    # `capability_path`): the patch INSERTS ~11 tool rows on top of
    # sdk-minimal's own, so the bare profile's roster is not what a turn ever
    # sees. Measured against DSH 0.2.0-rc.2: 25 tools. A count would only pin
    # today's number, so this asserts the NAMES that carry the two facts the
    # patches exist to establish — the capability bridge reached DSH, and
    # Rigma's own MCP server did too.
    body = json.loads(
        dump.read_text(encoding="utf-8").splitlines()[0])["body"]
    names = {t["function"]["name"] for t in body["tools"]}
    # capability patch: goal/todo/skill/subagent/workflow/fs
    # (src/rigma/data/dsh/agent-capabilities.patch.yml)
    assert {"create_goal", "todo_write", "skill", "subagent", "workflow",
            "glob", "grep", "read", "write"} <= names, sorted(names)
    # Rigma's MCP server, via the generated MCP patch
    # (harness_dsh.mcp_patch_file)
    assert {"mcp__rigma__recall", "mcp__rigma__remember",
            "mcp__rigma__undo_last_change"} <= names, sorted(names)
