"""A REAL DeepSeek Harness turn, against a fake OpenAI-compatible server.

Skipped unless the checkout is on this machine. When it IS, this is the only
test that proves the seam is real rather than plausible: Rigma's adapter spawns
the runner, the runner imports the SDK, the SDK drives the `dsh` CLI, the CLI
POSTs to an OpenAI-compatible endpoint, and the reply comes back through the
translation. Every layer is the shipping one — only the model is fake.

It also pins the two things that silently break the wire, both of which the
design note originally got wrong:
  * the request must land on `/v1/chat/completions`, not `/v1/messages`. The
    stock `sdk-minimal` profile leaves `protocol` at its `messages` default,
    which is Anthropic-shaped, so this is what proves the generated patch took.
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

pytestmark = pytest.mark.skipif(
    not harness_dsh.available(),
    reason="no DeepSeek Harness checkout on this machine "
           "(set RIGMA_DSH_HOME)")


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def fake_engine(tmp_path):
    """The fake model server, plus the log of what it was asked for."""
    port = _free_port()
    log = tmp_path / "requests.jsonl"
    proc = subprocess.Popen(
        [sys.executable, str(FAKE), "--port", str(port), "--log", str(log)],
        stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    try:
        for _ in range(80):
            try:
                socket.create_connection(("127.0.0.1", port), 0.2).close()
                break
            except OSError:
                time.sleep(0.1)
        yield port, log
    finally:
        proc.terminate()


def test_a_real_dsh_turn_answers_through_the_adapter(fake_engine, tmp_path):
    port, log = fake_engine
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
    """The patch, the auth header and the token ceiling, on the wire."""
    port, log = fake_engine
    list(harness_dsh.drive_turn(
        base_url=f"http://127.0.0.1:{port}/v1",
        model="local-test", prompt="hi",
        session_id=f"live-{uuid.uuid4().hex[:12]}",
        cwd=str(tmp_path), max_tokens=256, context_window=8192,
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
    # sdk-minimal advertises a deliberately tiny roster: one tool on Windows
    assert first["tools"] <= 2, first
