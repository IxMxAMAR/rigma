"""Rigma as an MCP HOST: consuming other people's stdio MCP servers.

The mirror of `tests/test_mcp_server.py`, which covers Rigma being one. Three
fixes live here, each with the failure it exists to stop:

* **The launcher is resolved through PATH/PATHEXT.** `"command": "npx"` is the
  block every MCP README and Claude Desktop config writes, and it is a
  `FileNotFoundError` on Windows because `subprocess` goes to CreateProcess,
  which does not consult PATHEXT. Node-based clients work because they spawn
  through a shell. That is a compatibility bug in exactly the "same MCP in
  Claude and in Rigma" path.

* **A non-text result becomes a file on disk.** A ComfyUI-style server that
  generates an image answered `content[0] = {"type": "image", "data": ...}` and
  the model was handed the literal string `[image]` — the whole point of
  connecting the server, replaced by six characters.

* **A changed or deleted server is reconciled.** `mcp.json` is re-read every
  turn, but a running server is compared against the spec it was STARTED under,
  so editing one server must restart that one server and deleting one must stop
  it. A restart throws away the server's in-process state, so it must happen
  only when the spec genuinely differs — never on every turn.
"""
import base64
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time

import pytest

from rigma import mcp_client, tools

# A real (if tiny) MCP server over stdio: the handshake, tools/list, and tools
# that answer each content shape the host has to render. `whoami` reports the
# last argv element, which is how a test proves WHICH spec a process is running.
FAKE_SERVER = textwrap.dedent("""
    import base64, json, sys
    PNG = base64.b64encode(b"\\x89PNG\\r\\n\\x1a\\n" + bytes(range(64))).decode()
    TOOLS = [
        {"name": "shout", "description": "Uppercase the input",
         "inputSchema": {"type": "object", "properties": {
             "text": {"type": "string"}}, "required": ["text"]}},
        {"name": "whoami", "description": "Which spec is this process running",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "png", "description": "A generated image",
         "inputSchema": {"type": "object", "properties": {}}},
        {"name": "blob", "description": "A resource with a blob body",
         "inputSchema": {"type": "object", "properties": {}}}]
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        m, mid = msg.get("method"), msg.get("id")
        if m == "initialize":
            out = {"jsonrpc": "2.0", "id": mid, "result": {
                "protocolVersion": msg["params"]["protocolVersion"],
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "fake", "version": "1.0"}}}
        elif m == "tools/list":
            out = {"jsonrpc": "2.0", "id": mid, "result": {"tools": TOOLS}}
        elif m == "tools/call":
            name = msg["params"]["name"]
            args = msg["params"].get("arguments") or {}
            if name == "shout":
                content = [{"type": "text",
                            "text": str(args.get("text", "")).upper()}]
            elif name == "whoami":
                content = [{"type": "text", "text": " ".join(sys.argv[1:])}]
            elif name == "png":
                content = [{"type": "text", "text": "rendered"},
                           {"type": "image", "data": PNG,
                            "mimeType": "image/png"}]
            elif name == "blob":
                content = [{"type": "resource", "resource": {
                    "uri": "file:///tmp/out.png", "mimeType": "image/png",
                    "blob": PNG}}]
            else:
                content = []
            out = {"jsonrpc": "2.0", "id": mid, "result": {"content": content}}
        elif mid is None:
            continue          # notifications need no reply
        else:
            out = {"jsonrpc": "2.0", "id": mid,
                   "error": {"code": -32601, "message": "unknown method"}}
        sys.stdout.write(json.dumps(out) + "\\n")
        sys.stdout.flush()
""")


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    # each test gets a fresh manager: the singleton caches the config
    mcp_client._manager = None
    yield tmp_path
    if mcp_client._manager is not None:
        mcp_client._manager.stop_all()
        mcp_client._manager = None


def _write_config(tmp_path, servers):
    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}), encoding="utf-8")


def _write_script(tmp_path):
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return script


def _cfg(script, tag):
    """A spec for the fake server, tagged so a restart is observable."""
    return {"command": sys.executable, "args": ["-u", str(script), tag]}


def _wait_dead(proc, timeout=15.0):
    end = time.monotonic() + timeout
    while proc.poll() is None and time.monotonic() < end:
        time.sleep(0.05)
    return proc.poll() is not None


# --- 1. the launcher is resolved, not passed through -------------------------

class _FakeStdin:
    def write(self, s):
        pass

    def flush(self):
        pass


class _StartProc:
    pid = 4242
    stdin = _FakeStdin()
    stdout = None

    def poll(self):
        return None


class _FakeThread:
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


def _stub_start(monkeypatch):
    """Capture the argv `start()` hands to Popen, without spawning anything."""
    recorded = {}

    def fake_popen(cmd, **kw):
        recorded["cmd"] = cmd
        return _StartProc()

    monkeypatch.setattr(mcp_client.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(mcp_client.threading, "Thread", _FakeThread)
    return recorded


def test_a_bare_command_is_resolved_through_path_before_spawning(monkeypatch):
    """The measured Windows bug, at the seam where it is fixed.

    `subprocess.Popen(["npx", ...])` reaches CreateProcess, which resolves only
    an `.exe` (it appends `.exe`, nothing else). `npx` is a `.cmd` shim, so the
    launch fails — while `shutil.which`, which consults PATHEXT, finds
    `npx.CMD`. The config is not rewritten and `npx.cmd` is not required: the
    bare name the user wrote is what gets resolved.
    """
    recorded = _stub_start(monkeypatch)
    seen = []

    def fake_which(name, path=None):
        seen.append((name, path))
        return r"C:\tools\npx.CMD" if name == "npx" else None

    monkeypatch.setattr(mcp_client.shutil, "which", fake_which)
    srv = mcp_client.McpServer("files", {"command": "npx", "args": ["-y", "pkg"]})
    monkeypatch.setattr(srv, "_send", lambda *a, **k: {})
    srv.start()
    assert recorded["cmd"] == [r"C:\tools\npx.CMD", "-y", "pkg"]
    # resolved against the PATH the CHILD will run with, because CreateProcess
    # searches the parent's PATH and never the `env` block it hands over
    assert seen == [("npx", os.environ.get("PATH"))]


def test_the_specs_own_path_is_where_the_launcher_is_looked_for(monkeypatch):
    """A spec that uses `env.PATH` to reach a privately installed launcher.

    CreateProcess searches the CALLING process's PATH, so resolving against
    Rigma's own PATH and then launching with the spec's would find nothing on
    either side.
    """
    recorded = _stub_start(monkeypatch)
    seen = []
    monkeypatch.setattr(mcp_client.shutil, "which",
                        lambda name, path=None: seen.append(path) or None)
    srv = mcp_client.McpServer("private", {
        "command": "mytool", "args": [],
        "env": {"PATH": r"C:\my\tools", "TOKEN": "x"}})
    monkeypatch.setattr(srv, "_send", lambda *a, **k: {})
    srv.start()
    assert seen == [r"C:\my\tools"]
    assert recorded["cmd"] == ["mytool"]      # unresolved -> raw, as written


def test_a_command_that_resolves_to_nothing_is_left_exactly_as_written(
        monkeypatch):
    """`shutil.which` returning None must not blank the command out.

    A user who wrote an absolute path, a `./script`, or something only their
    shell can resolve is not helped by Rigma "fixing" it to nothing. The raw
    value is passed through, and the failure names what they wrote.
    """
    recorded = _stub_start(monkeypatch)
    monkeypatch.setattr(mcp_client.shutil, "which", lambda n, path=None: None)
    srv = mcp_client.McpServer("odd", {"command": "./bin/serve", "args": []})
    monkeypatch.setattr(srv, "_send", lambda *a, **k: {})
    srv.start()
    assert recorded["cmd"] == ["./bin/serve"]


def test_a_launcher_that_cannot_be_found_names_itself_and_path(tmp_path):
    """The raw FileNotFoundError says "The system cannot find the file
    specified" and never names the config field — the one thing the user needs
    is which `command` was wrong, and that Rigma already looked on PATH."""
    srv = mcp_client.McpServer("ghost",
                               {"command": "rigma-no-such-launcher-xyz"})
    with pytest.raises(mcp_client.McpError) as ei:
        srv.start()
    msg = str(ei.value)
    assert "rigma-no-such-launcher-xyz" in msg
    assert "PATH" in msg
    assert srv.proc is None, "a failed launch must not leave a half-built server"


@pytest.mark.skipif(sys.platform != "win32",
                    reason="CreateProcess/PATHEXT resolution is Windows-only")
def test_a_cmd_shim_on_path_is_launched_from_its_bare_name(tmp_path,
                                                           monkeypatch):
    """End to end, with a real `.cmd` shim — the `npx` shape, exactly.

    This is the test that would have caught the bug: a `.cmd` reachable ONLY by
    its bare name, on PATH, run as an MCP server. The pre-fix code hands
    `["rigma-fake-mcp"]` to CreateProcess and gets FileNotFoundError; the fixed
    code resolves it through PATHEXT and the whole handshake works.
    """
    script = _write_script(tmp_path)
    shims = tmp_path / "shims"
    shims.mkdir()
    shim = shims / "rigma-fake-mcp.cmd"
    shim.write_text(f'@echo off\r\n"{sys.executable}" -u "{script}" %*\r\n',
                    encoding="utf-8")
    monkeypatch.setenv("PATH", str(shims) + os.pathsep + os.environ["PATH"])

    found = shutil.which("rigma-fake-mcp")
    assert found is not None, "the shim is not reachable by its bare name"
    assert os.path.basename(found).lower() == "rigma-fake-mcp.cmd"

    # The BEFORE half of the measurement, on this machine, for this shim.
    with pytest.raises(FileNotFoundError):
        subprocess.Popen(["rigma-fake-mcp"], stdin=subprocess.PIPE,
                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                         creationflags=subprocess.CREATE_NO_WINDOW)

    srv = mcp_client.McpServer("shim", {"command": "rigma-fake-mcp"})
    try:
        srv.start()
        assert [t["name"] for t in srv.tools][:1] == ["shout"]
        assert srv.call("shout", {"text": "it works"}) == "IT WORKS"
    finally:
        srv.stop()


# --- 2. non-text results become something the model can use ------------------

def _returning(monkeypatch, result, name="gen"):
    """A server whose tools/call answers `result`, without any process."""
    srv = mcp_client.McpServer(name, {"command": "x"})
    monkeypatch.setattr(srv, "_send", lambda *a, **k: result)
    return srv


PNG_BYTES = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


def _b64(payload: bytes) -> str:
    return base64.b64encode(payload).decode()


def test_an_image_result_becomes_a_path_the_model_can_read(tmp_path,
                                                           monkeypatch):
    """`[image]` is not a result.

    The old renderer kept `type == "text"` and replaced everything else with a
    marker, so a server whose entire value is the picture it generated gave the
    model a word. The bytes are written under $RIGMA_HOME and the PATH is what
    comes back, because a path is the one form of an image the model can act on.
    """
    result = {"content": [{"type": "text", "text": "rendered"},
                          {"type": "image", "data": _b64(PNG_BYTES),
                           "mimeType": "image/png"}]}
    out = _returning(monkeypatch, result).call("png", {})
    assert "[image]" not in out, out
    assert "rendered" in out, "the text block beside the image was lost"
    path = mcp_client.artifact_dir() / (hashlib.sha256(PNG_BYTES).hexdigest()
                                        + ".png")
    assert str(path) in out, out
    assert path.read_bytes() == PNG_BYTES, "the file is not the bytes sent"
    assert "file on disk" in out, "the model is not told it can read the file"


def test_the_same_image_twice_is_one_file(tmp_path, monkeypatch):
    """Content-addressed, so a server that re-renders the same picture every
    turn writes one file — and the path the model noted on turn 1 still points
    at the right bytes on turn 50, instead of at a file that was overwritten."""
    result = {"content": [{"type": "image", "data": _b64(PNG_BYTES),
                           "mimeType": "image/png"}]}
    srv = _returning(monkeypatch, result)
    first = srv.call("png", {})
    second = srv.call("png", {})
    assert first == second
    assert len(list(mcp_client.artifact_dir().iterdir())) == 1


def test_an_over_cap_payload_is_refused_and_never_written(tmp_path,
                                                          monkeypatch):
    """The cap is the price of turning a server's answer into writes.

    Checked BEFORE the write and never truncated onto disk: half a PNG at a
    content-addressed name would be read as a real image forever after, and the
    hash would claim the bytes were the ones the server sent.
    """
    monkeypatch.setattr(mcp_client, "_ARTIFACT_MAX", 64)
    big = b"x" * 65
    result = {"content": [{"type": "image", "data": _b64(big),
                           "mimeType": "image/png"}]}
    out = _returning(monkeypatch, result).call("png", {})
    assert "over the 64-byte cap" in out, out
    assert not mcp_client.artifact_dir().exists(), "an over-cap blob was written"


def test_the_cap_is_checked_against_the_decoded_size_not_the_base64(
        tmp_path, monkeypatch):
    """base64 is 4 characters per 3 bytes, so a cap applied to the ENCODED
    string would let 33% more bytes onto disk than it says."""
    monkeypatch.setattr(mcp_client, "_ARTIFACT_MAX", 100)
    raw = b"y" * 100                       # exactly at the cap: allowed
    out = _returning(monkeypatch, {"content": [
        {"type": "image", "data": _b64(raw), "mimeType": "image/png"}]}
    ).call("png", {})
    assert "saved to" in out, out
    assert len(_b64(raw)) > 100, "the encoded form is bigger than the cap"


def test_base64_that_is_not_base64_is_reported_not_raised(tmp_path,
                                                          monkeypatch):
    """`run_tool`'s contract is that a tool never raises and reports failure as
    text the model can react to; a malformed artifact is that case."""
    result = {"content": [{"type": "image", "data": "!!!not base64!!!",
                           "mimeType": "image/png"}]}
    out = _returning(monkeypatch, result).call("png", {})
    assert "base64" in out, out
    assert not mcp_client.artifact_dir().exists()


def test_base64_broken_across_lines_still_decodes(tmp_path, monkeypatch):
    """Some servers pretty-print the payload. Whitespace is not corruption."""
    raw = b"z" * 90
    folded = "\n".join(_b64(raw)[i:i + 20] for i in range(0, 120, 20))
    out = _returning(monkeypatch, {"content": [
        {"type": "image", "data": folded, "mimeType": "image/png"}]}
    ).call("png", {})
    assert "saved to" in out, out
    path = mcp_client.artifact_dir() / (hashlib.sha256(raw).hexdigest() + ".png")
    assert path.read_bytes() == raw


def test_an_unknown_media_type_is_stored_as_bin_not_guessed(tmp_path,
                                                            monkeypatch):
    """The extension is a claim about the bytes, so a media type Rigma does not
    know must not be rounded to a familiar one."""
    raw = b"mystery"
    out = _returning(monkeypatch, {"content": [
        {"type": "image", "data": _b64(raw),
         "mimeType": "image/x-made-up"}]}).call("png", {})
    path = mcp_client.artifact_dir() / (hashlib.sha256(raw).hexdigest() + ".bin")
    assert str(path) in out, out


def test_audio_takes_the_same_path_as_an_image(tmp_path, monkeypatch):
    """MCP gives `audio` the identical `data` + `mimeType` shape, so it is the
    same problem and gets the same answer rather than another marker."""
    raw = b"RIFF....WAVE"
    out = _returning(monkeypatch, {"content": [
        {"type": "audio", "data": _b64(raw), "mimeType": "audio/wav"}]}
    ).call("speak", {})
    path = mcp_client.artifact_dir() / (hashlib.sha256(raw).hexdigest() + ".wav")
    assert str(path) in out, out
    assert path.read_bytes() == raw


def test_a_resource_with_a_text_body_is_inlined_with_its_uri(tmp_path,
                                                             monkeypatch):
    """A text resource is already text: writing it to a file to hand back a path
    would be a detour the model has to pay for."""
    result = {"content": [{"type": "resource", "resource": {
        "uri": "file:///notes/todo.md", "mimeType": "text/markdown",
        "text": "- buy milk"}}]}
    out = _returning(monkeypatch, result).call("read", {})
    assert "- buy milk" in out
    assert "file:///notes/todo.md" in out, "the source of the text was dropped"
    assert not mcp_client.artifact_dir().exists()


def test_a_resource_with_a_blob_body_is_written_like_an_image(tmp_path,
                                                              monkeypatch):
    result = {"content": [{"type": "resource", "resource": {
        "uri": "file:///tmp/out.png", "mimeType": "image/png",
        "blob": _b64(PNG_BYTES)}}]}
    out = _returning(monkeypatch, result).call("read", {})
    path = mcp_client.artifact_dir() / (hashlib.sha256(PNG_BYTES).hexdigest()
                                        + ".png")
    assert str(path) in out, out
    assert path.read_bytes() == PNG_BYTES


def test_a_resource_link_is_passed_through_as_text(tmp_path, monkeypatch):
    """The host has no `resources/read`, so a link is only a URI — but a URI is
    something the model can act on, and a marker is not."""
    result = {"content": [{"type": "resource_link",
                           "uri": "file:///data/big.csv",
                           "name": "big.csv"}]}
    out = _returning(monkeypatch, result).call("list", {})
    assert "file:///data/big.csv" in out, out


def test_an_unknown_content_type_keeps_a_marker_rather_than_vanishing(
        tmp_path, monkeypatch):
    """Something unrecognised must still be visible: silently dropping a block
    makes a partial answer look complete."""
    result = {"content": [{"type": "text", "text": "a"},
                          {"type": "video", "data": "..."}]}
    out = _returning(monkeypatch, result).call("t", {})
    assert "[video]" in out, out


def test_a_non_object_result_is_reported_not_raised(tmp_path, monkeypatch):
    """The 09-2 trap one level down: `{"result": "ok"}` used to reach
    `result.get(...)` and surface as "'str' object has no attribute 'get'",
    which tells the model nothing about the server."""
    out = _returning(monkeypatch, "ok").call("t", {})
    assert out.startswith("error"), out
    assert "non-object" in out, out


def test_an_error_result_is_still_prefixed_and_rendered(tmp_path, monkeypatch):
    result = {"isError": True,
              "content": [{"type": "image", "data": _b64(PNG_BYTES),
                           "mimeType": "image/png"}]}
    out = _returning(monkeypatch, result).call("t", {})
    assert out.startswith("error: "), out
    assert "saved to" in out, out


# --- 3. a changed or deleted server takes effect -----------------------------

def test_a_changed_spec_restarts_that_server_and_only_that_server(tmp_path):
    """A restart DISCARDS in-process state — an indexed corpus, a loaded model,
    a browser session. The old code compared one key for the whole file and
    restarted EVERY server on any difference, so adding a second server threw
    away the first one's state for an edit that never mentioned it."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one"), "b": _cfg(script, "two")})
    mgr = mcp_client.manager()
    mgr._ensure()
    a, b = mgr._servers["a"], mgr._servers["b"]
    assert a.call("whoami", {}) == "one"

    _write_config(tmp_path, {"a": _cfg(script, "one"),
                             "b": _cfg(script, "two-edited")})
    mgr._ensure()
    assert mgr._servers["a"] is a, "a server nobody edited was restarted"
    assert mgr._servers["b"] is not b, "the edited server kept running"
    assert mgr._servers["b"].call("whoami", {}) == "two-edited"
    assert a.call("whoami", {}) == "one", "the untouched server was disturbed"


def test_an_unchanged_config_restarts_nothing(tmp_path):
    """The honest half of the trade: reconciliation must be free on the turns
    where nothing changed, or every turn would restart every server and no
    server could ever hold state."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one")})
    mgr = mcp_client.manager()
    mgr._ensure()
    a = mgr._servers["a"]
    for _ in range(3):
        mgr._ensure()
    assert mgr._servers["a"] is a
    assert a.call("whoami", {}) == "one"


def test_env_key_order_is_not_a_change(tmp_path):
    """`env` is a mapping: the same three variables written in a different order
    are the same server, and must not cost it its state."""
    script = _write_script(tmp_path)
    base = _cfg(script, "one")
    _write_config(tmp_path, {"a": {**base, "env": {"X": "1", "Y": "2"}}})
    mgr = mcp_client.manager()
    mgr._ensure()
    a = mgr._servers["a"]
    _write_config(tmp_path, {"a": {**base, "env": {"Y": "2", "X": "1"}}})
    mgr._ensure()
    assert mgr._servers["a"] is a


def test_a_removed_server_is_stopped_not_left_running(tmp_path):
    """Nothing else will ever stop it, and the model cannot discover that the
    server it is still calling is no longer configured."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one"), "b": _cfg(script, "two")})
    mgr = mcp_client.manager()
    mgr._ensure()
    b_proc = mgr._servers["b"].proc

    _write_config(tmp_path, {"a": _cfg(script, "one")})
    mgr._ensure()
    assert "b" not in mgr._servers
    assert _wait_dead(b_proc), "the removed server kept running"
    assert mgr._servers["a"].call("whoami", {}) == "one"


def test_removing_the_last_server_stops_it_through_the_tool_roster(tmp_path):
    """The removal the whole-config restart could not see: with an EMPTY config
    `tools.tool_specs` skipped `_ensure()` entirely ("no config -> zero
    overhead"), so the last server deleted from mcp.json stayed alive until
    Rigma restarted."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one")})
    names = {s["function"]["name"] for s in tools.tool_specs(allow_code=True)}
    assert any(n.startswith("mcp__") for n in names), names
    proc = mcp_client.manager()._servers["a"].proc

    _write_config(tmp_path, {})
    names = {s["function"]["name"] for s in tools.tool_specs(allow_code=True)}
    assert not any(n.startswith("mcp__") for n in names), names
    assert _wait_dead(proc), "the last configured server kept running"


def test_a_server_known_dead_under_this_config_is_not_respawned(tmp_path,
                                                                monkeypatch):
    """A failed server is retried when the config changes, not on every turn —
    otherwise every turn pays a process spawn that is known to fail."""
    _write_config(tmp_path, {"broken": {"command": sys.executable,
                                        "args": ["-c", "import sys; sys.exit(3)"]}})
    mgr = mcp_client.manager()
    mgr._ensure()
    assert "broken" in mgr._failed

    started = []
    real_start = mcp_client.McpServer.start

    def counting_start(self):
        started.append(self.name)
        return real_start(self)

    monkeypatch.setattr(mcp_client.McpServer, "start", counting_start)
    mgr._ensure()
    assert started == [], "a server known dead under this config was respawned"
    assert "broken" in mgr._failed


def test_editing_a_failed_server_retries_it(tmp_path):
    """The other half: a config edit is the only evidence that anything the
    failure rested on has changed, so it must clear the remembered death."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"flaky": {"command": sys.executable,
                                       "args": ["-c", "import sys; sys.exit(3)"]}})
    mgr = mcp_client.manager()
    mgr._ensure()
    assert "flaky" in mgr._failed

    _write_config(tmp_path, {"flaky": _cfg(script, "fixed")})
    mgr._ensure()
    assert "flaky" in mgr._servers, "the fixed server was never retried"
    assert mgr._failed == {}
    assert mgr._servers["flaky"].call("whoami", {}) == "fixed"


def test_a_dead_reader_is_stopped_without_touching_its_neighbours(tmp_path):
    """A server whose reader exited can never answer again (AUDIT F55), so it is
    stopped and remembered dead — but only it. The old code stopped and rebuilt
    EVERY server because of one dead reader, which is the same over-reaction as
    for a config edit, and just as destructive."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one"), "b": _cfg(script, "two")})
    mgr = mcp_client.manager()
    mgr._ensure()
    a, b = mgr._servers["a"], mgr._servers["b"]
    b_proc = b.proc
    b.reader_error = "simulated: the reader exited"

    mgr._ensure()
    assert mgr._servers["a"] is a, "the healthy neighbour was restarted"
    assert "b" not in mgr._servers
    assert "simulated" in mgr._failed["b"]
    assert _wait_dead(b_proc), "the dead server's process was left running"

    # ...and the death is not permanent: the config changing is the evidence
    # that whatever it died of may no longer hold, so it is retried.
    _write_config(tmp_path, {"a": _cfg(script, "one"), "b": _cfg(script, "two"),
                             "c": _cfg(script, "three")})
    mgr._ensure()
    assert mgr._servers["b"].call("whoami", {}) == "two"
    assert "b" not in mgr._failed


def test_a_new_server_is_picked_up_without_disturbing_the_running_ones(
        tmp_path):
    """The case the task starts from: a model installs a server mid-session."""
    script = _write_script(tmp_path)
    _write_config(tmp_path, {"a": _cfg(script, "one")})
    mgr = mcp_client.manager()
    mgr._ensure()
    a = mgr._servers["a"]

    _write_config(tmp_path, {"a": _cfg(script, "one"), "b": _cfg(script, "two")})
    mgr._ensure()
    assert mgr._servers["a"] is a
    assert mgr._servers["b"].call("whoami", {}) == "two"
