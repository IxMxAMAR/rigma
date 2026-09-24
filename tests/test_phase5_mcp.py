"""Phase 5: MCP host — a real stdio JSON-RPC round-trip against a scripted
MCP server (a python one-liner file), plus gating and failure modes."""
import json
import sys
import textwrap
import types

import pytest

from rigma import mcp_client, tools

FAKE_SERVER = textwrap.dedent("""
    import json, sys
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
            out = {"jsonrpc": "2.0", "id": mid, "result": {"tools": [
                {"name": "shout",
                 "description": "Uppercase the input text",
                 "inputSchema": {"type": "object", "properties": {
                     "text": {"type": "string"}}, "required": ["text"]}},
                {"name": "huge",
                 "description": "Return a very large single response",
                 "inputSchema": {"type": "object", "properties": {
                     "n": {"type": "integer"}}}},
                {"name": "silent",
                 "description": "Never answers",
                 "inputSchema": {"type": "object", "properties": {}}},
                {"name": "flood",
                 "description": "Streams one endless line, forever",
                 "inputSchema": {"type": "object", "properties": {}}}]}}
        elif m == "tools/call":
            args = msg["params"].get("arguments") or {}
            if msg["params"]["name"] == "shout":
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": str(args.get("text", "")).upper()}]}}
            elif msg["params"]["name"] == "huge":
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": "x" * int(args.get("n", 4000))}]}}
            elif msg["params"]["name"] == "silent":
                continue      # never answers: a caller that must be woken
            elif msg["params"]["name"] == "flood":
                # ONE ENDLESS LINE: a newline is never written, so a reader
                # that gives up is the only way this can end. (A finite
                # 100KB line only failed the reader if the drain budget
                # happened to run out first, which is a race.)
                sys.stdout.write('{"jsonrpc": "2.0", "id": %s, "result": '
                                 '{"content": [{"type": "text", "text": "'
                                 % mid)
                sys.stdout.flush()
                while True:
                    sys.stdout.write("x" * 8192)
                    sys.stdout.flush()
            else:
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "isError": True,
                    "content": [{"type": "text", "text": "no such tool"}]}}
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
    # each test gets a fresh manager (the singleton caches per config)
    mcp_client._manager = None
    yield tmp_path
    if mcp_client._manager is not None:
        mcp_client._manager.stop_all()
        mcp_client._manager = None


def _write_config(tmp_path, servers):
    (tmp_path / "mcp.json").write_text(
        json.dumps({"mcpServers": servers}), encoding="utf-8")


def _fake_server_config(tmp_path, name="fake"):
    script = tmp_path / "fake_mcp.py"
    script.write_text(FAKE_SERVER, encoding="utf-8")
    return {name: {"command": sys.executable,
                   "args": ["-u", str(script)]}}


def test_discovers_and_calls_a_real_stdio_server(tmp_path):
    _write_config(tmp_path, _fake_server_config(tmp_path))
    specs = tools.tool_specs(allow_code=True)
    names = {s["function"]["name"] for s in specs}
    assert "mcp__fake__shout" in names
    spec = next(s for s in specs
                if s["function"]["name"] == "mcp__fake__shout")
    assert "[fake]" in spec["function"]["description"]
    out = tools.run_tool("mcp__fake__shout", {"text": "hello rigma"},
                         {"allow_code": True})
    assert out == "HELLO RIGMA"


def test_mcp_error_results_are_prefixed(tmp_path):
    _write_config(tmp_path, _fake_server_config(tmp_path))
    out = tools.run_tool("mcp__fake__nonexistent", {},
                         {"allow_code": True})
    assert out.startswith("error")


def test_no_config_means_no_mcp_tools():
    names = {s["function"]["name"] for s in tools.tool_specs(allow_code=True)}
    assert not any(n.startswith("mcp__") for n in names)


def test_mcp_gated_like_code_and_profiles(tmp_path):
    _write_config(tmp_path, _fake_server_config(tmp_path))
    # no allow_code: not offered, not runnable
    base = {s["function"]["name"] for s in tools.tool_specs()}
    assert not any(n.startswith("mcp__") for n in base)
    assert "not enabled" in tools.run_tool("mcp__fake__shout",
                                           {"text": "x"}, {})
    # restrictive profiles exclude MCP wholesale
    for prof in ("no-network", "confined"):
        offered = {s["function"]["name"]
                   for s in tools.tool_specs(allow_code=True, profile=prof)}
        assert not any(n.startswith("mcp__") for n in offered), prof
        out = tools.run_tool("mcp__fake__shout", {"text": "x"},
                             {"allow_code": True, "profile": prof})
        assert "disabled" in out


def test_dead_server_reports_not_crashes(tmp_path):
    _write_config(tmp_path, {"broken": {
        "command": sys.executable,
        "args": ["-c", "import sys; sys.exit(3)"]}})
    # discovery survives the corpse; the failure is remembered and reported
    specs = tools.tool_specs(allow_code=True)
    assert isinstance(specs, list)
    out = tools.run_tool("mcp__broken__anything", {}, {"allow_code": True})
    assert out.startswith("error") and "unavailable" in out
    status = mcp_client.manager().status()
    assert "broken" in status["failed"]


def test_malformed_name_and_missing_server(tmp_path):
    _write_config(tmp_path, _fake_server_config(tmp_path))
    assert tools.run_tool("mcp__nosuchserver__t", {},
                          {"allow_code": True}).startswith("error")


# --- 09-2/09-3/09-8: a JSON value that is not an object is not a mapping ------

class _Lines:
    """A fake stdout: bounded readline, and iteration for the pre-fix path."""
    def __init__(self, lines):
        self._lines = list(lines)

    def readline(self, *a):
        return self._lines.pop(0) if self._lines else ""

    def __iter__(self):
        return self

    def __next__(self):
        if not self._lines:
            raise StopIteration
        return self._lines.pop(0)


def test_a_non_object_json_line_does_not_kill_the_reader():
    """json.loads may return a list, string, number or null, and `.get` then
    raises inside `_reader`'s try — which ends the reader for good and makes
    _ensure() tear down EVERY configured server (09-2)."""
    srv = mcp_client.McpServer("s", {"command": "x"})
    srv.proc = types.SimpleNamespace(
        stdout=_Lines(['[1, 2]\n', '123\n', '"hello"\n', 'null\n',
                       '{"jsonrpc": "2.0", "id": 1, "result": {}}\n']))
    srv._reader()
    assert srv.reader_error == ""


def test_an_unhashable_jsonrpc_id_is_skipped_not_fatal():
    """`{"id": [1]}` fails the same way via `self._replies.get([1])` → TypeError
    (09-2)."""
    srv = mcp_client.McpServer("s", {"command": "x"})
    srv._handle_line('{"jsonrpc": "2.0", "id": [1], "result": {}}')
    assert srv.reader_error == ""


def test_a_non_object_mcp_json_reads_as_empty(tmp_path):
    """A user edits ~/.rigma/mcp.json into a JSON array; `raw.get` then raised
    AttributeError, which is not in the caught tuple, so every MCP tool vanished
    and /api/mcp answered the Python error text (09-8)."""
    (tmp_path / "mcp.json").write_text("[]", encoding="utf-8")
    assert mcp_client.load_config() == {}
    (tmp_path / "mcp.json").write_text("null", encoding="utf-8")
    assert mcp_client.load_config() == {}


# --- 09-5: a server that accepts but never answers must be marked dead --------

class _FakeStdin:
    def write(self, s):
        pass

    def flush(self):
        pass


class _MuteProc:
    """A live child whose stdin swallows requests and whose stdout never
    answers. No process is started."""
    pid = 4242
    stdin = _FakeStdin()
    stdout = None

    def poll(self):
        return None


def _wedge(srv):
    for _ in range(mcp_client._WEDGE_AFTER):
        with pytest.raises(mcp_client.McpError):
            srv._send("tools/call", {"name": "t"}, timeout=0.01)


def test_a_server_that_accepts_but_never_answers_is_marked_wedged():
    """A timeout changed no state, so `reader_error` stayed empty and `_ensure`
    kept the mute process in `_servers` forever (09-5)."""
    srv = mcp_client.McpServer("mute", {"command": "x"})
    srv.proc = _MuteProc()
    _wedge(srv)
    assert srv.wedged is True
    assert srv.timeouts == mcp_client._WEDGE_AFTER
    assert "timed out" in srv.reader_error


def test_one_real_reply_clears_the_consecutive_timeout_count():
    """One slow call is not a dead server: the count has to be CONSECUTIVE."""
    import threading
    import time
    srv = mcp_client.McpServer("mute", {"command": "x"})
    srv.proc = _MuteProc()
    with pytest.raises(mcp_client.McpError):
        srv._send("m", {}, timeout=0.01)
    assert srv.timeouts == 1
    out = {}

    def call():
        out["r"] = srv._send("m", {}, timeout=5)

    t = threading.Thread(target=call)
    t.start()
    deadline = time.monotonic() + 5
    while not srv._replies and time.monotonic() < deadline:
        time.sleep(0.01)
    mid = next(iter(srv._replies))
    srv._replies[mid].put({"jsonrpc": "2.0", "id": mid, "result": "ok"})
    t.join(timeout=5)

    assert out.get("r") == "ok"
    assert srv.timeouts == 0 and srv.wedged is False


def test_a_wedged_server_is_rebuilt_on_the_next_ensure(monkeypatch):
    mgr = mcp_client.McpManager()
    srv = mcp_client.McpServer("mute", {"command": "x"})
    srv.proc = _MuteProc()
    _wedge(srv)
    mgr._servers = {"mute": srv}
    mgr._started = True
    mgr._cfg_key = json.dumps({"mute": {"command": "x"}}, sort_keys=True)
    monkeypatch.setattr(mcp_client, "load_config",
                        lambda: {"mute": {"command": "x"}})
    stopped = []
    monkeypatch.setattr(srv, "stop", lambda: stopped.append("old"))
    monkeypatch.setattr(
        mcp_client.McpServer, "start",
        lambda self: (_ for _ in ()).throw(mcp_client.McpError("down")))
    mgr._ensure()
    assert stopped == ["old"], "the wedged server must be stopped and rebuilt"
    assert "mute" in mgr._failed


def test_status_reports_the_wedged_server_and_its_count(monkeypatch):
    mgr = mcp_client.McpManager()
    srv = mcp_client.McpServer("mute", {"command": "x"})
    srv.proc = _MuteProc()
    _wedge(srv)
    mgr._servers = {"mute": srv}
    mgr._started = True
    mgr._cfg_key = json.dumps({"mute": {"command": "x"}}, sort_keys=True)
    monkeypatch.setattr(mcp_client, "load_config",
                        lambda: {"mute": {"command": "x"}})
    monkeypatch.setattr(mgr, "_ensure", lambda: None)
    st = mgr.status()
    assert st["wedged"] == {"mute": mcp_client._WEDGE_AFTER}
    assert "mute" in st["dead"]


# --- F54: shutdown must actually end the server -------------------------------
def test_stop_kills_the_server_and_releases_its_pipes(tmp_path):
    """`stop()` called terminate() and dropped the reference: no wait, no kill
    escalation, no pipe close — so a server that survived terminate could never
    be killed afterwards and a repeat stop_all() was a no-op, and the reader
    thread stayed blocked in readline() holding the stdout pipe (AUDIT F54)."""
    import time
    _write_config(tmp_path, _fake_server_config(tmp_path))
    mgr = mcp_client.manager()
    mgr._ensure()
    srv = mgr._servers["fake"]
    proc = srv.proc
    assert proc.poll() is None
    srv.stop()
    deadline = time.monotonic() + 10
    while proc.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert proc.poll() is not None, "the server survived stop()"
    assert srv.proc is None
    srv.stop()                    # a repeat is a harmless no-op, not a crash


def test_stop_all_is_idempotent_and_releases_every_server(tmp_path):
    _write_config(tmp_path, _fake_server_config(tmp_path))
    mgr = mcp_client.manager()
    mgr._ensure()
    procs = [s.proc for s in mgr._servers.values()]
    assert procs
    mgr.stop_all()
    mgr.stop_all()
    assert all(p.poll() is not None for p in procs)
    assert mgr._servers == {}


def test_a_waiting_caller_is_woken_when_the_server_is_stopped(tmp_path):
    """A caller blocked on a reply used to sit out the full 60s timeout."""
    import threading
    import time
    _write_config(tmp_path, _fake_server_config(tmp_path))
    mgr = mcp_client.manager()
    mgr._ensure()
    srv = mgr._servers["fake"]
    got = {}

    def waiter():
        try:
            srv._send("tools/call", {"name": "silent", "arguments": {}})
            got["out"] = "returned"
        except Exception as e:
            got["out"] = str(e)

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.5)               # the call is issued and now waiting
    srv.stop()
    t.join(timeout=10)
    assert not t.is_alive(), "the waiter was never woken"
    assert "no longer answering" in got.get("out", ""), got


# --- F55: one large result must not be buffered whole -------------------------
def test_an_oversize_frame_is_dropped_and_the_caller_told(tmp_path, monkeypatch):
    """The reader used an unbounded readline(), so one response was held in
    memory whole (twice, after parsing) inside the process that also holds the
    chat sessions (AUDIT F55)."""
    import time
    monkeypatch.setattr(mcp_client, "_FRAME_MAX", 2000)
    _write_config(tmp_path, _fake_server_config(tmp_path))
    mgr = mcp_client.manager()
    mgr._ensure()
    srv = mgr._servers["fake"]
    started = time.monotonic()
    out = tools.run_tool("mcp__fake__huge", {"n": 4000}, {"allow_code": True})
    elapsed = time.monotonic() - started
    assert elapsed < 30, f"the caller waited {elapsed:.1f}s for a dropped frame"
    assert out.startswith("error") and "exceeded" in out, out
    assert srv.oversize_frames == 1
    # framing stays in sync: the very next call is served normally
    assert tools.run_tool("mcp__fake__shout", {"text": "still here"},
                          {"allow_code": True}) == "STILL HERE"


def test_a_server_that_streams_one_endless_line_is_reported_dead(
        tmp_path, monkeypatch):
    """A reader that exited was invisible: proc.poll() still said alive, so
    every later call blocked a worker for the full timeout and _ensure() would
    not rebuild it (AUDIT F55)."""
    import time
    monkeypatch.setattr(mcp_client, "_FRAME_MAX", 2000)
    _write_config(tmp_path, _fake_server_config(tmp_path))
    mgr = mcp_client.manager()
    mgr._ensure()
    srv = mgr._servers["fake"]
    out = tools.run_tool("mcp__fake__flood", {}, {"allow_code": True})
    assert out.startswith("error"), out
    # The caller is woken by the oversize frame; the READER sets reader_error a
    # moment later, after it has given up on the rest of the line. Poll for it
    # rather than assuming the ordering — asserting immediately is what made
    # this fail only under full-suite load.
    end = time.monotonic() + 15
    while time.monotonic() < end and not srv.reader_error:
        time.sleep(0.05)
    assert srv.reader_error and "exceeded" in srv.reader_error, srv.reader_error
    # and the next call reports it instead of blocking for a minute
    second = tools.run_tool("mcp__fake__shout", {"text": "x"},
                            {"allow_code": True})
    assert second.startswith("error") and "exceeded" in second, second
    assert "fake" in mgr.status()["failed"]
