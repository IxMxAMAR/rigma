"""Phase 5: MCP host — a real stdio JSON-RPC round-trip against a scripted
MCP server (a python one-liner file), plus gating and failure modes."""
import json
import sys
import textwrap

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
                     "text": {"type": "string"}}, "required": ["text"]}}]}}
        elif m == "tools/call":
            args = msg["params"].get("arguments") or {}
            if msg["params"]["name"] == "shout":
                out = {"jsonrpc": "2.0", "id": mid, "result": {
                    "content": [{"type": "text",
                                 "text": str(args.get("text", "")).upper()}]}}
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
