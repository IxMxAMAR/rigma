"""Rigma as an MCP tool PROVIDER.

The mirror of `tests/test_mcp_client.py`: that one covers Rigma consuming other
people's MCP servers, this one covers Rigma being one. The protocol shapes are
pinned against what `mcp_client.McpServer` actually sends and expects, because a
server that is only compatible with itself is not compatible with anything.
"""
import io
import json

import pytest

from rigma import mcp_server


def _drive(*messages) -> list[dict]:
    """Run messages through the real stdio loop and parse what came back."""
    inp = io.StringIO("".join(json.dumps(m) + "\n" for m in messages))
    out = io.StringIO()
    mcp_server.serve(inp, out)
    return [json.loads(ln) for ln in out.getvalue().splitlines()]


def _req(mid, method, params=None):
    m = {"jsonrpc": "2.0", "id": mid, "method": method}
    if params is not None:
        m["params"] = params
    return m


@pytest.fixture
def docs(tmp_path, monkeypatch):
    """A Rigma home with the RAG sidecar recorded — i.e. documents ARE indexed."""
    (tmp_path / "rag").mkdir(parents=True, exist_ok=True)
    (tmp_path / "rag" / "sidecar.json").write_text('{"port": 11699}',
                                                   encoding="utf-8")
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def nodocs(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "empty"))
    return tmp_path


# -- the handshake the client actually performs -------------------------------

def test_the_handshake_answers_what_the_client_sends(nodocs):
    """`mcp_client` sends initialize with protocolVersion + clientInfo, then a
    notifications/initialized NOTIFICATION, then tools/list. All three have to
    work or Rigma cannot consume its own server — which is the cheapest possible
    proof that the shapes are right."""
    got = _drive(
        _req(1, "initialize", {"protocolVersion": mcp_server.PROTOCOL_VERSION,
                               "capabilities": {},
                               "clientInfo": {"name": "rigma", "version": "0.9"}}),
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        _req(2, "tools/list", {}),
    )
    assert [r["id"] for r in got] == [1, 2]
    init = got[0]["result"]
    assert init["protocolVersion"] == mcp_server.PROTOCOL_VERSION
    assert init["serverInfo"]["name"] == "rigma"
    assert "tools" in init["capabilities"]
    # the memory tools need only a file, so they are here even with nothing
    # indexed; the documents tool is not
    assert "search_my_documents" not in [t["name"] for t in got[1]["result"]["tools"]]


def test_a_notification_is_never_answered(nodocs):
    """A notification has no id and MUST NOT get a reply. Answering one is a
    protocol violation that some clients treat as fatal, and the client here
    sends one on every single startup."""
    got = _drive(
        {"jsonrpc": "2.0", "method": "notifications/initialized"},
        {"jsonrpc": "2.0", "method": "notifications/cancelled",
         "params": {"requestId": 1}},
        _req(7, "ping"),
    )
    assert [r["id"] for r in got] == [7]


def test_an_unknown_method_is_a_proper_jsonrpc_error(nodocs):
    got = _drive(_req(3, "resources/list"))
    assert got[0]["error"]["code"] == -32601
    assert "resources/list" in got[0]["error"]["message"]


def test_a_parse_error_does_not_kill_the_server(nodocs):
    """One malformed line must not end the session — the next request still has
    to be answered, or a single bad byte costs the whole turn."""
    inp = io.StringIO("not json at all\n" + json.dumps(_req(9, "ping")) + "\n")
    out = io.StringIO()
    mcp_server.serve(inp, out)
    got = [json.loads(ln) for ln in out.getvalue().splitlines()]
    assert got[0]["error"]["code"] == -32700
    assert got[1]["id"] == 9


def test_a_batch_is_answered(nodocs):
    """MCP allows a batch. A client that batches and gets silence back sees a
    hung server, which is the worst way to fail."""
    inp = io.StringIO(json.dumps([_req(1, "ping"), _req(2, "tools/list")]) + "\n")
    out = io.StringIO()
    mcp_server.serve(inp, out)
    got = [json.loads(ln) for ln in out.getvalue().splitlines()]
    assert [r["id"] for r in got] == [1, 2]


def test_only_jsonrpc_ever_reaches_stdout(nodocs, capsys):
    """One stray print corrupts the stream and the arm sees a broken server.
    Every line out has to parse, and nothing may be written outside the loop."""
    out = io.StringIO()
    mcp_server.serve(io.StringIO(json.dumps(_req(1, "initialize")) + "\n"), out)
    for line in out.getvalue().splitlines():
        json.loads(line)              # raises if anything else got in
    assert capsys.readouterr().out == ""


# -- what is offered, and why -------------------------------------------------

def test_the_roster_is_exactly_what_was_justified():
    """Every offered tool is a schema on EVERY request, forever, and mcode
    already sends 18. So the roster is the tools the arm CANNOT do — and adding
    one is a deliberate act that has to be argued for in the module docstring,
    not a line somebody appended.

    `remember`/`recall` earn their two small schemas because they are the one
    kind of memory the arm does not have: its session remembers the
    conversation, but not the durable facts about the user.
    """
    assert mcp_server._ROSTER == ("search_my_documents", "remember", "recall")


def test_the_memory_tools_are_offered_without_any_documents(docs):
    """They are a plain file, so they do not depend on the RAG sidecar — and
    they must not disappear just because nothing is indexed."""
    names = [t["name"] for t in mcp_server.offered()]
    assert names == ["search_my_documents", "remember", "recall"]

    (docs / "rag" / "sidecar.json").unlink()
    assert [t["name"] for t in mcp_server.offered()] == ["remember", "recall"]


def test_the_documents_tool_is_offered_only_when_documents_exist(docs):
    """Offered-and-empty costs tokens on every turn to say "nothing is indexed".
    Absent says the same thing for free."""
    tools = _drive(_req(1, "tools/list"))[0]["result"]["tools"]
    assert "search_my_documents" in [t["name"] for t in tools]
    t = next(t for t in tools if t["name"] == "search_my_documents")
    # the schema the arm is shown comes from RIGMA's registry, so a tool whose
    # arguments change cannot drift from it
    assert t["inputSchema"]["required"] == ["query"]
    assert "indexed documents" in t["description"]


def test_no_documents_means_no_documents_tool_rather_than_a_broken_server(nodocs):
    """Not an empty roster — the memory tools need nothing but a file, so they
    stay. The rule is that a tool which would answer "nothing here" is not
    offered, not that a missing sidecar empties the server."""
    names = [t["name"] for t in mcp_server.offered()]
    assert "search_my_documents" not in names
    assert names == ["remember", "recall"]


# -- calling ------------------------------------------------------------------

def test_calling_a_tool_that_is_not_offered_is_refused_not_run(nodocs):
    """The roster is a boundary, not a menu. A name outside it must not reach
    `run_tool` — that is the difference between offering tools and exposing
    Rigma."""
    got = _drive(_req(1, "tools/call",
                      {"name": "run_shell", "arguments": {"command": "ls"}}))
    res = got[0]["result"]
    assert res["isError"] is True
    assert "not offered" in res["content"][0]["text"]


def test_a_result_goes_back_as_content_not_as_a_protocol_error(docs, monkeypatch):
    """A tool failure has to be visible to the MODEL, which is the party who can
    react to it. A JSON-RPC error is invisible to the model — the arm would just
    see a server that does not work."""
    monkeypatch.setattr(mcp_server, "call",
                        lambda name, args: mcp_server._text("error: nope", True))
    got = _drive(_req(1, "tools/call", {"name": "search_my_documents",
                                        "arguments": {"query": "x"}}))
    assert "error" not in got[0]
    assert got[0]["result"]["isError"] is True


def test_a_tool_that_raises_is_reported_not_propagated(docs, monkeypatch):
    from rigma import tools as toolkit

    def _boom(name, args, ctx=None):
        raise RuntimeError("kaboom")

    monkeypatch.setattr(toolkit, "run_tool", _boom)
    res = mcp_server.call("search_my_documents", {"query": "x"})
    assert res["isError"] is True
    assert "kaboom" in res["content"][0]["text"]


def test_arguments_that_are_not_an_object_do_not_crash_the_call(docs):
    """A weak model sends `arguments: "query=x"` often enough that it has to be
    survivable, not fatal."""
    res = mcp_server.call("search_my_documents", {})
    assert isinstance(res["content"][0]["text"], str)


def test_the_server_is_pessimistic_about_code(monkeypatch):
    """This is a SECOND execution path into Rigma's tools. One that assumed
    permission would be a hole rather than a feature."""
    monkeypatch.delenv("RIGMA_MCP_ALLOW_CODE", raising=False)
    assert mcp_server.ctx()["allow_code"] is False
    monkeypatch.setenv("RIGMA_MCP_ALLOW_CODE", "1")
    assert mcp_server.ctx()["allow_code"] is True


def test_the_profile_is_read_from_the_environment(monkeypatch):
    monkeypatch.delenv("RIGMA_MCP_PROFILE", raising=False)
    assert mcp_server.profile() == "all"
    monkeypatch.setenv("RIGMA_MCP_PROFILE", "confined")
    assert mcp_server.profile() == "confined"


def test_the_workspace_is_passed_not_guessed(monkeypatch):
    monkeypatch.delenv("RIGMA_MCP_WORKSPACE", raising=False)
    assert mcp_server.workspace() == ""
    monkeypatch.setenv("RIGMA_MCP_WORKSPACE", r"C:\work")
    assert mcp_server.workspace() == r"C:\work"
