"""A7b: `mcp_server.offered(prof=...)` silently accepted an unknown profile.

A7 fixed the run-creation route (`serve.py`) to return 400 for a profile that
is not one of `runs.PROFILES`. This is the twin that was left: `offered` — and
the `profile()`/`ctx()` it reads from the environment — passed any string
straight to `tool_specs`, where an unrecognised profile matches none of the
restrictive branches and therefore behaves exactly like "all", the profile that
grants the full network and delete surface. A typo, or a registrar that sent
"read-only", silently produced the MOST permissive roster.

Absent still means "all" (the owner's OD-1 choice, unchanged); a value that is
PRESENT must name one of the four.
"""
import pytest

from rigma import mcp_server, runs

ALLOWED = sorted(runs.PROFILES)
BAD = ["read-only", "", "ALL", "all ", "no_network", "no network", 123, 1.5,
       True, [], {}]


@pytest.mark.parametrize("bad", BAD)
def test_offered_refuses_an_unknown_profile_and_names_the_allowed_set(bad):
    with pytest.raises(ValueError) as ei:
        mcp_server.offered(ws="C:/w", prof=bad)
    msg = str(ei.value)
    assert "profile" in msg, msg
    for name in ALLOWED:
        assert name in msg, f"{name!r} missing from {msg!r}"


@pytest.mark.parametrize("good", ALLOWED)
def test_a_valid_profile_is_accepted(good):
    mcp_server.offered(ws="C:/w", prof=good)      # must not raise


def test_an_absent_profile_still_defaults_to_all(monkeypatch):
    """OD-1: the DEFAULT is the owner's and is unchanged. Absent means "all"."""
    monkeypatch.delenv("RIGMA_MCP_PROFILE", raising=False)
    assert mcp_server.offered(ws="C:/w") == mcp_server.offered(
        ws="C:/w", prof="all")


@pytest.mark.parametrize("bad", ["read-only", "", "ALL", "all "])
def test_the_environment_profile_is_validated_too(bad, monkeypatch):
    """`offered(prof=None)` and every tool call read the profile from the
    environment, so the env is the same hole through a different door."""
    monkeypatch.setenv("RIGMA_MCP_PROFILE", bad)
    with pytest.raises(ValueError):
        mcp_server.profile()
    with pytest.raises(ValueError):
        mcp_server.ctx()
    with pytest.raises(ValueError):
        mcp_server.offered(ws="C:/w")


def test_tools_list_answers_an_unknown_profile_with_a_protocol_error(
        monkeypatch):
    """A bad environment must not crash the read loop: the arm gets a named
    error, the same refusal the route gives."""
    monkeypatch.setenv("RIGMA_MCP_PROFILE", "read-only")
    reply = mcp_server.handle(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"})
    assert reply.get("error"), reply
    assert "profile" in reply["error"]["message"], reply
    for name in ALLOWED:
        assert name in reply["error"]["message"], reply
