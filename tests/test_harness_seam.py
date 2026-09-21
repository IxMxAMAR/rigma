"""The harness seam: one interface, the built-in loop as the default.

These tests pin the two properties that make a seam worth having rather than a
place to hide a fallback: the default is unchanged, and asking for a backend
that cannot run a turn says so instead of quietly running the native loop.
"""
import pytest

from rigma import harness


def test_the_built_in_harness_is_the_default():
    assert harness.resolve(None).name == harness.NATIVE
    assert harness.resolve("").name == harness.NATIVE
    assert harness.resolve("  ").name == harness.NATIVE
    assert harness.resolve(harness.NATIVE).runnable is True


def test_the_endpoint_is_rigmas_own_server():
    """The point of the seam is a different AGENT, not a different model — so
    every backend is pointed at the model Rigma tuned for this machine."""
    assert harness.endpoint_for(11500) == "http://127.0.0.1:11500/v1"


def test_an_unknown_harness_is_refused_by_name():
    with pytest.raises(harness.HarnessError, match="unknown harness"):
        harness.resolve("not-a-harness")
    # the message must name what IS available, or it is a dead end
    with pytest.raises(harness.HarnessError, match="native"):
        harness.resolve("not-a-harness")


def test_a_known_but_unwired_harness_refuses_rather_than_falling_back():
    """The property this module exists for. A session that asked for MiniMax
    Code and silently got the native loop would be a lie the user cannot see."""
    with pytest.raises(harness.HarnessError) as e:
        harness.resolve(harness.MCODE)
    assert "cannot run a turn yet" in str(e.value)
    assert harness.BACKENDS[harness.MCODE].label in str(e.value)


def test_a_wired_harness_still_refuses_when_it_is_not_on_this_machine(
        monkeypatch):
    """Wiring DSH up does not make it present. `runnable` says a turn CAN be
    handed over; `installed` says the thing is actually here. A machine without
    the checkout gets a refusal naming what to set, never a silent native turn.
    """
    from rigma import harness_dsh
    monkeypatch.setattr(harness_dsh, "available", lambda: False)
    with pytest.raises(harness.HarnessError) as e:
        harness.resolve(harness.DSH)
    assert "not installed" in str(e.value)
    assert "RIGMA_DSH_HOME" in str(e.value)


def test_every_backend_is_listed_with_its_honest_cost():
    rows = {r["name"]: r for r in harness.list_harnesses()}
    assert set(rows) == {harness.NATIVE, harness.DSH, harness.MCODE}
    # the built-in is the default; DSH is wired up; MiniMax Code is not
    assert rows[harness.NATIVE]["runnable"] is True
    assert rows[harness.DSH]["runnable"] is True
    assert rows[harness.MCODE]["runnable"] is False
    # an external harness must declare what it does NOT inherit, or a user
    # chooses it expecting Rigma's tools and undo
    for name in (harness.DSH, harness.MCODE):
        assert rows[name]["unsupported"], name
        assert rows[name]["wire"], name
    # only a backend that CANNOT run has to explain why not
    assert rows[harness.MCODE]["pending"]
    assert not rows[harness.DSH]["pending"]


def test_installed_is_separate_from_runnable():
    """A probe must not be mistaken for a working integration: MiniMax Code is
    listed and probeable and still cannot run a turn."""
    rows = {r["name"]: r for r in harness.list_harnesses()}
    assert rows[harness.NATIVE]["installed"] is True
    assert rows[harness.MCODE]["runnable"] is False     # regardless of installed
    # and no probe may raise, on any machine
    for name in harness.known():
        assert isinstance(harness.installed(harness.BACKENDS[name]), bool)


def test_a_session_asking_for_an_unwired_harness_is_refused(monkeypatch,
                                                            tmp_path):
    """The seam is load-bearing, not decorative: a session that names another
    harness gets a refusal, never a silent turn on the built-in loop."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    s = sessions.create(title="t")
    s["harness"] = "mcode"
    sessions.save(s)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 400
    assert "cannot run a turn yet" in r.json()["error"]
    assert "MiniMax Code" in r.json()["error"]


def test_a_session_asking_for_dsh_without_the_checkout_is_refused(monkeypatch,
                                                                 tmp_path):
    """Wired up is not the same as present. The refusal must name what to set,
    because "not installed" with no instruction is a dead end."""
    from fastapi.testclient import TestClient

    from rigma import harness_dsh, serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_dsh, "available", lambda: False)
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    s = sessions.create(title="t")
    s["harness"] = "dsh"
    sessions.save(s)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 400
    assert "RIGMA_DSH_HOME" in r.json()["error"]


def test_a_new_session_defaults_to_the_built_in(monkeypatch, tmp_path):
    from rigma import sessions
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    assert sessions.create(title="t")["harness"] == harness.NATIVE


def test_the_api_lists_the_harnesses(monkeypatch, tmp_path):
    """Reachable, not library-only."""
    from fastapi.testclient import TestClient

    from rigma import serve
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        body = c.get("/api/harnesses").json()
    assert body["built_in"] == "native"
    assert body["endpoint"] == "http://127.0.0.1:11500/v1"
    names = {h["name"] for h in body["harnesses"]}
    assert {"native", "dsh", "mcode"} <= names
    # the built-in and DSH can be selected; MiniMax Code is listed but not wired
    assert [h["name"] for h in body["harnesses"] if h["runnable"]] == \
        ["dsh", "native"]


def test_the_advertised_endpoint_is_rigmas_not_the_engines(monkeypatch):
    """Every backend is pointed at RIGMA's /v1, never at llama-server's.

    `build_app(upstream_port)` proxies to the engine, so handing that port to
    `endpoint_for` aimed every external harness straight at llama-server —
    bypassing the session, the repair layer and the idle-unload bookkeeping,
    which is exactly the silent bypass the seam exists to prevent. The public
    port is read from the same state the OpenAI base is built from.
    """
    from fastapi.testclient import TestClient

    from rigma import serve
    from rigma import state as st

    monkeypatch.setattr(st, "read_state",
                        lambda: {"public_port": 11500, "model": "m"})
    with TestClient(serve.build_app(11499)) as c:
        body = c.get("/api/harnesses").json()
    assert body["endpoint"] == "http://127.0.0.1:11500/v1"
    assert "11499" not in body["endpoint"]


def test_a_dsh_session_is_driven_by_the_adapter(monkeypatch, tmp_path):
    """The seam hands the turn over for real.

    The adapter's neutral events become Rigma's OWN SSE vocabulary, and the
    reply is persisted into Rigma's session — so an externally-driven turn lands
    in the same rail, the same transcript and the same export as a native one.
    The backend owns its own tools, system prompt and transcript; that is the
    trade. Rigma keeps the MODEL (the endpoint handed over is Rigma's own /v1,
    never llama-server) and keeps the record the user reads.
    """
    import os

    from fastapi.testclient import TestClient

    from rigma import harness_dsh, serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_dsh, "available", lambda: True)
    seen: dict = {}

    def _fake_drive(**kw):
        seen.update(kw)
        yield harness_dsh.TurnEvent("notice", text="starting")
        yield harness_dsh.TurnEvent("tool", name="pwsh", args={"cmd": "ls"})
        yield harness_dsh.TurnEvent("tool_result", name="pwsh", text="a.txt")
        yield harness_dsh.TurnEvent("text", text="all ")
        yield harness_dsh.TurnEvent("text", text="done")

    monkeypatch.setattr(harness_dsh, "drive_turn", _fake_drive)
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    _real_read = st.read_state
    monkeypatch.setattr(st, "read_state",
                        lambda: {**(_real_read() or {}),
                                 "public_port": 11500, "ctx": 32768})
    s = sessions.create(title="t")
    s["harness"] = "dsh"
    sessions.save(s)

    with TestClient(serve.build_app(upstream_port=11499)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "do it"})
    assert r.status_code == 200, r.text
    body = r.text

    # the adapter's events reached the wire in Rigma's own vocabulary
    assert "event: notice" in body
    assert "event: tool" in body
    assert "event: tool_result" in body
    assert '"delta": "all "' in body
    assert "[DONE]" in body

    # RIGMA's endpoint, not the engine's — the whole point of the seam
    assert seen["base_url"] == "http://127.0.0.1:11500/v1"
    assert seen["prompt"] == "do it"
    assert seen["session_id"] == s["id"]
    assert seen["context_window"] == 32768

    # and the reply is in Rigma's transcript, where the rail reads it
    stored = sessions.load(s["id"])
    assert stored["messages"][-1]["role"] == "assistant"
    assert stored["messages"][-1]["content"] == "all done"
    assert stored["messages"][-1]["tool_trace"][0]["name"] == "pwsh"
