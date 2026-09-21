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


def test_a_wired_harness_that_is_absent_refuses_rather_than_falling_back(
        monkeypatch):
    """The property this module exists for. A session that asked for MiniMax
    Code and silently got the native loop would be a lie the user cannot see.

    Both external backends are WIRED now; neither is present on a machine that
    has not got it. `runnable` is a promise about the code, `installed` is a
    fact about this machine, and the refusal is what keeps the two apart."""
    from rigma import harness_mcode
    monkeypatch.setattr(harness_mcode, "available", lambda: False)
    with pytest.raises(harness.HarnessError) as e:
        harness.resolve(harness.MCODE)
    assert "not installed" in str(e.value)
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
    # all three can run a turn now — `runnable` is about the code, not about
    # this machine (see the installed/runnable test below)
    assert all(r["runnable"] for r in rows.values())
    # an external harness must declare what it does NOT inherit, or a user
    # chooses it expecting Rigma's tools and undo
    for name in (harness.DSH, harness.MCODE):
        assert rows[name]["unsupported"], name
        assert rows[name]["wire"], name
    # `pending` is for a backend that cannot run a turn at all, and none is
    assert not any(r["pending"] for r in rows.values())
    # mcode only stays account-free if its provider is SELECTED, and that is
    # the one step an integrator can silently skip — so it is on the menu
    assert "--use" in rows[harness.MCODE]["wire"]
    assert "MiniMax login" in rows[harness.MCODE]["wire"]


def test_installed_is_separate_from_runnable(monkeypatch):
    """A probe must not be mistaken for a working integration, and wiring a
    backend up must not be mistaken for it being present."""
    from rigma import harness_dsh, harness_mcode
    monkeypatch.setattr(harness_dsh, "available", lambda: False)
    monkeypatch.setattr(harness_mcode, "available", lambda: False)
    rows = {r["name"]: r for r in harness.list_harnesses()}
    assert rows[harness.NATIVE]["installed"] is True
    assert rows[harness.NATIVE]["runnable"] is True
    for name in (harness.DSH, harness.MCODE):
        assert rows[name]["runnable"] is True, name
        assert rows[name]["installed"] is False, name
        with pytest.raises(harness.HarnessError, match="not installed"):
            harness.resolve(name)
    # and no probe may raise, on any machine
    for name in harness.known():
        assert isinstance(harness.installed(harness.BACKENDS[name]), bool)


def test_a_session_asking_for_an_absent_harness_is_refused(monkeypatch,
                                                           tmp_path):
    """The seam is load-bearing, not decorative: a session that names another
    harness gets a refusal, never a silent turn on the built-in loop.

    `available` is pinned false so the test says the same thing on a machine
    that happens to have mcode installed.
    """
    from fastapi.testclient import TestClient

    from rigma import harness_mcode, serve, sessions
    from rigma import state as st

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: False)
    st.write_state("m", "Q4", 11500, engine_pid=1, ui_pid=1)
    s = sessions.create(title="t")
    s["harness"] = "mcode"
    sessions.save(s)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 400
    assert "not installed" in r.json()["error"]
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
    # all three can be selected, on a machine that has them
    assert [h["name"] for h in body["harnesses"] if h["runnable"]] == \
        ["dsh", "mcode", "native"]


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


# --------------------------------------------------------------------------
# The seam has to be REACHABLE from the product. `harness` was absent from
# `sessions.MUTABLE_FIELDS`, so `POST /api/sessions/{sid}` could not set it and
# the only way to point a chat at DSH was to edit its JSON by hand — the whole
# integration was library-only, and no UI could have fixed that.


def _patched_dsh(monkeypatch, present=True):
    """DSH's availability is a source checkout, so it is monkeypatched rather
    than installed: these tests are about the SELECTION, not the adapter."""
    from rigma import harness_dsh
    monkeypatch.setattr(harness_dsh, "available", lambda: present)


def test_a_session_can_be_pointed_at_a_harness_through_the_api(
        monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _patched_dsh(monkeypatch)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}", json={"harness": "dsh"})
    assert r.status_code == 200, r.text
    assert r.json()["harness"] == "dsh"
    assert sessions.load(sid)["harness"] == "dsh"


def test_the_stored_harness_name_is_canonical(monkeypatch, tmp_path):
    """The picker compares its own value against what the server stored, so the
    stored name has to be the name the menu uses."""
    from fastapi.testclient import TestClient

    from rigma import serve

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _patched_dsh(monkeypatch)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}", json={"harness": " DSH "})
    assert r.status_code == 200, r.text
    assert r.json()["harness"] == "dsh"


def test_an_unknown_harness_is_refused_at_the_write(monkeypatch, tmp_path):
    """Refused at the WRITE, not on every turn afterwards.

    A session holding a name that cannot resolve would 400 on every turn, and
    the user's only clue would be an error on a turn they had no reason to
    connect to a setting — with nothing in the UI that offers to fix it.
    """
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}", json={"harness": "not-a-harness"})
    assert r.status_code == 400
    assert "unknown harness" in r.json()["error"]
    assert sessions.load(sid)["harness"] == "native"      # left alone


def test_a_listed_but_absent_harness_cannot_be_selected(monkeypatch, tmp_path):
    """MiniMax Code is on the menu so its cost is visible; on a machine without
    it, it is not selectable — and the refusal has to say what is missing or it
    is a dead end."""
    from fastapi.testclient import TestClient

    from rigma import harness_mcode, serve, sessions

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.setattr(harness_mcode, "available", lambda: False)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}", json={"harness": "mcode"})
    assert r.status_code == 400
    assert "not installed" in r.json()["error"]
    assert sessions.load(sid)["harness"] == "native"


def test_a_harness_that_is_not_on_this_machine_cannot_be_selected(
        monkeypatch, tmp_path):
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    _patched_dsh(monkeypatch, present=False)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        sid = c.post("/api/sessions", json={}).json()["id"]
        r = c.post(f"/api/sessions/{sid}", json={"harness": "dsh"})
    assert r.status_code == 400
    assert "RIGMA_DSH_HOME" in r.json()["error"]
    assert sessions.load(sid)["harness"] == "native"


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

    # and the client is told WHICH backend drove the turn, before the reply:
    # an external turn can run for a minute before its first token, so a badge
    # that only appears with the reply is one the reader waits for
    assert "event: harness" in body
    assert '"name": "dsh"' in body
    assert '"label": "DeepSeek Harness"' in body

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
    # the badge survives a reload, and only an EXTERNAL reply carries one: a
    # native message is not stamped "native", or the label would be on the
    # ordinary case to explain the rare one
    assert stored["messages"][-1]["harness"] == "dsh"


def test_every_adapter_accepts_a_cancel_event():
    """A stop has to reach the adapter, so `cancel` is part of the contract.

    Checked against the seam's OWN table rather than a list kept here by hand:
    an adapter added later is covered the moment it is registered, which is the
    only way a contract stays true instead of becoming a comment.
    """
    import inspect

    assert harness._ADAPTERS, "the table is the thing being tested"
    for name in harness._ADAPTERS:
        mod = harness.adapter(name)
        assert mod is not None, name
        params = inspect.signature(mod.drive_turn).parameters
        assert "cancel" in params, f"{name} cannot be stopped"


def test_stopping_a_chat_that_is_not_running_says_so():
    """A stop that reports success when nothing was running teaches a UI to
    lie about what it did."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    with TestClient(serve.build_app(upstream_port=59999)) as client:
        s = sessions.create()
        r = client.post(f"/api/sessions/{s['id']}/stop")
        assert r.status_code == 200
        assert r.json() == {"ok": True, "stopped": False}


def test_stopping_a_chat_that_IS_running_sets_its_event():
    """The route's whole job: set the event the turn is already watching. It
    does not do the stopping itself, because only the backend knows what
    stopping means there."""
    import threading

    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    app = serve.build_app(upstream_port=59999)
    with TestClient(app) as client:
        s = sessions.create()
        ev = threading.Event()
        _cancels = _cancels_of(app)
        _cancels[s["id"]] = ev
        try:
            r = client.post(f"/api/sessions/{s['id']}/stop")
            assert r.json() == {"ok": True, "stopped": True}
            assert ev.is_set()
        finally:
            _cancels.pop(s["id"], None)


def test_the_permission_mode_is_a_session_choice():
    """How much the agent may do unasked is a trade, not a fact: headless has
    nobody to ask, so the mode that asks FAILS THE RUN. A hardcoded choice
    hides that; a per-chat one states it."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    with TestClient(serve.build_app(upstream_port=59999)) as client:
        s = sessions.create()
        r = client.post(f"/api/sessions/{s['id']}", json={"permission": "smart"})
        assert r.status_code == 200, r.text
        assert sessions.load(s["id"])["permission"] == "smart"


def test_a_permission_mode_the_backend_cannot_take_is_refused_at_the_write():
    """`ask` is refused by mcode ITSELF headlessly — "requires an interactive
    host" — so offering it would be a setting that breaks the turn it is set
    on. Refused here, where the reason can be read."""
    from fastapi.testclient import TestClient

    from rigma import serve, sessions

    with TestClient(serve.build_app(upstream_port=59999)) as client:
        s = sessions.create()
        r = client.post(f"/api/sessions/{s['id']}", json={"permission": "ask"})
        assert r.status_code == 400
        assert "full/smart/off" in r.json()["error"]
        assert "permission" not in (sessions.load(s["id"]) or {})


def _cancels_of(app):
    """The live `_cancels` registry for an app built by `build_app`.

    It is a closure local — deliberately, so two apps cannot share a stop
    registry — so the only way in is through the route that owns it, and that
    route is exactly what the test above is exercising. Reaching it any other
    way would mean the test no longer tested the route.
    """
    import inspect

    for route in app.routes:
        endpoint = getattr(route, "endpoint", None)
        if endpoint is not None and getattr(endpoint, "__name__", "") == "stop_chat":
            return inspect.getclosurevars(endpoint).nonlocals["_cancels"]
    raise AssertionError("no stop_chat route on the app")
