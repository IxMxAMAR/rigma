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
    """The property this module exists for. A session that asked for DSH and
    silently got the native loop would be a lie the user cannot see."""
    for name in (harness.DSH, harness.MCODE):
        with pytest.raises(harness.HarnessError) as e:
            harness.resolve(name)
        assert "cannot run a turn yet" in str(e.value)
        assert harness.BACKENDS[name].label in str(e.value)


def test_every_backend_is_listed_with_its_honest_cost():
    rows = {r["name"]: r for r in harness.list_harnesses()}
    assert set(rows) == {harness.NATIVE, harness.DSH, harness.MCODE}
    # the built-in is the only thing that can run a turn, and says so
    assert rows[harness.NATIVE]["runnable"] is True
    assert rows[harness.DSH]["runnable"] is False
    assert rows[harness.MCODE]["runnable"] is False
    # an external harness must declare what it does NOT inherit, or a user
    # chooses it expecting Rigma's tools and undo
    for name in (harness.DSH, harness.MCODE):
        assert rows[name]["unsupported"], name
        assert rows[name]["wire"], name
        assert rows[name]["pending"], name


def test_installed_is_separate_from_runnable():
    """A probe must not be mistaken for a working integration: DSH's SDK being
    importable does not make the backend wired up."""
    rows = {r["name"]: r for r in harness.list_harnesses()}
    assert rows[harness.NATIVE]["installed"] is True
    assert rows[harness.DSH]["runnable"] is False      # regardless of installed
    # and the probe itself must not raise, on any machine
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
    s["harness"] = "dsh"
    sessions.save(s)
    with TestClient(serve.build_app(upstream_port=11500)) as c:
        r = c.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert r.status_code == 400
    assert "cannot run a turn yet" in r.json()["error"]
    assert "DeepSeek Harness" in r.json()["error"]


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
    # the built-in is the only one a session may select today
    assert [h["name"] for h in body["harnesses"] if h["runnable"]] == ["native"]
