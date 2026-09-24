"""R3-1: every mutable session field is type-checked on write.

`validate_field_types` was added by run 2 (AUDIT 01-3) but only `confirm_exec`
got a real type. The other booleans in MUTABLE_FIELDS were copied through, and
`bool("false")` is True — so a client that sent the quoted form of "no" turned
the switch (two of them grants: `allow_absolute_reads`, `allow_outbound_post`)
ON, and the UI's `grants.readGrants`, which honours only a literal `true`,
displayed the grant as off. `authors_note_depth` and `max_tool_rounds` accepted
a string that reached `int(...)` in build_messages / _round_cap.

Run 2 had already learned this lesson twice — 09-6 for methods/macros
(`macros._as_bool`) and 13-3 for `confirm_exec` alone — and never applied it to
the HTTP PATCH surface.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import serve, sessions


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


@pytest.fixture
def client(home):
    return TestClient(serve.build_app(upstream_port=1, default_prompt=""))


# Every switch/grant on the PATCH surface, with the value the UI means by OFF.
_SWITCHES = ("use_tools", "use_rag", "allow_code", "auto_compact",
             "one_action", "carry_reasoning", "allow_absolute_reads",
             "allow_outbound_post", "confirm_exec")


def test_every_mutable_field_has_a_declared_type():
    """The structural half: the hole existed because the type table was a
    hand-written subset of MUTABLE_FIELDS. Adding a field to the PATCH surface
    must not be able to reintroduce it."""
    missing = sorted(set(sessions.MUTABLE_FIELDS) - set(sessions._FIELD_TYPES))
    assert missing == [], (
        "these MUTABLE_FIELDS have no entry in _FIELD_TYPES, so update_session "
        f"copies them through unvalidated: {missing}")


@pytest.mark.parametrize("field", _SWITCHES)
@pytest.mark.parametrize("quoted", ["false", "0", "no", "off", "FALSE"])
def test_a_quoted_switch_is_refused_not_read_as_true(client, field, quoted):
    """`bool("false")` is True. The switch is set to OFF first, so the value
    actually flips if the quoted form is read as a truthy string — which is
    exactly what granted the capability before this guard."""
    sid = client.post("/api/sessions", json={"title": "sw"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}",
                       json={field: False}).status_code == 200
    r = client.post(f"/api/sessions/{sid}", json={field: quoted})
    assert r.status_code == 400, (field, quoted, r.status_code, r.text)
    assert field in r.json()["error"]
    after = sessions.load(sid).get(field)
    assert after is False, (
        f"{field} is {after!r} after a refused quoted write — the string form "
        "of 'no' was read as a grant")


@pytest.mark.parametrize("field", _SWITCHES)
def test_a_real_boolean_still_writes(client, field):
    """The guard must not break the honest client: true and false both land."""
    sid = client.post("/api/sessions", json={"title": "sw"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}",
                       json={field: True}).status_code == 200
    assert sessions.load(sid)[field] is True
    assert client.post(f"/api/sessions/{sid}",
                       json={field: False}).status_code == 200
    assert sessions.load(sid)[field] is False


@pytest.mark.parametrize("field", _SWITCHES)
def test_one_is_not_a_boolean(client, field):
    """isinstance(1, int) is True, so a naive int branch would accept 1 for a
    switch — and `tools._exec_confirmed` reads a truthy value as a grant."""
    sid = client.post("/api/sessions", json={"title": "sw"}).json()["id"]
    r = client.post(f"/api/sessions/{sid}", json={field: 1})
    assert r.status_code == 400, (field, r.status_code, r.text)


def test_a_non_numeric_count_is_refused(client):
    """`authors_note_depth` reached `int(...)` in build_messages and
    `max_tool_rounds` reached `int(...)` in _round_cap, both uncaught."""
    sid = client.post("/api/sessions", json={"title": "n"}).json()["id"]
    for field in ("authors_note_depth", "max_tool_rounds"):
        r = client.post(f"/api/sessions/{sid}", json={field: "abc"})
        assert r.status_code == 400, (field, r.status_code, r.text)
        assert client.post(f"/api/sessions/{sid}",
                           json={field: 7}).status_code == 200
        assert sessions.load(sid)[field] == 7
        # a boolean where a count belongs is a client bug worth naming
        assert client.post(f"/api/sessions/{sid}",
                           json={field: True}).status_code == 400


def test_null_still_coerces_to_the_empty_value(client):
    """The documented behaviour the guard must keep: null is an omitted box."""
    sid = client.post("/api/sessions", json={"title": "n"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}",
                       json={"title": None, "use_tools": None,
                             "authors_note_depth": None}).status_code == 200
    s = sessions.load(sid)
    assert s["title"] == "" and s["use_tools"] is False
    assert s["authors_note_depth"] == 0


def test_the_read_side_agrees_with_the_ui_about_a_grant(client):
    """The asymmetry R3-1 closes: `grants.readGrants` honours only a literal
    `true`, so a stored "yes" was DISPLAYED as off while the server's bool()
    read it as a grant. After the write guard no HTTP write can store one.

    The second half is the latent KeyError: 13-2's two grants were never added
    to `_SESSION_DEFAULTS` (13-3's `confirm_exec` was), so a chat that predates
    them loads without the keys and a subscripting reader raises.
    """
    sid = client.post("/api/sessions", json={"title": "g"}).json()["id"]
    assert client.post(f"/api/sessions/{sid}",
                       json={"confirm_exec": "yes"}).status_code == 400
    assert client.post(f"/api/sessions/{sid}",
                       json={"allow_absolute_reads": "yes"}).status_code == 400
    s = sessions.load(sid)
    assert s["confirm_exec"] is False
    assert s["allow_absolute_reads"] is False
    assert s["allow_outbound_post"] is False
