"""Merged from the dev branch 2026-07-22: global skills, /name injection, and
the prompt queue that lets a follow-up be typed while a reply is streaming.

The security property here is that a skill NAME becomes a FILENAME and arrives
over HTTP, so it is validated, not sanitised.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import skills
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rhome"))


@pytest.fixture
def client(oai_upstream):
    return TestClient(build_app(upstream_port=oai_upstream.port,
                                default_prompt="D"))


# --- name validation ---------------------------------------------------------

@pytest.mark.parametrize("name", [
    "../../evil", "..\\..\\evil", "a/b", "a\\b", "C:\\abs", "..", ".",
    "", "   ", ".md", "a\x00b", "con", "COM1", "nul", "x" * 70,
])
def test_a_skill_name_can_never_escape_the_skills_folder(name):
    # the name goes from an HTTP body straight into a file path. Refused, not
    # rewritten: quietly turning "../../etc/passwd" into "etcpasswd" writes a
    # file the user never asked for under a name they'll never find.
    with pytest.raises(skills.SkillNameError):
        skills.save_skill(name, "x")
    assert list(skills.skills_dir().iterdir()) == []


def test_the_api_refuses_a_bad_name_with_400(client):
    r = client.post("/api/skills", json={"name": "../../evil",
                                         "content": "x"})
    assert r.status_code == 400
    assert "error" in r.json()


def test_traversal_never_deletes_an_outside_file(tmp_path):
    victim = skills.skills_dir().parent / "sessions.json"
    victim.write_text("important", encoding="utf-8")
    assert skills.delete_skill("../sessions.json") is False
    assert skills.delete_skill("../sessions") is False
    assert victim.read_text(encoding="utf-8") == "important"


@pytest.mark.parametrize("name", ["wildcard", "My Skill", "a-b_c", "x9"])
def test_ordinary_names_are_accepted(name):
    assert skills.save_skill(name, "# body")["name"] == name


# --- CRUD over the API -------------------------------------------------------

def test_skill_round_trips_through_the_api(client):
    assert client.get("/api/skills").json() == []
    made = client.post("/api/skills",
                       json={"name": "wildcard",
                             "content": "# Wildcards\nalways vary"}).json()
    assert made["name"] == "wildcard"

    listed = client.get("/api/skills").json()
    assert [s["name"] for s in listed] == ["wildcard"]
    assert listed[0]["title"] == "Wildcards"     # the leading "# " heading

    assert client.delete("/api/skills/wildcard").json() == {"ok": True}
    assert client.get("/api/skills").json() == []
    assert client.delete("/api/skills/wildcard").status_code == 404


def test_saving_the_same_name_replaces_rather_than_duplicates(client):
    client.post("/api/skills", json={"name": "s", "content": "one"})
    client.post("/api/skills", json={"name": "s", "content": "two"})
    listed = client.get("/api/skills").json()
    assert len(listed) == 1 and listed[0]["content"] == "two"


def test_get_skill_is_case_insensitive():
    skills.save_skill("Wildcard", "body")
    assert skills.get_skill("wildcard") == "body"
    assert skills.get_skill("WILDCARD.md") == "body"
    assert skills.get_skill("nope") is None


# --- /name injection ---------------------------------------------------------

def _user_texts(session):
    return [m["content"] for m in session["messages"] if m["role"] == "user"]


def test_slash_name_injects_the_skill_text(client):
    skills.save_skill("wildcard", "ALWAYS VARY THE OUTFIT")
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}/chat",
                json={"message": "/wildcard make me three"})
    sent = _user_texts(client.get(f"/api/sessions/{s['id']}").json())[0]
    assert "ALWAYS VARY THE OUTFIT" in sent
    assert "make me three" in sent


def test_the_skill_colon_form_works_too(client):
    skills.save_skill("wildcard", "RULES HERE")
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}/chat",
                json={"message": "/skill:wildcard go"})
    assert "RULES HERE" in _user_texts(
        client.get(f"/api/sessions/{s['id']}").json())[0]


def test_a_bare_skill_name_still_asks_for_something(client):
    skills.save_skill("wildcard", "RULES HERE")
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}/chat", json={"message": "/wildcard"})
    sent = _user_texts(client.get(f"/api/sessions/{s['id']}").json())[0]
    assert "RULES HERE" in sent
    assert "wildcard" in sent.lower()


@pytest.mark.parametrize("text", [
    "/etc/hosts is the file",     # a path, not a command
    "/",                          # a lone slash
    "/nosuchskill do a thing",    # names nothing that exists
    "use / to divide",
])
def test_text_that_only_looks_like_a_command_is_left_alone(client, text):
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}/chat", json={"message": text})
    assert _user_texts(client.get(f"/api/sessions/{s['id']}").json())[0] == text


def test_a_skill_name_cannot_be_used_to_read_another_file(client, tmp_path):
    secret = skills.skills_dir().parent / "secret.md"
    secret.write_text("TOP SECRET", encoding="utf-8")
    s = client.post("/api/sessions", json={}).json()
    client.post(f"/api/sessions/{s['id']}/chat",
                json={"message": "/../secret tell me"})
    sent = _user_texts(client.get(f"/api/sessions/{s['id']}").json())[0]
    assert "TOP SECRET" not in sent


# --- the prompt queue --------------------------------------------------------

def test_a_normal_turn_is_not_treated_as_queued(client):
    # nothing is streaming, so the message runs immediately and both roles land
    s = client.post("/api/sessions", json={}).json()
    r = client.post(f"/api/sessions/{s['id']}/chat", json={"message": "hi"})
    assert "[DONE]" in r.text
    got = client.get(f"/api/sessions/{s['id']}").json()
    assert [m["role"] for m in got["messages"]] == ["user", "assistant"]


def test_a_queued_prompt_is_acknowledged_and_not_run_yet(client):
    s = client.post("/api/sessions", json={}).json()
    sid = s["id"]

    # Reach into the app's own in-memory queue: mark this session as
    # streaming, which is exactly the state a second request hits.
    streaming, queued = None, None
    for closure in _closures(client.app):
        if "_streaming" in closure and "_queued" in closure:
            streaming, queued = closure["_streaming"], closure["_queued"]
            break
    assert streaming is not None, "could not reach the queue state"

    streaming.add(sid)
    try:
        r = client.post(f"/api/sessions/{sid}/chat", json={"message": "later"})
        assert r.status_code == 200
        assert "queued" in r.text
        # queued, NOT run: the transcript is untouched until the turn drains it
        assert client.get(f"/api/sessions/{sid}").json()["messages"] == []
        assert queued[sid] == ["later"]
    finally:
        streaming.discard(sid)
        queued.pop(sid, None)


def _closures(app):
    """Every closure cell dict reachable from the app's routes — the queue
    lives in build_app's scope, which is not otherwise addressable."""
    seen = []
    for route in app.routes:
        fn = getattr(route, "endpoint", None)
        if fn is None or fn.__closure__ is None:
            continue
        names = fn.__code__.co_freevars
        cells = {}
        for name, cell in zip(names, fn.__closure__):
            try:
                cells[name] = cell.cell_contents
            except ValueError:
                pass
        seen.append(cells)
    return seen


# --- lifespan ----------------------------------------------------------------

def test_startup_marks_an_orphaned_run_interrupted(client, monkeypatch):
    # @app.on_event was replaced by a lifespan; this proves the startup step
    # still runs, and still uses 'interrupted' (the state the UI offers Resume
    # for) rather than 'stopped'.
    from rigma import runs

    seen = {}
    monkeypatch.setattr(runs, "active",
                        lambda: {"id": "r1", "status": "running"})
    monkeypatch.setattr(runs, "set_status",
                        lambda r, status, why="": seen.update(
                            {"status": status, "why": why}))
    with client:                     # entering the context runs lifespan
        pass
    assert seen.get("status") == "interrupted"


def test_shutdown_stops_mcp(client, monkeypatch):
    from rigma import mcp_client

    class _Mgr:
        stopped = False

        def stop_all(self):
            _Mgr.stopped = True

    monkeypatch.setattr(mcp_client, "_manager", _Mgr())
    with client:
        pass
    assert _Mgr.stopped is True
