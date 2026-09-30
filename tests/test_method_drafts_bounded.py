"""10-R3-17: the draft store is bounded and a client can discard a draft.

A draft was removed only by promote(), so a creation chat the user abandons
left its file in ~/.rigma/method_drafts for the life of the process and no
route could delete one.
"""
import os
import time

import pytest
from fastapi.testclient import TestClient

from rigma import method_drafts
from rigma import state as st
from rigma.serve import build_app


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid())
    return tmp_path


@pytest.fixture
def client():
    return TestClient(build_app(upstream_port=1))


def _files():
    return list(method_drafts.drafts_dir().glob("*.json"))


def _backdate(did, seconds):
    p = method_drafts.drafts_dir() / f"{did}.json"
    old = time.time() - seconds
    os.utime(p, (old, old))


def test_stale_drafts_are_reaped_down_to_the_cap():
    cap = method_drafts.MAX_DRAFTS
    made = [method_drafts.new_draft(name=f"Draft {i}") for i in range(cap + 3)]
    for i, d in enumerate(made[:-1]):   # abandoned, with distinct ages so
        # "oldest first" is deterministic rather than a random-id tiebreak
        _backdate(d["id"],
                  method_drafts._DRAFT_LIVE_SECONDS * 2 + (len(made) - i))
    fresh = method_drafts.new_draft(name="Newest")   # this creation reaps
    assert len(_files()) == cap
    assert method_drafts.load(fresh["id"]) is not None
    assert method_drafts.load(made[-1]["id"]) is not None   # recent -> kept
    assert method_drafts.load(made[4]["id"]) is not None    # newest stale
    assert method_drafts.load(made[3]["id"]) is None        # 4th oldest, reaped
    assert method_drafts.load(made[0]["id"]) is None        # oldest, reaped


def test_a_live_draft_is_never_evicted_even_over_the_cap():
    cap = method_drafts.MAX_DRAFTS
    made = [method_drafts.new_draft(name=f"Live {i}") for i in range(cap + 2)]
    # every draft was written just now, so every one is in use; the cap yields
    # to the one rule it exists to respect
    assert len(_files()) == cap + 2
    for d in made:
        assert method_drafts.load(d["id"]) is not None


def test_delete_route_discards_a_draft(client):
    d = client.post("/api/methods/draft", json={}).json()
    r = client.delete(f"/api/methods/drafts/{d['draft_id']}")
    assert r.status_code == 200, r.text
    assert r.json() == {"deleted": d["draft_id"]}
    assert method_drafts.load(d["draft_id"]) is None


def test_delete_route_also_answers_on_the_singular_sibling_path(client):
    d = client.post("/api/methods/draft", json={}).json()
    r = client.delete(f"/api/methods/draft/{d['draft_id']}")
    assert r.status_code == 200, r.text
    assert method_drafts.load(d["draft_id"]) is None


def test_delete_route_unknown_id_is_the_repo_404(client):
    r = client.delete("/api/methods/drafts/nope")
    assert r.status_code == 404
    assert r.json() == {"error": "no such draft"}


def test_delete_route_rejects_an_id_that_cannot_name_a_draft(client):
    r = client.delete("/api/methods/drafts/-nope")
    assert r.status_code == 404
    assert r.json() == {"error": "no such draft"}
