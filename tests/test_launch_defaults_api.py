"""D2 backend: the launch-defaults API can support a first-load dialog.

The recon (`.scratch/orchestrator/recon-W5-frontend.md`, D2) established seven
facts by measurement; each was re-verified against the tree at the time of this
commit:

1. no GET for defaults (405)                     — was TRUE, now a GET exists
2. `backend` missing from the POST's `allowed`   — was TRUE, now accepted
3. "send null to clear" 500s for 6 of the 7      — was TRUE, now cleared
4. `set_launch_defaults` refuses non-custom      — TRUE and deliberately kept
   (owner decision: a shipped spec's launch settings are hand-authored)
5. the UI's defaults write sends 3 of the 7      — TRUE (frontend, later wave)
6. the UI's KV list is a stale 4-value subset    — TRUE (frontend, later wave)
7. no first-load signal                          — was TRUE, now `first_load`

Nothing here launches an engine: the routes are storage + one file read.
"""
import json
import struct

import pytest
from fastapi.testclient import TestClient

from rigma.serve import build_app

T_U32, T_STR = 4, 8


def _s(b):
    return struct.pack("<Q", len(b)) + b


def _gguf(tmp_path, name=b"Defaults Dialog 4B",
          fname="DefaultsDialog-Q5_K_M.gguf"):
    """A minimal valid gguf header, enough for install to read a slug."""
    kvs = [
        _s(b"general.architecture") + struct.pack("<I", T_STR) + _s(b"qwen3"),
        _s(b"general.name") + struct.pack("<I", T_STR) + _s(name),
        _s(b"qwen3.block_count") + struct.pack("<I", T_U32) + struct.pack("<I", 8),
        _s(b"qwen3.context_length") + struct.pack("<I", T_U32) + struct.pack("<I", 16384),
        _s(b"qwen3.embedding_length") + struct.pack("<I", T_U32) + struct.pack("<I", 1024),
        _s(b"qwen3.attention.head_count") + struct.pack("<I", T_U32) + struct.pack("<I", 16),
        _s(b"qwen3.attention.head_count_kv") + struct.pack("<I", T_U32) + struct.pack("<I", 2),
    ]
    p = tmp_path / fname
    p.write_bytes(b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", 0)
                  + struct.pack("<Q", len(kvs)) + b"".join(kvs) + b"\x00" * 256)
    return p


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)
    return tmp_path / "home"


@pytest.fixture
def client(home, tmp_path):
    """A custom (installed) model, which is the only kind that may store
    defaults."""
    c = TestClient(build_app(upstream_port=1))
    c.__enter__()
    r = c.post("/api/models/install", json={"path": str(_gguf(tmp_path))})
    assert r.status_code == 200, r.text
    try:
        yield c, r.json()["slug"]
    finally:
        c.__exit__(None, None, None)


# the six fields the route already accepted; `backend` is covered separately
# because it was dropped by the allowlist and would mask the null-clear failure
ALL_SET = {"quant": "Q5_K_M", "ctx": 8192, "kv": "q5_1", "vision": True,
           "spec_type": "draft-mtp", "spec_n_max": 1}


def test_a_get_returns_the_defaults_and_the_first_load_signal(client):
    """The dialog cannot open on a POST: there was no GET at all (405), so it
    had no way to show what is already pinned."""
    c, slug = client
    r = c.get(f"/api/models/{slug}/defaults")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["launch"] is None            # nothing pinned yet
    assert body["custom"] is True            # …so saving is allowed
    assert body["last_used"] is None
    assert body["first_load"] is True
    assert c.get("/api/models/no-such-model/defaults").status_code == 404


def test_the_post_accepts_backend_and_every_reader_sees_it(client):
    """`backend` was dropped by the route's allowlist while `LaunchDefaults`
    stored and consumed it, so a per-model backend was silently unsettable."""
    c, slug = client
    r = c.post(f"/api/models/{slug}/defaults",
               json={"ctx": 8192, "kv": "q5_1", "backend": "rocm"})
    assert r.status_code == 200, r.text
    launch = r.json()["launch"]
    assert launch["ctx"] == 8192 and launch["kv"] == "q5_1"
    assert launch["backend"] == "rocm"
    got = c.get(f"/api/models/{slug}/defaults").json()
    assert got["launch"] == launch
    assert got["first_load"] is False
    # …and on the list the UI already loads, so the first-load trigger does not
    # need a second request per model
    row = {m["slug"]: m for m in c.get("/api/models").json()["models"]}[slug]
    assert row["launch"]["ctx"] == 8192
    assert row["launch"]["backend"] == "rocm"
    # and `backend: null` clears it like every other field
    r = c.post(f"/api/models/{slug}/defaults", json={"backend": None})
    assert r.status_code == 200, r.text
    assert not r.json()["launch"].get("backend")
    assert r.json()["launch"]["ctx"] == 8192        # siblings untouched


def test_null_clears_every_field(client):
    """The route docstring promises it and the UI needs it: removing a default
    must not need a separate verb. Six of the seven fields are non-optional, so
    assigning None made `LaunchDefaults(**current)` raise ValidationError inside
    pydantic — and the route catches only HangarError, so the documented way to
    clear answered 500."""
    c, slug = client
    r = c.post(f"/api/models/{slug}/defaults", json=ALL_SET)
    assert r.status_code == 200, r.text
    # `launch` always carries all seven fields, with the falsy sentinel for the
    # ones nobody pinned
    assert {k: r.json()["launch"][k] for k in ALL_SET} == ALL_SET
    order = list(ALL_SET)
    for i, field in enumerate(order):
        r = c.post(f"/api/models/{slug}/defaults", json={field: None})
        assert r.status_code == 200, (field, r.status_code, r.text)
        launch = r.json()["launch"]
        if i == len(order) - 1:
            # everything cleared: no opinion at all, stored as no launch object
            assert launch is None, launch
            continue
        assert not launch.get(field), f"{field} did not clear: {launch}"
        for other in order[i + 1:]:
            assert launch[other] == ALL_SET[other], (field, other, launch)
    assert c.get(f"/api/models/{slug}/defaults").json()["launch"] is None


def test_a_model_that_has_been_used_is_not_a_first_load(client, home):
    """The first-load signal is honest about its evidence: `last_used` is
    written per completed turn, so "never used" is what it can say. It is
    exposed next to the flag so the UI can apply its own rule."""
    c, slug = client
    (home / "stats.json").write_text(
        json.dumps({"last_used": {slug: 1700000000}}), encoding="utf-8")
    got = c.get(f"/api/models/{slug}/defaults").json()
    assert got["launch"] is None
    assert got["last_used"] == 1700000000
    assert got["first_load"] is False


def test_a_registry_model_still_refuses_to_store_defaults(client):
    """Owner decision, unchanged: a shipped spec's launch settings are
    hand-authored and are not overwritten over HTTP. The dialog disables Save
    for `custom: false` and says why — but the GET must still answer, or the
    card cannot even say what the model will do."""
    c, _slug = client
    r = c.post("/api/models/qwen3-0.6b/defaults", json={"ctx": 4096})
    assert r.status_code == 400, r.text
    assert "registry model" in r.json()["error"]
    got = c.get("/api/models/qwen3-0.6b/defaults").json()
    assert got["custom"] is False and got["launch"] is None


def test_a_bad_kv_is_still_refused(client):
    """The new field must not have loosened the one validation the route had."""
    c, slug = client
    r = c.post(f"/api/models/{slug}/defaults", json={"kv": "q3_k"})
    assert r.status_code == 400 and "kv must be one of" in r.json()["error"]


def test_the_post_accepts_batch_ubatch_and_ngl(client):
    """C10: Rigma could already emit -b/-ub/-ngl, but no caller could ask for
    them, so they were unreachable from the UI. They must persist and read back
    on the surface the UI already loads."""
    c, slug = client
    r = c.post(f"/api/models/{slug}/defaults",
               json={"batch": 16384, "ubatch": 2048, "ngl": 40})
    assert r.status_code == 200, r.text
    launch = r.json()["launch"]
    assert (launch["batch"], launch["ubatch"], launch["ngl"]) == (16384, 2048,
                                                                  40)
    assert c.get(f"/api/models/{slug}/defaults").json()["launch"] == launch
    row = {m["slug"]: m for m in c.get("/api/models").json()["models"]}[slug]
    assert row["launch"]["ngl"] == 40
    # `ngl: 0` is a real request (every layer on the CPU), not "clear it"
    r = c.post(f"/api/models/{slug}/defaults", json={"ngl": 0})
    assert r.status_code == 200, r.text
    assert r.json()["launch"]["ngl"] == 0
    # …while null clears it back to "no opinion"
    r = c.post(f"/api/models/{slug}/defaults", json={"ngl": None})
    assert r.status_code == 200, r.text
    assert r.json()["launch"]["ngl"] == -1


def test_an_illegal_batch_pair_is_a_400_not_a_500(client):
    """llama.cpp refuses to start with `-ub` > `-b`, so the pair is refused at
    write time with a reason. The engine default batch (2048) applies when
    `batch` is unset, so a lone `ubatch: 4096` is refused too — and neither may
    surface as an uncaught pydantic ValidationError (a 500)."""
    c, slug = client
    r = c.post(f"/api/models/{slug}/defaults",
               json={"batch": 4096, "ubatch": 8192})
    assert r.status_code == 400, r.text
    assert "ubatch 8192 exceeds batch 4096" in r.json()["error"]
    r = c.post(f"/api/models/{slug}/defaults", json={"ubatch": 4096})
    assert r.status_code == 400, r.text
    assert "engine default" in r.json()["error"]
    assert c.get(f"/api/models/{slug}/defaults").json()["launch"] is None
