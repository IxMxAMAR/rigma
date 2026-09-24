"""R3-2: `_fuzzy_file`'s near-miss recovery hands back a file the credential
denylist had just refused.

`_read_path` (and `_write_path`) decide on the name the MODEL gave. `_fuzzy_file`
then runs on the NOT-FOUND path and substitutes the closest real sibling — so
`read_file('.en')` resolves onto `.env`, `read_file('id_rs')` onto `id_rsa`, and
the recovery note even names the file it used. The credential check is never
re-run on the substitute, so the denylist the 13-2 fix added is one dropped
character deep.
"""
import pytest

from rigma import tools


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    (d / ".env").write_text("SECRET=sk-live-1234\n", encoding="utf-8")
    (d / "id_rsa").write_text("PRIVATE-KEY-MATERIAL\n", encoding="utf-8")
    (d / "credentials").write_text("aws_secret=AKIA\n", encoding="utf-8")
    (d / "notes.txt").write_text("ordinary\n", encoding="utf-8")
    return d


@pytest.mark.parametrize("typo,real", [(".en", ".env"),
                                       ("id_rs", "id_rsa"),
                                       ("credential", "credentials")])
def test_the_denylist_survives_a_near_miss(ws, typo, real):
    # the exact name IS refused (the 13-2 fix, still true)
    refused = tools.run_tool("read_file", {"path": real},
                             {"workspace": str(ws)})
    assert refused.startswith("error") and "credential" in refused.lower()

    # ...and so must the typo that fuzzy-resolves onto it
    out = tools.run_tool("read_file", {"path": typo}, {"workspace": str(ws)})
    assert out.startswith("error"), out
    assert "SECRET" not in out and "PRIVATE-KEY" not in out and "AKIA" not in out


def test_a_near_miss_on_an_ordinary_file_still_works(ws):
    """The recovery itself must not be lost — that is why it exists."""
    out = tools.run_tool("read_file", {"path": "notes.tx"}, {"workspace": str(ws)})
    assert out.strip().endswith("ordinary"), out


def test_move_files_cannot_launder_a_credential_by_typo(ws):
    out = tools.run_tool(
        "move_files", {"paths": ["id_rs"], "dest": "public"},
        {"workspace": str(ws), "allow_code": True})
    assert out.startswith("error"), out
    assert not (ws / "public" / "id_rsa").exists()


def test_the_recovery_note_never_claims_it_used_a_denied_file(ws):
    """The note used to read "you asked for '.en' — used '.env'" and then hand
    back the bytes. Naming a sibling is not a secret (list_directory shows the
    same names); reading it is."""
    out = tools.run_tool("read_file", {"path": ".en"}, {"workspace": str(ws)})
    assert "used" not in out, out
    assert "SECRET" not in out, out
    assert out.startswith("error"), out


def test_fuzzy_file_itself_refuses_a_denied_candidate(ws):
    got, note = tools._fuzzy_file(ws / ".en", {"workspace": str(ws)})
    assert got is None and note == ""
