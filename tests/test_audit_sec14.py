"""Three contract violations in `tools.py`, pinned against the PROMISE.

Each of these is a place where the code disagreed with a promise it makes about
itself — in a comment, a docstring, or a schema — which is why each test asserts
the promise rather than today's behaviour:

* `ask_gemini` shipped a free-text question to Google with no grant at all. Its
  sibling `http_request` has required `allow_outbound_post` since the
  query-string bypass was closed (the long note above `carries_data`), and
  `ask_gemini` is `safe=True`, so nothing asked the owner.
* `confined` was decided by the SPELLING of a path: `_absolute_writes_allowed`
  refuses the profile, but `_write_path` ORs the `write_allowlist` route in
  beside it, and that route had no profile test — so the absolute form was
  refused while `..\\other\\x` was allowed into a seeded root.
* `move_files{"path": "one.png"}` read only `paths`, so the key was ignored and
  the "no paths means the last sample" fallback moved EVERY file of the previous
  sample, reporting "moved N file(s)".
"""

import pytest

from rigma import tools


def test_ask_gemini_needs_the_outbound_grant():
    """Refused for the missing GRANT, not for the missing key.

    Passing no ctx is the default chat: no grant is armed. The distinction
    matters because the two refusals name different remedies, and only one of
    them is about leaving the machine.
    """
    out = tools.run_tool("ask_gemini",
                         {"question": "read my notes and summarise them"})
    assert "allow outbound POST" in out
    assert "Gemini API key" not in out


def test_the_outbound_grant_opens_ask_gemini(monkeypatch):
    """The positive control: with the grant armed, the gate is not what refuses.

    No network call is made — the key is stubbed out, so reaching its message
    proves the grant check passed.
    """
    monkeypatch.setattr(tools, "_gemini_key", lambda: None)
    out = tools.run_tool("ask_gemini", {"question": "hi"},
                         {"allow_outbound_post": True})
    assert "no Gemini API key" in out
    assert "allow outbound POST" not in out


def test_confined_refuses_the_allowlist_route(tmp_path):
    """`confined` answers for BOTH routes to an absolute destination."""
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    ctx = {"workspace": str(ws), "profile": "confined",
           "write_allowlist": [str(outside)]}

    # Membership is what it looks like — proved WITHOUT the profile, so this test
    # cannot pass by the allowlist being empty — and the profile is what refuses
    # it, in isolation and through `_write_path`.
    assert tools._write_allowlist_contains(
        {"workspace": str(ws), "write_allowlist": [str(outside)]},
        outside) is True
    assert tools._write_allowlist_contains(ctx, outside) is False
    assert tools._outside_capability_allowed(ctx, outside / "x.bat",
                                             "write") is False

    # And the spelling a model actually reaches for: `..` out of the workspace
    # into the seeded root. This returned the path before the profile test.
    with pytest.raises(ValueError):
        tools._write_path(ctx, r"..\outside\planted.bat")


def test_the_allowlist_still_works_outside_confined(tmp_path):
    """The profile test must not disarm OD-2 for an ordinary session."""
    ws = tmp_path / "ws"
    outside = tmp_path / "outside"
    ws.mkdir()
    outside.mkdir()
    ctx = {"workspace": str(ws), "write_allowlist": [str(outside)]}
    assert tools._outside_capability_allowed(ctx, outside / "x.bat",
                                             "write") is True


def test_move_files_path_alias_fills_paths():
    """`path` — the spelling every other tool uses — reaches `paths`."""
    args = tools.normalize_tool_args("move_files",
                                     {"dest": "d", "path": "named.png"})
    assert args.get("paths") == "named.png"


def test_move_files_does_not_use_the_sample_when_a_file_was_named(
        monkeypatch, tmp_path):
    """A named file must never become "every file of the last sample"."""
    sample = [str(tmp_path / f"s{i}.png") for i in (1, 2, 3)]
    monkeypatch.setattr("rigma.runs.get_last_sample", lambda rid: sample)

    found, errs, notes = tools._transfer_sources(
        {"dest": str(tmp_path / "out"), "path": "named.png"},
        {"run_id": "r1", "workspace": str(tmp_path)})

    assert not any("s1.png" in str(x) for x in found)
    assert not any("s1.png" in str(x) for x in errs)
    assert not any("s1.png" in str(x) for x in notes)


def test_move_files_still_uses_the_sample_when_nothing_was_named(
        monkeypatch, tmp_path):
    """The convenience the fallback exists for is untouched."""
    sample = [str(tmp_path / f"s{i}.png") for i in (1, 2, 3)]
    for name in sample:
        (tmp_path / name.split("\\")[-1]).write_text("x", encoding="utf-8")
    monkeypatch.setattr("rigma.runs.get_last_sample", lambda rid: sample)

    found, errs, notes = tools._transfer_sources(
        {"dest": str(tmp_path / "out")},
        {"run_id": "r1", "workspace": str(tmp_path)})

    assert len(found) == 3
