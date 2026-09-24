"""R3-11: the credential denylist had a second door, and the index was behind it.

The 13-2 denylist is enforced on the TOOL read path. `rag.add_source` took any path
with no validation at all, and everything under it was embedded into the local
vector index — so adding a home directory or Documents put `.env`, `.ssh/id_rsa`,
`.git-credentials` and browser cookie DBs into the index, where
`search_my_documents` (a `safe=True`, AUTO-RUN tool) retrieved them with no
confirmation and no human in the loop. That defeats the premise the denylist is
built on — "the model has no legitimate reason to put a key into the conversation" —
through a door the fix never considered part of the same surface.

The fix derives raggity's `exclude` list from the SAME tuples the read path uses, so
a pattern added there is excluded from indexing by construction. A second
hand-maintained copy of a denylist is exactly how R3-2 and R3-11 both happened.

MEASURED AGAINST REAL RAGGITY 0.13.0, which is the only reason the `.*` variants
below exist. With the plain derived list, `credentials.md`, `my.api_key.md`,
`credentials.json.md`, `token_api_key.txt` and `server.pem.txt` were all STILL
indexed and all still retrievable — `**/credentials` and `**/*.pem` are exact
basenames, and an appended extension defeats both. A README about key rotation was
carrying its name into an index an auto-run tool reads. The reachable corpus that
proved it is reproduced in the commit message; a unit test cannot re-run raggity, so
what is pinned here is the SHAPE of the list, and the real-sidecar measurement is
recorded rather than asserted.
"""
import fnmatch

import pytest

from rigma import rag, tools


def test_every_denied_credential_file_is_excluded_from_the_index():
    """THE REGRESSION GUARD. If this fails, someone added a pattern to the read
    denylist and did not get it excluded from indexing — which is the bug."""
    globs = set(tools.credential_exclude_globs())
    for pat in tools._CREDENTIAL_FILES:
        assert f"**/{pat}" in globs, (
            f"{pat!r} is denied on the read path but would still be INDEXED")


def test_every_denied_credential_directory_is_excluded():
    globs = set(tools.credential_exclude_globs())
    for d in tools._CREDENTIAL_DIRS:
        assert f"**/{d}/**" in globs, (
            f"{d!r} is a denied directory but its contents would be INDEXED")


def test_a_credential_name_with_an_extension_appended_is_excluded():
    """The measured gap. Each of these WAS indexed and retrievable against real
    raggity before the `.*` variants were added."""
    globs = tools.credential_exclude_globs()
    for name in ("credentials.md", "my.api_key.md", "credentials.json.md",
                 "token_api_key.txt", "server.pem.txt", "notes.pem.txt"):
        assert any(fnmatch.fnmatch(name, g.replace("**/", "")) or
                   fnmatch.fnmatch(name, g) for g in globs), (
            f"{name!r} would be indexed")


def test_the_list_is_derived_not_copied():
    """A hand-maintained second copy is how this class of bug happens. Adding a
    pattern to the read denylist must be the ONLY edit needed."""
    before = set(tools.credential_exclude_globs())
    extra = ("a_pattern_that_did_not_exist",)
    original = tools._CREDENTIAL_FILES
    try:
        tools._CREDENTIAL_FILES = original + extra
        after = set(tools.credential_exclude_globs())
    finally:
        tools._CREDENTIAL_FILES = original
    assert f"**/{extra[0]}" in after - before


def test_the_browser_profile_secrets_are_excluded():
    """A profile directory holds saved logins and cookies; the read path matches it
    by path SHAPE, which a glob list cannot express, so the concrete file names it
    protects are named explicitly."""
    globs = set(tools.credential_exclude_globs())
    for name in ("Login Data", "Cookies", "cookies.sqlite", "logins.json",
                 "key4.db"):
        assert f"**/{name}" in globs


def test_the_generated_config_carries_the_exclusions(tmp_path, monkeypatch):
    """End to end through the config raggity actually reads: `write_rag_config` must
    emit an `exclude` key, not just an `include` one."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    src = tmp_path / "docs"
    src.mkdir()
    rag.add_source(str(src))
    text = (rag.rag_dir() / "raggity.toml").read_text(encoding="utf-8")
    assert "exclude = [" in text
    assert "**/.env" in text
    assert "**/.ssh/**" in text
    # and it is still valid TOML, which is what raggity will do with it
    import tomllib
    parsed = tomllib.loads(text)
    assert parsed["sources"]["exclude"], "exclude list came out empty"
    assert parsed["sources"]["include"] == [f"{src.as_posix()}/**/*"]


def test_an_excluded_list_does_not_break_the_include(tmp_path, monkeypatch):
    """The two keys are separate; adding excludes must not drop sources."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    rag.add_source(str(a))
    rag.add_source(str(b))
    import tomllib
    text = (rag.rag_dir() / "raggity.toml").read_text(encoding="utf-8")
    parsed = tomllib.loads(text)
    assert parsed["sources"]["include"] == [f"{a.as_posix()}/**/*",
                                            f"{b.as_posix()}/**/*"]
    assert len(parsed["sources"]["exclude"]) > 20


@pytest.mark.parametrize("pat", [".env", "id_rsa", "*.pem", "credentials"])
def test_the_globs_are_glob_shaped(pat):
    """raggity matches these against a POSIX path, so a bare basename with no `**/`
    would only match at the root of a source folder — the bug would then be
    invisible for every nested file."""
    assert f"**/{pat}" in tools.credential_exclude_globs()
