"""R3-7: `_resolve_image` never consulted the credential denylist.

`read_file`, `list_directory`, `find_files` and `sample_files` all resolve
through `_read_path`, which refuses a credential path even when absolute reads
are granted. The image tools take the other branch (`_resolve_image`), which
applied neither the denylist nor the grant.

R3-7 closed the denylist half. R3-TOOL-5 closed the grant half: an absolute
image outside the workspace is no longer readable without `allow_absolute_reads`,
which is what `view_images(folder=…)`, `list_directory` and `read_file` always
required. The pinned tests in `test_run_tools.py` that asserted the ungated
behaviour now supply the grant, so the capability is still covered — it is just
no longer automatic.
"""
import pytest

from rigma import tools


def _png(p):
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)
    return p


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    (d / ".ssh").mkdir(parents=True)
    _png(d / ".ssh" / "a.png")
    _png(d / "pic.png")
    return d


def test_view_image_refuses_a_credential_directory(ws):
    out = tools.run_tool("view_image", {"path": ".ssh/a.png"},
                         {"workspace": str(ws), "has_vision": True})
    assert out.startswith("error") and "credential" in out.lower(), out


def test_view_images_refuses_a_credential_directory(ws):
    out = tools.run_tool("view_images", {"paths": [".ssh/a.png"]},
                         {"workspace": str(ws), "has_vision": True})
    assert out.startswith("error") and "credential" in out.lower(), out


def test_view_images_folder_mode_agrees(ws):
    """Folder mode already went through `_read_path`; the two must agree."""
    out = tools.run_tool("view_images", {"folder": ".ssh"},
                         {"workspace": str(ws), "has_vision": True})
    assert out.startswith("error"), out


def test_a_browser_profile_image_is_refused(ws):
    prof = ws / "AppData" / "Local" / "Google" / "Chrome" / "User Data"
    prof.mkdir(parents=True)
    _png(prof / "avatar.png")
    out = tools.run_tool(
        "view_image",
        {"path": str(prof / "avatar.png")},
        {"workspace": str(ws), "has_vision": True, "allow_absolute_reads": True})
    assert out.startswith("error") and "refusing to read" in out, out


def test_an_ordinary_image_still_works(ws, tmp_path):
    rel = tools.run_tool("view_image", {"path": "pic.png"},
                         {"workspace": str(ws), "has_vision": True})
    assert rel.startswith(tools.IMAGE_SENTINEL)
    # R3-TOOL-5: an image OUTSIDE the workspace now needs the reads grant. It was
    # the one read path that skipped it — `view_images(folder=…)`,
    # `list_directory` and `read_file` all enforced it, and `view_image` was how a
    # prompt-injected model pulled any image off the disk into the conversation,
    # from where `fetch_url` could carry it out. Images outside the workspace are
    # a real need, which is exactly why the grant exists.
    other = _png(tmp_path / "elsewhere.png")
    base = {"workspace": str(ws), "has_vision": True}
    refused = tools.run_tool("view_image", {"path": str(other)}, base)
    assert "outside the workspace" in refused, refused
    assert tools.IMAGE_SENTINEL not in refused

    absolute = tools.run_tool("view_image", {"path": str(other)},
                              {**base, "allow_absolute_reads": True})
    assert absolute.startswith(tools.IMAGE_SENTINEL)
