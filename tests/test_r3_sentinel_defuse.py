"""R3-3: declaring `sentinel` at registration exempted the tool's ORDINARY
results from control-byte defusing.

04-7 replaced "sniff the result text for a sentinel" with "a property of the
tool". The property is right, but `run_tool` then returns the result verbatim
for the WHOLE tool — and a sentinel tool's error path embeds the model's own
argument (`_resolve_image`'s "no such file: <the path you gave>"), so
`view_image`/`view_images` became the one way to deliver a raw NUL run to the
model: the exact instant-EOS poison `_defuse_control_bytes` exists to stop, and
the one tool class that cannot be defused by the choke point.
"""
import pytest

from rigma import tools

NULS = "\u0000" * 300


def _ctx(**kw):
    return {"workspace": kw.pop("workspace", ""), "has_vision": True, **kw}


@pytest.fixture
def ws(tmp_path):
    d = tmp_path / "ws"
    d.mkdir()
    return d


def test_view_image_error_path_is_defused(ws):
    out = tools.run_tool("view_image",
                         {"path": "C:\\nope\\" + NULS + "x.png"},
                         _ctx(workspace=str(ws)))
    assert "\x00" not in out, f"{out.count(chr(0))} NUL(s) survived"
    assert "unreadable control byte" in out


def test_view_images_error_path_is_defused(ws):
    out = tools.run_tool("view_images",
                         {"paths": ["C:\\nope\\" + NULS + "x.png"]},
                         _ctx(workspace=str(ws)))
    assert "\x00" not in out


def test_a_confined_view_images_refusal_is_defused(ws):
    out = tools.run_tool("view_images", {"folder": "C:\\x\\" + NULS},
                         _ctx(workspace=str(ws), profile="confined"))
    assert "\x00" not in out


def test_a_real_image_sentinel_still_round_trips(ws, tmp_path):
    """The fix must not corrupt the sentinel: serve.py splits on the first NUL
    and the path list must survive intact."""
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 32)
    out = tools.run_tool("view_image", {"path": str(img)},
                         _ctx(workspace=str(ws), allow_absolute_reads=True))
    assert out.startswith(tools.IMAGE_SENTINEL)
    payload = out[len(tools.IMAGE_SENTINEL):].partition("\x00")[0]
    assert payload.strip() == str(img.resolve())


def test_the_other_sentinels_still_round_trip(ws):
    loop = _ctx(workspace=str(ws), can_host_loop_tools=True)
    assert tools.run_tool("use_tools", {"names": ["view_image"]},
                          loop).startswith(tools.USE_TOOLS_SENTINEL)
    assert tools.run_tool("delegate", {"question": "q"},
                          loop).startswith(tools.DELEGATE_SENTINEL)
