"""store.js sseParse unit tests - run with node when available."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

STORE_JS = Path(__file__).parent.parent / "src" / "rigma" / "data" / "ui" / "store.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")


def parse(src: str):
    driver = (STORE_JS.read_text(encoding="utf-8")
              + f"\nprocess.stdout.write(JSON.stringify(sseParse({json.dumps(src)})));")
    res = subprocess.run(["node", "-"], input=driver, capture_output=True,
                         text=True, timeout=10)
    assert res.returncode == 0, res.stderr
    return json.loads(res.stdout)


def test_plain_data_frame():
    out = parse('data: {"delta": "hi"}\n\n')
    assert out == {"events": [{"event": "", "data": '{"delta": "hi"}'}], "rest": ""}


def test_named_event_frame():
    out = parse('event: meta\ndata: {"ctx": 4096}\n\n')
    assert out["events"] == [{"event": "meta", "data": '{"ctx": 4096}'}]


def test_partial_frame_stays_in_rest():
    out = parse('data: {"delta": "a"}\n\ndata: {"del')
    assert len(out["events"]) == 1 and out["rest"] == 'data: {"del'


def test_done_and_multiple_frames():
    src = ('data: {"delta": "a"}\n\nevent: error\ndata: {"message": "x"}\n\n'
           "data: [DONE]\n\n")
    out = parse(src)
    assert [e["event"] for e in out["events"]] == ["", "error", ""]
    assert out["events"][2]["data"] == "[DONE]" and out["rest"] == ""
