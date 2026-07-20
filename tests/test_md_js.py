"""md.js unit tests - run with node when available (zero-dep UI has no JS harness)."""
import json
import shutil
import subprocess
from pathlib import Path

import pytest

MD_JS = Path(__file__).parent.parent / "src" / "rigma" / "data" / "ui" / "md.js"

pytestmark = pytest.mark.skipif(shutil.which("node") is None,
                                reason="node not installed")


def render(src: str) -> str:
    driver = (MD_JS.read_text(encoding="utf-8")
              + f"\nprocess.stdout.write(renderMarkdown({json.dumps(src)}));")
    res = subprocess.run(["node", "-"], input=driver, capture_output=True,
                         text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return res.stdout


def test_four_backtick_fence_terminates():
    out = render("````\ncode\n````\nafter")
    assert "<pre" in out and "after" in out


def test_fence_with_trailing_text_terminates():
    out = render("```js extra\ncode\n```")
    assert "<pre" in out and "code" in out


def test_unclosed_fence_renders():
    assert "still code" in render("```py\nstill code")


def test_html_is_escaped():
    out = render("<script>alert(1)</script>")
    assert "<script>" not in out and "&lt;script&gt;" in out


def test_javascript_url_not_linked():
    assert "<a" not in render("[x](javascript:alert(1))")


def test_blockquote_and_heading():
    out = render("# title\n> quoted")
    assert "<h3>title</h3>" in out and "<blockquote>quoted</blockquote>" in out
