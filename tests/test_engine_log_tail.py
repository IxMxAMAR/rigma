"""IMP-11: the engine-log route serves a BOUNDED tail.

The old reader pulled the WHOLE newest log into memory and sliced it, so a
server that had been up for days answered a 200-line panel request by pushing
hundreds of megabytes through the event loop. The bound is the fix; the tail
content is the feature.
"""
import pytest
from fastapi.testclient import TestClient

from rigma import server_ops
from rigma.serve import build_app


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _log(home, text, name="server-11500.log"):
    d = home / "logs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / name
    p.write_text(text, encoding="utf-8")
    return p


def _lines(n, width=50):
    return "".join(f"line {i}".ljust(width) + "\n" for i in range(n))


def test_tail_is_bounded_and_does_not_read_the_whole_file(home, monkeypatch):
    _log(home, "HEAD-MARKER\n" + _lines(5000))

    # The whole-file read IS the defect: fail if the reader touches read_text
    # on the log. read_tail_bytes opens and seeks instead.
    import pathlib
    real = pathlib.Path.read_text

    def _no_read_text(self, *a, **k):
        if str(self).endswith(".log"):
            raise AssertionError("the engine log was read whole")
        return real(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "read_text", _no_read_text)

    text, truncated = server_ops.log_tail_bounded(10)
    assert truncated is True
    assert text.splitlines() == [f"line {i}".ljust(50)
                                 for i in range(4990, 5000)]
    assert "HEAD-MARKER" not in text


def test_log_route_reports_truncation(home):
    _log(home, _lines(3000))                 # ~150 KB, past the 64 KB bound
    c = TestClient(build_app(upstream_port=1))
    r = c.get("/api/server/log", params={"lines": 12})
    assert r.status_code == 200
    assert r.headers["x-log-truncated"] == "1"
    assert r.text.splitlines() == [f"line {i}".ljust(50)
                                   for i in range(2988, 3000)]


def test_log_route_on_a_small_log_is_not_truncated(home):
    _log(home, "alpha\nbeta\n")
    c = TestClient(build_app(upstream_port=1))
    r = c.get("/api/server/log")
    assert r.status_code == 200
    assert r.text == "alpha\nbeta"
    assert r.headers["x-log-truncated"] == "0"


def test_log_route_with_no_log_is_empty(home):
    c = TestClient(build_app(upstream_port=1))
    r = c.get("/api/server/log")
    assert r.status_code == 200 and r.text == ""
    assert r.headers["x-log-truncated"] == "0"
