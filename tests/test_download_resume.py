"""A multi-GB pull WILL have its connection dropped. Resume existed, but nothing
retried, so one drop threw the whole transfer away:

    peer closed connection without sending complete message body
    (received 4438196071 bytes, expected 11845682560)

These tests use a server that drops mid-body on the first attempt and honours
Range on the retry.
"""
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest

from rigma import hangar

# 8 MiB: the reader pulls 1 MiB chunks, so a mid-body drop must land
# AFTER several chunks have already been written to the .part file
BODY = bytes(range(256)) * 32768


class _Flaky(BaseHTTPRequestHandler):
    drops_left = 1                        # how many transfers to cut short
    requests = []                         # the Range header of each attempt

    def do_GET(self):
        rng = self.headers.get("range", "")
        _Flaky.requests.append(rng)
        start = int(rng.split("=", 1)[1].split("-")[0]) if rng.startswith(
            "bytes=") else 0
        chunk = BODY[start:]
        self.send_response(206 if start else 200)
        self.send_header("content-length", str(len(chunk)))
        self.end_headers()
        if _Flaky.drops_left > 0:
            _Flaky.drops_left -= 1
            # send HALF the body, then hang up without finishing it
            self.wfile.write(chunk[: len(chunk) // 2])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(chunk)

    def log_message(self, *a):
        pass


@pytest.fixture
def flaky(monkeypatch):
    srv = HTTPServer(("127.0.0.1", 0), _Flaky)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _Flaky.drops_left, _Flaky.requests = 1, []
    port = srv.server_address[1]
    real = httpx.stream

    def _stream(method, url, **kw):       # send HF URLs to the test server
        return real(method,
                    url.replace("https://huggingface.co",
                                f"http://127.0.0.1:{port}"), **kw)

    monkeypatch.setattr(httpx, "stream", _stream)
    yield port
    srv.shutdown()


def test_download_resumes_after_a_dropped_connection(flaky, tmp_path):
    seen = []
    dest = tmp_path / "model.gguf"
    got = hangar._download_file("some/repo", "model.gguf", dest, seen.append)

    assert got == len(BODY)
    assert dest.read_bytes() == BODY, "resumed file must be byte-identical"
    assert not dest.with_name(dest.name + ".part").exists()
    # it really did drop once, then resume with a Range request
    assert len(_Flaky.requests) == 2
    assert _Flaky.requests[0] == ""
    assert _Flaky.requests[1].startswith("bytes=")
    assert seen and seen[-1] == len(BODY)


def test_download_gives_up_with_a_resumable_message(flaky, tmp_path,
                                                    monkeypatch):
    _Flaky.drops_left = 99                       # never completes
    monkeypatch.setattr(hangar, "DOWNLOAD_ATTEMPTS", 2)
    with pytest.raises(hangar.HangarError) as ei:
        hangar._download_file("some/repo", "model.gguf",
                              tmp_path / "m.gguf", lambda n: None)
    msg = str(ei.value)
    assert "resume" in msg and "bytes saved" in msg   # tells the user what to do
    # the partial is KEPT, so pressing Download again continues from there
    assert (tmp_path / "m.gguf.part").stat().st_size > 0
