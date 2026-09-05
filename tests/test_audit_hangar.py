"""Hangar downloads: what a `.part` file is allowed to become.

A resume file used to be keyed on the destination basename alone. Two repos
that ship the same generic filename — `mmproj-F16.gguf` is in the registry
twice over, and quantiser names collide routinely — therefore shared one, so a
cancelled pull of the first model could finish as the second. Nothing reaped
those partials either, and a quant stored in a repo subdirectory could not be
downloaded at all.

Covers docs/audit-2026-09-04-full.md F27 and F28. Every byte here is
synthetic.
"""
import hashlib
import threading

import pytest

from rigma import hangar
from rigma.hangar import HangarError
from rigma.models import GgufFile, ModelSpec


class _Resp:
    """What httpx.stream returns: a context manager over a chunked body."""

    def __init__(self, code, body=b"", headers=None):
        self.status_code = code
        self._body = body
        self.headers = headers or {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def iter_bytes(self, n=1 << 20):
        for i in range(0, len(self._body), n):
            yield self._body[i:i + n]


class _Origin:
    """A stand-in for the HF file endpoint that behaves like one.

    It honours Range, answers 416 when the range starts past EOF, and declares
    lengths the way a real origin does — on a 206 `content-length` is the
    slice and only `content-range` carries the whole object. `truncate` cuts
    that many bodies in half while still promising the full length: the
    clean-but-short transfer that raises no exception at all.
    """

    def __init__(self, body, truncate=0):
        self.body, self.truncate = body, truncate
        self.requests = []                 # the range header of each attempt

    def stream(self, method, url, headers=None, **kw):
        self.requests.append((headers or {}).get("range", ""))
        total = len(self.body)
        start = 0
        rng = self.requests[-1]
        if rng.startswith("bytes="):
            start = int(rng.split("=", 1)[1].split("-")[0])
            if start >= total:
                return _Resp(416, b"", {"content-range": f"bytes */{total}"})
        chunk = self.body[start:]
        hdrs = {"content-length": str(len(chunk))}
        if start:
            hdrs["content-range"] = f"bytes {start}-{total - 1}/{total}"
        if self.truncate:
            self.truncate -= 1
            chunk = chunk[:len(chunk) // 2]
        return _Resp(206 if start else 200, chunk, hdrs)


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "home"))
    return tmp_path / "home"


@pytest.fixture
def nosleep(monkeypatch):
    """Retry backoff, recorded instead of waited: the pre-fix path for a
    missing directory slept 1+2+4+8+16 = 31 seconds before giving up."""
    slept = []
    monkeypatch.setattr("time.sleep", slept.append)
    return slept


def _legacy_part(dest, body):
    """A partial left by an older build: the raw `<name>.part` layout, with
    nothing recording which repo or file the bytes came from."""
    p = dest.with_name(dest.name + ".part")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(body)
    return p


def _spec(*ggufs):
    return ModelSpec(slug="s", family="qwen35", kind="dense", n_layers=8,
                     full_attn_layers=2, kv_heads=2, head_dim=64,
                     native_ctx=262144, custom=True, ggufs=list(ggufs))


# ---- F27: a partial has to prove what it is ---------------------------------

def test_a_partial_left_by_another_file_is_not_resumed_onto(tmp_path,
                                                            monkeypatch):
    """The splice. Resuming 1MB of model A onto model B appends B's tail to
    A's head and lands on EXACTLY B's length, so a size check cannot see it."""
    origin = _Origin(b"B" * (3 << 20))
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "mmproj-F16.gguf"
    _legacy_part(dest, b"A" * (1 << 20))
    n = hangar._download_file("owner/second", "mmproj-F16.gguf", dest,
                              lambda b: None)
    assert n == 3 << 20
    assert dest.read_bytes() == b"B" * (3 << 20)
    assert origin.requests == [""]        # started over, never asked to resume


def test_a_partial_bigger_than_the_object_is_not_installed_as_complete(
        tmp_path, monkeypatch):
    """A stale partial past EOF makes the origin answer 416. That used to be
    read as "the partial is already complete" and renamed into place."""
    origin = _Origin(b"B" * (2 << 20))
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model-Q4_K_M.gguf"
    _legacy_part(dest, b"A" * (5 << 20))
    n = hangar._download_file("owner/second", "model-Q4_K_M.gguf", dest,
                              lambda b: None)
    assert n == 2 << 20
    assert dest.read_bytes() == b"B" * (2 << 20)


def test_a_partial_this_file_wrote_is_still_resumed(tmp_path, monkeypatch):
    """The point of the .part file survives: matching bytes resume."""
    body = b"C" * (3 << 20)
    origin = _Origin(body)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"
    _legacy_part(dest, body[:1 << 20])
    hangar._claim_partial(dest, "owner/first", "model.gguf")
    n = hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None)
    assert origin.requests == [f"bytes={1 << 20}-"]
    assert n == len(body) and dest.read_bytes() == body


def test_a_complete_partial_that_gets_a_416_is_installed_not_refetched(
        tmp_path, monkeypatch):
    """The one honest 416: the process died between the last chunk and the
    rename. Re-downloading gigabytes there would be its own bug."""
    body = b"D" * (2 << 20)
    origin = _Origin(body)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"
    _legacy_part(dest, body)
    hangar._claim_partial(dest, "owner/first", "model.gguf")
    n = hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None)
    assert n == len(body) and dest.read_bytes() == body
    assert origin.requests == [f"bytes={2 << 20}-"]   # one request, no refetch


def test_a_body_that_ends_early_is_not_installed_as_the_whole_file(
        tmp_path, monkeypatch, nosleep):
    """A short body that closes cleanly raises nothing. It used to be renamed
    into place as a complete model, which fails later as a corrupt gguf."""
    body = b"E" * (4 << 20)
    origin = _Origin(body, truncate=1)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"
    n = hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None)
    assert n == len(body) and dest.read_bytes() == body
    assert len(origin.requests) == 2 and origin.requests[1].startswith("bytes=")


def test_a_file_that_fails_its_sha256_is_discarded_not_installed(tmp_path,
                                                                 monkeypatch):
    """GgufFile.sha256 was declared in models.py and read nowhere. It is the
    only check that can catch a spliced file, whose length is exactly right."""
    origin = _Origin(b"F" * (1 << 20))
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"
    with pytest.raises(HangarError, match="sha256"):
        hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None, sha256="0" * 64)
    assert not dest.exists()
    # and the bad bytes are gone, not left for the next attempt to resume onto
    assert not dest.with_name(dest.name + ".part").exists()


def test_a_file_that_matches_its_sha256_installs(tmp_path, monkeypatch):
    body = b"G" * (1 << 20)
    origin = _Origin(body)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"
    n = hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None,
                              sha256=hashlib.sha256(body).hexdigest().upper())
    assert n == len(body) and dest.read_bytes() == body


def test_a_pull_tells_the_downloader_what_the_registry_says_the_file_is(
        home, monkeypatch):
    """gguf.bytes was in scope in start_pull and drove nothing but the
    progress bar; gguf.sha256 was never read at all."""
    spec = _spec(GgufFile(repo="owner/first", file="a.gguf", bytes=4242,
                          quant="Q4_K_M", sha256="ab" * 32))

    class _Reg:
        models = {"s": spec}

    seen = {}

    def _fake(repo, file, dest, report, *, expect_bytes=0, sha256=None):
        seen.update(repo=repo, file=file, expect_bytes=expect_bytes,
                    sha256=sha256)
        return 0

    monkeypatch.setattr(hangar, "_download_file", _fake)
    monkeypatch.setattr(hangar, "_PULLS", {})
    hangar.start_pull("s", "a.gguf", registry=_Reg())
    for t in threading.enumerate():
        if t.name == "pull:a.gguf":
            t.join(timeout=5)
    assert seen["repo"] == "owner/first" and seen["file"] == "a.gguf"
    assert seen["expect_bytes"] == 4242
    assert seen["sha256"] == "ab" * 32


def test_deleting_a_file_takes_its_resume_file_with_it(home, monkeypatch):
    """Nothing reaped a partial: it outlived the model, and glob("*.gguf")
    does not match ".part", so multi-GB of it was invisible in the library."""
    spec = _spec(GgufFile(repo="owner/first", file="a.gguf", bytes=16,
                          quant="Q4_K_M"))

    class _Reg:
        models = {"s": spec}

    target = hangar.models_dir() / "a.gguf"
    target.write_bytes(b"g" * 16)
    part = _legacy_part(target, b"A" * 32)
    hangar.delete_file("s", "a.gguf", registry=_Reg())
    assert not target.exists()
    assert not part.exists()


def test_deleting_a_model_takes_its_resume_files_with_it(home):
    spec = _spec(GgufFile(repo="owner/first", file="a.gguf", bytes=16,
                          quant="Q4_K_M"))
    hangar._write_spec(spec)
    target = hangar.models_dir() / "a.gguf"
    target.write_bytes(b"g" * 16)
    part = _legacy_part(target, b"A" * 32)
    hangar.delete_model("s")
    assert not target.exists()
    assert not part.exists()


# ---- F28: quants that live in a repo subdirectory ---------------------------

def test_a_quant_nested_in_a_repo_subdir_downloads(tmp_path, monkeypatch,
                                                   nosleep):
    """repo_files lists the tree recursively, so `file` can be
    "Q4_K_M/model.gguf". Opening the .part without creating the directory
    raised FileNotFoundError inside the retry loop, so the button reported six
    dropped connections for a download that had never started."""
    body = b"H" * (2 << 20)
    origin = _Origin(body)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "models" / "Q4_K_M" / "model.gguf"
    n = hangar._download_file("owner/first", "Q4_K_M/model.gguf", dest,
                              lambda b: None)
    assert n == len(body) and dest.read_bytes() == body
    assert len(origin.requests) == 1                  # no retry storm


def test_a_filesystem_error_is_not_reported_as_a_dropped_connection(
        tmp_path, monkeypatch, nosleep):
    """31 seconds of "resuming" and then "your connection kept dropping" for
    something no retry can fix."""
    origin = _Origin(b"I" * 4096)
    monkeypatch.setattr("httpx.stream", origin.stream)
    dest = tmp_path / "model.gguf"

    # Fail the write itself, rather than putting a directory in its place. A
    # directory is not a portable stand-in for "the filesystem said no": its
    # st_size is 4096 on Linux and 0 on Windows, and `_download_file` reads that
    # as bytes already fetched. With a 4096-byte body the Linux run therefore
    # decided the download was ALREADY COMPLETE and never opened anything, so
    # the test passed on Windows and failed on the runner.
    real_open = open

    def _refuse_the_part(path, *a, **kw):
        if str(path).endswith(".part"):
            raise OSError(28, "No space left on device")
        return real_open(path, *a, **kw)

    monkeypatch.setattr("builtins.open", _refuse_the_part)
    with pytest.raises(HangarError) as ei:
        hangar._download_file("owner/first", "model.gguf", dest,
                              lambda b: None)
    msg = str(ei.value)
    assert "kept dropping" not in msg and "resume" not in msg
    assert nosleep == []                          # gave up at once
    assert len(origin.requests) <= 1


def test_refresh_keeps_a_quant_that_lives_in_a_repo_subdirectory(home):
    """`on_disk` decides whether a file the repo no longer lists is forgotten.
    A name-only glob never saw a nested quant, so refreshing dropped it."""
    nested = hangar.models_dir() / "Q4_K_M" / "model.gguf"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_bytes(b"g" * 16)
    old = _spec(GgufFile(repo="acme/spicy", file="Q4_K_M/model.gguf",
                         bytes=16, quant="Q4_K_M"))
    new = hangar.merge_repo_files(
        old, {"ggufs": [{"file": "other.gguf", "bytes": 100}],
              "mmproj": None, "split_skipped": 0})
    assert "Q4_K_M/model.gguf" in {g.file for g in new.ggufs}


def test_refresh_still_forgets_a_flat_quant_that_was_never_downloaded(home):
    """The mirror rule has to keep working: reading the tree must not start
    treating everything as present."""
    old = _spec(GgufFile(repo="acme/spicy", file="gone.gguf", bytes=100,
                         quant="Q3_K_M"))
    new = hangar.merge_repo_files(
        old, {"ggufs": [{"file": "kept.gguf", "bytes": 100}],
              "mmproj": None, "split_skipped": 0})
    assert [g.file for g in new.ggufs] == ["kept.gguf"]
