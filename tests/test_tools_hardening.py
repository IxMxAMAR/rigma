"""Gemini 4-agent review 2026-07-18: DoS/SSRF/robustness fixes."""
import sys
import time

from rigma import tools


def test_calculator_pow_chain_fails_fast():
    t0 = time.monotonic()
    out = tools.run_tool("calculator", {"expression": "9**9**9**9"})
    assert "too large" in out
    assert time.monotonic() - t0 < 1.0        # rejected BEFORE computing


def test_calculator_still_does_normal_powers():
    assert tools.run_tool("calculator", {"expression": "2 ** 10"}) == "1024"
    assert tools.run_tool("calculator", {"expression": "2**64"}) \
        == "18446744073709551616"


def test_ssrf_ipv4_mapped_ipv6_blocked():
    # ::ffff:127.0.0.1 reports is_loopback=False on the raw v6 object
    assert tools._is_public_host("::ffff:127.0.0.1") is False
    assert tools._is_public_host("::ffff:169.254.169.254") is False


def test_run_python_stdin_does_not_hang():
    t0 = time.monotonic()
    out = tools.run_tool("run_python", {"code": "print(input())"},
                         {"allow_code": True})
    # EOF on stdin -> instant EOFError, not a 30s timeout
    assert time.monotonic() - t0 < 10
    assert "timed out" not in out


def test_view_image_gated_on_vision():
    # not offered without vision, and refused if called anyway
    base = {t["function"]["name"] for t in tools.tool_specs()}
    assert "view_image" not in base
    withvis = {t["function"]["name"] for t in tools.tool_specs(has_vision=True)}
    assert "view_image" in withvis
    assert "can't see images" in tools.run_tool(
        "view_image", {"path": "x.png"}, {"has_vision": False})


def test_view_image_returns_sentinel_for_real_image(tmp_path):
    img = tmp_path / "pic.png"
    img.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 64)   # .png ext is enough
    out = tools.run_tool("view_image", {"path": str(img)},
                         {"has_vision": True})
    assert out.startswith(tools.IMAGE_SENTINEL)
    assert out.endswith("pic.png")


def test_view_image_rejects_non_image(tmp_path):
    doc = tmp_path / "notes.txt"
    doc.write_text("hi")
    out = tools.run_tool("view_image", {"path": str(doc)}, {"has_vision": True})
    assert "not an image" in out


def test_view_image_missing_file(tmp_path):
    # the path must be guaranteed-missing on THIS machine: a hard-coded "D:/..."
    # depends on the state of one drive (D: is BitLocker-locked here, so Windows
    # answers WinError -2144272384 instead of "no such file").
    out = tools.run_tool("view_image",
                         {"path": str(tmp_path / "nope" / "gone.png")},
                         {"has_vision": True})
    assert "no such file" in out


def test_view_image_refuses_a_directory_named_like_an_image(tmp_path):
    """A DIRECTORY named `adir.png` must not come back as a viewable image.

    `_fuzzy_file`'s exists() accepts a directory, the suffix check passes, and
    stat() succeeds on a directory — so `_resolve_image` returned the path and
    `_view_image` emitted the IMAGE_SENTINEL: a FALSE SUCCESS that hands the
    model a "picture" it can never describe. `read_file` and `view_images` both
    re-check is_file() after the fuzzy repair; view_image is the odd one out."""
    d = tmp_path / "adir.png"
    d.mkdir()
    out = tools.run_tool("view_image", {"path": str(d)}, {"has_vision": True})
    assert tools.IMAGE_SENTINEL not in out
    assert "no such file" in out


def _raise_stat_for_locked(monkeypatch, message="device not ready"):
    """Make every stat of a path containing 'locked' raise errno 22 — what a
    BitLocker-locked volume, an offline network share or a device-not-ready
    answer actually does. errno 22 is outside pathlib's ignored set, so
    `exists()`/`is_file()`/`is_dir()` all RAISE rather than return False."""
    import pathlib
    real_stat = pathlib.Path.stat

    def stat(self, *a, **k):
        if "locked" in str(self):
            raise OSError(22, message)
        return real_stat(self, *a, **k)

    monkeypatch.setattr(pathlib.Path, "stat", stat)


def test_fuzzy_file_answers_none_when_the_parent_cannot_be_statd(
        tmp_path, monkeypatch):
    """_fuzzy_file has exactly two outcomes — "here it is" and "not there" — and
    an unstatable path is the second. Before the guard, `p.exists()` re-raised
    the raw OS error instead of returning (None, "")."""
    _raise_stat_for_locked(monkeypatch)
    got, note = tools._fuzzy_file(tmp_path / "locked" / "gone.png")
    assert got is None and note == ""


def test_view_image_unstatable_path_answers_no_such_file(tmp_path, monkeypatch):
    """The tool must answer its OWN "no such file" (which the model can act on),
    not leak the drive's error text — the live D:-drive BitLocker failure."""
    _raise_stat_for_locked(monkeypatch)
    out = tools.run_tool("view_image",
                         {"path": str(tmp_path / "locked" / "gone.png")},
                         {"has_vision": True})
    assert "no such file" in out
    assert "device not ready" not in out
    assert "Errno 22" not in out


def test_read_file_unstatable_path_answers_no_such_file(tmp_path, monkeypatch):
    """read_file reaches _fuzzy_file through the same is_file() probe, so it gets
    the same answer rather than the raw OS error."""
    _raise_stat_for_locked(monkeypatch)
    out = tools.run_tool("read_file",
                         {"path": str(tmp_path / "locked" / "gone.txt")},
                         {"workspace": str(tmp_path)})
    assert "no such file" in out
    assert "device not ready" not in out


def test_ask_gemini_is_a_safe_tool():
    base = {t["function"]["name"] for t in tools.tool_specs()}
    assert "ask_gemini" in base                # available without opt-in


def test_ask_gemini_empty_question():
    assert "empty question" in tools.run_tool("ask_gemini", {"question": " "})


def test_ask_gemini_no_key(monkeypatch):
    monkeypatch.setattr(tools, "_gemini_key", lambda: None)
    # The outbound grant is checked FIRST (`_ask_gemini`), so reaching the key
    # answer at all now requires it — otherwise this test would be asserting the
    # gate's message under a name that claims to test the key's.
    out = tools.run_tool("ask_gemini", {"question": "hi"},
                         {"allow_outbound_post": True})
    assert "no Gemini API key" in out


def test_view_images_batch(tmp_path):
    for i in range(3):
        (tmp_path / f"p{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * 40)
    paths = [str(tmp_path / f"p{i}.png") for i in range(3)]
    out = tools.run_tool("view_images", {"paths": paths}, {"has_vision": True})
    assert out.startswith(tools.IMAGE_SENTINEL)
    body = out[len(tools.IMAGE_SENTINEL):]
    assert body.count("\n") == 2                 # 3 paths, newline-separated


def test_view_images_gated_and_caps(tmp_path):
    base = {t["function"]["name"] for t in tools.tool_specs()}
    assert "view_images" not in base
    assert "view_images" in {t["function"]["name"]
                             for t in tools.tool_specs(has_vision=True)}
    # >8 paths: only first 8 kept, with a note
    for i in range(10):
        (tmp_path / f"q{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    out = tools.run_tool("view_images",
                         {"paths": [str(tmp_path / f"q{i}.png") for i in range(10)]},
                         {"has_vision": True})
    paths_part = out[len(tools.IMAGE_SENTINEL):].partition("\x00")[0]
    assert paths_part.count("\n") == 7           # 8 paths
    assert "first 8 of 10" in out


def test_encode_image_data_uri():
    from rigma import tools as t
    import tempfile
    import os
    fd, path = tempfile.mkstemp(suffix=".png")
    os.write(fd, b"\x89PNG\r\n\x1a\n" + b"\0" * 40)
    os.close(fd)
    uri = t.encode_image_data_uri(path)
    os.unlink(path)
    assert uri.startswith("data:image/")
    assert "base64," in uri


def test_cached_run_serves_repeat_from_cache(monkeypatch):
    tools._cache.clear()
    n = {"c": 0}
    monkeypatch.setattr(tools, "run_tool",
                        lambda *a, **k: (n.__setitem__("c", n["c"] + 1) or "R"))
    a = tools.cached_run("web_search", {"query": "x"})
    b = tools.cached_run("web_search", {"query": "x"})
    assert a == b == "R" and n["c"] == 1          # 2nd served from cache


def test_cached_run_skips_noncacheable(monkeypatch):
    tools._cache.clear()
    n = {"c": 0}
    monkeypatch.setattr(tools, "run_tool",
                        lambda *a, **k: (n.__setitem__("c", n["c"] + 1) or "R"))
    tools.cached_run("read_file", {"path": "x"})
    tools.cached_run("read_file", {"path": "x"})
    assert n["c"] == 2                            # file reads never cached


def test_cached_run_never_caches_errors(monkeypatch):
    tools._cache.clear()
    n = {"c": 0}
    monkeypatch.setattr(tools, "run_tool",
                        lambda *a, **k: (n.__setitem__("c", n["c"] + 1)
                                         or "error: boom"))
    tools.cached_run("fetch_url", {"url": "http://x"})
    tools.cached_run("fetch_url", {"url": "http://x"})
    assert n["c"] == 2                            # failures aren't cached


def test_http_request_get_cacheable_post_not():
    assert tools._is_cacheable("http_request", {"url": "u"}) is True     # GET
    assert tools._is_cacheable("http_request", {"url": "u", "method": "POST"}) \
        is False
    assert tools._is_cacheable("http_request",
                               {"url": "u", "headers": {"A": "b"}}) is False


def test_view_image_accepts_webp(tmp_path):
    # regression: .webp was rejected because mimetypes doesn't know it on Windows
    for ext in (".webp", ".avif", ".jpeg", ".gif"):
        f = tmp_path / ("pic" + ext)
        f.write_bytes(b"\x00" * 32)
        out = tools.run_tool("view_image", {"path": str(f)}, {"has_vision": True})
        assert out.startswith(tools.IMAGE_SENTINEL), ext


def test_view_image_still_rejects_nonimage(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("hi")
    assert "not an image" in tools.run_tool(
        "view_image", {"path": str(f)}, {"has_vision": True})


# ---- rescue parser for engine-missed tool calls (live 2026-07-20) ----

def test_rescue_parses_the_exact_leaked_call():
    # verbatim from the stalled live run: llama-server yielded NO tool_calls
    # and this arrived as content
    from rigma.tools import rescue_xml_tool_call
    text = ("<tool_call><function=list_directory>\n<parameter=path>\n.\n"
            "</parameter>\n</function>\n</tool_call>")
    name, args = rescue_xml_tool_call(text)
    assert name == "list_directory"
    assert args == {"path": "."}


def test_rescue_handles_multiline_code_params():
    from rigma.tools import rescue_xml_tool_call
    text = ("<tool_call><function=run_python>\n<parameter=code>\n"
            "import os\nprint(len(os.listdir('.')))\n</parameter>\n"
            "</function></tool_call>")
    name, args = rescue_xml_tool_call(text)
    assert name == "run_python"
    assert "os.listdir" in args["code"]


def test_rescue_ignores_prose_and_mentions():
    from rigma.tools import rescue_xml_tool_call
    for text in ("I will call list_directory now.",
                 "the function=thing syntax is documented",
                 "", None):
        assert rescue_xml_tool_call(text) == (None, None)


def test_rescue_takes_only_the_first_call():
    # one-action mode must hold even through the rescue path
    from rigma.tools import rescue_xml_tool_call
    text = ("<function=read_file><parameter=path>a.md</parameter></function>"
            "<function=write_file><parameter=path>b.md</parameter></function>")
    name, args = rescue_xml_tool_call(text)
    assert name == "read_file"


def test_rescue_structured_values_stay_structured():
    from rigma.tools import rescue_xml_tool_call
    text = ('<function=view_images><parameter=paths>["a.png", "b.png"]'
            '</parameter></function>')
    name, args = rescue_xml_tool_call(text)
    assert args["paths"] == ["a.png", "b.png"]


def test_tool_results_defuse_control_byte_runs(tmp_path, monkeypatch):
    """read_file on a crash-corrupted file (NUL flood mid-text) must hand the
    model a readable marker, not the raw bytes — a NUL run reads to a language
    model as end-of-document and produced instant-EOS turns (2026-07-21)."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path / "rigma-home"))
    from rigma import tools
    f = tmp_path / "bible.md"
    f.write_bytes(b"CAST: Corwin\n" + b"\x00" * 207 + b"STORY SO FAR: ch1-7\n")
    out = tools.run_tool("read_file", {"path": str(f)},
                         {"workspace": str(tmp_path), "allow_code": True})
    assert "\x00" not in out
    assert "207 unreadable control byte(s)" in out
    assert "CAST: Corwin" in out and "STORY SO FAR" in out


def test_write_file_never_authors_control_bytes(tmp_path, monkeypatch):
    """The model must not be able to write a poisoned file (echoed corruption,
    pasted binary) — control bytes are stripped on the way to disk."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    from rigma import tools
    tools.run_tool("write_file",
                   {"path": "out.txt", "content": "good\x00\x00\x00text"},
                   {"workspace": str(tmp_path), "allow_code": True})
    assert (tmp_path / "out.txt").read_text(encoding="utf-8") == "goodtext"


def test_file_content_cannot_switch_off_the_control_byte_guard(tmp_path):
    """AUDIT 04-7. `_defuse_control_bytes` used to return the text UNCHANGED
    when it contained a loop sentinel, assuming only the agent loop could make
    one. A corrupted file can contain that marker too, which switched the guard
    off and delivered its NUL run verbatim — the instant-EOS poison the guard
    exists to prevent. The defusing decision is now made by tool NAME."""
    poison = ("\x00__RIGMA_DELEGATE__\x00" + "\x00" * 30 + "tail")
    # the function itself no longer sniffs: it defuses whatever it is given
    assert "\x00" not in tools._defuse_control_bytes(poison)
    # ...and read_file cannot smuggle one in
    f = tmp_path / "poison.md"
    f.write_bytes(poison.encode("utf-8"))
    out = tools.run_tool("read_file", {"path": str(f)},
                         {"workspace": str(tmp_path), "allow_code": True})
    assert "\x00" not in out
    assert "unreadable control byte(s)" in out


def test_the_sentinel_tools_still_return_their_sentinel(tmp_path):
    """The guard is passed around by name for the handlers whose output IS a
    loop sentinel — defusing one would break the image/delegate/unlock paths."""
    ctx = {"workspace": str(tmp_path), "allow_code": True,
           "can_host_loop_tools": True}
    out = tools.run_tool("use_tools", {"names": ["view_image"]}, ctx)
    assert out.startswith(tools.USE_TOOLS_SENTINEL)
    assert "view_image" in out


# --- AUDIT 04-10: a reserved device name is not a file ------------------------
def test_reserved_device_names_are_refused_before_the_disk(tmp_path):
    """`write_file('nul')` used to report "wrote 7 chars" while writing nothing:
    NUL is a device, so Path.exists() is true, write_text discards into it, and
    read_file then answers "no such file" — a silent success the model cannot
    recover from."""
    for name in ("nul", "NUL.gguf", "con", "aux", "prn", "com1", "lpt9.txt"):
        out = tools.run_tool("write_file", {"path": name, "content": "PAYLOAD"},
                             {"workspace": str(tmp_path), "allow_code": True})
        assert out.startswith("error"), (name, out)
        assert "reserved device name" in out, (name, out)


def test_the_write_path_guard_still_rejects_globs_and_allows_real_names():
    assert tools._bad_write_char("*") == "*"
    assert tools._bad_write_char("a?b.txt") == "?"
    assert tools._bad_write_char("notes.txt") is None
    assert tools._bad_write_char("aux_data.txt") is None
    assert tools._bad_write_char("com10.gguf") is None
    assert tools._reserved_device_name("Q4_K_M/nul.gguf") == "nul.gguf"


# --- AUDIT 05-4: the background-job table is bounded --------------------------
# Fakes only: a test that spawns and kills a real child hangs in this sandbox
# (taskkill is denied), and the defect is pure bookkeeping anyway.
class _FakeProc:
    pid = 4242

    def __init__(self, rc=0):
        import io
        self.stdout = io.StringIO("")
        self.stderr = io.StringIO("")
        self.rc = rc

    def poll(self):
        return self.rc


def _fake_job(rc=0):
    import collections
    import threading
    return {"proc": _FakeProc(rc), "chunks": collections.deque(),
            "buflen": 0, "lock": threading.Lock(), "cmd": "echo hi",
            "started": 0.0}


def _with_saved_jobs(fn):
    saved = dict(tools._JOBS)
    try:
        tools._JOBS.clear()
        return fn()
    finally:
        tools._JOBS.clear()
        tools._JOBS.update(saved)


def test_prune_jobs_keeps_live_jobs_and_the_newest_finished():
    def run():
        for jid in range(1, 11):
            tools._JOBS[jid] = _fake_job(rc=0)      # finished
        tools._JOBS[11] = _fake_job(rc=None)        # live
        assert tools._prune_jobs(keep_finished=3) == 7
        assert sorted(tools._JOBS) == [8, 9, 10, 11]
    _with_saved_jobs(run)


def test_start_job_evicts_old_finished_records(monkeypatch, tmp_path):
    """`_JOBS[jid] = job` was the only writer and nothing ever popped, so every
    finished Popen and its output window lived for the server's uptime."""
    def run():
        monkeypatch.setattr(tools, "_launch_killable",
                            lambda *a, **k: _FakeProc(0))
        monkeypatch.setattr(tools, "_JOB_KEEP_FINISHED", 4)
        ctx = {"allow_code": True, "confirm_exec": True, "profile": "all",
               "workspace": str(tmp_path)}
        for _ in range(25):
            out = tools.run_tool("start_job", {"command": "echo hi"}, ctx)
            assert out.startswith("started job"), out
        assert len(tools._JOBS) <= 4
    _with_saved_jobs(run)


def _shell_output(argv, tmp_path):
    p = tools._launch_killable(argv, False, str(tmp_path))
    sink, overflow = [], []
    tools._read_capped(p.stdout, sink, 100000, overflow)
    p.wait()
    return "".join(sink)


def test_utf8_output_with_a_byte_cp1252_cannot_decode_is_not_lost(tmp_path):
    # The UTF-8 for "ŝ" contains 0x9D, which a strict cp1252 decode refuses; the
    # reader swallowed the error and the model got EMPTY output.
    code = ("import sys; sys.stdout.buffer.write("
            "'one caf\\u00e9\\ntwo \\u015d \\u4e2d\\nthree\\n'.encode('utf-8'))")
    out = _shell_output([sys.executable, "-c", code], tmp_path)
    assert out == "one caf\u00e9\ntwo \u015d \u4e2d\nthree\n"


def test_a_python_child_print_reaches_the_model_intact(tmp_path):
    out = _shell_output([sys.executable, "-c", "print('caf\u00e9 \u015d \u4e2d')"],
                        tmp_path)
    assert out.strip() == "caf\u00e9 \u015d \u4e2d"
