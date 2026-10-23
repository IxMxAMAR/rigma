"""The terminal view (owner request 2026-10-23).

`rigma up` installed no log handler, so every `_log.info` was dropped and the
only lines that reached the console came from third parties that bring their own
logging — fastembed's missing-model ERRORs. A healthy server therefore looked
broken in the one place the owner was watching, while the turn timings, tool
calls and job lifecycle it actually recorded were invisible.

These cover the two halves of that: the handler is installed exactly once, and a
tool call's arguments survive being put on one line.
"""
import io
import logging

from rigma import console_log, serve

# `tests/conftest.py` patches `serve.run_ui` to raise for every test (an unmocked
# one would serve forever and squat on port 11500). This module is the one place
# that has to reach the REAL function, so it keeps its own reference — captured
# at import, before any fixture has replaced the attribute.
_REAL_RUN_UI = serve.run_ui


def _ours(root: logging.Logger) -> list:
    return [h for h in root.handlers
            if getattr(h, console_log._HANDLER_ATTR, False)]


def _fresh() -> logging.Logger:
    """Start from a root logger with no handler of ours on it.

    The root logger is process-wide state and pytest replaces `sys.stderr`
    between tests, so a handler left behind by anything else binds a stream that
    no longer exists — and a test asserting "install added nothing" would be
    measuring that, not its own call.
    """
    console_log.uninstall()
    return logging.getLogger()


def _restore(root: logging.Logger, before: list) -> None:
    console_log.uninstall()
    root.handlers[:] = before


def test_install_adds_exactly_one_handler(monkeypatch):
    monkeypatch.delenv("RIGMA_LOG", raising=False)
    monkeypatch.delenv("RIGMA_LOG_LEVEL", raising=False)
    root = _fresh()
    before = list(root.handlers)
    try:
        console_log.install()
        console_log.install()          # a second call must not double every line
        assert len(_ours(root)) == 1
        assert root.level == logging.INFO
    finally:
        _restore(root, before)


def test_rigma_log_zero_installs_nothing(monkeypatch):
    monkeypatch.setenv("RIGMA_LOG", "0")
    root = _fresh()
    before = list(root.handlers)
    try:
        console_log.install()
        assert _ours(root) == []
    finally:
        _restore(root, before)


def test_info_from_a_rigma_logger_reaches_the_stream(monkeypatch):
    monkeypatch.delenv("RIGMA_LOG", raising=False)
    root = _fresh()
    before = list(root.handlers)
    # This test's own sink: `capsys` cannot be trusted to hold the stream a
    # handler was created against, and the assertion is about the LINE, not
    # about which stream pytest happened to hand out.
    buf = io.StringIO()
    try:
        console_log.install()
        _ours(root)[0].setStream(buf)
        logging.getLogger("rigma.serve").info("round 2/5 - 3 tool call(s)")
    finally:
        _restore(root, before)
    out = buf.getvalue()
    assert "round 2/5 - 3 tool call(s)" in out
    assert "serve" in out, "the line should name the module it came from"


def test_shortname_strips_only_rigma_prefix():
    fmt = console_log._Formatter("%(shortname)s", None)
    ours = logging.LogRecord("rigma.serve", logging.INFO, "f", 1, "m", (), None)
    theirs = logging.LogRecord("fastembed.common", logging.ERROR, "f", 1, "m",
                               (), None)
    assert fmt.format(ours) == "serve"
    # A third party's name is the one thing that says the line is NOT ours.
    assert fmt.format(theirs) == "fastembed.common"


def test_brief_args_is_one_ascii_line_and_counts_what_it_dropped():
    line = serve._brief_args({"path": "C:/a/b/c.txt", "q": "x" * 200,
                              "n": 3, "extra": True, "more": [1, 2]})
    assert "\n" not in line
    assert line.startswith("(path=C:/a/b/c.txt, q=")
    assert "+2 more" in line, "dropped arguments must be counted, not implied"
    assert line.isascii(), "a Windows console is not UTF-8"


def test_brief_args_empty_for_no_arguments():
    assert serve._brief_args(None) == ""
    assert serve._brief_args({}) == ""


def test_run_ui_installs_the_handler_before_serving(monkeypatch):
    """The wiring, not just the module: `run_ui` is the only server entry point.

    Installing the handler anywhere later would leave the app's own start-up —
    the embedder probe, the plan, the engine — invisible, which is precisely the
    state this whole change exists to end.
    """
    import sys
    import types

    seen = {}

    def fake_run(app, **kwargs):
        seen["installed"] = bool(_ours(logging.getLogger()))
        seen["log_level"] = kwargs.get("log_level")

    fake_uvicorn = types.ModuleType("uvicorn")
    fake_uvicorn.run = fake_run          # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "uvicorn", fake_uvicorn)
    monkeypatch.setattr(serve, "build_app", lambda port: object())
    monkeypatch.setattr(serve, "run_ui", _REAL_RUN_UI)

    root = logging.getLogger()
    before = list(root.handlers)
    try:
        _REAL_RUN_UI(11500, 11499)
    finally:
        _restore(root, before)

    assert seen["installed"] is True
    # uvicorn's access log would bury the activity lines under one line per
    # asset request, so it stays at warning.
    assert seen["log_level"] == "warning"
