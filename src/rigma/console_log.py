"""Live activity on the terminal (owner request 2026-10-23).

`rigma up` installed NO log handler. Rigma's own loggers therefore fell through
to `logging.lastResort`, which prints WARNING and above as a bare message with
no time, level or source — and dropped every `_log.info` outright. So the only
lines that ever reached the console came from third parties that bring their own
logging: fastembed's loguru (`ERROR ... Could not find the model tar.gz file`)
and, at `log_level="warning"`, almost nothing from uvicorn.

The result was a terminal that looked broken and told you nothing. On a cold
start the owner saw four fastembed ERRORs and concluded memory was failing,
while the turn timings, compaction runs, engine retries and every tool call the
server actually recorded were invisible. That is also why "is it still working?"
had to be answered from the UI — where the thinking block is a collapsible
element that disappears when a turn completes, so the one place with the truth
was the one place that printed nothing.

This module installs exactly one handler, from the server entry point only.
Importing it must never reconfigure a caller's logging: a test run, a library
user and the CLI's own command output all share this interpreter.

Controls:
  RIGMA_LOG=0            install nothing (a quiet server, for a supervisor that
                         captures its own output)
  RIGMA_LOG_LEVEL=DEBUG  raise/lower the root level (default INFO)
"""

from __future__ import annotations

import logging
import os
import sys

#: Marks the handler this module owns, so `install()` is idempotent even when
#: called twice (the app is built once, but a reload or a test can call it again)
_HANDLER_ATTR = "_rigma_console"

#: Third-party loggers that are chatty at INFO and say nothing a person running
#: a chat server wants. Pinned to WARNING rather than left to the root level:
#: `httpx` logs every request it makes, and a turn makes several.
_NOISY = ("httpx", "httpcore", "urllib3", "huggingface_hub", "filelock",
          "uvicorn.access", "python_multipart")

_FORMAT = "%(asctime)s %(levelname)-5s %(shortname)-12s %(message)s"
_DATEFMT = "%H:%M:%S"


class _Formatter(logging.Formatter):
    """`rigma.serve` prints as `serve`; everything else keeps its own name.

    The prefix is the same on every line the server writes, so it costs column
    width on all of them and identifies nothing. Third-party names are left
    alone — `fastembed` in the margin is the one thing that tells a reader the
    line is not ours.
    """

    def format(self, record: logging.LogRecord) -> str:
        name = record.name
        record.shortname = name[len("rigma."):] if name.startswith("rigma.") \
            else name
        return super().format(record)


def install(level: str | int | None = None) -> logging.Logger:
    """Install the console handler on the root logger. Idempotent.

    Returns the root logger so a caller can log immediately without importing
    `logging` itself.
    """
    root = logging.getLogger()
    if os.environ.get("RIGMA_LOG", "1").strip() in ("0", "off", "false", "no"):
        return root
    want = level if level is not None else os.environ.get("RIGMA_LOG_LEVEL", "INFO")
    try:
        root.setLevel(want)
    except (TypeError, ValueError):
        root.setLevel(logging.INFO)

    handler = None
    for h in root.handlers:
        if getattr(h, _HANDLER_ATTR, False):
            handler = h
            break
    if handler is None:
        handler = logging.StreamHandler(sys.stderr)
        setattr(handler, _HANDLER_ATTR, True)
        handler.setFormatter(_Formatter(_FORMAT, _DATEFMT))
        root.addHandler(handler)
    # A handler installed by a previous `install()` keeps its level; only the
    # root's changes. Keeping one handler is what stops a second call from
    # doubling every line.

    for name in _NOISY:
        logging.getLogger(name).setLevel(logging.WARNING)
    return root


def uninstall() -> None:
    """Remove the handler. For tests, and for a caller that wants its own."""
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, _HANDLER_ATTR, False):
            root.removeHandler(h)
