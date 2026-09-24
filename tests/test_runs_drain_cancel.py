"""AUDIT 03-4: /stop must cancel the in-flight engine request, not orphan it.

`_drain_turn` awaited `asyncio.wait({task}, ...)` around a long-lived
`agen.__anext__()` task. Cancelling the DRIVER task (which is what stop_run
does) only cancelled the outer coroutine: the inner task was orphaned, so the
generator stayed alive, kept reading the engine's SSE response and persisted a
partial message after the run had already been reported stopped. `aclose()` on
the still-running generator raised RuntimeError, which the bare except
swallowed — so the httpx response was never closed either.

The scripted generator below stands in for `_llm_turn`: no network, no engine.
"""
import asyncio

import pytest

from rigma import serve

_CHUNK = b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'


def _drain(app):
    return app.state.drain_turn


def test_drain_turn_cancels_the_in_flight_engine_request(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    drain = _drain(serve.build_app(upstream_port=1))

    state = {"reading": False, "closed": False, "resumed": False}
    hold = asyncio.Event()

    async def gen():
        try:
            yield _CHUNK
            state["reading"] = True
            await hold.wait()             # the in-flight engine request
            state["resumed"] = True
        finally:
            state["closed"] = True

    async def main():
        task = asyncio.ensure_future(
            drain({"id": "s", "messages": []}, agen=gen()))
        # wait until the generator is suspended inside the request
        for _ in range(200):
            if state["reading"]:
                break
            await asyncio.sleep(0.01)
        assert state["reading"], "the scripted turn never reached the engine wait"

        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        # give any orphaned inner task a chance to run. The assertion must
        # happen HERE, while the loop is still alive: asyncio.run() cancels
        # every remaining task on exit, which would close the orphan anyway and
        # hide the leak.
        for _ in range(5):
            await asyncio.sleep(0.01)
        assert state["closed"] is True, (
            "the in-flight generator was never cancelled — it is still reading "
            "the engine after the stop")
        assert state["resumed"] is False, "the abandoned turn ran on after the stop"

    asyncio.run(main())


def test_drain_turn_returns_the_engine_error(tmp_path, monkeypatch):
    """The override must not break the ordinary path: an `event: error` chunk
    is still surfaced as the turn's error."""
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    drain = _drain(serve.build_app(upstream_port=1))

    async def gen():
        yield b'event: error\ndata: {"message": "engine exploded"}\n\n'

    async def main():
        return await drain({"id": "s", "messages": []}, agen=gen())

    assert asyncio.run(main()) == "engine exploded"
