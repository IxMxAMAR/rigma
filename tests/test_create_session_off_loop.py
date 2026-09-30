"""ODR-9: `create_session` ran the allowlist seed on the event loop.

`create_session` is async but called `sessions.create()` synchronously, and the
seed runs `json_extract` over every full session body. On a big store, "New
chat" stalled every live stream. It must go through `asyncio.to_thread`.
"""
import asyncio
import time

from fastapi.testclient import TestClient

from rigma import serve, sessions


def test_create_session_offloads_the_seed(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    monkeypatch.delenv("RIGMA_KEEP_ALIVE_MIN", raising=False)

    calls = []
    real = sessions.create

    def slow_create(*a, **k):
        time.sleep(0.05)
        return real(*a, **k)

    monkeypatch.setattr(sessions, "create", slow_create)

    real_to_thread = asyncio.to_thread

    async def spy(func, *a, **k):
        calls.append(func)
        return await real_to_thread(func, *a, **k)

    monkeypatch.setattr(asyncio, "to_thread", spy)

    c = TestClient(serve.build_app(upstream_port=1))
    sid = c.post("/api/sessions", json={"title": "t"}).json()["id"]
    assert sid
    assert slow_create in calls, (
        "create_session did not offload sessions.create via asyncio.to_thread")
