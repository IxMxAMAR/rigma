"""Auto-compact: when a turn leaves the window ~full, older messages are
summarized before the next turn — reusing the manual /compact machinery."""
import json
import os
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from fastapi.testclient import TestClient

from rigma import sessions
from rigma import state as st
from rigma.serve import build_app


class _Upstream(BaseHTTPRequestHandler):
    prompt_tokens = 950          # reported by the streamed chat turn
    compact_status = 200         # summarizer (non-stream) response status

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        body = json.loads(self.rfile.read(n))
        if not body.get("stream"):                      # the summarizer call
            self.send_response(_Upstream.compact_status)
            self.send_header("content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(
                {"choices": [{"message": {"content": "COMPACT DIGEST"}}]}).encode())
            return
        self.send_response(200)                          # the streamed chat turn
        self.send_header("content-type", "text/event-stream")
        self.end_headers()

        def sse(o):
            self.wfile.write(b"data: " + json.dumps(o).encode() + b"\n\n")

        sse({"choices": [{"delta": {"content": "ok"}}]})
        sse({"choices": [{"delta": {}}],
             "usage": {"prompt_tokens": _Upstream.prompt_tokens},
             "timings": {"predicted_per_second": 40}})
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


@pytest.fixture
def upstream():
    srv = HTTPServer(("127.0.0.1", 0), _Upstream)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv.server_address[1]
    srv.shutdown()


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("RIGMA_HOME", str(tmp_path))
    return tmp_path


def _seed(client, n=24):
    sid = client.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["messages"] = [{"role": "user", "content": f"m{i}"} for i in range(n)]
    sessions.save(s)
    return sid


def test_auto_compact_fires_when_nearly_full(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 200   # 950/1000 > .92
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert "event: compacted" in r.text
    s = sessions.load(sid)
    assert s["digest"] == "COMPACT DIGEST"
    assert len(s["messages"]) <= 17           # trimmed to the recent tail
    assert s["archive"]                        # older messages preserved


def test_no_compact_below_threshold(home, upstream):
    _Upstream.prompt_tokens = 100              # 100/1000 well under .92
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: compacted" not in r.text
    assert sessions.load(sid)["digest"] == ""


def test_auto_compact_respects_toggle(home, upstream):
    _Upstream.prompt_tokens = 950
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    client.post(f"/api/sessions/{sid}", json={"auto_compact": False})
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: compacted" not in r.text
    assert sessions.load(sid)["digest"] == ""


def test_summarizer_failure_does_not_break_turn(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 500
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200 and "[DONE]" in r.text     # turn still completes
    assert "event: compacted" not in r.text
    s = sessions.load(sid)
    assert s["digest"] == "" and s["messages"][-1]["content"] == "ok"  # answer saved


def test_runs_compact_against_a_small_budget():
    from rigma import serve
    assert serve.RUN_CTX_BUDGET == 32768
    # a run session compacts against RUN_CTX_BUDGET, not the engine context
    assert serve.compact_budget({"run_id": "r1"}, 131072) == serve.RUN_CTX_BUDGET
    # ...but never above what the engine actually has
    assert serve.compact_budget({"run_id": "r1"}, 16384) == 16384
    # a normal chat still uses the engine's context
    assert serve.compact_budget({}, 131072) == 131072


def test_compaction_keeps_enough_actions():
    # one action now costs TWO messages (assistant + TOOL RESULT), so the keep
    # window must retain a useful number of ACTIONS, not just messages
    from rigma import serve
    assert serve.AUTO_COMPACT_KEEP >= 16


def test_archive_is_bounded(home, upstream):
    # a run compacts often and re-serialises the whole session each save, so an
    # unbounded archive is real write amplification over a long run
    from rigma import serve as _s
    assert _s.ARCHIVE_MAX <= 1000
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client, n=24)
    s = sessions.load(sid)
    s["archive"] = [{"role": "user", "content": f"old{i}"}
                    for i in range(_s.ARCHIVE_MAX + 50)]
    sessions.save(s)
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 200
    client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert len(sessions.load(sid)["archive"]) <= _s.ARCHIVE_MAX


def _seed_run(client, n_obs=20, body=800):
    """A RUN session whose bulk is tool-result observations."""
    sid = client.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["run_id"] = "run-1"
    msgs = []
    for i in range(n_obs):
        msgs.append({"role": "assistant", "content": f"step {i} reasoning"})
        msgs.append({"role": "user", "kind": "tool_result",
                     "tools": [{"name": "sample_files", "ok": True}],
                     "content": "TOOL RESULT sample_files: " + "x" * body})
    s["messages"] = msgs
    sessions.save(s)
    return sid


def test_runs_mask_observations_instead_of_summarizing(home, upstream):
    # masking is deterministic and lossless where it matters; the digest keeps
    # the gist and destroys the exact strings an agent navigates by. Masking
    # must be TRIED FIRST, and when it succeeds the summarizer must not run.
    # ctx must exceed what keep_recent preserves, or masking cannot possibly
    # reach budget and falling back to the digest is correct.
    _Upstream.prompt_tokens, _Upstream.compact_status = 3900, 200
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=4000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed_run(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert r.status_code == 200
    assert "event: masked" in r.text, r.text[-400:]
    s = sessions.load(sid)
    assert s["digest"] == "", "masking was enough — the summarizer should not run"
    assert any("masked" in m.get("content", "") for m in s["messages"])


def test_masking_preserves_the_models_own_turns(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 3900, 200
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=4000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed_run(client)
    before = [m["content"] for m in sessions.load(sid)["messages"]
              if m["role"] == "assistant"]
    client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    after = [m["content"] for m in sessions.load(sid)["messages"]
             if m["role"] == "assistant"]
    assert before == after[:len(before)], "the model's reasoning must be intact"


def test_chats_still_summarize_and_never_mask(home, upstream):
    # no run_id: the prose IS the content, and a digest is the right abstraction
    _Upstream.prompt_tokens, _Upstream.compact_status = 950, 200
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=1000)
    client = TestClient(build_app(upstream_port=upstream))
    sid = _seed(client)
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "hi"})
    assert "event: masked" not in r.text
    assert sessions.load(sid)["digest"] == "COMPACT DIGEST"


def test_chat_auto_titles_after_a_few_turns(home, upstream):
    # the rail was "Sup bro", "Hello", and three identical truncations —
    # after the 4th message one tiny non-streaming call names the chat.
    # _Upstream answers non-streaming calls with COMPACT DIGEST, which here
    # plays the generated title.
    _Upstream.prompt_tokens, _Upstream.compact_status = 100, 200
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=131072)
    client = TestClient(build_app(upstream_port=upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}/chat", json={"message": "Sup bro"})
    assert sessions.load(sid)["title"] == "Sup bro"      # too early
    r = client.post(f"/api/sessions/{sid}/chat", json={"message": "more"})
    s = sessions.load(sid)
    assert s["title"] == "COMPACT DIGEST"
    assert s["title_source"] == "auto"
    assert "event: meta" in r.text


def test_user_rename_is_never_overwritten(home, upstream):
    _Upstream.prompt_tokens, _Upstream.compact_status = 100, 200
    st.write_state("m", "Q4", 11500, engine_pid=os.getpid(),
                   ui_pid=os.getpid(), ctx=131072)
    client = TestClient(build_app(upstream_port=upstream))
    sid = client.post("/api/sessions", json={}).json()["id"]
    client.post(f"/api/sessions/{sid}", json={"title": "My Novel"})
    for m in ("a", "b", "c"):
        client.post(f"/api/sessions/{sid}/chat", json={"message": m})
    assert sessions.load(sid)["title"] == "My Novel"


# --- the timeout that made compaction impossible ------------------------------
# Live 2026-08-18. Compaction posts the WHOLE session as one non-streaming
# request with a hardcoded timeout=120.0 (serve.py). On the owner's machine the
# measured prefill for the running model is 104 tok/s, so 120 seconds buys about
# 12,500 tokens of prefill — while auto-compaction only fires at 0.92 * 65,536 =
# 60,293 tokens. The trigger threshold was FIVE TIMES larger than the timeout
# could process, so compaction could only ever be attempted on sessions
# guaranteed to time out. 33 sessions, 0 digests, and a manual attempt that sat
# at "Compacting..." for exactly two minutes and died.
#
# The existing tests never caught it because the fake upstream answers instantly.

def test_compact_timeout_scales_with_the_payload():
    from rigma.serve import compact_timeout
    small = compact_timeout(4_000, pp_tps=104.0)
    big = compact_timeout(230_000, pp_tps=104.0)      # the owner's session
    assert big > small
    # 57.5K tokens at 104 tok/s is ~550s of prefill alone; 120 cannot work
    assert big > 550, f"{big}s would still time out before prefill finishes"


def test_compact_timeout_uses_the_machines_measured_prefill():
    """A slower machine needs longer, and we HAVE the measurement now."""
    from rigma.serve import compact_timeout
    assert compact_timeout(230_000, pp_tps=20.0) > \
        compact_timeout(230_000, pp_tps=400.0)


def test_compact_timeout_is_conservative_when_nothing_was_measured():
    """No calibration yet must not mean a 120s timeout again."""
    from rigma.serve import compact_timeout
    assert compact_timeout(230_000, pp_tps=0.0) > 550


def test_compact_timeout_has_a_floor_and_a_ceiling():
    """A tiny session still gets a sane minimum; a wedged engine must not hang
    the request forever."""
    from rigma.serve import compact_timeout
    assert compact_timeout(10, pp_tps=104.0) >= 120
    assert compact_timeout(50_000_000, pp_tps=1.0) <= 3600


# --- compaction folds in bounded chunks ---------------------------------------
# One call over the whole session is self-defeating: the thing that reduces
# context had to read all of it first, in a single request that could not
# finish. Fold instead — summarise a bounded slice, carry the result into the
# next slice. Each call is small enough to complete, progress is observable, and
# a session that is too big for one request is no longer too big to compact.

def test_chunks_never_split_a_message():
    from rigma.serve import compact_chunks
    msgs = [{"role": "user", "content": "a" * 5000},
            {"role": "assistant", "content": "b" * 5000},
            {"role": "user", "content": "c" * 5000}]
    chunks = compact_chunks(msgs, budget=6000)
    assert sum(len(c) for c in chunks) == 3          # every message appears once
    for c in chunks:
        assert c, "no empty chunk"


def test_a_small_session_is_still_one_call():
    from rigma.serve import compact_chunks
    msgs = [{"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi"}]
    assert len(compact_chunks(msgs, budget=24000)) == 1


def test_a_big_session_is_split_into_several_bounded_chunks():
    from rigma.serve import compact_chunks
    msgs = [{"role": "user", "content": "x" * 4000} for _ in range(30)]
    chunks = compact_chunks(msgs, budget=24000)
    assert len(chunks) >= 4, "120,000 chars must not go in one request"
    for c in chunks:
        size = sum(len(str(m.get("content") or "")) for m in c)
        # one oversized message is allowed through alone; otherwise stay bounded
        assert size <= 24000 or len(c) == 1


def test_one_enormous_message_still_gets_its_own_chunk():
    """The owner's 20,000-token chapter arrives as ONE message. It cannot be
    split, but it must not drag other messages into an oversized request."""
    from rigma.serve import compact_chunks
    msgs = [{"role": "user", "content": "s" * 100},
            {"role": "assistant", "content": "L" * 200_000},
            {"role": "user", "content": "t" * 100}]
    chunks = compact_chunks(msgs, budget=24000)
    big = [c for c in chunks if any(len(str(m.get("content"))) > 100_000 for m in c)]
    assert len(big) == 1 and len(big[0]) == 1, "the giant message travels alone"


def test_chunking_handles_vision_parts_without_crashing():
    from rigma.serve import compact_chunks
    msgs = [{"role": "user", "content": [{"type": "text", "text": "look"},
                                         {"type": "image_url"}]}]
    assert len(compact_chunks(msgs, budget=24000)) == 1


def test_a_failed_fold_leaves_the_session_untouched(home, monkeypatch):
    """The archive is the user's manuscript. A compaction that dies half way
    must not leave messages moved and no digest to replace them."""
    import httpx
    from fastapi.testclient import TestClient
    from rigma.serve import build_app
    from rigma import sessions

    calls = {"n": 0}

    class _Boom:
        async def post(self, *a, **k):
            calls["n"] += 1
            if calls["n"] >= 2:            # succeed once, then fail
                raise httpx.ReadTimeout("engine gave up")
            req = httpx.Request("POST", "http://x")
            return httpx.Response(200, request=req, json={
                "choices": [{"message": {"content": "partial digest"}}]})

    app = build_app(upstream_port=1)
    client = TestClient(app)
    sid = client.post("/api/sessions", json={}).json()["id"]
    s = sessions.load(sid)
    s["messages"] = [{"role": "user", "content": "y" * 30_000}
                     for _ in range(4)]
    sessions.save(s)
    before = len(sessions.load(sid)["messages"])

    import rigma.serve as srv
    monkeypatch.setattr(srv.httpx, "AsyncClient", lambda *a, **k: _Boom())
    r = client.post(f"/api/sessions/{sid}/compact", json={"keep": 1})
    assert r.status_code >= 400                      # reported, not silent
    after = sessions.load(sid)
    assert len(after["messages"]) == before          # nothing moved
    assert not (after.get("digest") or "")           # no half-written digest
    assert not after.get("archive")
