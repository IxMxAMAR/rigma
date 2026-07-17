"""Test server for browser smoke: fake OpenAI upstream + rigma UI app."""
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["RIGMA_HOME"] = tempfile.mkdtemp(prefix="rigma-smoke-")

import subprocess  # noqa: E402
import sys  # noqa: E402

from rigma import state  # noqa: E402

# a real (dummy) engine process so Unload has something to genuinely kill
_engine = subprocess.Popen([sys.executable, "-c",
                            "import time; time.sleep(600)"])
state.write_state("qwen3.6-35b-a3b", "Q0", 18500, engine_pid=_engine.pid,
                  ui_pid=os.getpid(), use_case="creative", ctx=4096)

# offline stand-ins for the Hugging Face browser
import rigma.hf_browse as hf_browse  # noqa: E402

hf_browse.search = lambda q, limit=12: [
    {"repo": "cool/WebTune-GGUF", "downloads": 153994, "likes": 56,
     "updated": "2026-07-01"}]
hf_browse.inspect_repo = lambda rid, registry=None: {
    "repo": rid, "name": "web-tune-7b", "family": "llama", "kind": "dense",
    "native_ctx": 131072, "capabilities": ["tools", "vision"],
    "already": False, "mmproj": {"file": "mmproj-F16.gguf",
                                 "bytes": 800 * 2**20},
    "split_skipped": 1, "recommended": "Q4_K_M",
    "ggufs": [
        {"file": "WebTune-Q8_0.gguf", "quant": "Q8_0", "bytes": 30 * 2**30,
         "fit": {"ok": True, "ctx": 8192, "n_cpu_moe": 0, "speed": "offload"}},
        {"file": "WebTune-Q4_K_M.gguf", "quant": "Q4_K_M",
         "bytes": 4 * 2**30, "fit": {"ok": True, "ctx": 131072,
                                     "n_cpu_moe": 0, "speed": "gpu"}},
    ]}


def _fake_add(rid, registry=None):
    from rigma.hangar import _write_spec
    from rigma.models import GgufFile, ModelSpec
    spec = ModelSpec(slug="web-tune-7b", family="llama", kind="dense",
                     n_layers=32, full_attn_layers=32, kv_heads=8,
                     head_dim=128, native_ctx=131072,
                     ggufs=[GgufFile(repo=rid, file="WebTune-Q4_K_M.gguf",
                                     bytes=4 * 2**30, quant="Q4_K_M")],
                     use_cases=["general"], capabilities=["tools", "vision"],
                     custom=True)
    _write_spec(spec)
    return spec


hf_browse.add_model = _fake_add


class FakeUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        req = self.rfile.read(n)
        if b"OVERFLOW" in req:
            body = json.dumps({"error": {
                "code": 400, "type": "exceed_context_size_error",
                "message": "request (9999 tokens) exceeds the available "
                           "context size (4096 tokens), try increasing it"}}
                ).encode()
            self.send_response(400)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if b'"stream": false' in req or b'"stream":false' in req:
            body = json.dumps({"choices": [{"message": {
                "role": "assistant",
                "content": "Digest: twelve turns of testing."}}]}).encode()
            self.send_response(200)
            self.send_header("content-type", "application/json")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
        if b"THINK" in req:
            for tok in ("pondering ", "deeply"):
                chunk = {"choices": [{"delta": {"reasoning_content": tok}}]}
                self.wfile.write(b"data: " + json.dumps(chunk).encode()
                                 + b"\n\n")
                self.wfile.flush()
                time.sleep(0.1)
        for tok in ("Hello", " from", " the", " **smoke**", " test."):
            chunk = {"choices": [{"delta": {"content": tok}}]}
            self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            self.wfile.flush()
            time.sleep(0.25)   # slow enough for the Stop-morph check
        tail = {"choices": [], "usage": {"prompt_tokens": 1234},
                "timings": {"predicted_per_second": 42.5}}
        self.wfile.write(b"data: " + json.dumps(tail).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")

    def log_message(self, *a):
        pass


srv = HTTPServer(("127.0.0.1", 0), FakeUpstream)
threading.Thread(target=srv.serve_forever, daemon=True).start()

import uvicorn  # noqa: E402

from rigma.serve import build_app  # noqa: E402

app = build_app(upstream_port=srv.server_address[1],
                default_prompt="You are a smoke-test narrator.")
print(f"SMOKE_READY port=18500 upstream={srv.server_address[1]}", flush=True)
uvicorn.run(app, host="127.0.0.1", port=18500, log_level="warning")
