"""Test server for browser smoke: fake OpenAI upstream + rigma UI app."""
import json
import os
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

os.environ["RIGMA_HOME"] = tempfile.mkdtemp(prefix="rigma-smoke-")

from rigma import state  # noqa: E402

state.write_state("smoke-model", "Q0", 18500, engine_pid=os.getpid(),
                  ui_pid=os.getpid(), use_case="creative", ctx=4096)


class FakeUpstream(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        self.rfile.read(n)
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.end_headers()
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
