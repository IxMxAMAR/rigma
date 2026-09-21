# llama-server stand-in for tests: /health + /v1/chat/completions with timings.
#
# STREAMS when the caller asks for it, and answers in one JSON body when it does
# not. Rigma's own proxy reads a body; an agent backend driving the server (DSH)
# asks for `stream: true` and reads SSE, and a fake that ignored that produced an
# empty reply that looked like an adapter bug (measured 2026-09-21).
#
# `--log PATH` appends one JSON line per POST describing what was asked for, so a
# test can assert the SHAPE of the request — the path, the auth header, the
# token ceiling — and not merely that something answered.
import http.server
import json
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])
logpath = sys.argv[sys.argv.index("--log") + 1] if "--log" in sys.argv else ""
CHUNKS = ("hello ", "from ", "dsh")


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("content-length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            body = {}
        if logpath:
            with open(logpath, "a", encoding="utf-8") as f:
                f.write(json.dumps({
                    "path": self.path,
                    "auth": self.headers.get("authorization", ""),
                    "stream": body.get("stream"),
                    "model": body.get("model"),
                    "max_tokens": body.get("max_tokens"),
                    "tools": len(body.get("tools") or []),
                    "messages": len(body.get("messages") or []),
                }) + "\n")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for tok in CHUNKS:
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [{"delta": {"content": tok}}]}).encode() + b"\n\n")
                self.wfile.flush()
            self.wfile.write(b"data: " + json.dumps(
                {"choices": [{"delta": {}, "finish_reason": "stop"}],
                 "usage": {"prompt_tokens": 8,
                           "completion_tokens": len(CHUNKS)}}).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return
        data = json.dumps({"choices": [{"message": {"content": "ok"}}],
                           "usage": {"prompt_tokens": 2048,
                                     "completion_tokens": 128},
                           "timings": {"prompt_per_second": 650.0,
                                       "predicted_per_second": 55.5}}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
