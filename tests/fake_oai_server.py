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
# `--tool NAME` makes the fake model ask for a tool on its FIRST request and
# answer in words on the next one, so a test can see how an agent backend
# projects a tool call AND its result. The name is deliberately one no backend
# has: the call is reported either way, and an unknown tool cannot touch this
# machine. One shot, not every request — otherwise the backend calls the tool,
# is told it failed, and calls it again until it hits its step limit.
toolname = sys.argv[sys.argv.index("--tool") + 1] if "--tool" in sys.argv else ""
# `--no-count` answers 404 to the token-counting route. MiniMax Code asks
# `/v1/responses/input_tokens` before every turn, which llama-server does not
# serve — so whether an agent backend survives that 404 decides whether Rigma's
# proxy has to learn the route or not.
count404 = "--no-count" in sys.argv
# `--dump PATH` writes each request body whole. The `--log` line is a summary
# for asserting on; this is for finding out what a client actually SENDS —
# which tool schemas, which system prompt, how much history it replays.
dumppath = sys.argv[sys.argv.index("--dump") + 1] if "--dump" in sys.argv else ""
_requests = {"n": 0}
CHUNKS = ("hello ", "from ", "dsh")


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        if count404 and "input_tokens" in self.path:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
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
        if dumppath:
            with open(dumppath, "a", encoding="utf-8") as f:
                f.write(json.dumps({"path": self.path, "body": body}) + "\n")
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            first = _requests["n"] == 0
            _requests["n"] += 1
            if toolname and first:
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [{"delta": {"tool_calls": [
                        {"index": 0, "id": "call_probe_1", "type": "function",
                         "function": {"name": toolname,
                                      "arguments": "{\"probe\": 1}"}}]}}]}
                ).encode() + b"\n\n")
                self.wfile.write(b"data: " + json.dumps(
                    {"choices": [{"delta": {}, "finish_reason": "tool_calls"}]}
                ).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
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
        # A COMPLETE OpenAI chat completion, not the two fields a reader needs.
        # An agent backend's `provider add --use` runs this through a strict
        # schema before it will save and select the provider, and a thin body is
        # rejected as "Invalid response from model provider" — which then looks
        # like the provider being unusable rather than the fixture being short.
        data = json.dumps({
            "id": "chatcmpl-probe",
            "object": "chat.completion",
            "created": 1789990501,
            "model": body.get("model") or "local-test",
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": "ok"}}],
            "usage": {"prompt_tokens": 2048, "completion_tokens": 128,
                      "total_tokens": 2176},
            "timings": {"prompt_per_second": 650.0,
                        "predicted_per_second": 55.5},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
