# llama-server stand-in for tests: /health + /v1/chat/completions with timings.
import http.server
import json
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def do_POST(self):
        body = {"choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2048, "completion_tokens": 128},
                "timings": {"prompt_per_second": 650.0,
                            "predicted_per_second": 55.5}}
        data = json.dumps(body).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
