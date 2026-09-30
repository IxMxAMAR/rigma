# raggity-serve stand-in for tests. Honors:
#   fake_raggity_server.py serve --config X --port N
import http.server
import json
import os
import sys

port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000
# REC-1: echo a per-run token (RIGMA_FAKE_TOKEN) in /healthz so a test can prove
# the answer came from the sidecar IT launched, not a foreign raggity on the
# same port. Absent, no token field is sent.
token = os.environ.get("RIGMA_FAKE_TOKEN", "")


class H(http.server.BaseHTTPRequestHandler):
    def _send(self, obj):
        data = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/healthz":
            body = {"status": "ok", "version": "0.12.0",
                    "index_backend": "lancedb", "documents": 42}
            if token:
                body["token"] = token
            self._send(body)
        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/retrieve":
            self._send({"chunks": [{"text": "alpha", "score": 0.9,
                                    "source": "a.md", "metadata": {}}],
                        "packed_context": "alpha", "token_count": 1,
                        "tokenizer": "chars/4-approx"})
        elif self.path == "/ask":
            self._send({"answer": f"grounded: {body.get('question', '')}",
                        "abstained": False, "citations": []})
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
