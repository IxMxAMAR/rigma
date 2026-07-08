# raggity-serve stand-in for tests. Honors:
#   fake_raggity_server.py serve --config X --port N
import http.server
import json
import sys

port = int(sys.argv[sys.argv.index("--port") + 1]) if "--port" in sys.argv else 8000


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
            self._send({"status": "ok", "version": "0.12.0",
                        "index_backend": "lancedb", "documents": 42})
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
