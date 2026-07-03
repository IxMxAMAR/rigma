# Stand-in for llama-server in tests: honors --port, serves /health, ignores the rest.
import http.server
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
