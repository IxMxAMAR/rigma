# Stand-in for llama-server in tests: honors --port, serves /health, ignores the rest.
import http.server
import sys

port = int(sys.argv[sys.argv.index("--port") + 1])
# `--token VALUE` is echoed as the X-Rigma-Fake-Token header on /health so a test
# can prove the health 200 came from the process IT launched and not from a
# foreign listener that happens to own the port (REC-1). Absent, no header.
token = sys.argv[sys.argv.index("--token") + 1] if "--token" in sys.argv else ""


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        health = self.path == "/health"
        self.send_response(200 if health else 404)
        if health and token:
            self.send_header("X-Rigma-Fake-Token", token)
        self.end_headers()

    def log_message(self, *a):
        pass


http.server.HTTPServer(("127.0.0.1", port), H).serve_forever()
