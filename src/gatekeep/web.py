"""Local web app: serves the page and streams one question's events as JSON lines (standard library only)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

STATIC = Path(__file__).parent / "static"
TYPES = {".html": "text/html; charset=utf-8", ".css": "text/css; charset=utf-8", ".js": "text/javascript; charset=utf-8"}
MODES = ("fast", "careful")


class _Server(ThreadingHTTPServer):
    allow_reuse_address = False  # on Windows a second server could otherwise bind the same port silently


def make_server(corpora, ask, port=8000):
    """corpora: {name: {"label", "index" (None = not ready), "prepare", "examples"}}; ask(question, index, mode) -> events."""
    lock = threading.Lock()  # one GPU: one question at a time

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):  # keep the console quiet
            pass

        def _send(self, code, body, ctype="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _line(self, event):
            self.wfile.write((json.dumps(event) + "\n").encode())
            self.wfile.flush()

        def do_GET(self):
            if self.path == "/api/info":
                return self._send(200, {"corpora": [{"name": n, "label": c["label"], "ready": c["index"] is not None,
                                                     "prepare": c["prepare"], "examples": c["examples"]}
                                                    for n, c in corpora.items()]})
            name = "index.html" if self.path == "/" else self.path.removeprefix("/static/")
            f = STATIC / name
            if f.parent != STATIC or f.suffix not in TYPES or not f.is_file():  # flat folder only
                return self._send(404, {"error": "not found"})
            self._send(200, f.read_bytes(), TYPES[f.suffix])

        def do_POST(self):
            if self.path != "/api/ask":
                return self._send(404, {"error": "not found"})
            port = self.server.server_address[1]  # refuse cross-site posts and DNS rebinding
            if self.headers.get("Content-Type") != "application/json" or self.headers.get("Host") not in (f"127.0.0.1:{port}", f"localhost:{port}"):
                return self._send(403, {"error": "local page only"})
            try:
                req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                question, mode, corpus = req["question"].strip(), req["mode"], req["corpus"]
            except (ValueError, KeyError, TypeError, AttributeError):
                return self._send(400, {"error": "send JSON with question, mode and corpus"})
            if not isinstance(corpus, str) or not question or mode not in MODES or corpora.get(corpus, {}).get("index") is None:
                return self._send(400, {"error": "empty question, unknown mode, or a corpus that is not ready"})
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            with lock:
                try:
                    for event in ask(question, corpora[corpus]["index"], mode):
                        self._line(event)
                except Exception as e:  # shown on this turn; the server keeps running
                    self._line({"type": "error", "message": f"{type(e).__name__}: {e}"})

    return _Server(("127.0.0.1", port), Handler)
