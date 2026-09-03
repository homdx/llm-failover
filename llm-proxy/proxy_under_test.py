"""Shared plumbing for the standalone proxy test scripts.

Each script takes the folder holding the proxy and its config.toml:

    python3 test_truncation.py ../llm-proxy

The proxy module is imported under whichever name it has in that folder
(main.py, python_proxy2.py, ...), with the working directory switched to
it first — the proxy reads config.toml from the CWD at import time.
"""
import io
import os
import sys

PROXY_CANDIDATES = ("main.py", "python_proxy2.py", "python_proxy.py", "proxy.py")


def load(argv=None):
    """Import the proxy module from the folder named on the command line."""
    argv = argv if argv is not None else sys.argv
    if len(argv) < 2:
        sys.exit(f"usage: python3 {os.path.basename(argv[0])} <folder-with-the-proxy>")

    folder = os.path.abspath(argv[1])
    if not os.path.isdir(folder):
        sys.exit(f"not a folder: {folder}")

    for name in PROXY_CANDIDATES:
        if os.path.exists(os.path.join(folder, name)):
            module_name = name[:-3]
            break
    else:
        sys.exit(f"no proxy script in {folder} (looked for {', '.join(PROXY_CANDIDATES)})")

    if not os.path.exists(os.path.join(folder, "config.toml")):
        sys.exit(f"no config.toml in {folder} — the proxy reads it at import time")

    os.chdir(folder)
    sys.path.insert(0, folder)
    proxy = __import__(module_name)
    proxy.LOG_ENABLED = False  # keep the test run out of logs/*.jsonl
    return proxy


def make_handler(proxy, real_headers=False):
    """Build a ProxyHandler subclass that talks to a BytesIO, not a socket.

    real_headers=True keeps the inherited send_response/end_headers, so a
    second status line lands in the captured bytes exactly as it would on
    a live connection. Otherwise they're stubbed, which keeps the captured
    body free of HTTP framing and easier to assert on.
    """

    class FakeHandler(proxy.ProxyHandler):
        def __init__(self):
            self.path = "/v1/chat/completions"
            self.headers = {"Content-Type": "application/json", "Content-Length": "2"}
            self.rfile = io.BytesIO(b"{}")
            self.wfile = io.BytesIO()
            self.request_version = "HTTP/1.1"
            self.requestline = "POST /v1/chat/completions HTTP/1.1"
            self.client_address = ("127.0.0.1", 0)
            self.sent = []

        def send_response(self, code, message=None):
            self.sent.append(code)
            if real_headers:
                super().send_response(code, message)

        if not real_headers:
            def send_header(self, key, value):
                self.sent.append((key, value))

            def end_headers(self):
                pass

        def log_message(self, *args):
            pass

        def status(self):
            """The first status code handed to the client."""
            return next(c for c in self.sent if isinstance(c, int))

        def statuses(self):
            return [c for c in self.sent if isinstance(c, int)]

        def header(self, name):
            for item in self.sent:
                if isinstance(item, tuple) and item[0] == name:
                    return item[1]
            return None

        def body(self):
            return self.wfile.getvalue()

    return FakeHandler


class FakeUpstream:
    """Mimics http.client.HTTPResponse closely enough for the relay paths.

    chunks are returned one per read(); raise_at_end is raised once the
    chunks run out, which is how a body that stops arriving part-way
    through actually presents itself.
    """

    def __init__(self, status, headers, chunks, raise_at_end=None):
        self.status = status
        self._headers = headers
        self._chunks = list(chunks)
        self._raise = raise_at_end

    def getheaders(self):
        return self._headers

    def read(self, amt=None):
        if self._chunks:
            return self._chunks.pop(0)
        if self._raise is not None:
            exc, self._raise = self._raise, None
            raise exc
        return b""

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


JSON_HEADERS = [("Content-Type", "application/json")]
SSE_HEADERS = [("Content-Type", "text/event-stream")]
TEXT_HEADERS = [("Content-Type", "text/plain")]

HEALTHY_JSON = (
    b'{"id":"1","choices":[{"index":0,"message":{"content":"hi"},'
    b'"finish_reason":"stop"}],"usage":{"prompt_tokens":1,'
    b'"completion_tokens":1,"total_tokens":2}}'
)
