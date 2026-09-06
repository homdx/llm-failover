"""Does the client-side keepalive hold a connection open without breaking
anything that worked before it existed?

The feature under test: when the upstream stays silent longer than
heartbeat_after_sec and the request asked for a stream, the proxy commits
to a 200 text/event-stream early and drips ignorable ": keepalive" SSE
comments until the real response arrives. That defeats the client's own
idle timeout (undici's default headersTimeout is 300s) while a cold model
loads.

Everything here is about the boundaries of that behaviour, because the
behaviour itself is cheap and the boundaries are where it can do damage:

  * OFF by default. heartbeat_after_sec = 0 must reproduce the old code
    byte for byte, including on a slow upstream.
  * Never on a non-stream request. There is nowhere to hide a keepalive
    in a plain JSON response, and committing 200 early would be
    unrecoverable.
  * Never when the upstream answers in time — which is what keeps
    ordinary 4xx/5xx coming back as real status codes and keeps the
    silent-retry machinery intact.
  * Exactly one status line on the wire, ever. Splicing a second
    "HTTP/1.1 ..." into a body in flight is the original bug this proxy
    exists to prevent, and an early 200 is a new way to hit it.
  * No keepalive spliced into a real chunk. The stop() join is what
    guarantees it, so the ordering is asserted on the delivered bytes.
  * An error that arrives after the commit still reaches the client, as
    a data chunk plus [DONE], rather than as a silently truncated stream.

    python3 test_heartbeat.py ../llm-proxy
"""
import io
import sys
import urllib.error
import time
import urllib.request

import proxy_under_test as put

proxy = put.load()

if not hasattr(proxy, "HEARTBEAT_AFTER_SEC"):
    sys.exit("this proxy has no heartbeat support (HEARTBEAT_AFTER_SEC missing)")

# Same shape as the other scripts: retries on, but with the waits taken
# out so a whole case runs in well under a second.
proxy.RETRY_ENABLED = True
proxy.RETRY_MAX_ATTEMPTS = 3
proxy.RETRY_PAUSE_SECONDS = 0.0
proxy.RETRY_MAX_PAUSE_SECONDS = 1800
proxy.BODY_RETRY_PAUSE_SECONDS = 0.0
proxy.VALIDATE_RESPONSE_BODY = False
if hasattr(proxy, "MAX_TOTAL_RETRY_SECONDS"):
    proxy.MAX_TOTAL_RETRY_SECONDS = 0
    proxy.COOLDOWN_JITTER_SECONDS = 0.0

# Short enough to keep the suite fast, long enough that a "fast" upstream
# (which answers instantly) is never mistaken for a slow one.
HB_AFTER = 0.25
HB_EVERY = 0.05
SLOW = HB_AFTER + 0.35

STREAM_BODY = b'{"stream":true,"messages":[]}'
PLAIN_BODY = b'{"messages":[]}'

SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n',
    b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n',
    b'data: [DONE]\n\n',
]

# real_headers=True: the status line lands in the captured bytes exactly
# as it would on a live socket, which is the only way to count them.
Base = put.make_handler(proxy, real_headers=True)


class Handler(Base):
    """Base handler with a request body we control per case."""

    request_body = STREAM_BODY

    def __init__(self):
        super().__init__()
        self.headers = {
            "Content-Type": "application/json",
            "Content-Length": str(len(self.request_body)),
        }
        self.rfile = io.BytesIO(self.request_body)


results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"         {detail}")


def run(body_bytes, hb_after, upstream, delay=0.0):
    """Drive one request through the proxy against a faked upstream.

    upstream is called with no arguments to produce the response (or to
    raise). delay is how long it pretends to think before answering —
    the whole point of the feature is what happens during that window.
    """
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        if delay:
            time.sleep(delay)
        return upstream()

    class Case(Handler):
        request_body = body_bytes

    original = urllib.request.urlopen
    saved_after, saved_every = proxy.HEARTBEAT_AFTER_SEC, proxy.HEARTBEAT_INTERVAL_SEC
    urllib.request.urlopen = fake_urlopen
    proxy.urllib.request.urlopen = fake_urlopen
    proxy.HEARTBEAT_AFTER_SEC = hb_after
    proxy.HEARTBEAT_INTERVAL_SEC = HB_EVERY
    handler = Case()
    try:
        handler._proxy("POST")
    finally:
        urllib.request.urlopen = original
        proxy.urllib.request.urlopen = original
        proxy.HEARTBEAT_AFTER_SEC = saved_after
        proxy.HEARTBEAT_INTERVAL_SEC = saved_every
    return handler, calls["n"]


def sse_ok():
    return put.FakeUpstream(200, put.SSE_HEADERS, list(SSE_CHUNKS))


def http_error(code, body=b'{"error":{"message":"boom"}}', headers=None):
    def raise_it():
        raise urllib.error.HTTPError(
            "http://x/v1/chat/completions", code, "err",
            headers or {"Content-Type": "application/json"}, io.BytesIO(body))
    return raise_it


print("\nclient-side keepalive (heartbeat)\n")

# ------------------------------------------------------------------ 1
# The regression that matters most: with the feature off, a slow upstream
# has to behave exactly as it did before the feature was written.
h, n = run(STREAM_BODY, 0, sse_ok, delay=SLOW)
body = h.body()
check("1. heartbeat_after_sec = 0 — old behaviour on a slow upstream",
      b"keepalive" not in body and h.statuses() == [200]
      and body.count(b"HTTP/1.1 ") == 1 and b"[DONE]" in body,
      f"keepalives=0 statuses={h.statuses()} attempts={n}")

# ------------------------------------------------------------------ 2
h, n = run(PLAIN_BODY, HB_AFTER, lambda: put.FakeUpstream(
    200, put.JSON_HEADERS, [put.HEALTHY_JSON]), delay=SLOW)
body = h.body()
check("2. non-stream request — never gets a keepalive, however slow",
      b"keepalive" not in body and h.statuses() == [200]
      and put.HEALTHY_JSON in body,
      f"statuses={h.statuses()}")

# ------------------------------------------------------------------ 3
h, n = run(STREAM_BODY, HB_AFTER, sse_ok, delay=0.0)
body = h.body()
check("3. fast upstream — no keepalive, so error statuses stay real",
      b"keepalive" not in body and h.statuses() == [200],
      f"statuses={h.statuses()}")

# ------------------------------------------------------------------ 4
h, n = run(STREAM_BODY, HB_AFTER, sse_ok, delay=SLOW)
body = h.body()
beats = body.count(b": keepalive")
check("4. slow stream — keepalives, then the real chunks, one status line",
      beats >= 1 and h.statuses() == [200] and body.count(b"HTTP/1.1 ") == 1
      and b"[DONE]" in body and b'"content":"hi"' in body,
      f"keepalives={beats} status lines={body.count(b'HTTP/1.1 ')} attempts={n}")

# ------------------------------------------------------------------ 5
# Ordering, not just presence: every keepalive must be flushed before the
# first real chunk. A comment landing inside a data: line would corrupt
# the stream in a way the client can't recover from.
first_data = body.find(b"data: ")
last_beat = body.rfind(b": keepalive")
check("5. no keepalive spliced into a real chunk",
      last_beat != -1 and first_data != -1 and last_beat < first_data,
      f"last keepalive at {last_beat}, first data: at {first_data}")

# ------------------------------------------------------------------ 6
# Content-Type has to be committed as text/event-stream, or the client
# won't treat what follows as a stream at all.
check("6. early commit declares text/event-stream",
      b"text/event-stream" in body.split(b"\r\n\r\n")[0],
      "declared in the headers written at commit time")

# ------------------------------------------------------------------ 7
# Non-retryable error after the commit: the status code is spent, so it
# travels as a chunk. What must NOT happen is a second status line.
h, n = run(STREAM_BODY, HB_AFTER, http_error(404), delay=SLOW)
body = h.body()
check("7. 404 after the commit — delivered in-stream, not as a 2nd status",
      h.statuses() == [200] and body.count(b"HTTP/1.1 ") == 1
      and b"boom" in body and b"[DONE]" in body,
      f"statuses={h.statuses()} status lines={body.count(b'HTTP/1.1 ')} attempts={n}")

# ------------------------------------------------------------------ 8
# Retryable error: retries still run under a live heartbeat (a comment is
# not an answer), and only the exhausted end result goes in-stream.
h, n = run(STREAM_BODY, HB_AFTER, http_error(502), delay=SLOW)
body = h.body()
check("8. 502 — silent retries still run, exhausted result goes in-stream",
      n == 3 and h.statuses() == [200] and body.count(b"HTTP/1.1 ") == 1
      and b"rate_limit_error" in body and b"[DONE]" in body,
      f"attempts={n} (want 3) statuses={h.statuses()}")

# ------------------------------------------------------------------ 9
# The gate that decides whether any of this applies at all.
cases = [
    (b'{"stream":true}', True),
    (b'{"stream":false}', False),
    (b'{"messages":[]}', False),
    (b'', False),
    (b'not json at all', False),
    (b'{"stream":true', False),
]
bad = [b for b, want in cases if proxy._wants_stream(b) is not want]
check("9. _wants_stream — only a real streaming request qualifies",
      not bad, f"misjudged: {bad}" if bad else "6/6 bodies classified correctly")

print(f"\n{sum(results)}/{len(results)} pass")
sys.exit(0 if all(results) else 1)
