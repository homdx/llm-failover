"""Does the tightened body validation leave healthy responses alone?

A validator that rejects too much is worse than no validator: every false
positive becomes ten pointless upstream calls and a fabricated 429 in
place of a response that was fine.

    python3 test_false_positives.py ../llm-proxy
"""
import sys
import threading
import urllib.request

import proxy_under_test as put

proxy = put.load()
Handler = put.make_handler(proxy)

proxy.RETRY_ENABLED = True
proxy.RETRY_MAX_ATTEMPTS = 3
proxy.RETRY_PAUSE_SECONDS = 0.0
proxy.BODY_RETRY_PAUSE_SECONDS = 0.0
proxy.VALIDATE_RESPONSE_BODY = True
if hasattr(proxy, "MAX_TOTAL_RETRY_SECONDS"):
    proxy.MAX_TOTAL_RETRY_SECONDS = 0
    proxy.COOLDOWN_JITTER_SECONDS = 0.0
    proxy.MAX_CONCURRENT_UPSTREAM = 0
    proxy._upstream_slots = None

results = []


def check(name, status, headers, chunks, want_status, want_attempts, method="POST"):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return put.FakeUpstream(status, headers, list(chunks))

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    proxy.urllib.request.urlopen = fake_urlopen
    handler = Handler()
    try:
        handler._proxy(method)
    finally:
        urllib.request.urlopen = original
        proxy.urllib.request.urlopen = original

    ok = handler.status() == want_status and calls["n"] == want_attempts
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         status={handler.status()} attempts={calls['n']} "
          f"(want {want_status}/{want_attempts})")


print("\nhealthy responses must not be retried or masked\n")

check("SSE finished via finish_reason, upstream sends no [DONE]",
      200, put.SSE_HEADERS,
      [b'data: {"choices":[{"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
       b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'],
      200, 1)

check("SSE finished via [DONE], no chunk ever carried a finish_reason",
      200, put.SSE_HEADERS,
      [b'data: {"choices":[{"delta":{"content":"hi"}}]}\n\n'
       b'data: [DONE]\n\n'],
      200, 1)

check("SSE with a usage-only trailing chunk after the finish_reason",
      200, put.SSE_HEADERS,
      [b'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
       b'data: {"choices":[],"usage":{"total_tokens":9}}\n\n'],
      200, 1)

check("non-stream JSON with finish_reason=length (a real max_tokens cutoff, "
      "complete JSON, nothing to retry)",
      200, put.JSON_HEADERS,
      [b'{"choices":[{"message":{"content":"partial"},"finish_reason":"length"}]}'],
      200, 1)

check("text/plain body", 200, put.TEXT_HEADERS, [b"pong"], 200, 1)

check("explicit Content-Length: 0",
      200, put.JSON_HEADERS + [("Content-Length", "0")], [], 200, 1)

check("304 Not Modified", 304, put.JSON_HEADERS, [], 304, 1)

check("a genuinely broken body is still caught (control)",
      200, put.JSON_HEADERS, [b"x"], 429, 3)

# An over-sized response must be delivered, not turned into a 429.
saved = proxy.MAX_BUFFER_BYTES
proxy.MAX_BUFFER_BYTES = 1000
check("response past max_buffer_bytes is streamed through, not failed",
      200, put.JSON_HEADERS,
      [b'{"choices":[{"message":{"content":"' + b"a" * 5000 + b'"}}]}'],
      200, 1)
proxy.MAX_BUFFER_BYTES = saved

print(f"\n{sum(results)}/{len(results)} pass")
sys.exit(0 if all(results) else 1)
