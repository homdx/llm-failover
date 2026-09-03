"""Does a response that never arrived intact get retried instead of
forwarded broken?

Six ways a body can be truncated, plus a control. The one that matters
most in practice is case 2: a long reasoning phase outruns the gateway
and the connection drops part-way through the body. Nothing has been
written to the client yet at that point, so it can and should be retried
silently.

    python3 test_truncation.py ../llm-proxy
"""
import http.client
import socket
import sys
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

results = []


def run_case(name, upstream_factory, want_status, want_attempts):
    calls = {"n": 0}

    def fake_urlopen(req, timeout=None):
        calls["n"] += 1
        return upstream_factory()

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    proxy.urllib.request.urlopen = fake_urlopen
    handler = Handler()
    try:
        handler._proxy("POST")
    finally:
        urllib.request.urlopen = original
        proxy.urllib.request.urlopen = original

    ok = handler.status() == want_status and calls["n"] == want_attempts
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         status={handler.status()} attempts={calls['n']} "
          f"(want {want_status}/{want_attempts})")
    return handler


print("\ntruncated / corrupted upstream bodies\n")

run_case(
    "1. body arrives complete but cut off mid-JSON",
    lambda: put.FakeUpstream(200, put.JSON_HEADERS, [put.HEALTHY_JSON[:60]]),
    429, 3)

run_case(
    "2. connection dies mid-body (IncompleteRead)",
    lambda: put.FakeUpstream(200, put.JSON_HEADERS, [put.HEALTHY_JSON[:60]],
                             raise_at_end=http.client.IncompleteRead(b"", 400)),
    429, 3)

run_case(
    "3. socket times out mid-body, e.g. a stalled reasoning phase",
    lambda: put.FakeUpstream(200, put.JSON_HEADERS, [put.HEALTHY_JSON[:60]],
                             raise_at_end=socket.timeout("timed out")),
    429, 3)

run_case(
    "4. SSE cut on a clean chunk boundary: every line parses, no "
    "finish_reason ever arrived",
    lambda: put.FakeUpstream(200, put.SSE_HEADERS, [
        b'data: {"choices":[{"delta":{"content":"th"},"finish_reason":null}]}\n\n'
        b'data: {"choices":[{"delta":{"content":"ink"},"finish_reason":null}]}\n\n']),
    429, 3)

run_case(
    "5. legitimately empty body (204) must pass straight through",
    lambda: put.FakeUpstream(204, put.JSON_HEADERS, []),
    204, 1)

run_case(
    "6. healthy response, control",
    lambda: put.FakeUpstream(200, put.JSON_HEADERS, [put.HEALTHY_JSON]),
    200, 1)

# ---------------------------------------------------------------- case 7
# The streaming path (validate_response_body = false). Headers and some
# chunks are already with the client when upstream dies, so a retry is
# impossible — but writing a 502 on top of the body in flight would
# splice a raw status line into it, which is worse than saying nothing.
StreamHandler = put.make_handler(proxy, real_headers=True)


def stream_case():
    proxy.VALIDATE_RESPONSE_BODY = False

    def fake_urlopen(req, timeout=None):
        return put.FakeUpstream(
            200, put.SSE_HEADERS,
            [b'data: {"choices":[{"delta":{"content":"th"}}]}\n\n'],
            raise_at_end=http.client.IncompleteRead(b"", 400))

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    proxy.urllib.request.urlopen = fake_urlopen
    handler = StreamHandler()
    try:
        handler._proxy("POST")
    finally:
        urllib.request.urlopen = original
        proxy.urllib.request.urlopen = original
        proxy.VALIDATE_RESPONSE_BODY = True

    status_lines = handler.body().count(b"HTTP/1.1 ")
    ok = len(handler.statuses()) == 1 and status_lines == 1
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] 7. streaming path, upstream dies "
          f"after the response started")
    print(f"         status lines in the delivered bytes={status_lines} (want 1)")


stream_case()

print(f"\n{sum(results)}/{len(results)} pass")
sys.exit(0 if all(results) else 1)
