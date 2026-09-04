"""Does a stalled-but-still-alive upstream body get tolerated instead of
being killed by the plain per-read timeout \u2014 while the extended
budget still eventually cuts off a trickle that never stops, and a
genuinely dead upstream is left completely unaffected?

Exercises the buffered/validated relay path (_relay_buffered), since
that's the one where a failure can be told apart cleanly from success
by status code alone \u2014 nothing is written to the client until the
whole body is in, so a request that ultimately fails still comes back
as a plain 429, never a half-sent 200.

Three cases:
  1. The first byte takes a moment (no exception \u2014 modeling a wait
     under timeout_sec), then the rest of the body trickles in about
     one byte at a time, with timeout_sec set low enough that every
     single gap between bytes would, on its own, time out a plain
     read. Every one of those gaps is tolerated because the upstream
     has already proven it's alive, and the whole response finishes
     comfortably inside stream_max_wait_sec.
  2. The exact same trickle pattern, except it never stops. This must
     still fail once stream_max_wait_sec is exhausted \u2014 the budget
     widens the timeout, it does not remove it.
  3. Control: an upstream that never sends a single byte fails on the
     very first read, exactly as before this feature. Patience is
     conditional on the upstream having sent something already, so a
     flat-out dead connection isn't held open for the extended budget
     for no reason.

    python3 test_stream_stall_patience.py ../llm-proxy
"""
import sys

import proxy_under_test as put

proxy = put.load()
Handler = put.make_handler(proxy)

proxy.RETRY_ENABLED = False           # single attempt \u2014 keep the cases clean
proxy.VALIDATE_RESPONSE_BODY = True   # exercise _relay_buffered
proxy.UPSTREAM_TIMEOUT_SEC = 0.5      # any gap over this alone would time out
proxy.STREAM_MAX_WAIT_SEC = 10.0      # hard ceiling for a trickling response

results = []


class FakeClock:
    """Fake time.monotonic(), advanced explicitly instead of by sleeping,
    so the whole test suite runs instantly and deterministically."""

    def __init__(self):
        self.t = 0.0

    def now(self):
        return self.t

    def advance(self, dt):
        self.t += dt


def run_case(name, chunks, advances, want_status, want_body=None, want_reads=None):
    clock = FakeClock()
    pending = list(advances)

    def tick():
        if pending:
            clock.advance(pending.pop(0))

    upstream = put.FakeUpstream(200, put.TEXT_HEADERS, chunks, clock=tick)

    def fake_urlopen(req, timeout=None):
        return upstream

    original_urlopen = proxy.urllib.request.urlopen
    original_monotonic = proxy.time.monotonic
    proxy.urllib.request.urlopen = fake_urlopen
    proxy.time.monotonic = clock.now
    handler = Handler()
    try:
        handler._proxy("POST")
    finally:
        proxy.urllib.request.urlopen = original_urlopen
        proxy.time.monotonic = original_monotonic

    ok = handler.status() == want_status
    if ok and want_body is not None:
        ok = handler.body() == want_body
    if ok and want_reads is not None:
        ok = want_reads(upstream.read_calls)
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         status={handler.status()} (want {want_status}) "
          f"read_calls={upstream.read_calls}")
    return handler


print("\nstalled-but-alive upstream vs. a trickle that never ends vs. a dead one\n")

# Case 1: "H" arrives after a brief wait (no exception raised \u2014 that
# wait stayed under timeout_sec), then five more bytes trickle in about
# 1s apart, each gap alone enough to time out a single read at 0.5s.
trickle = [b"H"]
for byte in b"ELLO!":
    trickle.append(TimeoutError("timed out"))
    trickle.append(bytes([byte]))
run_case(
    "1. first byte slow-but-on-time, then ~1 byte/sec \u2014 tolerated, "
    "request succeeds",
    trickle,
    advances=[0.4] + [1.0] * 10,
    want_status=200,
    want_body=b"HELLO!",
    want_reads=lambda n: n > 1,  # proves timeouts really were retried, not dodged
)

# Case 2: same trickle shape, but it keeps going long enough to blow
# through the 10s stream_max_wait_sec budget. Must still fail.
never_stops = [b"H"]
for _ in range(30):
    never_stops.append(TimeoutError("timed out"))
    never_stops.append(b".")
run_case(
    "2. trickle that never stops still hits the stream_max_wait_sec "
    "ceiling \u2014 request fails (\"time out not removed\")",
    never_stops,
    advances=[0.4] + [1.0] * 200,
    want_status=429,
    want_reads=lambda n: n >= 10,  # failed only after real patience, not instantly
)

# Case 3 (control): upstream never sends a single byte. Fails on the
# very first read \u2014 patience only kicks in once something has
# actually arrived, so a flat-out dead connection isn't held open for
# the extended budget.
run_case(
    "3. dead upstream (never sends a byte) fails on the first read, "
    "unaffected by the extended budget",
    [TimeoutError("timed out")],
    advances=[0.6],
    want_status=429,
    want_reads=lambda n: n == 1,
)

print()
if all(results):
    print(f"{len(results)}/{len(results)} passed\n")
    sys.exit(0)
else:
    print(f"{sum(results)}/{len(results)} passed \u2014 see FAIL lines above\n")
    sys.exit(1)
