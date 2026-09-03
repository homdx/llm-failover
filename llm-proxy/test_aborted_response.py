"""Once bytes have gone out to the client, an upstream failure must not
trigger a retry.

The client already holds half a body, so the retry can never be
delivered — and the upstream regenerates (and bills) the whole completion
to produce something nobody will read. Worse, starting a second response
writes a raw status line into the middle of the first.

Reached through the passthrough path: a response past max_buffer_bytes
stops being buffered and starts streaming, and then the upstream dies.

    python3 test_aborted_response.py ../llm-proxy
"""
import http.client
import sys
import urllib.request

import proxy_under_test as put

main = put.load()
main.RETRY_ENABLED = True
main.RETRY_MAX_ATTEMPTS = 5
main.BODY_RETRY_PAUSE_SECONDS = 0.0
main.RETRY_PAUSE_SECONDS = 0.0
main.VALIDATE_RESPONSE_BODY = True
main.MAX_BUFFER_BYTES = 1000
if hasattr(main, "MAX_TOTAL_RETRY_SECONDS"):
    main.MAX_TOTAL_RETRY_SECONDS = 0
    main.COOLDOWN_JITTER_SECONDS = 0.0

# Real send_response, so a spliced status line shows up in the bytes.
FakeHandler = put.make_handler(main, real_headers=True)

BIG = b"x" * 4000
calls = {"n": 0}


class DyingUpstream(put.FakeUpstream):
    def __init__(self):
        super().__init__(200, put.JSON_HEADERS, [BIG, BIG],
                         raise_at_end=http.client.IncompleteRead(b"", 9999))


def dying(req, timeout=None):
    calls["n"] += 1
    return DyingUpstream()


urllib.request.urlopen = dying
main.urllib.request.urlopen = dying

h = FakeHandler()
h._proxy("POST")

statuses = h.statuses()
body = h.body()

print("\nupstream dies AFTER the passthrough response has started")
print(f"  upstream requests sent : {calls['n']}   (must be 1 — a retry is "
      f"undeliverable and re-bills the completion)")
print(f"  status lines written   : {statuses}   (must be exactly one)")
print(f"  bytes delivered        : {len(body)}")
n_status_lines = body.count(b"HTTP/1.1 ")
print(f"  status lines IN the body: {n_status_lines}   (must be 1 — the real "
      f"one; more means a second response was spliced into the first)")

ok = calls["n"] == 1 and len(statuses) == 1 and n_status_lines == 1
print(f"  --> {'PASS' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
