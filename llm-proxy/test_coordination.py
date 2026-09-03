"""Does coordinating the retries break the cases that used to work?

Sharing one cooldown across threads and letting a single request probe a
lifted rate limit only pays off if a healthy upstream still runs fully in
parallel and recovery still happens promptly.

    python3 test_coordination.py ../llm-proxy
"""
import io
import sys
import threading
import time
import urllib.error
import urllib.request

import proxy_under_test as put

main = put.load()
main.RETRY_ENABLED = True
main.RETRY_MAX_ATTEMPTS = 10
main.RETRY_STATUS_CODES = {429, 500, 502, 529}
main.RETRY_PAUSE_SECONDS = 1.5
main.RETRY_MAX_PAUSE_SECONDS = 18
main.VALIDATE_RESPONSE_BODY = True

PATCHED = hasattr(main, "_await_turn")
if PATCHED:
    main.MAX_CONCURRENT_UPSTREAM = 2
    main._upstream_slots = threading.BoundedSemaphore(2)
    main.MAX_TOTAL_RETRY_SECONDS = 6.0
    main.COOLDOWN_JITTER_SECONDS = 0.05
    main.RETRY_BACKOFF_FACTOR = 2.0

GOOD = put.HEALTHY_JSON


class FakeResp(put.FakeUpstream):
    def __init__(self):
        super().__init__(200, put.JSON_HEADERS, [GOOD])


FakeHandler = put.make_handler(main)


def drive(n_clients, responder):
    if PATCHED:
        main._gate_clear()
    urllib.request.urlopen = responder
    main.urllib.request.urlopen = responder
    handlers = [FakeHandler() for _ in range(n_clients)]
    t0 = time.monotonic()
    threads = [threading.Thread(target=h._proxy, args=("POST",)) for h in handlers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return handlers, time.monotonic() - t0


results = []


def report(name, ok, detail):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


print(f"\n{'patched' if PATCHED else 'HEAD'} build\n")

# 1. healthy upstream, 5 parallel requests: must not be slowed or serialised
calls = []
lock = threading.Lock()


def healthy(req, timeout=None):
    with lock:
        calls.append(1)
    return FakeResp()


hs, wall = drive(5, healthy)
report("healthy upstream, 5 parallel",
       all(h.status() == 200 for h in hs) and len(calls) == 5 and wall < 1.0,
       f"statuses={[h.status() for h in hs]} upstream_calls={len(calls)} wall={wall:.2f}s")

# 2. recovery: 429 for the first 4 calls, then healthy
state = {"n": 0}


def recovers(req, timeout=None):
    with lock:
        state["n"] += 1
        n = state["n"]
    if n <= 4:
        raise urllib.error.HTTPError(
            req.full_url, 429, "Too Many Requests",
            {"Retry-After": "0.3"}, io.BytesIO(b'{"error":"rate limit"}'))
    return FakeResp()


state["n"] = 0
hs, wall = drive(5, recovers)
report("upstream recovers after 4 rejections",
       all(h.status() == 200 for h in hs),
       f"statuses={[h.status() for h in hs]} upstream_calls={state['n']} wall={wall:.2f}s")

# 2b. that success must leave no cooldown armed behind it
if PATCHED:
    report("gate cleared after a success", main._gate_remaining() == 0,
           f"remaining={main._gate_remaining():.2f}s")

# 3. Retry-After handed to the client must not undercut what upstream asked
def slow_limit(req, timeout=None):
    raise urllib.error.HTTPError(
        req.full_url, 429, "Too Many Requests",
        {"Retry-After": "4"}, io.BytesIO(b'{"error":"rate limit"}'))


main.RETRY_MAX_ATTEMPTS = 2
if PATCHED:
    main.MAX_TOTAL_RETRY_SECONDS = 0  # let it exhaust attempts, not the budget
hs, wall = drive(1, slow_limit)
ra = int(hs[0].header("Retry-After"))
report("Retry-After >= what upstream asked (4s)",
       ra >= 4,
       f"upstream asked 4s, config pause_seconds=1.5s, told client {ra}s")
main.RETRY_MAX_ATTEMPTS = 10
if PATCHED:
    main.MAX_TOTAL_RETRY_SECONDS = 6.0

print(f"\n{sum(results)}/{len(results)} checks pass")
sys.exit(0 if all(results) else 1)
