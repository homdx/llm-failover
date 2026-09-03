"""How much traffic does the proxy generate when the upstream is rate
limiting it?

Five requests in parallel, upstream answers 429 with a Retry-After every
single time — the situation from the production log. Times are scaled
down 10x so the run finishes quickly (0.5s here stands for 5s there).

Run it against both the old and the new proxy to compare:

    python3 bench_ratelimit_storm.py ../llm-proxy
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

# The upstream keeps asking for the same short wait. Real time throughout:
# the shared cooldown reads time.monotonic(), so faking sleep would put the
# two clocks out of step and the measurement would be meaningless.
RETRY_AFTER = "0.5"

FakeHandler = put.make_handler(main)

hits = []
hits_lock = threading.Lock()


def rate_limited(req, timeout=None):
    with hits_lock:
        hits.append(time.monotonic())
    raise urllib.error.HTTPError(
        req.full_url, 429, "Too Many Requests",
        {"Retry-After": RETRY_AFTER}, io.BytesIO(b'{"error":"rate limit"}'))


def run(n_clients=5):
    hits.clear()
    if hasattr(main, "_gate_clear"):
        main._gate_clear()
    urllib.request.urlopen = rate_limited
    main.urllib.request.urlopen = rate_limited

    handlers = [FakeHandler() for _ in range(n_clients)]
    t0 = time.monotonic()
    threads = [threading.Thread(target=h._proxy, args=("POST",)) for h in handlers]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.monotonic() - t0

    retry_afters = [int(h.header("Retry-After")) for h in handlers
                    if h.header("Retry-After")]
    statuses = [h.status() for h in handlers]
    return len(hits), wall, statuses, retry_afters


print("\nscenario: 5 parallel requests, upstream answers 429 Retry-After: 0.5s every time")
print("(times scaled 10x down from the real log: 0.5s here = 5s there)")

if hasattr(main, "MAX_CONCURRENT_UPSTREAM"):
    main.MAX_CONCURRENT_UPSTREAM = 2
    main._upstream_slots = threading.BoundedSemaphore(2)
    main.MAX_TOTAL_RETRY_SECONDS = 4.5
    main.COOLDOWN_JITTER_SECONDS = 0.05
    main.RETRY_BACKOFF_FACTOR = 2.0
    label = "PATCHED (gate + probe + 2 slots + 4.5s budget)"
else:
    label = "HEAD (each thread retries on its own)"

n, wall, statuses, retry_afters = run()
print(f"\n{label}")
print(f"  requests sent upstream : {n}")
print(f"  wall time held         : {wall:.1f}s")
print(f"  upstream request rate  : {n / wall:.1f}/s")
print(f"  statuses to client     : {statuses}")
print(f"  Retry-After to client  : {retry_afters}")
