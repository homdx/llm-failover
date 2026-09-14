"""Does the sqlite-backed per-key upstream routing behave transparently?

The feature (key_store.py + main._resolve_upstream) is meant to be a
no-op unless a database file is actually present at KEY_STORE_DB_PATH:

  1. No db file at all -> every request goes to [upstream].host/scheme
     from config.toml, exactly as before this feature existed.
  2. A db file exists and the client's api key matches a row -> that
     row's host/scheme are used instead, silently.
  3. A db file exists but the key isn't in it -> falls back to
     [upstream].host/scheme, and this must be visible on the console
     (a person watching the log needs to know why a request went to the
     "wrong" place).
  4. Two different upstream hosts must not share a rate-limit gate or a
     max_concurrent_upstream semaphore -- each host is calculated
     independently, never pooled.

    python3 test_key_routing.py ../llm-proxy
"""
import contextlib
import io
import sys
import tempfile
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import api_manager as am
import proxy_under_test as put

main = put.load()
Handler = put.make_handler(main)

DB_KEY = "sk-or-v1-abcdefghijklmnop"
DB_HOST = "openrouter.ai"

results = []


def report(name, ok, detail=""):
    results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" -- {detail}" if detail else ""))


def run_one(headers_extra=None):
    """Fire one request through _resolve_upstream and _proxy, capturing the
    upstream URL actually contacted and anything printed to the console.
    """
    seen = {}

    def fake_urlopen(req, timeout=None):
        seen["url"] = req.full_url
        return put.FakeUpstream(200, put.JSON_HEADERS, [put.HEALTHY_JSON])

    handler = Handler()
    if headers_extra:
        handler.headers.update(headers_extra)

    original = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    main.urllib.request.urlopen = fake_urlopen
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            handler._proxy("POST")
    finally:
        urllib.request.urlopen = original
        main.urllib.request.urlopen = original

    return seen.get("url"), buf.getvalue(), handler


print("\ntransparent per-key upstream routing (key_store.py)\n")

# --- 1. no db file at KEY_STORE_DB_PATH: must behave exactly as before ---
main.KEY_STORE_DB_PATH = Path("/nonexistent/path/entries.db")
url, log, handler = run_one({"Authorization": f"Bearer {DB_KEY}"})
report(
    "no db file -> uses config.toml default host, no db is even touched",
    url is not None and main.NVIDIA_HOST in url and handler.status() == 200,
    f"url={url}",
)

# --- 2 & 3 need a real db, built with the same tool that ships it ---
tmpdir = tempfile.mkdtemp(prefix="key_routing_test_")
db_path = Path(tmpdir) / "entries.db"
with am.ApiStore(db_path) as store:
    store.add("openrouter", api_key=DB_KEY, base_url=f"https://{DB_HOST}/api/v1", host=DB_HOST)
main.KEY_STORE_DB_PATH = db_path

url, log, handler = run_one({"Authorization": f"Bearer {DB_KEY}"})
report(
    "db has the key -> routed to that entry's host, transparently",
    url is not None and DB_HOST in url and "api key not found" not in log,
    f"url={url}",
)

url, log, handler = run_one({"Authorization": "Bearer sk-not-in-the-db"})
report(
    "db exists but key is missing -> falls back to config default AND says so on the console",
    url is not None and main.NVIDIA_HOST in url and "api key not found" in log,
    f"url={url} logged={'yes' if 'api key not found' in log else 'no'}",
)

url, log, handler = run_one(None)
report(
    "db exists, request carries no key at all -> falls back and logs it",
    url is not None and main.NVIDIA_HOST in url and "api key not found" in log,
    f"url={url}",
)

# --- 4. per-host gate/semaphore must not be pooled across hosts ---
main.MAX_CONCURRENT_UPSTREAM = 1
state_a = main._get_host_state("host-a.example")
state_b = main._get_host_state("host-b.example")
report(
    "two different hosts get two different _HostState objects",
    state_a is not state_b,
)
report(
    "each host's semaphore is its own -- not the same object",
    state_a.slots is not state_b.slots and state_a.slots is not None and state_b.slots is not None,
)

main._gate_penalize(state_a, 30.0)
report(
    "penalizing host A's gate leaves host B's gate untouched",
    main._gate_remaining(state_a) > 0 and main._gate_remaining(state_b) == 0,
    f"a={main._gate_remaining(state_a):.1f}s b={main._gate_remaining(state_b):.1f}s",
)

# the default (config.toml) host must resolve to a stable, reusable state
same_default_a = main._get_host_state(main.NVIDIA_HOST)
same_default_b = main._get_host_state(main.NVIDIA_HOST)
report(
    "the same host always resolves to the same state object",
    same_default_a is same_default_b,
)

print(f"\n{sum(results)}/{len(results)} checks pass")
sys.exit(0 if all(results) else 1)
