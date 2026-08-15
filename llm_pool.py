"""
llm_pool.py — several model servers instead of one (POOL-1).

WHAT IT IS

A wrapper over a set of LLMClients. PooledClient mirrors the LLMClient
interface (chat / chat_json), so it drops into any code that used to hold a
single client, with no change on the caller's side.

HOW ACCESS WORKS

A free server is leased per CALL and returned afterwards. Per call, not per
caller: with five callers and two servers, pinning would give 3/2 and half
the capacity idle.

FAULT TOLERANCE

If a server errors, the call is repeated on a DIFFERENT free server. That is
the point of the pool beyond speed: one stuck model no longer kills the call.
A server that errors several times in a row is parked and returns after a
cooldown, so every later call does not keep spending an attempt on it.

One server in the config means exactly the old behaviour: the lease always
succeeds instantly and there is no control flow.

PARITY-1 (THE KEY PROPERTY OF THIS FILE)

Pool clients are built by the SAME LLMClient.from_config() as a single
client, only with an explicit section name. The pool does NOT construct
LLMClient by hand and does not re-list [api] settings. Otherwise every new
setting in llm_client.py (as already happened with error_retries,
max_retry_after_sec, rotate_on_429, rate_limit_cycles) reaches single-server
mode and silently misses the pool — and the same config behaves differently
depending on whether it names one section or two. That drift is impossible
here by construction: there is exactly one set of defaults, and it lives in
LLMClient.from_config().
"""

from __future__ import annotations

import threading
import time

import llm_client as _llm            # the MODULE, not the name: tests and
                                     # stub mode patch llm_client.LLMClient
                                     # AFTER import, and a name bound at
                                     # import time would miss that patch.
from llm_client import LLMUnavailable

# How many consecutive errors park a server.
ENDPOINT_FAIL_THRESHOLD = 3
# For how long. Not forever: a remote gateway that returned 500 usually
# recovers on its own, and writing it off for good loses half the pool.
ENDPOINT_COOLDOWN_SEC = 120.0


class _Endpoint:
    """One server: its client, busy flag, consecutive-error counter."""

    def __init__(self, client, name: str):
        self.client = client
        self.name = name
        self.busy = False
        self.fails = 0
        self.blocked_until = 0.0

    def available(self, now: float) -> bool:
        return not self.busy and now >= self.blocked_until

    def note_ok(self):
        self.fails = 0

    def note_fail(self) -> bool:
        """True if this error parked the server."""
        self.fails += 1
        if self.fails >= ENDPOINT_FAIL_THRESHOLD:
            self.blocked_until = time.monotonic() + ENDPOINT_COOLDOWN_SEC
            self.fails = 0
            return True
        return False


class LLMPool:
    """
    Set of servers with leasing. Thread-safe: independent calls enter it from
    several threads at once.
    """

    def __init__(self, clients: list, names: list[str] | None = None):
        if not clients:
            raise ValueError("LLMPool: at least one client is required")
        names = names or [f"ep{i}" for i in range(len(clients))]
        self._eps = [_Endpoint(c, n) for c, n in zip(clients, names)]
        self._cv = threading.Condition()
        self.on_event = None          # optional log sink: on_event(str)

    def __len__(self):
        return len(self._eps)

    @property
    def size(self) -> int:
        return len(self._eps)

    def _log(self, msg: str):
        if self.on_event:
            try:
                self.on_event(msg)
            except Exception:
                pass

    def acquire(self, exclude: set | None = None, timeout: float = 300.0):
        """
        Lease a free server. `exclude` holds names already tried: after an
        error we need a DIFFERENT one.

        Returns an _Endpoint, or None if nothing suitable turned up within
        timeout. None is not a failure — the caller falls back to any available
        server, including one already tried.
        """
        exclude = exclude or set()
        deadline = time.monotonic() + timeout
        with self._cv:
            while True:
                now = time.monotonic()
                for ep in self._eps:
                    if ep.name not in exclude and ep.available(now):
                        ep.busy = True
                        return ep
                # All are busy, parked, or excluded.
                if not any(ep.name not in exclude for ep in self._eps):
                    return None
                left = deadline - now
                if left <= 0:
                    return None
                # We wait for both a release and a cooldown expiry, so we
                # wake up on our own, not only on notify.
                self._cv.wait(min(left, 1.0))

    def release(self, ep):
        with self._cv:
            ep.busy = False
            self._cv.notify_all()


class PooledClient:
    """
    Drop-in replacement for LLMClient. Same interface, a pool inside.

    The number of attempts is NOT the number of servers: two servers must not
    turn into two retries of every call. A repeat on another server happens
    only on error, at most max_failover times.
    """

    def __init__(self, pool: LLMPool, max_failover: int = 1, on_retry=None):
        self.pool = pool
        self.max_failover = max(0, int(max_failover))
        self.on_retry = on_retry

    # Callers read these off the client (logging, debugging). Expose the
    # first server as the representative one, or third-party code hits an
    # AttributeError for nothing.
    @property
    def model(self):
        return self.pool._eps[0].client.model

    @property
    def base_url(self):
        return self.pool._eps[0].client.base_url

    @property
    def models(self):
        # PARITY-3: LLMClient has .models (the MODEL-FALLBACK list); the
        # wrapper had nothing, so code that merely logged client.models
        # raised AttributeError exactly when moving from one section to a
        # pool. Same representative server as .model/.base_url above.
        return getattr(self.pool._eps[0].client, "models",
                       [self.pool._eps[0].client.model])

    def _call(self, method: str, *a, **kw):
        tried: set = set()
        last = None
        for attempt in range(self.max_failover + 1):
            ep = self.pool.acquire(exclude=tried)
            if ep is None:
                # No other free server — take any, rather than stall.
                ep = self.pool.acquire()
                if ep is None:
                    break
            try:
                result = getattr(ep.client, method)(*a, **kw)
                ep.note_ok()
                return result
            except LLMUnavailable:
                # The breaker counts calls across ALL servers (_Breaker is
                # global), so this is a statement about the whole pool.
                # Another server will not help here.
                raise
            except Exception as e:
                last = e
                tried.add(ep.name)
                if ep.note_fail():
                    self.pool._log(f"endpoint {ep.name} parked for "
                                   f"{ENDPOINT_COOLDOWN_SEC:.0f}s after "
                                   f"repeated failures")
                if attempt < self.max_failover and self.on_retry:
                    self.on_retry(f"endpoint {ep.name} failed ({type(e).__name__}"
                                  f": {e}) — retrying on another server")
            finally:
                self.pool.release(ep)
        raise last if last else RuntimeError("LLMPool: could not lease a server")

    def chat(self, *a, **kw):
        return self._call("chat", *a, **kw)

    def chat_json(self, *a, **kw):
        return self._call("chat_json", *a, **kw)


# ── config ───────────────────────────────────────────────────────────────

def clients_from_config(cfg) -> tuple[list, list[str]]:
    """
    Read the pool from the config. The format is backwards compatible:

        [api]
        active = remote
        pool = api_remote, api_remote2      ; optional

        [api_remote]
        base_url = https://api.one/
        model    = qwen3:30b

        [api_remote2]
        base_url = https://api.two/
        model    = qwen3:8b

    Without the `pool` key a single api_<active> section is used, as before.
    Different models across the pool are allowed on purpose: a spare server
    with a weaker model beats a failed call.
    """
    active = cfg.get("api", "active", fallback="local")
    raw = cfg.get("api", "pool", fallback="").strip()
    sections = ([s.strip() for s in raw.split(",") if s.strip()]
                if raw else [f"api_{active}"])

    clients, names = [], []
    for sec in sections:
        if not cfg.has_section(sec):
            raise ValueError(f"[api] pool refers to section {sec!r}, "
                             f"which is missing from the config")
        clients.append(_client_for_section(cfg, sec))
        names.append(sec)
    return clients, names


def _client_for_section(cfg, sec: str):
    """
    Client for an explicitly named server section.

    PARITY-1: this used to list constructor arguments by hand, and the list
    LAGGED BEHIND LLMClient.from_config() — the pool never received
    rotate_on_429 or rate_limit_cycles at all (so a 429 in pool mode slept
    inside chat() instead of rotating models), while error_retries and
    max_retry_after_sec had to be patched in here after the fact. Now the
    section is simply passed to from_config: any [api] setting applies to the
    pool and to a single client identically and automatically, with no edit
    here.
    """
    return _llm.LLMClient.from_config(cfg, api_section=sec)


def build_client(cfg, on_retry=None):
    """
    Main entry point: what to hand to the caller.

    One server → a plain LLMClient (no threads or locks for nothing). Two or
    more → a PooledClient.
    """
    clients, names = clients_from_config(cfg)
    if len(clients) == 1:
        c = clients[0]
        c.on_retry = on_retry
        return c
    # PARITY-2: on_retry goes on EVERY pool client, not just the wrapper.
    # Without it, pool mode silently dropped every message LLMClient emits
    # itself: a 429 pause, a RATE-ROT model switch, HTTP-RETRY on 402/5xx.
    # Single-server mode shows them (`c.on_retry = on_retry` above) — the
    # same config, different observability.
    for c in clients:
        c.on_retry = on_retry
    pool = LLMPool(clients, names)
    return PooledClient(pool, max_failover=cfg.getint("api", "max_failover",
                                                      fallback=1),
                        on_retry=on_retry)


_SHARED_POOL: LLMPool | None = None
_SHARED_LOCK = threading.Lock()


def shared_client(cfg, on_retry=None, factory=None):
    """
    Same as build_client, but ONE POOL PER PROCESS.

    This is a correctness condition, not an optimisation. Each caller builds
    its own client; if each also built its own pool, leasing would stop
    meaning anything — five of them would independently decide a server is
    free and fire five parallel requests at it. The same mistake the
    per-instance json_format flag once made.

    The wrapper is per caller (it carries that caller's on_retry for
    logging), but the LLMPool behind them is shared.
    """
    global _SHARED_POOL
    # One server — the old path to the letter, including the caller's
    # FACTORY. Building a client here by hand is wrong: the calling module
    # may patch LLMClient under its own name in tests, and bypassing that
    # patch would turn a stub into a real network call.
    if not cfg.get("api", "pool", fallback="").strip():
        c = factory() if factory else _llm.LLMClient.from_config(cfg)
        c.on_retry = on_retry
        return c
    clients, names = clients_from_config(cfg)
    # PARITY-2: see build_client — retry logs must be visible in pool mode
    # too.
    for c in clients:
        c.on_retry = on_retry
    with _SHARED_LOCK:
        if _SHARED_POOL is None or _SHARED_POOL.size != len(clients):
            _SHARED_POOL = LLMPool(clients, names)
    return PooledClient(_SHARED_POOL,
                        max_failover=cfg.getint("api", "max_failover", fallback=1),
                        on_retry=on_retry)


def reset_shared_pool():
    """For tests: the next shared_client() rebuilds the pool."""
    global _SHARED_POOL
    _SHARED_POOL = None


# ── parallel execution of independent calls ──────────────────────────────

def run_parallel(tasks: list, workers: int, on_error=None) -> list:
    """
    Run independent tasks (callables with no arguments).

    workers<=1 → runs sequentially, IN THE SAME ORDER and without threading.
    That matters more than it looks: with one server the behaviour must stay
    byte-for-byte as before, or runs with and without a pool cannot be
    compared.

    One task raising does not cancel the rest: the tasks are independent by
    definition, so one failure is no reason to lose the others. The result is
    a list of the same length, with None where a task failed.
    """
    if workers <= 1 or len(tasks) <= 1:
        out = []
        for t in tasks:
            try:
                out.append(t())
            except Exception as e:
                if on_error:
                    on_error(e)
                out.append(None)
        return out

    from concurrent.futures import ThreadPoolExecutor
    results: list = [None] * len(tasks)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(t): i for i, t in enumerate(tasks)}
        for fut, i in futs.items():
            try:
                results[i] = fut.result()
            except Exception as e:
                if on_error:
                    on_error(e)
    return results
