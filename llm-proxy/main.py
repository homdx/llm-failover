"""
Local HTTP proxy that forwards OpenAI-compatible requests from Kilo to
NVIDIA's build.nvidia.com / NIM API (https://integrate.api.nvidia.com/v1),
with full request/response logging in JSON Lines format so you can see
exactly what Kilo is sending.

Settings live in config.toml next to this file.

Run directly:
    python3 main.py

Run tunneled through an external SOCKS5 proxy (this is what fixes the 451
geo-block — leave proxy.use_socks5 = false in config.toml when doing this,
the two mechanisms shouldn't both be active):
    proxychains4 python3 main.py

Point Kilo's baseUrl at:
    http://<server.host>:<server.port>/v1   (see config.toml — currently 127.0.0.1:8081)
"""

import http.client
import http.server
import json
import os
import random
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone

try:
    import tomllib  # stdlib on Python 3.11+
except ModuleNotFoundError:
    import tomli as tomllib  # pip install tomli   (Python < 3.11)

import key_store

CONFIG_PATH = "config.toml"

with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

HOST = CONFIG["server"]["host"]
PORT = CONFIG["server"]["port"]
NVIDIA_HOST = CONFIG["upstream"]["host"]
# "https" for the original build.nvidia.com target; set to "http" in
# config.toml when pointing upstream.host at a plain local server such as
# Ollama (e.g. localhost:11434), which doesn't speak TLS.
UPSTREAM_SCHEME = CONFIG["upstream"].get("scheme", "https")

# --- transparent per-key upstream routing via the api_manager sqlite store ---
# Optional [keys] section in config.toml; .get(...) so an existing
# config.toml without it still works and this is a no-op. When the
# client's api key (Authorization: Bearer ..., api-key, or x-api-key
# header) matches a row in that database, its host/scheme are used for
# this request INSTEAD of upstream.host/scheme above. No DB file, no
# match, or a DB that fails to open all fall back to upstream.host/scheme
# unchanged — the feature is entirely additive.
_KEYS_CFG = CONFIG.get("keys", {})
KEY_STORE_DB_PATH = key_store.resolve_db_path(_KEYS_CFG.get("db_path"))
UPSTREAM_TIMEOUT_SEC = CONFIG["upstream"].get("timeout_sec", 120)
# Hard ceiling on the TOTAL time one response body may take to arrive
# when the upstream keeps trickling data (even a byte at a time)
# without ever going idle for longer than timeout_sec on its own. A
# single fp.read() stalling past timeout_sec is normally treated as a
# dead connection and fails the request — but if the upstream has
# already sent at least one byte for this response, that one stall is
# tolerated and the read is retried in place instead, as long as the
# total time spent on this response hasn't yet passed
# stream_max_wait_sec. This does NOT remove the timeout: an upstream
# that trickles forever still eventually gets cut off here, it's just
# given more patience than a single timeout_sec. A completely silent
# upstream (never sends a byte) is unaffected and still fails at the
# plain timeout_sec as before.
STREAM_MAX_WAIT_SEC = CONFIG["upstream"].get("stream_max_wait_sec", 900)

# --- keeping the CLIENT's connection alive while the upstream is silent ---
# Nothing above helps with the other side of the wire: while we wait for
# the upstream's first byte (a cold model can take minutes just to load),
# the client sees a socket that has never produced a single byte, and its
# own idle timeout fires — undici's default headersTimeout is 300s, which
# is where the "5 minute" cutoff comes from regardless of how patient this
# proxy is willing to be.
#
# When heartbeat_after_sec > 0 and the incoming request asked for a
# stream, this many seconds of upstream silence make the proxy commit to a
# 200 text/event-stream response early and start dripping SSE comment
# lines (": keepalive") until the real response arrives. Comments are
# ignorable by spec, so the eventual real chunks are unaffected — but the
# STATUS CODE is now fixed at 200 before the upstream's is known. Silent
# retries still work (a comment line is not an answer); what changes is
# the give-up path: a masked 429 can no longer be sent as a status, so it
# is delivered as an error chunk inside the stream instead.
#
# 0 = off, and the proxy behaves exactly as it did before this existed.
# Set it comfortably below the client's own timeout but above a normal
# error round-trip (240 is a good default for a 300s client), so ordinary
# 4xx/5xx responses still come back as real status codes.
HEARTBEAT_AFTER_SEC = max(0.0, float(CONFIG["upstream"].get("heartbeat_after_sec", 0)))
HEARTBEAT_INTERVAL_SEC = max(1.0, float(CONFIG["upstream"].get("heartbeat_interval_sec", 20)))

LOG_ENABLED = CONFIG["logging"]["enabled"]
LOG_DIR = CONFIG["logging"]["log_dir"]
LOG_BODY_LIMIT = CONFIG["logging"].get("body_limit_bytes", 200_000)

# Silent retry: on these upstream status codes, hold the client connection
# open, wait, and re-send the SAME request — Kilo never sees the failed
# attempt(s), only the eventual outcome. .get(...) so an existing
# config.toml without a [retry] section still works (retries just off).
_RETRY_CFG = CONFIG.get("retry", {})
RETRY_ENABLED = bool(_RETRY_CFG.get("enabled", False))
RETRY_STATUS_CODES = set(_RETRY_CFG.get("status_codes", [400, 425]))
RETRY_MAX_ATTEMPTS = max(1, int(_RETRY_CFG.get("max_attempts", 3)))  # total tries, incl. the first
RETRY_PAUSE_SECONDS = max(0.0, float(_RETRY_CFG.get("pause_seconds", 15)))  # fallback when upstream gives no hint
RETRY_MAX_PAUSE_SECONDS = max(0.0, float(_RETRY_CFG.get("max_pause_seconds", 180)))

# Buffer-and-validate: catches a 200-status body that is truncated or has
# garbage spliced into it mid-stream (e.g. a stray "HTTP/1.1 502 Bad
# Gateway" landing inside an SSE data: line) — a failure a status-code
# check alone can't see. See the long comment in config.toml [retry].
VALIDATE_RESPONSE_BODY = bool(_RETRY_CFG.get("validate_response_body", False))
MAX_BUFFER_BYTES = int(_RETRY_CFG.get("max_buffer_bytes", 50_000_000))
REQUIRE_STREAM_DONE = bool(_RETRY_CFG.get("require_stream_done", True))
# A completions stream that stops before ANY chunk carries a finish_reason
# was cut off mid-generation, even though every line in it parsed as valid
# JSON and the status was 200. This is what a response killed during a long
# reasoning phase actually looks like on the wire, and it is detectable
# whether or not the upstream bothers to send a closing "data: [DONE]".
REQUIRE_STREAM_FINISH_REASON = bool(_RETRY_CFG.get("require_stream_finish_reason", True))
# A truncated body is not a rate limit, so it does not deserve the
# rate-limit-sized wait: pausing pause_seconds before each of max_attempts
# retries here would hold the client for minutes over a hiccup that clears
# in one retry.
BODY_RETRY_PAUSE_SECONDS = max(0.0, float(_RETRY_CFG.get("body_retry_pause_seconds", 2.0)))

# --- keeping parallel requests from fighting each other over a rate limit ---
# Each in-flight request runs in its own thread with its own retry loop, so
# without these three the proxy answers an upstream "slow down" by sending
# MORE traffic: N requests x max_attempts tries, all inside the window the
# upstream just asked everyone to sit out.
#
# max_concurrent_upstream caps how many requests may be talking to the
# upstream at once (0 = unlimited, the old behaviour).
MAX_CONCURRENT_UPSTREAM = max(0, int(_RETRY_CFG.get("max_concurrent_upstream", 0)))
# Hard ceiling on how long one request may spend retrying before it gives
# up and answers the client (0 = unlimited, the old behaviour). Retrying
# past the client's own timeout is wasted effort: it has stopped listening.
MAX_TOTAL_RETRY_SECONDS = max(0.0, float(_RETRY_CFG.get("max_total_retry_seconds", 0)))
# Random spread applied when threads come off a shared cooldown, so they
# don't all fire at the same instant and re-trigger the limit together.
COOLDOWN_JITTER_SECONDS = max(0.0, float(_RETRY_CFG.get("cooldown_jitter_seconds", 0.5)))
# Growth factor for the FALLBACK wait only (pause_seconds, used when the
# upstream gave no Retry-After and no body hint) across attempts within one
# request. An explicit upstream number is always honoured as-is. Capped by
# max_pause_seconds. 1.0 = flat, the old behaviour.
RETRY_BACKOFF_FACTOR = max(1.0, float(_RETRY_CFG.get("backoff_factor", 2.0)))

# Optional in-process SOCKS5 tunneling. Leave this off if you're already
# wrapping the process with `proxychains4 python3 main.py`.
if CONFIG["proxy"]["use_socks5"]:
    import socket
    try:
        import socks  # pip install PySocks
    except ModuleNotFoundError as e:
        raise SystemExit(
            "config.toml has [proxy] use_socks5 = true, but the 'PySocks' package "
            "isn't installed, so this process exits immediately and nothing ever "
            "binds to the port \u2014 which is exactly what makes Kilo report "
            "'Cannot connect to API: Unable to connect.'\n"
            "Fix with:  pip install PySocks\n"
            "or set use_socks5 = false in config.toml if you launch with "
            "`proxychains4 python3 main.py` instead."
        ) from e

    socks.set_default_proxy(
        socks.SOCKS5,
        CONFIG["proxy"]["socks5_host"],
        CONFIG["proxy"]["socks5_port"],
        rdns=True,
    )
    socket.socket = socks.socksocket

STRIP_REQUEST_HEADERS = {"host", "content-length", "accept-encoding", "connection"}
STRIP_RESPONSE_HEADERS = {"transfer-encoding", "content-encoding", "connection"}
REDACT_HEADERS = {"authorization", "api-key", "x-api-key"}

_log_lock = threading.Lock()

# Raised internally when the client (Kilo) has already hung up — nothing
# more should be written to self.wfile once this fires. Kilo hanging up
# mid-retry (its own client-side timeout, most likely once total retry
# time gets long) previously crashed the request thread TWICE: once when
# the real write failed, then again when the exception handler tried to
# write an error response to the same dead socket.
_CLIENT_GONE_ERRORS = (BrokenPipeError, ConnectionResetError, ConnectionAbortedError)


class _ClientGone(Exception):
    pass


class _ResponseAborted(Exception):
    """The response had already started (status line + some body sent to
    Kilo) when the upstream connection failed. There is nothing safe left
    to do — retrying would mean sending a second status line into the
    middle of a body Kilo is already reading, which is the exact
    corruption this proxy exists to prevent. So: stop, log it plainly,
    and never treat it as retryable. Deliberately not _ClientGone: the
    client didn't necessarily go anywhere, the upstream did.
    """
    pass


class _InvalidUpstreamBody(Exception):
    """Raised when a 200-status upstream body fails structural validation
    (truncated stream, corrupted/unparsable SSE chunk, or the connection
    dying part-way through the body). Carries what we saw so it can be
    logged and, if the retry budget allows, silently retried exactly like
    an HTTPError — Kilo never sees the broken bytes.
    """

    def __init__(self, reason, raw=b"", headers=None, status=None, retryable=True):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw
        self.headers = headers or []
        self.status = status
        self.retryable = retryable


# Errors that mean "the upstream body stopped arriving part-way through".
# http.client raises IncompleteRead when a declared Content-Length or a
# chunked body ends early; a stalled generation trips the socket timeout;
# a dropped TLS connection surfaces as SSLError/OSError. All of them are
# the same event as far as this proxy is concerned: a truncated response.
# TimeoutError and the ConnectionError family are OSError subclasses, so
# OSError is the catch-all backstop rather than a separate case.
_UPSTREAM_READ_ERRORS = (
    http.client.IncompleteRead,
    http.client.HTTPException,
    ssl.SSLError,
    TimeoutError,
    OSError,
)

# Console-only running totals — never written to logs/*.jsonl.
_stats_lock = threading.Lock()
_stats = {
    "requests": 0, "in_tokens": 0, "out_tokens": 0, "in_bytes": 0, "out_bytes": 0,
    "ok": 0,
    # A request that hit >=1 retryable failure and STILL came back OK is a
    # 429 (or whatever the upstream really said) the client never had to
    # see — every one of those is a "saved". masked_429 is the opposite:
    # retry budget ran out and the client got sent home with a 429 anyway.
    "saved_429": 0,
    "masked_429": 0,
    "retries_absorbed": 0,
}


def _human_size(n: int) -> str:
    """Bytes as a human-readable size, auto-picking B / KB / MB / GB."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024.0:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}GB"


def _now_str() -> str:
    """Local wall-clock timestamp for console lines, e.g. '2026-09-12 15:04:22.123'.

    This is deliberately the machine's LOCAL time (not the UTC used in the
    logs/*.jsonl "ts" field) so it matches whatever clock the person
    watching the terminal is looking at.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _fmt_duration(seconds: float) -> str:
    """Seconds as a short human string: '0.42s', '12.30s', '2m14s', '1h02m03s'.

    Under a minute this is exactly the old "%.2fs" precision. Past a
    minute — realistic once a few retries and cooldowns stack up — it
    switches to a Xm/Xh breakdown so nobody has to do the division in
    their head while reading a live log.
    """
    if seconds < 60:
        return f"{seconds:.2f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m{s:04.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h{int(m):02d}m{int(s):02d}s"


class _ReqTrace:
    """Per-request scratchpad shared by the retry loop, the heartbeat
    thread, and the final _print_stats call — this is what lets the
    one-line console summary answer not just "how long did this take"
    but "where did that time actually go" and "did silent retry / the
    streaming keepalive pay for themselves on this request".

    Thread-safety note: the retry loop is the only writer of attempts /
    failures / upstream_sec / wait_sec. The heartbeat thread only ever
    writes heartbeat_committed / heartbeat_committed_at, and only before
    _stop_heartbeat() has been joined — after that join, the main thread
    reads it. Simple attribute assignments are already atomic under the
    GIL, and nothing here is read by one thread while genuinely still
    being written by another, so no extra lock is needed on top of that.
    """
    __slots__ = (
        "t0", "attempts", "failures", "upstream_sec", "wait_sec",
        "is_stream", "heartbeat_enabled", "heartbeat_committed",
        "heartbeat_committed_at", "bridged",
    )

    def __init__(self, t0, is_stream, heartbeat_enabled):
        self.t0 = t0
        self.attempts = 0
        self.failures = []  # short labels, e.g. ["502", "malformed body"]
        self.upstream_sec = 0.0
        self.wait_sec = 0.0
        self.is_stream = is_stream
        self.heartbeat_enabled = heartbeat_enabled
        self.heartbeat_committed = False
        self.heartbeat_committed_at = None
        self.bridged = False  # heartbeat fired AND a real response then arrived

    def elapsed(self) -> float:
        return time.monotonic() - self.t0

    def proxy_sec(self) -> float:
        return max(0.0, self.elapsed() - self.upstream_sec - self.wait_sec)

    def failures_note(self) -> str:
        return f" failures={self.failures}" if self.failures else ""

    def heartbeat_label(self) -> str:
        if not self.heartbeat_enabled:
            return "off"
        if not self.is_stream:
            return "n/a"
        if not self.heartbeat_committed:
            return "idle"  # eligible, but upstream answered before it fired
        return "bridged" if self.bridged else "stalled"


def _extract_usage(parsed_body):
    """Pull the OpenAI-style `usage` object out of a parsed response body.

    Handles both a plain JSON response (`{"usage": {...}, ...}`) and a
    streamed one (`{"stream_chunks": [...]}`) — Kilo sends
    `stream_options.include_usage: true`, so usage shows up on the final
    SSE chunk rather than at the top level.
    """
    if not isinstance(parsed_body, dict):
        return None
    usage = parsed_body.get("usage")
    if isinstance(usage, dict):
        return usage
    for chunk in reversed(parsed_body.get("stream_chunks") or []):
        if isinstance(chunk, dict) and isinstance(chunk.get("usage"), dict):
            return chunk["usage"]
    return None


def _print_stats(req_id, usage, trace: "_ReqTrace", in_bytes, out_bytes, outcome="OK"):
    """Print the ONE line that answers, for a request that just reached a
    FINAL outcome (succeeded, or the retry loop gave up on it): when did
    this happen, how long did the whole thing take, and where did that
    time actually go.

    This is the single call site for that line — every terminal branch of
    _proxy() (success, retries exhausted with a masked 429, a non-retryable
    error forwarded as-is, a network exception, or giving up before ever
    reaching the upstream because of a cooldown/concurrency limit) calls
    this exactly once, so the timestamp and elapsed time are never printed
    twice for the same request. `outcome` is "OK", "429" when the client
    ends up being told to back off (regardless of what the upstream really
    said underneath), or the real HTTP status/exception name otherwise.
    """
    in_tok  = usage.get("prompt_tokens")     if usage else None
    out_tok = usage.get("completion_tokens") if usage else None
    tot_tok = usage.get("total_tokens")      if usage else None

    elapsed = trace.elapsed()
    attempts = trace.attempts
    retries_this_req = max(0, attempts - 1)

    with _stats_lock:
        _stats["requests"] += 1
        if in_tok is not None:
            _stats["in_tokens"] += in_tok
        if out_tok is not None:
            _stats["out_tokens"] += out_tok
        _stats["in_bytes"]  += in_bytes
        _stats["out_bytes"] += out_bytes
        _stats["retries_absorbed"] += retries_this_req
        if outcome == "OK":
            _stats["ok"] += 1
            if trace.failures:
                # Hit at least one retryable failure and still came back
                # clean — that's a 429 (or whatever it really was) the
                # client never had to see.
                _stats["saved_429"] += 1
        elif outcome == "429":
            _stats["masked_429"] += 1
        n       = _stats["requests"]
        sum_in  = _stats["in_tokens"]
        sum_out = _stats["out_tokens"]
        sum_in_b  = _stats["in_bytes"]
        sum_out_b = _stats["out_bytes"]
        snap = dict(_stats)

    def _fmt(v):
        return str(v) if v is not None else "?"

    time_note = (
        f"time={_fmt_duration(elapsed)} "
        f"(upstream={_fmt_duration(trace.upstream_sec)} "
        f"wait={_fmt_duration(trace.wait_sec)} "
        f"proxy={_fmt_duration(trace.proxy_sec())})"
    )
    print(
        f"[{req_id}] {_now_str()} outcome={outcome} attempt={attempts} "
        f"{time_note} heartbeat={trace.heartbeat_label()} "
        f"IN={_fmt(in_tok)} OUT={_fmt(out_tok)} TOTAL={_fmt(tot_tok)}"
        f"{trace.failures_note()}",
        flush=True,
    )
    print(
        f"    \u03a3 requests={n} ok={snap['ok']} masked_429={snap['masked_429']} "
        f"saved_429={snap['saved_429']} retries_absorbed={snap['retries_absorbed']}",
        flush=True,
    )
    print(
        f"    \u03a3 tokens: IN={sum_in} OUT={sum_out} TOTAL={sum_in + sum_out}  "
        f"size: IN={_human_size(sum_in_b)} OUT={_human_size(sum_out_b)} "
        f"TOTAL={_human_size(sum_in_b + sum_out_b)}",
        flush=True,
    )


def _get_ci(headers, name, default=""):
    """Case-insensitive lookup in a list/dict of (key, value) header pairs."""
    items = headers.items() if isinstance(headers, dict) else headers
    for k, v in items:
        if k.lower() == name.lower():
            return v
    return default


# Same two hint patterns llm_stream.py's _parse_retry_after uses — some
# gateways (Groq-style "Please try again in 820ms", Gemini-style "Please
# retry in 57.06s.") put an exact wait in the error body text even when
# they don't set a Retry-After header.
_RETRY_AFTER_MS_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*ms", re.IGNORECASE)
_RETRY_AFTER_S_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*s(?:econds?)?\b", re.IGNORECASE)


def _resolve_pause(headers, err_body: bytes, default_seconds: float):
    """How long to wait before retrying, and where that number came from.

    Checked in order, same priority as llm_stream.py's _parse_retry_after:
      1. The Retry-After response header, if present and a plain number
         of seconds (an HTTP-date form isn't parsed — falls through).
      2. A "try again in Xms" / "retry in Xs" hint in the error body text.
      3. The configured default (config.toml [retry] pause_seconds).

    Returns (seconds, source) where source is a short label for the
    console line — never written to the JSONL log.
    """
    retry_after = _get_ci(headers, "Retry-After")
    if retry_after:
        try:
            return max(0.1, float(retry_after)), "Retry-After header"
        except ValueError:
            pass
    detail = err_body.decode("utf-8", errors="replace")[:500] if err_body else ""
    m = _RETRY_AFTER_MS_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1)) / 1000.0), "body hint"
        except ValueError:
            pass
    m = _RETRY_AFTER_S_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1))), "body hint"
        except ValueError:
            pass
    return default_seconds, "config default"


# ---------------------------------------------------------------------------
# Shared cooldown gate.
#
# A rate limit belongs to the account, not to one request, so the wait it
# asks for has to be observed by every thread — otherwise thread A sleeps
# out its 5 seconds while threads B..E keep hammering the same limit, and
# the limit never gets a chance to clear. One deadline, set by whoever hit
# the limit most recently, respected by everyone before their next attempt.
# ---------------------------------------------------------------------------
# A rate limit belongs to the account behind a given upstream HOST, not to
# one request or to the proxy as a whole \u2014 so with per-key routing (see
# key_store.py) each host gets its OWN gate and its OWN semaphore, sized
# from the same config.toml [retry] values, rather than one pool shared
# across every host the proxy happens to talk to. Pooling them would let a
# cooldown on host A block traffic to unrelated host B, and would let
# host B's requests eat into slots meant for host A's max_concurrent_upstream.
class _HostState:
    __slots__ = ("gate_lock", "gate_until", "probe_lock", "slots")

    def __init__(self):
        self.gate_lock = threading.Lock()
        self.gate_until = 0.0  # time.monotonic() deadline; no upstream call before this
        # Held by the one request allowed to test this host's upstream when
        # its cooldown lifts. Without it, every queued thread fires the
        # instant the deadline passes and the limit is hit N times over to
        # learn one fact.
        self.probe_lock = threading.Lock()
        self.slots = (
            threading.BoundedSemaphore(MAX_CONCURRENT_UPSTREAM)
            if MAX_CONCURRENT_UPSTREAM > 0 else None
        )


_host_states_lock = threading.Lock()
_host_states = {}  # host -> _HostState, created lazily as hosts are seen


def _get_host_state(host):
    with _host_states_lock:
        state = _host_states.get(host)
        if state is None:
            state = _HostState()
            _host_states[host] = state
        return state


def _gate_penalize(state, seconds: float):
    """Stand every thread bound for this host down for `seconds`.

    Never shortens an existing wait.
    """
    if seconds <= 0:
        return
    deadline = time.monotonic() + seconds
    with state.gate_lock:
        if deadline > state.gate_until:
            state.gate_until = deadline


def _gate_remaining(state) -> float:
    with state.gate_lock:
        return max(0.0, state.gate_until - time.monotonic())


def _gate_clear(state):
    """A request got through, so this host's limit isn't in force any more.

    If that read was wrong the very next 429 re-arms the gate, which costs
    one wasted request \u2014 cheaper than making everyone sit out a cooldown
    that has already expired.
    """
    with state.gate_lock:
        state.gate_until = 0.0


def _await_turn(state, req_id, deadline=None):
    """Wait for this host's shared cooldown, then for permission to go upstream.

    Returns (ok, probing). ok is False when waiting any longer would blow
    this request's time budget \u2014 the caller answers the client instead.

    While no cooldown is in force this returns immediately and requests
    run in parallel as before. Coming OUT of a cooldown is the part that
    matters: exactly one request (the probe) is let through to find out
    whether the limit has lifted. If it gets another 429 the gate is
    re-armed and everyone keeps waiting, so the upstream sees one request
    per window instead of one per waiting thread. Whoever holds the probe
    must hand it back with _end_turn().
    """
    waited = False
    while True:
        remaining = _gate_remaining(state)
        if remaining > 0:
            if deadline is not None and time.monotonic() + remaining > deadline:
                return False, False
            if not waited:
                print(
                    f".. [{req_id}] shared cooldown active ({remaining:.1f}s left) "
                    f"\u2014 holding before contacting upstream",
                    flush=True,
                )
                waited = True
            time.sleep(min(remaining, 0.5))
            continue

        if not waited:
            return True, False

        if state.probe_lock.acquire(blocking=False):
            if _gate_remaining(state) > 0:
                # Re-armed by another thread between the two checks.
                state.probe_lock.release()
                continue
            if COOLDOWN_JITTER_SECONDS:
                time.sleep(random.uniform(0.0, COOLDOWN_JITTER_SECONDS))
            return True, True

        # Someone else is probing. Wait for their verdict rather than
        # duplicating it.
        if deadline is not None and time.monotonic() > deadline:
            return False, False
        time.sleep(0.05)


def _end_turn(state, probing: bool):
    if probing:
        state.probe_lock.release()


def _acquire_slot(state, req_id, deadline=None) -> bool:
    """Take one of this host's max_concurrent_upstream slots. True if held."""
    if state.slots is None:
        return True
    timeout = None
    if deadline is not None:
        timeout = max(0.0, deadline - time.monotonic())
        if timeout <= 0:
            return False
    if state.slots.acquire(timeout=timeout):
        return True
    print(
        f".. [{req_id}] no upstream slot free within this request's time "
        f"budget \u2014 not queueing any longer",
        flush=True,
    )
    return False


def _release_slot(state):
    if state.slots is not None:
        state.slots.release()


def _redact_headers(headers: dict) -> dict:
    return {
        k: ("***redacted***" if k.lower() in REDACT_HEADERS else v)
        for k, v in headers.items()
    }


def _resolve_upstream(req_id, headers):
    """Return (host, scheme) for this request.

    Transparent by construction: with no [keys].db_path configured and no
    file at the default location, this returns the config.toml default on
    every request without even trying to touch sqlite. Only when that file
    exists do we bother parsing an api key out of the request and looking
    it up — a miss there (key missing, or not in the db) falls back to the
    same default and prints why, so a bad/rotated key doesn't fail silently.
    """
    if not key_store.db_available(KEY_STORE_DB_PATH):
        return NVIDIA_HOST, UPSTREAM_SCHEME
    api_key = key_store.extract_api_key(headers)
    entry = key_store.lookup(api_key, KEY_STORE_DB_PATH) if api_key else None
    if entry is not None:
        return entry.host, entry.scheme
    print(
        f".. [{req_id}] api key not found in {KEY_STORE_DB_PATH} — "
        f"falling back to default upstream {UPSTREAM_SCHEME}://{NVIDIA_HOST} "
        f"from config.toml",
        flush=True,
    )
    return NVIDIA_HOST, UPSTREAM_SCHEME


def _parse_body(raw: bytes, content_type: str):
    """Best-effort turn raw request/response bytes into loggable JSON."""
    if not raw:
        return None
    if "text/event-stream" in content_type:
        chunks = []
        for line in raw.decode(errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                chunks.append("[DONE]")
                continue
            try:
                chunks.append(json.loads(payload))
            except json.JSONDecodeError:
                chunks.append(payload)
        return {"stream_chunks": chunks}
    try:
        return json.loads(raw.decode())
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": raw[:2000].decode(errors="replace")}


# Statuses that carry no body at all by definition — an empty body for
# one of these is a complete response, not a truncated one.
_BODILESS_STATUSES = {204, 205, 304}


def _validate_sse(raw: bytes):
    """Structural check for a fully-buffered text/event-stream body."""
    saw_done = False
    saw_choices = False
    saw_finish_reason = False

    for line in raw.decode("utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            saw_done = True
            continue
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            return False, "malformed SSE chunk (invalid JSON in a data: line)"
        choices = chunk.get("choices") if isinstance(chunk, dict) else None
        if isinstance(choices, list) and choices:
            saw_choices = True
            for choice in choices:
                if isinstance(choice, dict) and choice.get("finish_reason"):
                    saw_finish_reason = True

    if REQUIRE_STREAM_DONE and not saw_done:
        return False, "stream ended without a closing data: [DONE]"
    # The truncation that a per-line JSON check cannot see: the stream
    # stopped on a clean chunk boundary, so every line parsed, but no
    # chunk ever reported why generation ended. That is a response cut
    # off mid-flight (reasoning tokens ran past the gateway's patience,
    # upstream hung up, ...) rather than a finished one. Only applied to
    # completion-shaped streams, and only when there is no [DONE] either
    # — an upstream that closed the stream properly is taken at its word.
    if (REQUIRE_STREAM_FINISH_REASON and saw_choices
            and not saw_finish_reason and not saw_done):
        return False, "stream ended mid-generation (no finish_reason in any chunk)"
    return True, ""


def _validate_body(content_type: str, raw: bytes, status: int = 200,
                   method: str = "POST", declared_length=None):
    """Check a fully-buffered response body is structurally intact.

    Returns (True, "") if it looks complete, else (False, reason).
    Mirrors _parse_body's parsing so "valid" here means "Kilo's own
    OpenAI-compatible client will be able to parse this too".

    status/method/declared_length exist so a legitimately empty body
    (204/304, a HEAD, an explicit Content-Length: 0) is not mistaken for
    a truncated one and retried into a fabricated 429.
    """
    ct = (content_type or "").lower()

    if status in _BODILESS_STATUSES or method.upper() == "HEAD" or declared_length == 0:
        return True, ""
    if not raw:
        return False, "empty body"
    if "text/event-stream" in ct:
        return _validate_sse(raw)
    if ct and "json" not in ct:
        # Not a shape this proxy knows how to check (text/plain health
        # endpoints and the like) — don't invent a failure for it.
        return True, ""
    try:
        json.loads(raw.decode())
        return True, ""
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "invalid JSON body"


def _tolerant_read(fp, amt, stream_start, has_data, req_id):
    """One fp.read(amt), retried in place across benign idle stalls.

    A read timing out (TimeoutError — what a stalled socket raises,
    whether it surfaces as the http.client/urllib built-in or its
    socket.timeout alias) only keeps this loop spinning when both:
      - has_data() is true, i.e. the upstream has already sent at
        least one byte for this response, so it's known to be alive
        rather than dead; and
      - less than STREAM_MAX_WAIT_SEC has passed since stream_start.

    Otherwise the TimeoutError is re-raised and handled exactly as
    before this feature existed. See STREAM_MAX_WAIT_SEC above for why
    this is a widened budget, not a removed timeout.
    """
    while True:
        try:
            return fp.read(amt)
        except TimeoutError:
            if not has_data() or time.monotonic() - stream_start >= STREAM_MAX_WAIT_SEC:
                raise
            print(
                f".. [{req_id}] upstream idle past timeout_sec ({UPSTREAM_TIMEOUT_SEC}s) "
                f"but has sent data before for this response \u2014 still inside the "
                f"stream_max_wait_sec budget ({STREAM_MAX_WAIT_SEC}s), continuing to wait",
                flush=True,
            )


def _wants_stream(body: bytes) -> bool:
    """True if this request body asks for a streamed (SSE) response.

    Heartbeats are only valid for streams: on a plain JSON response there
    is nowhere to hide a keepalive, and committing to 200 early would be
    unrecoverable. A body that isn't JSON, or doesn't say stream, answers
    False and takes the untouched code path.
    """
    if not body:
        return False
    try:
        return bool(json.loads(body).get("stream"))
    except Exception:
        return False


def write_log(entry: dict):
    if not LOG_ENABLED:
        return
    os.makedirs(LOG_DIR, exist_ok=True)
    entry.setdefault("ts", datetime.now(timezone.utc).isoformat())
    fname = os.path.join(LOG_DIR, f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl")
    line = json.dumps(entry, ensure_ascii=False)
    with _log_lock:
        with open(fname, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")


class ProxyHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    # True once a status line has gone out to the client for the request
    # in flight. Nothing may write a second response after that: doing so
    # splices a raw "HTTP/1.1 502 Bad Gateway" into the middle of the body
    # the client is still reading, which is exactly the corrupted-SSE
    # symptom this proxy is supposed to protect against.
    _response_started = False

    # True once a heartbeat thread has committed this request to an early
    # 200 text/event-stream. From then on the status code is spent, and
    # anything that would have been a status (an error, a masked 429) has
    # to go out as a chunk inside the stream instead.
    _heartbeat_active = False

    def log_message(self, format, *args):
        pass  # console output is now just the per-request stats line below

    def _proxy(self, method):
        req_id = uuid.uuid4().hex[:8]
        self._response_started = False
        self._heartbeat_active = False
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""

        write_log({
            "type": "request",
            "id": req_id,
            "method": method,
            "path": self.path,
            "headers": _redact_headers(dict(self.headers.items())),
            "body": _parse_body(body, self.headers.get("Content-Type", "")),
        })

        upstream_host, upstream_scheme = _resolve_upstream(req_id, self.headers)
        host_state = _get_host_state(upstream_host)

        upstream_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in STRIP_REQUEST_HEADERS
        }
        upstream_headers["Host"] = upstream_host
        upstream_headers["Accept-Encoding"] = "identity"

        url = f"{upstream_scheme}://{upstream_host}{self.path}"

        _t0 = time.monotonic()
        in_bytes = len(body)
        is_stream = _wants_stream(body)
        heartbeat_ok = HEARTBEAT_AFTER_SEC > 0 and is_stream
        max_attempts = RETRY_MAX_ATTEMPTS if RETRY_ENABLED else 1
        deadline = _t0 + MAX_TOTAL_RETRY_SECONDS if MAX_TOTAL_RETRY_SECONDS > 0 else None
        last_pause = RETRY_PAUSE_SECONDS

        # One scratchpad per request, so the final _print_stats line can
        # report not just "how long" but "attempt N, this much of it spent
        # actually waiting on a shared limit vs. talking to the upstream".
        trace = _ReqTrace(_t0, is_stream, HEARTBEAT_AFTER_SEC > 0)
        print(
            f">> [{req_id}] {_now_str()} {method} {self.path} "
            f"in={_human_size(in_bytes)} stream={'yes' if is_stream else 'no'} "
            f"(up to {max_attempts} attempt{'s' if max_attempts != 1 else ''})",
            flush=True,
        )

        for attempt in range(1, max_attempts + 1):
            trace.attempts = attempt
            # Another thread may have just been told to slow down. Honour
            # that before adding one more request to the pile.
            t_wait = time.monotonic()
            turn_ok, probing = _await_turn(host_state, req_id, deadline)
            trace.wait_sec += time.monotonic() - t_wait
            if not turn_ok:
                _print_stats(req_id, None, trace, in_bytes, 0, outcome="429")
                self._write_masked_retry(
                    req_id, "shared cooldown longer than this request's budget",
                    retry_after=_gate_remaining(host_state), trace=trace,
                    host_state=host_state,
                )
                return
            t_wait = time.monotonic()
            slot_ok = _acquire_slot(host_state, req_id, deadline)
            trace.wait_sec += time.monotonic() - t_wait
            if not slot_ok:
                _end_turn(host_state, probing)
                _print_stats(req_id, None, trace, in_bytes, 0, outcome="429")
                self._write_masked_retry(
                    req_id, "upstream concurrency limit", retry_after=last_pause,
                    trace=trace, host_state=host_state)
                return

            # Fresh Request object per attempt — cheap, and avoids any risk
            # of urllib mutating headers on a reused one across retries.
            req = urllib.request.Request(url, data=body or None, headers=upstream_headers, method=method)
            stop_heartbeat = self._start_heartbeat(req_id, trace) if heartbeat_ok else None
            t_upstream = time.monotonic()
            try:
                try:
                    with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_SEC) as resp:
                        # Stop the drip BEFORE the first real byte goes
                        # out: both write to the same wfile, and the join
                        # inside _stop_heartbeat is what keeps a keepalive
                        # comment from being spliced into a chunk.
                        if stop_heartbeat:
                            stop_heartbeat()
                            if trace.heartbeat_committed:
                                # The keepalive had already taken over the
                                # response by the time a real one showed up.
                                trace.bridged = True
                        if VALIDATE_RESPONSE_BODY:
                            usage, out_bytes = self._relay_buffered(
                                req_id, resp.status, resp.getheaders(), resp, method)
                        else:
                            usage, out_bytes = self._relay(req_id, resp.status, resp.getheaders(), resp)
                finally:
                    trace.upstream_sec += time.monotonic() - t_upstream
                    stop_heartbeat and stop_heartbeat()
                # Something got through, so whatever limit was in force has
                # lifted — let anyone still queued behind the gate move.
                _gate_clear(host_state)
                _print_stats(req_id, usage, trace, in_bytes, out_bytes, outcome="OK")
                return
            except _ClientGone:
                return
            except _ResponseAborted:
                return
            except _InvalidUpstreamBody as e:
                trace.failures.append(f"{e.status} body")
                retrying = RETRY_ENABLED and e.retryable and attempt < max_attempts
                if retrying and deadline is not None and \
                        time.monotonic() + BODY_RETRY_PAUSE_SECONDS > deadline:
                    retrying = False

                write_log({
                    "type": "response",
                    "id": req_id,
                    "status": e.status,
                    "attempt": attempt,
                    "retrying": retrying,
                    "validation_error": e.reason,
                    "headers": dict(e.headers) if e.headers else {},
                    "body_snippet": e.raw[:2000].decode(errors="replace") if e.raw else None,
                })

                if retrying:
                    print(
                        f".. [{req_id}] upstream {e.status} body failed validation "
                        f"({e.reason}) \u2014 retry {attempt}/{max_attempts - 1} in "
                        f"{BODY_RETRY_PAUSE_SECONDS:.1f}s (config default; not sent to Kilo)",
                        flush=True,
                    )
                    t_wait = time.monotonic()
                    time.sleep(BODY_RETRY_PAUSE_SECONDS)
                    trace.wait_sec += time.monotonic() - t_wait
                    continue

                _print_stats(req_id, None, trace, in_bytes, len(e.raw), outcome="429")

                # Retry budget used up on a body that never came back
                # intact. Forwarding it as-is is exactly the bug we're
                # fixing (Kilo choking on a spliced-in "HTTP/1.1 502 Bad
                # Gateway" mid-JSON-string) — mask it the same predictable
                # way as an exhausted status-code retry instead.
                self._write_masked_retry(req_id, f"malformed body: {e.reason}", trace=trace,
                                          host_state=host_state)
                return
            except urllib.error.HTTPError as e:
                trace.failures.append(str(e.code))
                err_body = e.read()
                headers = list(e.headers.items()) if e.headers else []
                code_is_retryable = e.code in RETRY_STATUS_CODES
                retrying = RETRY_ENABLED and code_is_retryable and attempt < max_attempts

                pause = pause_source = None
                if retrying:
                    # The fallback grows across attempts; an explicit
                    # Retry-After / body hint is always honoured as given.
                    fallback = min(
                        RETRY_PAUSE_SECONDS * (RETRY_BACKOFF_FACTOR ** (attempt - 1)),
                        RETRY_MAX_PAUSE_SECONDS,
                    )
                    pause, pause_source = _resolve_pause(headers, err_body, fallback)
                    last_pause = pause
                    if deadline is not None and time.monotonic() + pause > deadline:
                        print(
                            f".. [{req_id}] {pause:.1f}s wait would push this "
                            f"request past its {MAX_TOTAL_RETRY_SECONDS:.0f}s "
                            f"budget \u2014 not retrying",
                            flush=True,
                        )
                        retrying = False
                    elif pause > RETRY_MAX_PAUSE_SECONDS:
                        # A wait this long looks like a daily/monthly quota
                        # reset, not a transient hiccup — waiting would just
                        # hold Kilo's connection open for no good reason.
                        print(
                            f".. [{req_id}] upstream {e.code} asked for "
                            f"{pause:.0f}s ({pause_source}), over the "
                            f"{RETRY_MAX_PAUSE_SECONDS:.0f}s cap \u2014 not retrying",
                            flush=True,
                        )
                        retrying = False

                write_log({
                    "type": "response",
                    "id": req_id,
                    "status": e.code,
                    "attempt": attempt,
                    "retrying": retrying,
                    "headers": dict(headers),
                    "body": _parse_body(err_body, _get_ci(headers, "Content-Type")),
                })

                if retrying:
                    print(
                        f".. [{req_id}] upstream {e.code} \u2014 retry "
                        f"{attempt}/{max_attempts - 1} in {pause:.1f}s "
                        f"({pause_source}) (not sent to Kilo)",
                        flush=True,
                    )
                    # Publish the wait instead of sleeping it privately, so
                    # the other in-flight requests sit it out as well —
                    # _gate_wait at the top of the loop does the sleeping.
                    _gate_penalize(host_state, pause)
                    continue

                will_mask_as_429 = RETRY_ENABLED and code_is_retryable
                _print_stats(
                    req_id, None, trace, in_bytes, len(err_body),
                    outcome="429" if will_mask_as_429 else str(e.code),
                )

                if will_mask_as_429:
                    # Giving up here means the client will come back, so the
                    # number we hand it has to be one we'd honour ourselves:
                    # arm the gate for the same span rather than letting the
                    # next request walk straight back into the limit.
                    _gate_penalize(host_state, last_pause)
                    # Silent-retry budget is used up (or the resolved wait
                    # looked like a quota reset). Kilo never sees the real
                    # status or body here — upstreams shove this family of
                    # errors into wildly different shapes (bare 5xx bodies,
                    # {"name":"UnknownError","data":{...}}, etc.) and Kilo
                    # copes with a plain 429 + Retry-After far better than
                    # with whichever shape happened to come back last. The
                    # real status/body is still in the log above.
                    self._write_masked_retry(req_id, e.code, retry_after=last_pause,
                                              trace=trace, host_state=host_state)
                else:
                    self._write_final(req_id, e.code, headers, err_body, "error_forward")
                return
            except Exception as e:
                print(f"!! [{req_id}] {type(e).__name__}: {e}", flush=True)
                write_log({"type": "error", "id": req_id, "error": f"{type(e).__name__}: {e}"})
                _print_stats(req_id, None, trace, in_bytes, 0, outcome="502")
                self._write_final(req_id, 502, [], str(e).encode(), "network_error")
                return
            finally:
                # Runs on the retry `continue` too, so a waiting request
                # gets the slot instead of it being pinned for the pause.
                _release_slot(host_state)
                _end_turn(host_state, probing)

    def _start_heartbeat(self, req_id, trace: "_ReqTrace" = None):
        """Begin an SSE keepalive drip; returns a stop() callable.

        The thread sleeps HEARTBEAT_AFTER_SEC first and does nothing at
        all if the upstream answers within that window — the common case,
        including every error round-trip, which is why an ordinary 429 or
        502 still reaches the client as a real status code.
        """
        done = threading.Event()

        def _loop():
            if done.wait(HEARTBEAT_AFTER_SEC):
                return
            try:
                self._start_response(200, [
                    ("Content-Type", "text/event-stream; charset=utf-8"),
                    ("Cache-Control", "no-cache"),
                ])
                self._heartbeat_active = True
                if trace is not None:
                    trace.heartbeat_committed = True
                    trace.heartbeat_committed_at = time.monotonic()
                print(
                    f".. [{req_id}] upstream silent for {HEARTBEAT_AFTER_SEC:.0f}s \u2014 "
                    f"holding the client with SSE keepalives every "
                    f"{HEARTBEAT_INTERVAL_SEC:.0f}s (status now committed as 200)",
                    flush=True,
                )
                while not done.wait(HEARTBEAT_INTERVAL_SEC):
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except _CLIENT_GONE_ERRORS:
                # Client gave up anyway. The main thread will find out on
                # its own next write; nothing useful to do from here.
                print(f"xx [{req_id}] client disconnected during keepalive", flush=True)
            except Exception as e:
                print(f"!! [{req_id}] keepalive stopped: {type(e).__name__}: {e}", flush=True)

        thread = threading.Thread(target=_loop, daemon=True)
        thread.start()

        def stop():
            done.set()
            thread.join()

        return stop

    def _write_stream_error(self, req_id, status, body_bytes, stage, retry_after=None):
        """Deliver an error INSIDE an already-committed SSE stream.

        Only reachable with a heartbeat running. The client is mid-stream
        on a 200, so the failure travels as a data chunk carrying the same
        OpenAI-shaped error object the status path would have sent, then
        [DONE] so the stream closes cleanly instead of looking truncated.
        """
        try:
            payload = json.loads(body_bytes)
        except Exception:
            payload = {"error": {"message": body_bytes.decode(errors="replace"),
                                 "type": "upstream_error", "code": status}}
        try:
            if retry_after is not None:
                payload.setdefault("error", {})["retry_after"] = int(float(retry_after))
        except (TypeError, ValueError):
            pass  # Retry-After can also be an HTTP-date; not worth translating
        print(
            f"xx [{req_id}] {status} arrived after the keepalive committed a 200 "
            f"\u2014 sending it as an in-stream error chunk instead",
            flush=True,
        )
        write_log({"type": "response_stream_error", "id": req_id,
                   "stage": stage, "would_have_sent": status, "body": payload})
        try:
            self.wfile.write(b"data: " + json.dumps(payload).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
            return True
        except _CLIENT_GONE_ERRORS:
            print(f"xx [{req_id}] client disconnected before the error chunk landed", flush=True)
            write_log({"type": "client_disconnected", "id": req_id, "stage": stage})
            return False

    def _write_masked_retry(self, req_id, original, retry_after=None, trace: "_ReqTrace" = None,
                             host_state=None):
        """Send Kilo one predictable shape after we give up retrying.

        `original` is whatever actually went wrong — an upstream status
        code (500, 502, 529, ...) or a body-validation reason. It is
        logged, but never forwarded. Kilo only ever sees a plain 429 with
        a Retry-After header, so it backs off the same way regardless of
        which unfamiliar error shape the upstream used underneath.

        The number sent is the largest of: what the upstream last asked
        for, what the shared cooldown still has to run, and the configured
        pause_seconds. Sending a flat config value instead would send the
        client back before the limit it just hit has cleared.

        This is the ONLY call site through which a masked-429 response
        leaves the proxy — an exhausted retry loop, a cooldown longer than
        the request's own budget, or no free upstream slot in time all end
        up here. The console line deliberately does NOT repeat the date
        and elapsed time: _print_stats (called right alongside this, at
        every one of those call sites) is the single place that prints
        those, so a person tailing the log never sees the same timestamp
        twice for one request.
        """
        if retry_after is None:
            retry_after = RETRY_PAUSE_SECONDS
        gate_remaining = _gate_remaining(host_state) if host_state is not None else 0.0
        retry_after = int(round(max(1.0, retry_after, gate_remaining, RETRY_PAUSE_SECONDS)))
        body = json.dumps({
            "error": {
                "message": f"Upstream temporarily unavailable (was {original}). Retry after {retry_after}s.",
                "type": "rate_limit_error",
                "code": 429,
            }
        }).encode()
        headers = [
            ("Content-Type", "application/json"),
            ("Retry-After", str(retry_after)),
        ]
        print(
            f".. [{req_id}] giving up on upstream {original} \u2014 "
            f"masking to Kilo as 429 + Retry-After: {retry_after}s",
            flush=True,
        )
        write_log({
            "type": "response",
            "id": req_id,
            "status": 429,
            "masked_from": original,
            "attempts": trace.attempts if trace is not None else None,
            "elapsed_sec": round(trace.elapsed(), 3) if trace is not None else None,
            "headers": dict(headers),
            "body": json.loads(body),
        })
        self._write_final(req_id, 429, headers, body, "retry_exhausted_masked")

    def _start_response(self, status, headers):
        """Write the status line and headers, and latch _response_started.

        A no-op once a response is already on the wire — with a heartbeat
        running, the 200 and its headers went out minutes ago, and the
        relay that follows must append to that stream rather than start a
        second one. Without a heartbeat this is never reached twice.
        """
        if self._response_started:
            return
        self.send_response(status)
        for key, val in headers:
            if key.lower() not in STRIP_RESPONSE_HEADERS:
                self.send_header(key, val)
        self.send_header("Connection", "close")
        self.end_headers()
        self._response_started = True

    def _write_final(self, req_id, status, headers, body_bytes, stage):
        """Send a final (non-streamed) response. Returns True on success.

        If Kilo already hung up (its own client-side timeout, typically
        after we spent a while retrying), logs it quietly and returns
        False instead of letting a second, unhandled exception crash the
        request thread on top of whatever originally went wrong.
        """
        if self._response_started and self._heartbeat_active:
            # Committed to a 200 by the keepalive, but nothing real has
            # been sent yet — the error can still be delivered, just as a
            # chunk rather than as a status line.
            return self._write_stream_error(
                req_id, status, body_bytes, stage,
                retry_after=_get_ci(headers, "Retry-After"))
        if self._response_started:
            # A response is already on the wire (usually a streamed 200
            # that upstream then truncated). Appending another status
            # line here would corrupt the body the client is mid-way
            # through parsing; all that's left to do is stop.
            print(
                f"xx [{req_id}] upstream failed after the response had already "
                f"started \u2014 not writing a {status} on top of it",
                flush=True,
            )
            write_log({"type": "response_aborted", "id": req_id,
                       "stage": stage, "would_have_sent": status})
            return False
        try:
            self._start_response(status, headers)
            self.wfile.write(body_bytes)
            return True
        except _CLIENT_GONE_ERRORS:
            print(f"xx [{req_id}] client disconnected before the response could be sent", flush=True)
            write_log({"type": "client_disconnected", "id": req_id, "stage": stage})
            return False

    def _relay(self, req_id, status, headers, fp):
        try:
            self._start_response(status, headers)

            content_type = _get_ci(headers, "Content-Type")
            captured = bytearray()
            total_len = 0
            stream_start = time.monotonic()

            # Stream to the client in real time; separately buffer (up to
            # the configured cap) a copy for the log entry written after
            # the loop. total_len tracks the FULL size regardless of cap.
            while True:
                chunk = _tolerant_read(
                    fp, 4096, stream_start, lambda: total_len > 0, req_id)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                total_len += len(chunk)
                if len(captured) < LOG_BODY_LIMIT:
                    captured.extend(chunk)
        except _CLIENT_GONE_ERRORS:
            # Kilo hung up mid-response (its own client-side timeout is
            # the usual cause after a long retry sequence). Drain what
            # NVIDIA already sent so its connection closes cleanly, log
            # it once, and stop — nothing more can be written to Kilo.
            try:
                fp.read()
            except Exception:
                pass
            print(f"xx [{req_id}] client disconnected mid-response", flush=True)
            write_log({"type": "client_disconnected", "id": req_id, "stage": "relay"})
            raise _ClientGone from None

        parsed_body = _parse_body(bytes(captured), content_type)
        write_log({
            "type": "response",
            "id": req_id,
            "status": status,
            "headers": dict(headers),
            "body": parsed_body,
        })
        return _extract_usage(parsed_body), total_len

    def _relay_buffered(self, req_id, status, headers, fp, method="POST"):
        """Buffer the whole upstream body, validate it, THEN send it on.

        Unlike _relay (which streams live, chunk by chunk, so Kilo can
        already have half a broken response by the time anything looks
        wrong), nothing is written to Kilo here until the full body has
        been read and passed _validate_body. That's what makes a genuine
        silent retry possible for a body that's corrupted or truncated
        mid-stream: on failure this raises _InvalidUpstreamBody instead
        of touching self.wfile, so the caller can just try again.
        """
        content_type = _get_ci(headers, "Content-Type")
        try:
            declared_length = int(_get_ci(headers, "Content-Length"))
        except (TypeError, ValueError):
            declared_length = None

        captured = bytearray()
        over_limit = False
        stream_start = time.monotonic()
        try:
            while True:
                chunk = _tolerant_read(
                    fp, 65536, stream_start, lambda: len(captured) > 0, req_id)
                if not chunk:
                    break
                captured.extend(chunk)
                if len(captured) > MAX_BUFFER_BYTES:
                    # Too big to hold for validation. Deliver it anyway —
                    # an over-sized response is still a response, and
                    # turning it into a fabricated 429 (or retrying it,
                    # which just produces another over-sized response)
                    # helps nobody. Handled OUTSIDE this try block (see
                    # below): once _passthrough starts writing to the
                    # client, a failure in it is no longer a "nothing sent
                    # yet, safe to retry" case like the one this except
                    # clause exists for.
                    over_limit = True
                    break
        except _UPSTREAM_READ_ERRORS as e:
            # The body stopped arriving part-way through: a long reasoning
            # phase outran the gateway, upstream hung up, the socket timed
            # out. Nothing has been written to the client yet, so this is
            # retryable exactly like a body that arrived complete but
            # unparsable — which is the whole point of buffering. Without
            # this branch the exception escapes to the generic handler,
            # which sends a bare 502 and does not retry at all.
            captured.extend(getattr(e, "partial", b"") or b"")
            raise _InvalidUpstreamBody(
                f"upstream connection failed mid-body ({type(e).__name__}: {e})",
                raw=bytes(captured), headers=headers, status=status,
            ) from None

        if over_limit:
            return self._passthrough(
                req_id, status, headers, bytes(captured), fp, content_type)

        raw = bytes(captured)
        ok, reason = _validate_body(content_type, raw, status=status,
                                    method=method, declared_length=declared_length)
        if not ok:
            raise _InvalidUpstreamBody(reason, raw=raw, headers=headers, status=status)

        try:
            self._start_response(status, headers)
            self.wfile.write(raw)
            self.wfile.flush()
        except _CLIENT_GONE_ERRORS:
            print(f"xx [{req_id}] client disconnected before the validated response could be sent", flush=True)
            write_log({"type": "client_disconnected", "id": req_id, "stage": "relay_buffered"})
            raise _ClientGone from None

        parsed_body = _parse_body(raw[:LOG_BODY_LIMIT], content_type)
        write_log({
            "type": "response",
            "id": req_id,
            "status": status,
            "headers": dict(headers),
            "body": parsed_body,
        })
        return _extract_usage(parsed_body), len(raw)

    def _passthrough(self, req_id, status, headers, prefix, fp, content_type):
        """Stream an over-sized response through without validating it.

        Reached only when the body grew past max_buffer_bytes mid-read:
        the bytes already buffered go out first, then the rest is relayed
        live. Validation is impossible from here, so this deliberately
        gives up on retrying rather than on delivering.
        """
        print(
            f".. [{req_id}] response passed max_buffer_bytes ({MAX_BUFFER_BYTES}) "
            f"\u2014 streaming it through unvalidated",
            flush=True,
        )
        total_len = len(prefix)
        try:
            self._start_response(status, headers)
            self.wfile.write(prefix)
            self.wfile.flush()
            while True:
                chunk = fp.read(65536)
                if not chunk:
                    break
                self.wfile.write(chunk)
                self.wfile.flush()
                total_len += len(chunk)
        except _CLIENT_GONE_ERRORS:
            # NOTE: this same error type can also arise from fp.read() —
            # a reset upstream connection and a reset client connection
            # raise identical exception types, and both can occur inside
            # this one try block (it mixes reads from upstream with
            # writes to the client). There's no reliable way to tell them
            # apart from the exception alone, so this is reported as
            # "connection lost mid-response" rather than confidently
            # blamed on the client.
            print(f"xx [{req_id}] connection lost mid-response (passthrough)", flush=True)
            write_log({"type": "response_aborted", "id": req_id, "stage": "passthrough"})
            raise _ResponseAborted from None
        except _UPSTREAM_READ_ERRORS as e:
            # Half-delivered and unrecoverable: _response_started is set,
            # so this must never be recast as a retryable _InvalidUpstreamBody
            # (that would retry the request and call _start_response a
            # second time on top of the bytes already sent — a corrupted
            # response, which is exactly the bug this proxy exists to fix).
            print(
                f"xx [{req_id}] upstream failed mid-passthrough after the "
                f"response had already started ({type(e).__name__}: {e})",
                flush=True,
            )
            write_log({"type": "response_aborted", "id": req_id, "stage": "passthrough",
                       "error": f"{type(e).__name__}: {e}"})
            raise _ResponseAborted from None

        parsed_body = _parse_body(prefix[:LOG_BODY_LIMIT], content_type)
        write_log({
            "type": "response",
            "id": req_id,
            "status": status,
            "validated": False,
            "headers": dict(headers),
            "body": parsed_body,
        })
        return _extract_usage(parsed_body), total_len

    def do_POST(self):
        self._proxy("POST")

    def do_GET(self):
        self._proxy("GET")

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Connection", "close")
        self.end_headers()


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer((HOST, PORT), ProxyHandler)
#    print(f"NVIDIA proxy listening on http://{HOST}:{PORT}  (logs -> {LOG_DIR}/*.jsonl)", flush=True)
    print(
        f"Proxy listening on http://{HOST}:{PORT} -> upstream "
        f"{UPSTREAM_SCHEME}://{NVIDIA_HOST}  (logs -> {LOG_DIR}/*.jsonl)",
        flush=True,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
