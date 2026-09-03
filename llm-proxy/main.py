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

import http.server
import json
import os
import re
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

CONFIG_PATH = "config.toml"

with open(CONFIG_PATH, "rb") as f:
    CONFIG = tomllib.load(f)

HOST = CONFIG["server"]["host"]
PORT = CONFIG["server"]["port"]
NVIDIA_HOST = CONFIG["upstream"]["host"]
UPSTREAM_TIMEOUT_SEC = CONFIG["upstream"].get("timeout_sec", 120)

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


class _InvalidUpstreamBody(Exception):
    """Raised when a 200-status upstream body fails structural validation
    (truncated stream, corrupted/unparsable SSE chunk, or a response
    bigger than max_buffer_bytes). Carries what we saw so it can be
    logged and, if the retry budget allows, silently retried exactly like
    an HTTPError — Kilo never sees the broken bytes.
    """

    def __init__(self, reason, raw=b"", headers=None, status=None):
        super().__init__(reason)
        self.reason = reason
        self.raw = raw
        self.headers = headers or []
        self.status = status

# Console-only running totals — never written to logs/*.jsonl.
_stats_lock = threading.Lock()
_stats = {"requests": 0, "in_tokens": 0, "out_tokens": 0, "in_bytes": 0, "out_bytes": 0}


def _human_size(n: int) -> str:
    """Bytes as a human-readable size, auto-picking B / KB / MB / GB."""
    size = float(n)
    for unit in ("B", "KB", "MB"):
        if size < 1024.0:
            return f"{size:.0f}{unit}" if unit == "B" else f"{size:.2f}{unit}"
        size /= 1024.0
    return f"{size:.2f}GB"


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


def _print_stats(req_id, usage, elapsed, in_bytes, out_bytes):
    in_tok  = usage.get("prompt_tokens")     if usage else None
    out_tok = usage.get("completion_tokens") if usage else None
    tot_tok = usage.get("total_tokens")      if usage else None

    with _stats_lock:
        _stats["requests"] += 1
        if in_tok is not None:
            _stats["in_tokens"] += in_tok
        if out_tok is not None:
            _stats["out_tokens"] += out_tok
        _stats["in_bytes"]  += in_bytes
        _stats["out_bytes"] += out_bytes
        n       = _stats["requests"]
        sum_in  = _stats["in_tokens"]
        sum_out = _stats["out_tokens"]
        sum_in_b  = _stats["in_bytes"]
        sum_out_b = _stats["out_bytes"]

    def _fmt(v):
        return str(v) if v is not None else "?"

    print(
        f"[{req_id}] IN={_fmt(in_tok)} OUT={_fmt(out_tok)} "
        f"TOTAL={_fmt(tot_tok)} time={elapsed:.2f}s",
        flush=True,
    )
    print(
        f"    \u03a3 requests={n} IN={sum_in} OUT={sum_out} TOTAL={sum_in + sum_out}",
        flush=True,
    )
    print(
        f"    \u03a3 size: IN={_human_size(sum_in_b)}  OUT={_human_size(sum_out_b)}  "
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


def _redact_headers(headers: dict) -> dict:
    return {
        k: ("***redacted***" if k.lower() in REDACT_HEADERS else v)
        for k, v in headers.items()
    }


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


def _validate_body(content_type: str, raw: bytes):
    """Check a fully-buffered response body is structurally intact.

    Returns (True, "") if it looks complete, else (False, reason).
    Mirrors _parse_body's parsing so "valid" here means "Kilo's own
    OpenAI-compatible client will be able to parse this too".
    """
    if not raw:
        return False, "empty body"
    if "text/event-stream" in (content_type or ""):
        saw_done = False
        for line in raw.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                saw_done = True
                continue
            try:
                json.loads(payload)
            except json.JSONDecodeError:
                return False, "malformed SSE chunk (invalid JSON in a data: line)"
        if REQUIRE_STREAM_DONE and not saw_done:
            return False, "stream ended without a closing data: [DONE]"
        return True, ""
    try:
        json.loads(raw.decode())
        return True, ""
    except (UnicodeDecodeError, json.JSONDecodeError):
        return False, "invalid JSON body"


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

    def log_message(self, format, *args):
        pass  # console output is now just the per-request stats line below

    def _proxy(self, method):
        req_id = uuid.uuid4().hex[:8]
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

        upstream_headers = {
            k: v for k, v in self.headers.items()
            if k.lower() not in STRIP_REQUEST_HEADERS
        }
        upstream_headers["Host"] = NVIDIA_HOST
        upstream_headers["Accept-Encoding"] = "identity"

        url = f"https://{NVIDIA_HOST}{self.path}"

        _t0 = time.monotonic()
        in_bytes = len(body)
        max_attempts = RETRY_MAX_ATTEMPTS if RETRY_ENABLED else 1

        for attempt in range(1, max_attempts + 1):
            # Fresh Request object per attempt — cheap, and avoids any risk
            # of urllib mutating headers on a reused one across retries.
            req = urllib.request.Request(url, data=body or None, headers=upstream_headers, method=method)
            try:
                with urllib.request.urlopen(req, timeout=UPSTREAM_TIMEOUT_SEC) as resp:
                    if VALIDATE_RESPONSE_BODY:
                        usage, out_bytes = self._relay_buffered(req_id, resp.status, resp.getheaders(), resp)
                    else:
                        usage, out_bytes = self._relay(req_id, resp.status, resp.getheaders(), resp)
                _print_stats(req_id, usage, time.monotonic() - _t0, in_bytes, out_bytes)
                return
            except _ClientGone:
                return
            except _InvalidUpstreamBody as e:
                retrying = attempt < max_attempts

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
                        f"{RETRY_PAUSE_SECONDS:.1f}s (config default; not sent to Kilo)",
                        flush=True,
                    )
                    time.sleep(RETRY_PAUSE_SECONDS)
                    continue

                _print_stats(req_id, None, time.monotonic() - _t0, in_bytes, len(e.raw))

                # Retry budget used up on a body that never came back
                # intact. Forwarding it as-is is exactly the bug we're
                # fixing (Kilo choking on a spliced-in "HTTP/1.1 502 Bad
                # Gateway" mid-JSON-string) — mask it the same predictable
                # way as an exhausted status-code retry instead.
                self._write_masked_retry(req_id, f"malformed body: {e.reason}")
                return
            except urllib.error.HTTPError as e:
                err_body = e.read()
                headers = list(e.headers.items()) if e.headers else []
                code_is_retryable = e.code in RETRY_STATUS_CODES
                retrying = RETRY_ENABLED and code_is_retryable and attempt < max_attempts

                pause = pause_source = None
                if retrying:
                    pause, pause_source = _resolve_pause(headers, err_body, RETRY_PAUSE_SECONDS)
                    if pause > RETRY_MAX_PAUSE_SECONDS:
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
                    time.sleep(pause)
                    continue

                _print_stats(req_id, None, time.monotonic() - _t0, in_bytes, len(err_body))

                if RETRY_ENABLED and code_is_retryable:
                    # Silent-retry budget is used up (or the resolved wait
                    # looked like a quota reset). Kilo never sees the real
                    # status or body here — upstreams shove this family of
                    # errors into wildly different shapes (bare 5xx bodies,
                    # {"name":"UnknownError","data":{...}}, etc.) and Kilo
                    # copes with a plain 429 + Retry-After far better than
                    # with whichever shape happened to come back last. The
                    # real status/body is still in the log above.
                    self._write_masked_retry(req_id, e.code)
                else:
                    self._write_final(req_id, e.code, headers, err_body, "error_forward")
                return
            except Exception as e:
                print(f"!! [{req_id}] {type(e).__name__}: {e}", flush=True)
                write_log({"type": "error", "id": req_id, "error": f"{type(e).__name__}: {e}"})
                _print_stats(req_id, None, time.monotonic() - _t0, in_bytes, 0)
                self._write_final(req_id, 502, [], str(e).encode(), "network_error")
                return

    def _write_masked_retry(self, req_id, original_code):
        """Send Kilo one predictable shape after we give up retrying.

        original_code is whatever the upstream actually sent (500, 502,
        529, ...) — logged, but never forwarded. Kilo only ever sees a
        plain 429 with a Retry-After header pulled straight from config
        ([retry] pause_seconds), so it backs off the same way regardless
        of which unfamiliar error shape the upstream used underneath.
        """
        retry_after = int(round(RETRY_PAUSE_SECONDS))
        body = json.dumps({
            "error": {
                "message": f"Upstream temporarily unavailable (was {original_code}). Retry after {retry_after}s.",
                "type": "rate_limit_error",
                "code": 429,
            }
        }).encode()
        headers = [
            ("Content-Type", "application/json"),
            ("Retry-After", str(retry_after)),
        ]
        print(
            f".. [{req_id}] retries exhausted on upstream {original_code} \u2014 "
            f"masking to Kilo as 429 + Retry-After: {retry_after}s",
            flush=True,
        )
        write_log({
            "type": "response",
            "id": req_id,
            "status": 429,
            "masked_from": original_code,
            "headers": dict(headers),
            "body": json.loads(body),
        })
        self._write_final(req_id, 429, headers, body, "retry_exhausted_masked")

    def _write_final(self, req_id, status, headers, body_bytes, stage):
        """Send a final (non-streamed) response. Returns True on success.

        If Kilo already hung up (its own client-side timeout, typically
        after we spent a while retrying), logs it quietly and returns
        False instead of letting a second, unhandled exception crash the
        request thread on top of whatever originally went wrong.
        """
        try:
            self.send_response(status)
            for key, val in headers:
                if key.lower() not in STRIP_RESPONSE_HEADERS:
                    self.send_header(key, val)
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body_bytes)
            return True
        except _CLIENT_GONE_ERRORS:
            print(f"xx [{req_id}] client disconnected before the response could be sent", flush=True)
            write_log({"type": "client_disconnected", "id": req_id, "stage": stage})
            return False

    def _relay(self, req_id, status, headers, fp):
        try:
            self.send_response(status)
            for key, val in headers:
                if key.lower() not in STRIP_RESPONSE_HEADERS:
                    self.send_header(key, val)
            self.send_header("Connection", "close")
            self.end_headers()

            content_type = _get_ci(headers, "Content-Type")
            captured = bytearray()
            total_len = 0

            # Stream to the client in real time; separately buffer (up to
            # the configured cap) a copy for the log entry written after
            # the loop. total_len tracks the FULL size regardless of cap.
            while True:
                chunk = fp.read(4096)
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

    def _relay_buffered(self, req_id, status, headers, fp):
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
        captured = bytearray()
        over_limit = False

        while True:
            chunk = fp.read(65536)
            if not chunk:
                break
            if len(captured) + len(chunk) > MAX_BUFFER_BYTES:
                over_limit = True
                break
            captured.extend(chunk)

        if over_limit:
            raise _InvalidUpstreamBody(
                f"response exceeded max_buffer_bytes ({MAX_BUFFER_BYTES})",
                headers=headers, status=status,
            )

        raw = bytes(captured)
        ok, reason = _validate_body(content_type, raw)
        if not ok:
            raise _InvalidUpstreamBody(reason, raw=raw, headers=headers, status=status)

        try:
            self.send_response(status)
            for key, val in headers:
                if key.lower() not in STRIP_RESPONSE_HEADERS:
                    self.send_header(key, val)
            self.send_header("Connection", "close")
            self.end_headers()
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
    print(f"NVIDIA proxy listening on http://{HOST}:{PORT}  (logs -> {LOG_DIR}/*.jsonl)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
