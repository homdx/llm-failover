"""
Automated test suite for the Kilo <-> NVIDIA proxy script.

WHAT THIS TESTS
----------------
- The body-validation logic (_validate_body / _validate_sse): does it
  correctly detect a truncated or corrupted SSE stream, a truncated plain
  JSON body, and correctly leave alone bodies that are legitimately empty
  (204/304/HEAD) or not JSON-shaped at all?
- The retry loop end to end: a corrupted body gets silently retried and
  Kilo only ever sees the clean response; if every attempt is corrupted,
  Kilo gets exactly one clean, predictable 429 and never the garbage
  bytes; a retryable HTTP status (502) gets retried the same way; a
  non-retryable status (404) is forwarded immediately, unchanged.
- The oversized-response ("passthrough") path: a normal oversized
  response is delivered untouched, and -- this is the part that actually
  had bugs -- if the upstream connection drops PART-WAY THROUGH that
  passthrough, the proxy must never send a second status line on top of
  the bytes it already sent Kilo, and must not mislabel an upstream-side
  drop as "the client disconnected".
- Concurrency limiting (max_concurrent_upstream) and the retry-time
  budget (max_total_retry_seconds).
- The config.toml file itself: does it parse, and do the values look
  sane?
- The SOCKS5 startup check: if proxy.use_socks5 = true and PySocks isn't
  installed, does the process fail with a clear message instead of a
  bare traceback and a silently dead port?

HOW TO RUN
----------
1. Put this file in the SAME folder as your proxy script and config.toml.
   (It auto-detects the script under a few common names -- see
   PROXY_SCRIPT_CANDIDATES below. Rename that list if yours differs.)

2. From that folder, run:

       python3 -m unittest test_python_proxy -v

   or just:

       python3 test_python_proxy.py

   Either way you'll get a pass/fail line per test. No network access,
   no real SOCKS5 proxy, and no real NVIDIA/Kilo traffic is used --
   everything upstream is faked in-process, so this is safe to run
   anywhere, any time, including in CI.

3. A handful of tests spawn real background threads with small (tens of
   milliseconds) real sleeps to test concurrency limiting honestly; the
   whole suite should still finish in well under a second.

WHAT THIS DOES NOT DO
---------------------
It never touches your real config.toml's [proxy]/[upstream] settings for
the functional tests -- each test loads the proxy module fresh against
its own throwaway config in a temp directory, so nothing here depends on
whether SOCKS5 or the real NVIDIA host are reachable. The one exception
is RealConfigSanityTest, which reads your actual config.toml (next to
this file) just to check it parses and its values look sane -- it never
executes the proxy against it.
"""

import contextlib
import email.message
import importlib.util
import io
import os
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
from unittest import mock

THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# Adjust this if your script has a different name.
PROXY_SCRIPT_CANDIDATES = ["python_proxy2.py", "python_proxy.py", "main.py", "proxy.py"]


def _find_proxy_script():
    for name in PROXY_SCRIPT_CANDIDATES:
        path = os.path.join(THIS_DIR, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "Couldn't find the proxy script next to this test file. Looked for: "
        + ", ".join(PROXY_SCRIPT_CANDIDATES)
        + ". Edit PROXY_SCRIPT_CANDIDATES at the top of this file if yours "
          "has a different name."
    )


PROXY_SCRIPT_PATH = _find_proxy_script()

# Baseline config used by every functional test. Individual tests override
# just the keys they care about via load_proxy(**overrides). use_socks5 is
# always false here so tests never need PySocks or a real SOCKS5 proxy --
# that specific behaviour has its own dedicated test (SocksStartupTest).
BASE_CONFIG = """
[server]
host = "127.0.0.1"
port = 8080

[upstream]
host = "example.invalid"
timeout_sec = 5

[proxy]
use_socks5 = false
socks5_host = "127.0.0.1"
socks5_port = 1080

[logging]
enabled = false
log_dir = "logs"
body_limit_bytes = 200000

[retry]
enabled = {retry_enabled}
status_codes = [400, 425, 429, 500, 502, 529]
max_attempts = {max_attempts}
pause_seconds = {pause_seconds}
max_pause_seconds = {max_pause_seconds}
validate_response_body = {validate_response_body}
max_buffer_bytes = {max_buffer_bytes}
require_stream_done = {require_stream_done}
require_stream_finish_reason = {require_stream_finish_reason}
body_retry_pause_seconds = {body_retry_pause_seconds}
max_concurrent_upstream = {max_concurrent_upstream}
max_total_retry_seconds = {max_total_retry_seconds}
cooldown_jitter_seconds = {cooldown_jitter_seconds}
backoff_factor = {backoff_factor}
"""

DEFAULTS = dict(
    retry_enabled="true",
    max_attempts=3,
    pause_seconds=0.02,
    max_pause_seconds=180,
    validate_response_body="true",
    max_buffer_bytes=100_000,
    require_stream_done="true",
    require_stream_finish_reason="true",
    body_retry_pause_seconds=0.01,
    max_concurrent_upstream=0,
    max_total_retry_seconds=0,
    cooldown_jitter_seconds=0,
    backoff_factor=2.0,
)


def load_proxy(**overrides):
    """Import a fresh copy of the proxy module against a throwaway config.

    Fresh per call on purpose: module-level state (the shared cooldown
    gate, the concurrency semaphore, running stats) must not leak between
    tests, or tests become order-dependent and flaky.
    """
    cfg = dict(DEFAULTS)
    cfg.update(overrides)
    tmpdir = tempfile.mkdtemp(prefix="proxy_test_")
    with open(os.path.join(tmpdir, "config.toml"), "w") as f:
        f.write(BASE_CONFIG.format(**cfg))

    old_cwd = os.getcwd()
    os.chdir(tmpdir)
    try:
        spec = importlib.util.spec_from_file_location(
            f"proxy_under_test_{time.monotonic_ns()}", PROXY_SCRIPT_PATH
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        os.chdir(old_cwd)
        shutil.rmtree(tmpdir, ignore_errors=True)
    return mod


def make_handler(mod, method="POST", path="/v1/chat/completions", body=b"{}"):
    """A ProxyHandler instance with fake I/O, built without ever running
    BaseHTTPRequestHandler's real socket-based __init__."""
    handler = object.__new__(mod.ProxyHandler)
    handler.headers = {"Content-Length": str(len(body)), "Content-Type": "application/json"}
    handler.path = path
    handler.rfile = io.BytesIO(body)
    handler.wfile = io.BytesIO()
    handler._sent_statuses = []
    handler.send_response = lambda status, *a: handler._sent_statuses.append(status)
    handler.send_header = lambda k, v: None
    handler.end_headers = lambda: None
    return handler


def make_http_error(code, body=b'{"error":"boom"}', headers=None, content_type="application/json"):
    hdrs = email.message.Message()
    hdrs["Content-Type"] = content_type
    for k, v in (headers or {}).items():
        hdrs[k] = v
    return urllib.error.HTTPError(url="https://example.invalid/v1/x", code=code,
                                   msg="err", hdrs=hdrs, fp=io.BytesIO(body))


class FakeResp:
    """Minimal stand-in for the object urllib.request.urlopen() returns."""

    def __init__(self, body, status=200, content_type="text/event-stream", headers=None):
        self.status = status
        self._buf = io.BytesIO(body)
        self._headers = headers if headers is not None else [("Content-Type", content_type)]

    def getheaders(self):
        return self._headers

    def read(self, n):
        return self._buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class ScriptedUpstream:
    """Feeds a fixed sequence of behaviours to successive urlopen() calls.

    Each item is either a FakeResp instance, an exception instance (raised
    when its turn comes), or a zero-arg callable (invoked for its return
    value / raise). The last item repeats if urlopen is called more times
    than there are scripted items.
    """

    def __init__(self, items):
        self.items = items
        self.calls = 0

    def __call__(self, req, timeout=None):
        idx = min(self.calls, len(self.items) - 1)
        self.calls += 1
        item = self.items[idx]
        if isinstance(item, BaseException):
            raise item
        if callable(item) and not isinstance(item, FakeResp):
            return item()
        return item


# Real corruption pattern from the original bug report: a stray raw
# "HTTP/1.1 502 Bad Gateway" landed inside a data: line's JSON string,
# leaving it unterminated.
CORRUPTED_SSE = (
    b'data: {"id":"chatcmpl-edb57dd7","choices":[{"index":0,"delta":{"role":"assistant",'
    b'"reasoning_content":"."},"finish_reason":null,"logprobs":null}],"created":1788404109,'
    b'"model":"moonshotai/kimi-kHTTP/1.1 502 Bad Gateway. Error message: JSON Parse error: '
    b'Unterminated string\n\n'
)

CLEAN_SSE = (
    b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
    b'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],'
    b'"usage":{"prompt_tokens":5,"completion_tokens":2,"total_tokens":7}}\n\n'
    b'data: [DONE]\n\n'
)


# ---------------------------------------------------------------------------
# Unit tests: the pure validation / parsing helpers
# ---------------------------------------------------------------------------
class ValidateBodyTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_proxy()

    def test_corrupted_sse_from_the_original_bug_report_is_rejected(self):
        ok, reason = self.mod._validate_body("text/event-stream", CORRUPTED_SSE)
        self.assertFalse(ok)
        self.assertIn("malformed", reason)

    def test_clean_complete_sse_is_accepted(self):
        ok, reason = self.mod._validate_body("text/event-stream", CLEAN_SSE)
        self.assertTrue(ok, reason)

    def test_sse_missing_done_is_rejected_when_required(self):
        mod = load_proxy(require_stream_done="true", require_stream_finish_reason="false")
        truncated = b'data: {"id":"x","choices":[{"index":0,"delta":{"content":"hi"}}]}\n\n'
        ok, reason = mod._validate_body("text/event-stream", truncated)
        self.assertFalse(ok)
        self.assertIn("[DONE]", reason)

    def test_sse_missing_done_is_accepted_when_finish_reason_present_and_done_not_required(self):
        mod = load_proxy(require_stream_done="false", require_stream_finish_reason="true")
        # No [DONE], but the last chunk DID report finish_reason: complete.
        body = (
            b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}\n\n'
        )
        ok, reason = mod._validate_body("text/event-stream", body)
        self.assertTrue(ok, reason)

    def test_sse_cut_mid_generation_with_no_done_and_no_finish_reason_is_rejected(self):
        mod = load_proxy(require_stream_done="false", require_stream_finish_reason="true")
        body = b'data: {"choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":null}]}\n\n'
        ok, reason = mod._validate_body("text/event-stream", body)
        self.assertFalse(ok)
        self.assertIn("finish_reason", reason)

    def test_non_completion_stream_without_choices_is_not_flagged_by_finish_reason_check(self):
        mod = load_proxy(require_stream_done="false", require_stream_finish_reason="true")
        body = b'data: {"some_other_shape": true}\n\n'
        ok, reason = mod._validate_body("text/event-stream", body)
        self.assertTrue(ok, reason)

    def test_valid_plain_json_is_accepted(self):
        ok, reason = self.mod._validate_body("application/json", b'{"id":"x","object":"list"}')
        self.assertTrue(ok, reason)

    def test_truncated_plain_json_is_rejected(self):
        ok, reason = self.mod._validate_body("application/json", b'{"id":"x","object":')
        self.assertFalse(ok)

    def test_empty_body_is_rejected(self):
        ok, reason = self.mod._validate_body("application/json", b"")
        self.assertFalse(ok)

    def test_204_with_empty_body_is_not_a_failure(self):
        ok, reason = self.mod._validate_body("application/json", b"", status=204)
        self.assertTrue(ok, reason)

    def test_head_request_with_empty_body_is_not_a_failure(self):
        ok, reason = self.mod._validate_body("application/json", b"", method="HEAD")
        self.assertTrue(ok, reason)

    def test_declared_content_length_zero_is_not_a_failure(self):
        ok, reason = self.mod._validate_body("application/json", b"", declared_length=0)
        self.assertTrue(ok, reason)

    def test_non_json_content_type_is_left_alone(self):
        ok, reason = self.mod._validate_body("text/plain", b"pong")
        self.assertTrue(ok, reason)


class ResolvePauseTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_proxy()

    def test_retry_after_header_wins(self):
        seconds, source = self.mod._resolve_pause([("Retry-After", "7")], b"", default_seconds=99)
        self.assertEqual(seconds, 7.0)
        self.assertEqual(source, "Retry-After header")

    def test_body_hint_in_milliseconds(self):
        seconds, source = self.mod._resolve_pause([], b"Please try again in 820ms", default_seconds=99)
        self.assertAlmostEqual(seconds, 0.82)
        self.assertEqual(source, "body hint")

    def test_body_hint_in_seconds(self):
        seconds, source = self.mod._resolve_pause([], b"Please retry in 57.06s.", default_seconds=99)
        self.assertAlmostEqual(seconds, 57.06)

    def test_falls_back_to_default_when_no_hint(self):
        seconds, source = self.mod._resolve_pause([], b"nothing useful here", default_seconds=12)
        self.assertEqual(seconds, 12)
        self.assertEqual(source, "config default")


class ExtractUsageTest(unittest.TestCase):
    def setUp(self):
        self.mod = load_proxy()

    def test_top_level_usage(self):
        usage = self.mod._extract_usage({"usage": {"total_tokens": 7}})
        self.assertEqual(usage, {"total_tokens": 7})

    def test_usage_on_final_stream_chunk(self):
        parsed = {"stream_chunks": [{"choices": []}, {"usage": {"total_tokens": 7}}]}
        self.assertEqual(self.mod._extract_usage(parsed), {"total_tokens": 7})

    def test_no_usage_present(self):
        self.assertIsNone(self.mod._extract_usage({"stream_chunks": [{"choices": []}]}))


# ---------------------------------------------------------------------------
# Integration tests: the retry loop end to end
# ---------------------------------------------------------------------------
class RetryOnCorruptedBodyTest(unittest.TestCase):
    def test_corrupted_then_clean_is_retried_silently(self):
        mod = load_proxy()
        upstream = ScriptedUpstream([FakeResp(CORRUPTED_SSE), FakeResp(CLEAN_SSE)])
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 2, "should have retried exactly once")
        self.assertEqual(h._sent_statuses, [200])
        out = h.wfile.getvalue()
        self.assertNotIn(b"Unterminated", out)
        self.assertIn(b"[DONE]", out)

    def test_always_corrupted_exhausts_retries_and_masks_as_429(self):
        mod = load_proxy(max_attempts=3)
        upstream = ScriptedUpstream([FakeResp(CORRUPTED_SSE)])  # repeats forever
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 3)
        self.assertEqual(h._sent_statuses, [429])
        out = h.wfile.getvalue()
        self.assertNotIn(b"Unterminated", out)
        self.assertIn(b"malformed body", out)

    def test_validation_disabled_lets_corrupted_body_straight_through(self):
        # Documents the trade-off, doesn't imply it's a bug: with
        # validate_response_body off, the proxy is back to pure streaming
        # passthrough and cannot catch this class of corruption.
        mod = load_proxy(validate_response_body="false")
        mod.urllib.request.urlopen = ScriptedUpstream([FakeResp(CORRUPTED_SSE)])

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(h._sent_statuses, [200])
        self.assertIn(b"Unterminated", h.wfile.getvalue())


class HttpErrorRetryTest(unittest.TestCase):
    def test_retryable_status_is_retried_then_succeeds(self):
        mod = load_proxy()
        upstream = ScriptedUpstream([make_http_error(502), FakeResp(CLEAN_SSE)])
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 2)
        self.assertEqual(h._sent_statuses, [200])

    def test_retryable_status_exhausted_is_masked_as_429_not_forwarded_raw(self):
        mod = load_proxy(max_attempts=2)
        upstream = ScriptedUpstream([make_http_error(502, body=b"raw upstream 502 text")])
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 2)
        self.assertEqual(h._sent_statuses, [429])
        self.assertNotIn(b"raw upstream 502 text", h.wfile.getvalue())

    def test_non_retryable_status_is_forwarded_immediately_unchanged(self):
        mod = load_proxy()
        upstream = ScriptedUpstream([make_http_error(404, body=b'{"error":"not found"}')])
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 1, "a non-retryable status must not be retried")
        self.assertEqual(h._sent_statuses, [404])
        self.assertIn(b"not found", h.wfile.getvalue())

    def test_retry_after_header_is_honoured_in_masked_response(self):
        # max_attempts=2: attempt 1 reads the Retry-After header and
        # records it as last_pause; attempt 2 is the final one (no more
        # attempts left, so it doesn't re-resolve its own header) and
        # masks using that recorded value. _gate_penalize is neutered so
        # the test doesn't really sleep out a 42s cooldown between the
        # two attempts.
        mod = load_proxy(max_attempts=2)
        mod._gate_penalize = lambda seconds: None
        upstream = ScriptedUpstream([make_http_error(429, headers={"Retry-After": "42"})])
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(upstream.calls, 2)
        self.assertIn(b'"code": 429', h.wfile.getvalue())
        self.assertIn(b"42s", h.wfile.getvalue())


class OversizedResponseTest(unittest.TestCase):
    """The passthrough path, including the two bugs found in review:
    (A) an upstream-side drop during passthrough was mislabeled as "the
        client disconnected", and (B) a retryable upstream error type
        during passthrough could trigger a second, corrupting call to
        _start_response on top of the bytes already sent. Both of these
        tests fail against the pre-fix version of this file and pass
        against the fixed one.
    """

    def test_oversized_response_is_delivered_unvalidated_not_treated_as_a_failure(self):
        mod = load_proxy(max_buffer_bytes=50)
        big = CLEAN_SSE * 5  # comfortably over the 50-byte cap
        mod.urllib.request.urlopen = ScriptedUpstream([FakeResp(big)])

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(h._sent_statuses, [200])
        self.assertEqual(h.wfile.getvalue(), big)

    def test_upstream_timeout_mid_passthrough_never_sends_a_second_response(self):
        mod = load_proxy(max_buffer_bytes=50, max_attempts=3)

        class FlakyThenClean:
            def __init__(self):
                self.n = 0

            def __call__(self, req, timeout=None):
                self.n += 1
                if self.n == 1:
                    return _OversizedThenFailingResp(TimeoutError("stalled"))
                return FakeResp(CLEAN_SSE)

        upstream = FlakyThenClean()
        mod.urllib.request.urlopen = upstream

        h = make_handler(mod)
        h._proxy("POST")

        self.assertEqual(
            len(h._sent_statuses), 1,
            "a second status line was sent on top of an already-started "
            "response -- this is the corruption bug",
        )
        # Only the first (oversized) chunk's bytes should be present --
        # nothing from the "successful" second attempt.
        self.assertNotIn(b"[DONE]", h.wfile.getvalue())

    def test_upstream_reset_mid_passthrough_is_not_blamed_on_the_client(self):
        mod = load_proxy(max_buffer_bytes=50, max_attempts=3)
        mod.urllib.request.urlopen = ScriptedUpstream(
            [_OversizedThenFailingResp(ConnectionResetError("reset"))]
        )

        messages = []
        with mock.patch("builtins.print", side_effect=lambda *a, **k: messages.append(" ".join(map(str, a)))):
            h = make_handler(mod)
            h._proxy("POST")

        self.assertEqual(len(h._sent_statuses), 1)
        joined = "\n".join(messages)
        self.assertNotIn("client disconnected", joined)


class _OversizedThenFailingResp:
    """First read() returns an oversized chunk (forcing the passthrough
    path); the second read() raises the given exception, simulating the
    upstream connection failing mid-passthrough -- after the response
    header and first chunk have already gone out to the client."""

    def __init__(self, exc):
        self.status = 200
        self.exc = exc
        self.reads = 0

    def getheaders(self):
        return [("Content-Type", "text/event-stream")]

    def read(self, n):
        self.reads += 1
        if self.reads == 1:
            return b"data: " + b"x" * 200 + b"\n\n"
        raise self.exc

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
# Concurrency / retry-budget tests
# ---------------------------------------------------------------------------
class ConcurrencyLimitTest(unittest.TestCase):
    def test_max_concurrent_upstream_caps_simultaneous_calls(self):
        mod = load_proxy(max_concurrent_upstream=2, max_attempts=1, retry_enabled="false")

        in_flight = {"current": 0, "peak": 0}
        lock = threading.Lock()

        def fake_urlopen(req, timeout=None):
            with lock:
                in_flight["current"] += 1
                in_flight["peak"] = max(in_flight["peak"], in_flight["current"])
            time.sleep(0.05)
            with lock:
                in_flight["current"] -= 1
            return FakeResp(CLEAN_SSE)

        mod.urllib.request.urlopen = fake_urlopen

        threads = [threading.Thread(target=lambda: make_handler(mod)._proxy("POST")) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5)

        self.assertLessEqual(in_flight["peak"], 2, "more requests reached upstream at once than the configured cap")

    def test_max_total_retry_seconds_gives_up_instead_of_retrying_forever(self):
        # A huge fallback pause combined with a tiny budget should make
        # the very first failed attempt give up immediately rather than
        # waiting an amount of time it doesn't have.
        mod = load_proxy(pause_seconds=5, max_total_retry_seconds=0.01, max_attempts=5)
        upstream = ScriptedUpstream([make_http_error(502)])
        mod.urllib.request.urlopen = upstream

        started = time.monotonic()
        h = make_handler(mod)
        h._proxy("POST")
        elapsed = time.monotonic() - started

        self.assertEqual(upstream.calls, 1, "should not have retried past its time budget")
        self.assertEqual(h._sent_statuses, [429])
        self.assertLess(elapsed, 2.0, "gave up far slower than its own budget suggests")

    def test_backoff_factor_grows_the_published_pause_across_attempts(self):
        mod = load_proxy(pause_seconds=10, backoff_factor=2.0, max_pause_seconds=1000, max_attempts=4)
        recorded = []
        mod._gate_penalize = lambda seconds: recorded.append(seconds)  # don't actually arm the gate
        mod.urllib.request.urlopen = ScriptedUpstream([make_http_error(502)])

        h = make_handler(mod)
        h._proxy("POST")

        # pause_seconds * backoff_factor**(attempt-1) for attempts 1..3
        self.assertEqual(recorded[:3], [10.0, 20.0, 40.0])


# ---------------------------------------------------------------------------
# SOCKS5 startup check
# ---------------------------------------------------------------------------
class SocksStartupTest(unittest.TestCase):
    def test_missing_pysocks_fails_clearly_instead_of_a_bare_traceback(self):
        tmpdir = tempfile.mkdtemp(prefix="proxy_socks_test_")
        try:
            with open(os.path.join(tmpdir, "config.toml"), "w") as f:
                f.write(
                    '[server]\nhost = "127.0.0.1"\nport = 8080\n'
                    '[upstream]\nhost = "example.invalid"\ntimeout_sec = 5\n'
                    '[proxy]\nuse_socks5 = true\nsocks5_host = "127.0.0.1"\nsocks5_port = 1080\n'
                    '[logging]\nenabled = false\nlog_dir = "logs"\n'
                    "[retry]\nenabled = false\n"
                )
            shutil.copy(PROXY_SCRIPT_PATH, os.path.join(tmpdir, "proxy_script.py"))

            script = (
                "import builtins, runpy\n"
                "orig = builtins.__import__\n"
                "def blocked(name, *a, **kw):\n"
                "    if name == 'socks':\n"
                "        raise ModuleNotFoundError(\"No module named 'socks'\")\n"
                "    return orig(name, *a, **kw)\n"
                "builtins.__import__ = blocked\n"
                "runpy.run_path('proxy_script.py', run_name='__main__')\n"
            )
            with open(os.path.join(tmpdir, "run_blocked.py"), "w") as f:
                f.write(script)

            import subprocess
            result = subprocess.run(
                [sys.executable, "run_blocked.py"],
                cwd=tmpdir, capture_output=True, text=True, timeout=10,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("PySocks", result.stderr)
            self.assertIn("pip install", result.stderr)
            # Should NOT be a bare traceback with no actionable message.
            self.assertNotIn("Traceback (most recent call last)", result.stderr)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Sanity-check the real config.toml deployed next to this file (parse only
# -- never executes the proxy against it).
# ---------------------------------------------------------------------------
class RealConfigSanityTest(unittest.TestCase):
    def test_real_config_parses_and_has_sane_values(self):
        config_path = os.path.join(THIS_DIR, "config.toml")
        if not os.path.isfile(config_path):
            self.skipTest("no config.toml next to this test file")

        try:
            import tomllib
        except ModuleNotFoundError:
            import tomli as tomllib

        with open(config_path, "rb") as f:
            cfg = tomllib.load(f)

        for section in ("server", "upstream", "proxy", "logging"):
            self.assertIn(section, cfg)

        self.assertIsInstance(cfg["server"]["port"], int)
        self.assertTrue(1 <= cfg["server"]["port"] <= 65535)

        retry = cfg.get("retry", {})
        if retry.get("max_buffer_bytes") is not None:
            self.assertGreater(retry["max_buffer_bytes"], 0)
        if retry.get("max_attempts") is not None:
            self.assertGreaterEqual(retry["max_attempts"], 1)
        if retry.get("max_concurrent_upstream") is not None:
            self.assertGreaterEqual(retry["max_concurrent_upstream"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
