# Proxy checks

Standalone scripts for `llm-proxy`. No network, no real upstream, no real
client — every response is faked in-process, so they are safe to run
anywhere.

```sh
chmod +x run_all.sh
./run_all.sh ../llm-proxy
```

Each script also runs on its own and takes the folder holding the proxy
script and its `config.toml`:

```sh
python3 test_truncation.py ../llm-proxy
```

The proxy is imported under whichever name it has in that folder —
`main.py`, `python_proxy2.py`, `python_proxy.py` or `proxy.py`.

## What each one covers

| Script | Question it answers |
| --- | --- |
| `test_truncation.py` | A body that never arrived intact — cut off mid-JSON, connection dropped, socket timed out, SSE stopped on a clean chunk boundary — does it get retried instead of forwarded broken? |
| `test_false_positives.py` | Does the validation leave healthy responses alone? A `finish_reason: "length"` cutoff, a stream that ends with `[DONE]` and no `finish_reason`, `text/plain`, `304`, an over-sized response. |
| `test_coordination.py` | Does sharing one cooldown across threads still let a healthy upstream run in parallel, recover promptly, clear the gate after a success, and report a `Retry-After` that doesn't undercut what the upstream asked for? |
| `test_stream_stall_patience.py` | An upstream that trickles — does one slow byte keep the request alive, does a trickle that never ends still hit the `stream_max_wait_sec` ceiling, and does a completely dead upstream still fail on the first read? |
| `test_heartbeat.py` | The client-side keepalive: is it off when `heartbeat_after_sec = 0`, off on non-stream requests, off when the upstream answers in time, and when it does fire does it emit exactly one status line, keep every keepalive out of a real chunk, and still deliver a late error as an in-stream chunk? |
| `test_aborted_response.py` | Once bytes have gone to the client, does an upstream failure stop rather than retry? A retry there is undeliverable, re-bills the completion, and splices a second status line into the body. |
| `bench_ratelimit_storm.py` | Measurement, not pass/fail: five parallel requests against an upstream that answers 429 every time — how many requests does the proxy actually send, and for how long does it hold the client? |

## The keepalive settings these tests cover

`test_heartbeat.py` exercises two `[upstream]` keys that the rest of the
suite never touches:

| Key | Meaning |
| --- | --- |
| `heartbeat_after_sec` | Seconds of upstream silence after which the proxy commits to a `200 text/event-stream` and starts dripping ignorable `: keepalive` SSE comments, so the client's own idle timeout never fires while a cold model loads. `0` disables it, and the proxy behaves exactly as it did before the feature existed. |
| `heartbeat_interval_sec` | Gap between those comments once the drip has started. |

Only requests that asked for `"stream": true` are eligible — a plain JSON
response has nowhere to hide a keepalive.

The trade-off is worth stating plainly, because it is what the tests
guard: **the status code is spent early**. Once the drip has started, an
upstream error can no longer be sent as a status, so it is delivered as
an error chunk inside the stream followed by `[DONE]`. Silent retries are
unaffected — a comment line is not an answer, so the retry loop still
runs underneath a live heartbeat.

Two constraints on the value:

- It must be **well below `timeout_sec`**. `timeout_sec` is when the
  socket read gives up; a heartbeat scheduled at or after that point can
  never fire, because the connection is already gone. Setting the two
  equal is the same as switching the feature off, only harder to notice.
- It must be **above a normal error round-trip**, so ordinary 4xx/5xx
  responses still come back as real status codes instead of in-stream
  chunks. A few minutes is right for a cold-model upstream; a second or
  two is not.

If `max_total_retry_seconds` is smaller than `heartbeat_after_sec`, the
request gives up before the heartbeat would ever start — check the two
against each other as well.

## Relation to `test_python_proxy.py`

`test_python_proxy.py` is a unittest suite covering much of the same
ground in finer units, and it is the one to run in CI. These scripts
overlap with it deliberately and add two things it doesn't have:

- `bench_ratelimit_storm.py` measures traffic under a rate limit rather
  than asserting on it, which is what makes the effect of the shared
  cooldown visible as a number.
- The scripts print what they saw for every case, pass or fail, so they
  double as a way to watch the behaviour rather than only confirm it.

Run both.

## Reading a failure

Every script prints the observed value next to the expected one, so a
failing line says what actually happened:

```
  [FAIL] 2. connection dies mid-body (IncompleteRead)
         status=502 attempts=1 (want 429/3)
```

That one means the truncated body was never retried and the client got a
bare 502 — the failure the buffering exists to prevent.
