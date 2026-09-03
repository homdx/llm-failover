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
| `test_aborted_response.py` | Once bytes have gone to the client, does an upstream failure stop rather than retry? A retry there is undeliverable, re-bills the completion, and splices a second status line into the body. |
| `bench_ratelimit_storm.py` | Measurement, not pass/fail: five parallel requests against an upstream that answers 429 every time — how many requests does the proxy actually send, and for how long does it hold the client? |

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
