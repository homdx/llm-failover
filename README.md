# llm-failover

A small, dependency-free Python client for calling LLM APIs (Ollama, OpenAI-compatible
gateways, Groq, OpenRouter, Gemini, ...) that stays working when a model, a server, or a
provider's rate limit gets in the way.

Two files, no framework: `llm_client.py` wraps one server; `llm_pool.py` wraps several.
Everything else in a normal LLM client — retries, rate-limit handling, model fallback,
server failover — is built on top of those two.

## What's actually interesting here

- **It remembers what a server rejects, and stops asking.** The first time a server
  returns HTTP 400 on a field it doesn't support (`response_format`, a thinking-control
  field, ...), the client disables *just that field* for *that server* and never sends
  it again — for any client instance talking to the same server, not just the one that
  hit the error.

- **429 doesn't always mean "sleep."** With a single model, a 429 is handled the classic
  way: read `Retry-After` (or parse it out of the error text — providers phrase this
  differently), wait, retry. With several models configured, a 429 instead hands the
  turn to the next model immediately, because many providers rate-limit *per model*, not
  per account — sleeping would just waste the minute. Only if an entire pass over all
  models comes back 429 does it conclude the limit is per-key and sleep, for the
  *shortest* time any model asked for.

- **A model list is a fallback chain, not just a config value.** `model = a, b, c` in the
  config means: if `a` exhausts its retries, try `b` from scratch, then `c`. The position
  of the last model that actually worked is remembered across calls, so the next call
  doesn't restart the search at `a` every time.

- **A pool of servers behaves exactly like one server, minus the single point of
  failure.** One server in the config gives you a plain client with no threading. Two or
  more give you the same interface, but each call leases whichever server is free,
  retries on a different one on error, and temporarily parks a server that keeps failing.

- **Same settings, built one way.** A client for a pool and a client used alone are
  constructed by the exact same code path, just naming a different config section. New
  settings can't silently apply to one and not the other — a bug class that a naive
  "pool builds clients by hand" design invites.

- **A distinct circuit breaker for "the server is actually gone."** A handful of broken
  JSON responses or slow answers is normal and gets retried quietly. Six real failures in
  a row raise a dedicated exception instead, on the theory that a caller silently falling
  back to a stub answer six times in a row is worse than stopping and saying so.

- **Secrets in a prompt don't reach the LLM.** If a system/user message happens to
  contain an `api_key = ...` line (a config file's contents pulled into context, for
  example), the client masks the value before sending, regardless of which key it is.

## Minimal usage

```python
from llm_client import LLMClient
import configparser

cfg = configparser.ConfigParser()
cfg.read("config.ini")

client = LLMClient.from_config(cfg)
reply = client.chat(system="You are concise.", user="Say hello in one sentence.")
```

Asking for structured output instead of free text:

```python
data = client.chat_json(system="Return JSON only.", user="Give me {\"greeting\": ...}")
```

Talking to several servers instead of one — same interface either way:

```python
from llm_pool import build_client

client = build_client(cfg)          # returns a plain client for one server,
                                     # or a pool-backed client for several —
                                     # decided by whether [api] pool is set
reply = client.chat(system="...", user="...")
```

See `config.example.ini` for every setting the client reads and what it defaults to.
