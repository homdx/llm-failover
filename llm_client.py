"""
llm_client.py

Minimal chat-completion client that speaks both Ollama (``/api/chat``) and
any OpenAI-compatible server (``/v1/chat/completions``). Payload shapes are
a simplified version of tools/llm_stream.py from
https://github.com/homdx/jan-auto-agent (branch collect-fix-2), so configs
and call style match that project. A handful of fixes made upstream since
then (branch improvements45) — MASK-KEY-1 secret masking, AUTO-ZAITHINK-1
Z.ai/GLM thinking-control field, and two response-parsing bugfixes — were
ported back here; each is marked with its upstream tag below.

Usage:

    from llm_client import LLMClient

    client = LLMClient.from_config(cfg)                    # api_<[api].active>
    client = LLMClient.from_config(cfg, api_section="api_remote")
    text = client.chat(system="...", user="...")
"""

from __future__ import annotations

import inspect
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.request

# ── DEBUG ────────────────────────────────────────────────────────────────────
# Enable with the environment variable:  LLM_DEBUG=1 python your_script.py
# Or hard-code:  _LLM_DEBUG = True
_LLM_DEBUG: bool = os.environ.get("LLM_DEBUG", "0").strip() not in ("0", "", "false", "no")

def _dbg(*args):
    if _LLM_DEBUG:
        import sys, datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        print(f"[LLM_DEBUG {ts}]", *args, file=sys.stderr, flush=True)
# ─────────────────────────────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_think(text: str) -> str:
    """Strip <think>...</think> blocks (qwen3-style reasoning) from a reply."""
    if not text:
        return text
    out = _THINK_RE.sub("", text)
    if "</think>" in out:
        out = out.rsplit("</think>", 1)[-1]
    elif "<think>" in out:
        out = out.split("<think>", 1)[0]
    out = out.replace("<think>", "").replace("</think>", "")
    return out.strip()


def strip_json_fence(text: str) -> str:
    """Remove a ```json ... ``` / ``` ... ``` fence if present.

    BUGFIX (ported from jan-auto-agent, tools/llm_stream.py, branch
    improvements45): a lone/unpaired ``` — output truncated mid-stream, or
    a stray ``` occurring inside otherwise-unfenced content — used to fall
    through to split("```")[0], which silently discarded everything before
    that first ``` even though it was never a real fence pair. A closing
    ``` must now actually be found before anything is treated as a fenced
    block; otherwise the text is returned unchanged.
    """
    if "```json" in text:
        rest = text.split("```json", 1)[1]
        if "```" in rest:
            return rest.split("```")[0].strip()
        return text
    if "```" in text:
        before, _, rest = text.partition("```")
        if "```" in rest:
            return rest.split("```")[0].strip()
        return text
    return text


# MASK-KEY-1 (ported from jan-auto-agent, tools/llm_stream.py, branch
# improvements45): matches an ini-style `api_key = <value>` assignment
# line, including comment-prefixed variants such as `### api_key = ...`
# or `; api_key = ...` (any leading whitespace, optional comment chars
# [#;]+ before the key name; case-insensitive key name), so a real or
# test secret pasted/read into a prompt (a config file's contents, a
# support dump, ...) never reaches the LLM verbatim. Captures the prefix
# (indent + optional comment + "api_key = ") separately so only the
# value is swapped for the placeholder.
_API_KEY_LINE_RE = re.compile(
    r'(?im)^([ \t]*(?:[#;]+[ \t]*)?api_key[ \t]*=[ \t]*)(\S+)([ \t]*)$'
)


def mask_api_key(text: str) -> str:
    """Replace every ``api_key = <value>`` line in *text* with
    ``api_key = here_your_key``, regardless of what the value actually is
    (a placeholder like "test", a real key, anything non-blank).

    Plain string transform with no knowledge of which key is "real" — it
    masks unconditionally so a secret embedded in file content pulled
    into a prompt (this client's own system/user message text) can't leak
    to the LLM. Non-string input (``None``, text with no such line) is
    returned unchanged.
    """
    if not text:
        return text
    return _API_KEY_LINE_RE.sub(r"\1here_your_key\3", text)


def _make_unverified_context() -> ssl.SSLContext:
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


def _ollama_chat_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/api/chat"):
        return base
    if base.endswith("/api"):
        return f"{base}/chat"
    return f"{base}/api/chat"


_RETRY_AFTER_MS_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*ms", re.IGNORECASE)
# THINK-5: match "reasoning" as a field name without also matching the
# substring inside "reasoning_effort" — otherwise a rejection of one is
# miscounted as a rejection of the other. Lookahead instead of matching
# literal escaped quotes: servers quote field names inconsistently in
# JSON error bodies (\", 'reasoning', or bare).
_BARE_REASONING_FIELD_RE = re.compile(r"reasoning(?!_effort)")
# THINK-6: Groq says "`reasoning_effort` must be one of `none` or
# `default`" — pull the offered values out of the backticks. The
# "must be one of `a`, `b` or `c`" shape is common to many OpenAI-compatible
# validators, so the regex is not tied to a specific field name; the
# detector in chat() already scopes it to this error.
_MUST_BE_ONE_OF_RE = re.compile(r"must be one of\s*((?:`[^`]+`[,\s]*(?:or\s*)?)+)",
                                re.IGNORECASE)
_BACKTICK_VALUE_RE = re.compile(r"`([^`]+)`")
_RETRY_AFTER_S_RE = re.compile(r"(?:try again|retry)\s+in\s+([\d.]+)\s*s(?:econds?)?\b", re.IGNORECASE)
# AUTO-ZAITHINK-1: matches "thinking" as a rejected field name in an HTTP
# 400 body (ported from jan-auto-agent, tools/llm_stream.py, branch
# improvements45). Word-boundary only — the payload's "thinking" object is
# a distinct field from "reasoning"/"reasoning_effort" above, so this
# never collides with _BARE_REASONING_FIELD_RE.
_BARE_THINKING_FIELD_RE = re.compile(r"\bthinking\b", re.IGNORECASE)


def _parse_retry_after(e: urllib.error.HTTPError, detail: str):
    """RATE-2/RATE-4: how long to ACTUALLY wait before repeating a 429,
    instead of a flat "60 seconds" when the server stated an exact figure.

    Groq does not always send a Retry-After header but almost always writes
    the delay in the error body ("Please try again in 820ms"); Gemini uses a
    different verb ("Please retry in 57.062042596s."), which the original
    "try again in" regex missed entirely, so the code always waited 60s
    instead of the honest ~57. Both verbs are matched now.

    Returns seconds (float), or None if no figure was found anywhere — then
    the caller picks a fallback, usually error_retry_wait_sec.
    """
    if e.headers:
        retry_after = e.headers.get("Retry-After")
        if retry_after:
            try:
                return max(0.1, float(retry_after))
            except ValueError:
                pass
    m = _RETRY_AFTER_MS_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1)) / 1000.0)
        except ValueError:
            pass
    m = _RETRY_AFTER_S_RE.search(detail)
    if m:
        try:
            return max(0.1, float(m.group(1)))
        except ValueError:
            pass
    return None


class LLMUnavailable(RuntimeError):
    """
    FIX-17: the model server has failed too many times in a row.

    This is NOT an ordinary call error. Callers typically swallow those and
    fall back to a stub answer, which is exactly how a dead ollama once went
    unnoticed for a whole run. This exception is meant to be propagated, not
    swallowed, so work stops instead of continuing blind.
    """


class RateLimited(RuntimeError):
    """
    RATE-ROT: the server answered 429 for a SPECIFIC model.

    It means "not this model right now", not "the server is broken". Raised
    instead of sleeping only when the list holds more than one model and
    rotate_on_429 is set: then the wait-or-switch decision belongs to
    chat_json(), which sees the whole list, not to chat(), which sees one
    request.

    wait_s is what the server asked for (Retry-After or error text), None if
    it did not say. chat_json() takes the MINIMUM over a pass: if any model
    frees up sooner, waiting for the slowest is pointless.
    """

    def __init__(self, message: str, model: str = "", wait_s=None,
                 url: str = ""):
        super().__init__(message)
        self.model = model
        self.wait_s = wait_s
        self.url = url


def _is_timeout(exc) -> bool:
    """
    An expired timeout means a slow server, not a dead one.

    RETRY-1: timeouts used to bump the breaker like a dropped connection, and
    six in a row killed the run — even though ollama is alive and simply
    thinking hard about a few thousand tokens. On a CPU laptop that is normal
    operation, not a fault.
    """
    if isinstance(exc, TimeoutError):
        return True
    # socket.timeout -> OSError with a telltale message on older versions
    return isinstance(exc, OSError) and "timed out" in str(exc).lower()


def _note_failure(exc):
    if _is_timeout(exc):
        return
    _Breaker.failures += 1
    if _Breaker.failures >= _Breaker.threshold:
        raise LLMUnavailable(
            f"{_Breaker.failures} LLM-вызовов подряд завершились ошибкой "
            f"(последняя: {exc}). Раунд прерван, чтобы не доигрывать его на "
            f"аварийных заглушках."
        ) from None


def _chat_takes_json_mode(fn) -> bool:
    """Does this chat() implementation accept the json_mode kwarg?

    Cached per function object: chat_json is called constantly and
    inspect.signature() is not free. A function taking **kwargs counts as
    accepting it — that is how wrappers are written.
    """
    key = getattr(fn, "__func__", fn)
    cached = _JSON_MODE_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        params = inspect.signature(fn).parameters
        ok = ("json_mode" in params
              or any(p.kind is inspect.Parameter.VAR_KEYWORD
                     for p in params.values()))
    except (TypeError, ValueError):
        ok = False
    _JSON_MODE_CACHE[key] = ok
    return ok


_JSON_MODE_CACHE: dict = {}


class _ServerCaps:
    """
    What the model server CAN do. Module-level for the same reason as
    _Breaker: there are many clients, and one server behind them.

    The first version kept these flags per instance and was useless — the
    client that paid a request to learn "this gateway rejects format=json"
    kept that knowledge to itself, and every other client rediscovered it.
    Knowledge bought with a request has to be shared.

    Keyed by base_url, not global: one run may talk to a local ollama (which
    understands format) and a remote gateway (which does not), and a
    conclusion about one must not cripple the other.
    """
    no_json_format: dict[str, bool] = {}
    # THINK-5: one shared flag used to disable all three think fields
    # (chat_template_kwargs, reasoning_effort, reasoning) when ANY of them
    # was rejected. Groq rejects chat_template_kwargs specifically — that
    # one is vLLM/SGLang-only and never was in the OpenAI spec — but
    # reasoning_effort/reasoning are standard and might have worked. One
    # flag per field name, so a working mechanism is not thrown out with a
    # broken one.
    rejected_think_fields: dict[str, set] = {}
    # THINK-6: Groq rejects the VALUE, not the field: '`reasoning_effort`
    # must be one of `none` or `default`'. The field works, its allowed set
    # just differs from native OpenAI ("low"/"medium"/"high"). Hardcoding
    # "none" globally would break providers that expect "low", so the value
    # is learned from the error text and remembered per base_url.
    reasoning_effort_value: dict[str, str] = {}

    @classmethod
    def reasoning_effort_for(cls, base_url: str) -> str:
        return cls.reasoning_effort_value.get(base_url, "low")

    @classmethod
    def set_reasoning_effort_value(cls, base_url: str, value: str):
        cls.reasoning_effort_value[base_url] = value

    # MODEL-FALLBACK-2: every chat_json() call used to restart the model
    # scan from models[0], even when that model had been declared exhausted
    # a minute earlier. Gemini free tier hands out a DAILY quota
    # ("generate_content_free_tier_requests, limit: 20") that does not come
    # back in minutes, so each call burned a full retries/error_retries
    # cycle (~6-7 min) on a dead model before reaching a live one. Remember
    # the position of the last WORKING model per base_url; the next call
    # (from any client sharing that base_url) starts there, not at zero.
    current_model_index: dict[str, int] = {}

    @classmethod
    def get_model_index(cls, base_url: str) -> int:
        return cls.current_model_index.get(base_url, 0)

    @classmethod
    def set_model_index(cls, base_url: str, idx: int):
        cls.current_model_index[base_url] = idx

    @classmethod
    def rejects_json_format(cls, base_url: str) -> bool:
        return cls.no_json_format.get(base_url, False)

    @classmethod
    def mark_json_format_rejected(cls, base_url: str):
        cls.no_json_format[base_url] = True

    @classmethod
    def think_field_rejected(cls, base_url: str, field: str) -> bool:
        return field in cls.rejected_think_fields.get(base_url, ())

    @classmethod
    def mark_think_field_rejected(cls, base_url: str, field: str):
        cls.rejected_think_fields.setdefault(base_url, set()).add(field)

    @classmethod
    def reset(cls):
        cls.no_json_format.clear()
        cls.rejected_think_fields.clear()
        cls.reasoning_effort_value.clear()
        cls.current_model_index.clear()


class _Breaker:
    """
    FIX-17: breaker state, deliberately kept OUTSIDE LLMClient.

    This is the health of the model server, not a property of a wrapper
    class: many client instances, one server. Keeping the counter as a class
    attribute was also fragile — a global name lookup breaks if the module
    name is patched in tests, and `cls._failures += 1` from a subclass
    silently starts a second counter.
    """
    failures = 0
    threshold = 6


class LLMClient:
    """
    Wrapper over one API profile (see [api] / [api_local] / [api_remote] in
    config.ini). Supports api_format="ollama" (for `ollama serve`) and
    api_format="openai" (Jan and any OpenAI-compatible server).
    """

    def __init__(self, base_url: str, api_key: str, model: str,
                 api_format: str = "ollama", verify_ssl: bool = True,
                 num_ctx: int = 0, think: "bool | None" = None,
                 timeout: int = 120, retries: int = 1, on_retry=None,
                 json_format: bool = True,
                 error_retries: int = 0, error_retry_wait_sec: int = 60,
                 max_retry_after_sec: int = 180,
                 rotate_on_429: bool = False,
                 rate_limit_cycles: int = 1):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        # MODEL-FALLBACK-1: `model` may be a comma-separated list
        # ("model-a, model-b, model-c"). When the current model runs out of
        # retries, chat_json() moves to the next one and restarts attempts
        # there. self.model stays a SINGLE active name — that is what
        # _build_request reads — so all code that just reads self.model is
        # unaware of the list. A name without commas gives a one-item list
        # and behaves exactly as before.
        self.models = [m.strip() for m in str(model).split(",") if m.strip()]
        if not self.models:
            self.models = [model]
        # MODEL-FALLBACK-3: a real config listed the same model twice
        # ("gemini-3.6-flash, gemini-3.6-flash, gemini-3.5-flash") — most
        # likely a copy-paste typo. Without dedup, "switch to the next
        # model" honestly switched to the same name and repeated the same
        # failure, costing another full retries cycle on a model already
        # known to be dead (~7 min on Gemini free tier). First occurrence
        # wins, duplicates are dropped silently.
        seen = set()
        deduped = []
        for m in self.models:
            if m not in seen:
                seen.add(m)
                deduped.append(m)
        self.models = deduped
        self.model = self.models[0]
        self.api_format = api_format
        self.num_ctx = num_ctx
        self.think = think
        self.timeout = timeout
        # RETRY-1: how many EXTRA attempts to make on failure.
        # 0 is the old behaviour: one attempt, then give up.
        self.retries = max(0, int(retries))
        # HTTP-RETRY: extra attempts on ANY error status (402 "out of
        # credits", 5xx, ...), on top of the dedicated 429 and 400+json_mode
        # handling. Each attempt waits error_retry_wait_sec and repeats the
        # SAME request unchanged — unlike RETRY-1 in chat_json(), which
        # perturbs temperature/prompt and does not wait. 402/5xx used to
        # fail immediately: HF returns 402 for short-lived billing glitches
        # too, not only for a genuinely exhausted quota. Default 0 so a bare
        # LLMClient(...) in tests and stubs does not suddenly sleep a minute
        # on every HTTP error; from_config() turns it on for real runs.
        self.error_retries = max(0, int(error_retries))
        self.error_retry_wait_sec = max(1, int(error_retry_wait_sec))
        # RATE-3: Groq answered a DAILY-limit 429 (TPD, not TPM) with
        # Retry-After=1754s (29 min), and that figure grows toward HOURS
        # since TPD resets once a day. RATE-1/HTTP-RETRY slept exactly as
        # long as told, blocking the whole run on one request instead of
        # giving up fast so the pool can switch server/model. Ceiling: if
        # the server asks for longer than max_retry_after_sec, raise instead
        # of sleeping. Short TPM pauses stay under the cap and still wait.
        self.max_retry_after_sec = max(1, int(max_retry_after_sec))
        # RATE-ROT: with several models listed, a 429 raises RateLimited
        # instead of sleeping — chat_json() decides where to go next.
        # Default False so a bare LLMClient(...) in tests and stubs is not
        # changed by a setting it never passed; from_config() enables it.
        self.rotate_on_429 = bool(rotate_on_429)
        # How many EXTRA full passes over the model list to make when a
        # whole pass came back 429 (i.e. the limit looks per key). 0 means
        # one pass and straight up, with no pause at all.
        self.rate_limit_cycles = max(0, int(rate_limit_cycles))
        # Optional log(str) callback: the client knows nothing about the
        # caller's logger, but the caller can supply one to see retries.
        self.on_retry = on_retry
        self._ssl_context = None if verify_ssl else _make_unverified_context()
        # EOS-2/COMPAT: json_format=false in [api] is a claim about the
        # SERVER, so it goes to the shared cache, not onto the instance.
        if not json_format:
            _ServerCaps.mark_json_format_rejected(self.base_url)

    @classmethod
    def from_config(cls, cfg, section: str = None,
                    api_section: str = None) -> "LLMClient":
        """
        Build a client from a configparser.ConfigParser following the agents.ini
        scheme: [api].active selects the api_<active> section.

        PARITY-1: `api_section` names the server section EXPLICITLY, bypassing
        [api].active. The pool needs it (llm_pool._client_for_section): it has
        several sections but must build clients with the SAME code as
        single-server mode, or the [api] defaults drift apart one release at a
        time — as they did, when the pool knew nothing about rotate_on_429 and
        rate_limit_cycles because it constructed LLMClient by hand. One
        constructor, one set of defaults.

        The old positional `section` is still unused; it is kept so calls like
        from_config(cfg, "player") keep working.
        """
        active = cfg.get("api", "active", fallback="local")
        api_section = api_section or f"api_{active}"
        verify_ssl = cfg.getboolean("api", "verify_ssl", fallback=True)

        base_url = cfg.get(api_section, "base_url")
        api_key = cfg.get(api_section, "api_key", fallback="not-needed")
        model = cfg.get(api_section, "model")
        api_format = cfg.get(api_section, "api_format", fallback="ollama")
        num_ctx = cfg.getint(api_section, "num_ctx", fallback=0)
        think_raw = cfg.get(api_section, "think", fallback=None)
        think = None if think_raw is None else cfg.getboolean(api_section, "think")
        timeout = cfg.getint("api", "timeout_seconds", fallback=120)
        retries = cfg.getint("api", "retries", fallback=1)
        # HTTP-RETRY: see the __init__ docstring. 2 attempts, 60s apart —
        # what used to apply to 429 only, now extended to 402/5xx.
        error_retries = cfg.getint("api", "error_retries", fallback=2)
        error_retry_wait_sec = cfg.getint("api", "error_retry_wait_sec",
                                          fallback=60)
        # RATE-3: sane ceiling for a 429 wait — see __init__. 180s covers
        # typical TPM pauses without letting retries block the run for
        # tens of minutes when the server reports time to a DAILY/MONTHLY
        # reset rather than a per-minute one.
        max_retry_after_sec = cfg.getint("api", "max_retry_after_sec",
                                         fallback=180)
        # RATE-ROT: on by default for real runs — sleeping on 429 inside
        # chat() reliably bypassed MODEL-FALLBACK, the very reason the
        # model list exists. rotate_on_429=false restores the old path.
        rotate_on_429 = cfg.getboolean("api", "rotate_on_429", fallback=True)
        rate_limit_cycles = cfg.getint("api", "rate_limit_cycles", fallback=1)
        # Gateways in front of a remote server may not know the `format`
        # field — it can be disabled upfront instead of on the first 400.
        json_format = cfg.getboolean("api", "json_format", fallback=True)

        return cls(base_url=base_url, api_key=api_key, model=model,
                    api_format=api_format, verify_ssl=verify_ssl,
                    num_ctx=num_ctx, think=think, timeout=timeout,
                    retries=retries, json_format=json_format,
                    error_retries=error_retries,
                    error_retry_wait_sec=error_retry_wait_sec,
                    max_retry_after_sec=max_retry_after_sec,
                    rotate_on_429=rotate_on_429,
                    rate_limit_cycles=rate_limit_cycles)

    def _build_request(self, system: str, user: str, temperature: float,
                        max_tokens: int, json_mode: bool = False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            # UA-1: urllib without an explicit User-Agent sends
            # "Python-urllib/3.x". Groq (behind Cloudflare) answered HTTP
            # 403 "error code: 1010" — a block on the CLIENT SIGNATURE
            # (known script-library strings are banned before any JS
            # challenge). An honest non-library UA usually clears that
            # specific filter; it does not forge TLS/JA3 fingerprints and
            # will not defeat stronger protection, but 1010 is usually the UA.
            "User-Agent": "learn-in-play1-llm-client/1.0 (+https://github.com/homdx/learn-in-play1)",
        }
        # MASK-KEY-1: mask any `api_key = ...` line that ended up inside
        # the system/user text itself (e.g. a config file's contents
        # pulled into context) — NOT self.api_key, which is this
        # request's own credential and is used as-is in the
        # Authorization header above.
        messages = [
            {"role": "system", "content": mask_api_key(system)},
            {"role": "user", "content": mask_api_key(user)},
        ]
        if self.api_format == "ollama":
            url = _ollama_chat_url(self.base_url)
            options = {"temperature": temperature, "num_predict": max_tokens}
            if self.num_ctx:
                options["num_ctx"] = self.num_ctx
            payload = {"model": self.model, "messages": messages,
                       "stream": False, "options": options}
            if self.think is not None:
                payload["think"] = self.think
            # EOS-2: JSON grammar forbids EOS as the first token, so an
            # empty answer (eval_count=1, no eval_duration) is physically
            # impossible: the sampler has to start with '{'.
            if json_mode and not self._json_format_off():
                payload["format"] = "json"
        else:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "messages": messages,
                "stream": False,
            }
            if json_mode and not self._json_format_off():
                payload["response_format"] = {"type": "json_object"}
            # THINK-1: Qwen3 and other hybrid-thinking models behind
            # vLLM/SGLang disable <think> via
            # chat_template_kwargs.enable_thinking. Without it the model
            # spends the whole max_tokens on reasoning and stops on
            # 'length' with empty content. think=false now applies to
            # api_format=openai too, not just ollama.
            if self.think is not None and not _ServerCaps.think_field_rejected(
                    self.base_url, "chat_template_kwargs"):
                payload["chat_template_kwargs"] = {"enable_thinking": self.think}
            # THINK-3: chat_template_kwargs is vLLM/SGLang-specific.
            # OpenRouter uses its own unified parameter — a nested
            # `reasoning` object — and native OpenAI o1/o3/gpt-5 has a flat
            # `reasoning_effort`. Real case: openai/gpt-oss-20b:free through
            # OpenRouter burned all 700 max_tokens in a hidden reasoning
            # channel that enable_thinking does not touch at all. Send both
            # fields: what a provider does not understand it usually
            # ignores, cheaper than guessing which one applies here.
            # THINK-5: not every server ignores it — Groq fails the whole
            # request with HTTP 400 on chat_template_kwargs. That does not
            # imply the other two are rejected, so each field is tracked
            # independently (see THINK-5 in chat()).
            if self.think is False:
                if not _ServerCaps.think_field_rejected(self.base_url, "reasoning_effort"):
                    payload["reasoning_effort"] = _ServerCaps.reasoning_effort_for(self.base_url)
                if not _ServerCaps.think_field_rejected(self.base_url, "reasoning"):
                    payload["reasoning"] = {"effort": "low", "exclude": True}
            # AUTO-ZAITHINK-1 (ported from jan-auto-agent, tools/llm_stream.py,
            # branch improvements45): Z.ai/GLM's own thinking-control
            # convention — a top-level `thinking: {"type": "enabled"|
            # "disabled"}` object, distinct from chat_template_kwargs/
            # reasoning_effort/reasoning above (neither is Z.ai's documented
            # shape). Sent ADDITIONALLY, never instead of them, for either
            # value of self.think (not just False): a provider that doesn't
            # recognise `thinking` is expected to silently ignore the
            # unknown field, the same assumption the other three fields
            # already rely on, so trying all four costs nothing against a
            # provider that only speaks one. Upstream report: a GLM-5.2
            # endpoint ignored all three existing fields, defaulted to its
            # own "Max" reasoning effort, and burned the whole max_tokens
            # budget on <think> before any content was written.
            if self.think is not None and not _ServerCaps.think_field_rejected(
                    self.base_url, "thinking"):
                payload["thinking"] = {"type": "enabled" if self.think else "disabled"}
        _dbg(f"_build_request: url={url!r}, model={self.model!r}, "
             f"api_format={self.api_format!r}, think={self.think!r}, "
             f"num_ctx={self.num_ctx}, max_tokens={max_tokens}, temp={temperature}, "
             f"json_mode={json_mode}")
        _dbg(f"  payload keys: {list(payload.keys())}")
        return url, headers, payload

    def _extract_content(self, raw: dict) -> str:
        _dbg(f"_extract_content: api_format={self.api_format!r}")
        _dbg(f"  raw keys: {list(raw.keys())}")
        if self.api_format == "ollama":
            msg = raw.get("message", {})
            _dbg(f"  ollama message keys: {list(msg.keys())}")
            # BUGFIX (ported from jan-auto-agent, tools/llm_stream.py,
            # branch improvements45): some Ollama-compatible backends
            # return "content": null (JSON null) rather than omitting the
            # key or using "" — a `.get("content", "")` default only
            # covers a MISSING key, not an explicit null value, so this
            # returned None here and `.strip()` below raised
            # AttributeError on a reply that otherwise arrived
            # successfully, instead of degrading to an empty string the
            # same way a filtered/empty openai `choices` list already does.
            content = msg.get("content") or ""
            # Some thinking models in Ollama put the reasoning in
            # message.thinking and leave content empty.
            thinking = msg.get("thinking", "")
            _dbg(f"  content repr (first 200): {content[:200]!r}")
            _dbg(f"  thinking present: {bool(thinking)}, len={len(thinking)}")

            # DIAG-CTX: prompt_eval_count/eval_count are the only way to
            # tell "the model emitted an empty <think> block" from "the
            # prompt filled num_ctx, nothing left to generate". Both look
            # identically empty in list(raw.keys()), but need opposite fixes.
            prompt_eval_count = raw.get("prompt_eval_count")
            eval_count = raw.get("eval_count")
            has_eval_duration = "eval_duration" in raw
            _dbg(f"  prompt_eval_count={prompt_eval_count}, "
                 f"eval_count={eval_count}, num_ctx={self.num_ctx}, "
                 f"has_eval_duration={has_eval_duration}")
            # DIAG-CTX-3: stash for chat() — only there, AFTER
            # strip_think(), is it known whether the text vanished in the
            # strip rather than having been empty all along.
            self._last_ollama_diag = {
                "prompt_eval_count": prompt_eval_count,
                "eval_count": eval_count,
                "num_ctx": self.num_ctx,
            }

            if not content.strip() and not thinking:
                if eval_count == 0 or (eval_count is not None and not has_eval_duration):
                    near_limit = (
                        self.num_ctx and prompt_eval_count is not None
                        and prompt_eval_count >= self.num_ctx - 64
                    )
                    _dbg(
                        "  WARNING: 0 completion tokens generated "
                        f"(prompt_eval_count={prompt_eval_count}/{self.num_ctx}). "
                        + ("Prompt has filled the context window — this is "
                           "context overflow, NOT a thinking-only glitch. "
                           "Shrink dsyn/checklist/history sizes or raise num_ctx."
                           if near_limit else
                           "eval_count=0 but prompt is not near num_ctx limit — "
                           "investigate server-side (stop token / filter?).")
                    )
                elif thinking:
                    _dbg("  WARNING: content is empty but thinking is not — "
                         "model returned only a <think> block, no JSON output!")
            return content.strip()
        choices = raw.get("choices") or []
        _dbg(f"  openai choices count: {len(choices)}")
        if not choices:
            _dbg(f"  ERROR: empty choices. Full raw: {json.dumps(raw)[:500]}")
            raise ValueError(
                f"Пустой ответ LLM (choices=[]) - возможно, запрос был "
                f"отфильтрован сервером. Ключи ответа: {list(raw.keys())}"
            )
        msg = choices[0].get("message", {})
        _dbg(f"  openai message keys: {list(msg.keys())}")
        content = msg.get("content", "") or ""
        # OpenAI-compatible servers with thinking sometimes put the
        # reasoning in a separate field and leave content empty. Names
        # differ: vLLM/SGLang uses "reasoning_content", OpenRouter (incl.
        # gpt-oss through it) uses "reasoning". Only the first name used to
        # be checked, so LLM_DEBUG reported "reasoning_content present:
        # False" while 3000+ chars sat under "reasoning".
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        finish_reason = choices[0].get("finish_reason", "")
        _dbg(f"  finish_reason: {finish_reason!r}")
        _dbg(f"  content repr (first 200): {content[:200]!r}")
        _dbg(f"  reasoning present: {bool(reasoning)}, len={len(reasoning)}")
        if not content.strip() and reasoning:
            _dbg("  WARNING: content is empty but reasoning is not — "
                 "model returned only thinking, no actual JSON output!")
        # DIAG-CTX-2: openai-format counterpart of the context-overflow-vs-
        # sampling-glitch diagnosis, which previously existed only in the
        # ollama branch (prompt_eval_count/eval_count). Those fields do not
        # exist in an openai response, yet the final chat_json() error text
        # still told the reader to look for them. Here the same diagnosis is
        # built from usage.prompt_tokens/completion_tokens and finish_reason.
        #
        # DIAG-CTX-3: checking here is not enough. At this point content is
        # NOT empty — it is the literal "<think>...", and it only becomes
        # empty later in chat(), after strip_think(). So "not
        # content.strip()" never fires for that case (Groq, qwen3.6). Stash
        # usage/finish_reason on self so chat() can diagnose it after the
        # strip, where the fact "there was text and it vanished" is known.
        usage = raw.get("usage") or {}
        self._last_usage = usage
        self._last_finish_reason = finish_reason
        if usage:
            _dbg(f"  usage: prompt_tokens={usage.get('prompt_tokens')}, "
                f"completion_tokens={usage.get('completion_tokens')}, "
                f"total_tokens={usage.get('total_tokens')}")
        return content.strip()

    def chat(self, system: str, user: str, temperature: float = 0.4,
             max_tokens: int = 400, json_mode: bool = False,
             _retried_429: bool = False, _http_retry_n: int = 0) -> str:
        """Blocking chat-completion call. Returns the reply text, with any
        <think>...</think> already stripped out."""
        url, headers, payload = self._build_request(system, user, temperature,
                                                    max_tokens, json_mode)
        req = urllib.request.Request(
            url, data=json.dumps(payload).encode("utf-8"),
            headers=headers, method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout,
                                         context=self._ssl_context) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            detail = ""
            try:
                detail = e.read().decode("utf-8", errors="replace")[:500]
            except Exception:
                pass
            # RATE-1: a 429 from an OpenAI-compatible gateway means "wait
            # and repeat", not "server is down" — mixing it with a dropped
            # connection (RETRY-1) is wrong, since that path perturbs
            # temperature/prompt while this one must repeat the SAME
            # request. Wait for Retry-After if sent, else 60s, once here;
            # after that RETRY-1 in chat_json() takes over.
            # RATE-ROT: with several models listed, sleeping is wrong.
            # Waiting only pays off if the limit is per KEY; if it is per
            # MODEL (Groq/Gemini) the next model answers right now, and the
            # sleep is a wasted minute. chat_json() decides — it sees the
            # whole list and the outcome of a full pass.
            _rot = (e.code == 429
                    and getattr(self, "rotate_on_429", False)
                    and len(getattr(self, "models", None) or []) > 1)
            if _rot:
                rot_wait = _parse_retry_after(e, detail)
                asked = (f"просит ждать {rot_wait:.1f} сек"
                         if rot_wait is not None else "не сказал сколько ждать")
                msg = (f"RATE-ROT: HTTP 429 от {url} на модели "
                       f"{self.model!r} (сервер {asked}) — не сплю, "
                       f"отдаю ход следующей модели списка")
                on_retry = getattr(self, "on_retry", None)
                if on_retry:
                    on_retry(msg)
                _dbg(f"chat(): {msg} (detail={detail[:200]!r})")
                raise RateLimited(msg, model=self.model, wait_s=rot_wait,
                                  url=url) from None
            if e.code == 429 and not _retried_429 and not _rot:
                wait_s = _parse_retry_after(e, detail)
                if wait_s is None:
                    wait_s = 60.0
                if wait_s > self.max_retry_after_sec:
                    # RATE-3: longer than the sane ceiling — almost
                    # certainly a daily/monthly limit, not a per-minute one.
                    # Do not sleep for hours inside one call; raise and let
                    # the caller decide (the pool switches server/model).
                    msg = (f"HTTP 429 от {url}: сервер просит ждать "
                          f"{wait_s:.0f} сек — это дольше потолка "
                          f"{self.max_retry_after_sec} сек (похоже на "
                          f"дневной/месячный лимит, а не минутный), не жду")
                    on_retry = getattr(self, "on_retry", None)
                    if on_retry:
                        on_retry(msg)
                    _dbg(f"chat(): {msg}")
                    raise RuntimeError(msg) from None
                on_retry = getattr(self, "on_retry", None)
                if on_retry:
                    on_retry(f"HTTP 429 от {url} (rate limited), "
                             f"жду {wait_s:.1f} сек и повторяю тот же запрос")
                _dbg(f"chat(): HTTP 429, sleeping {wait_s:.1f}s before retry "
                     f"(detail={detail[:200]!r})")
                time.sleep(wait_s)
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=json_mode,
                                 _retried_429=True)
            # EOS-2/COMPAT: a 400 on "format": "json" almost always means a
            # gateway that validates the body strictly and rejects fields
            # outside its schema. Local ollama accepts it, a remote proxy
            # does not — which is why this only ever broke on api_remote and
            # looked like "it worked, then it stopped".
            #
            # Back off once and remember it: later requests go out plain.
            # The EOS grammar guard is lost, but it was insurance, not a
            # requirement, and failing instead is not an option.
            if json_mode and e.code == 400 and not self._json_format_off():
                _ServerCaps.mark_json_format_rejected(self.base_url)
                _dbg("chat(): HTTP 400 with format=json — server rejects the "
                     "field, retrying without it and disabling it for this "
                     "client")
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=False)
            # THINK-5: Groq does not silently ignore chat_template_kwargs /
            # reasoning_effort / reasoning — it fails the whole request with
            # its own HTTP 400 ("property 'chat_template_kwargs' is
            # unsupported"). This is a SECOND, separate check from the one
            # above: the server may have rejected response_format on the
            # first 400, and this one surfaces on the retry. Uncaught, the
            # request falls into generic HTTP-RETRY with the same field
            # still in the body and repeats until retries run out.
            #
            # Disable ONLY the field actually named in the error, not all
            # three (THINK-4 bundled them: a chat_template_kwargs rejection
            # also dropped reasoning_effort/reasoning, which the server may
            # accept and which do suppress thinking).
            #
            # THINK-6: for reasoning_effort the error may mean "bad value"
            # rather than "unsupported field" (Groq: "must be one of `none`
            # or `default`"). Then keep the field and adopt a value from the
            # server's hint (preferring "none", since the goal is to
            # suppress thinking), remembered for this base_url only.
            if e.code == 400 and self.api_format == "openai":
                newly_rejected = []
                reasoning_effort_fixed = False

                if "reasoning_effort" in detail:
                    m = _MUST_BE_ONE_OF_RE.search(detail)
                    candidates = (_BACKTICK_VALUE_RE.findall(m.group(1))
                                 if m else [])
                    chosen = ("none" if "none" in candidates
                             else (candidates[0] if candidates else None))
                    if (chosen and
                            _ServerCaps.reasoning_effort_for(self.base_url) != chosen):
                        _ServerCaps.set_reasoning_effort_value(self.base_url, chosen)
                        reasoning_effort_fixed = True
                        _dbg(f"chat(): HTTP 400 says reasoning_effort must "
                            f"be one of {candidates} — switching to "
                            f"{chosen!r} for this server and retrying "
                            f"(detail={detail[:200]!r})")
                    elif not _ServerCaps.think_field_rejected(
                            self.base_url, "reasoning_effort"):
                        newly_rejected.append("reasoning_effort")

                if ("chat_template_kwargs" in detail
                        and not _ServerCaps.think_field_rejected(
                            self.base_url, "chat_template_kwargs")):
                    newly_rejected.append("chat_template_kwargs")
                if (_BARE_REASONING_FIELD_RE.search(detail)
                        and not _ServerCaps.think_field_rejected(
                            self.base_url, "reasoning")):
                    newly_rejected.append("reasoning")
                # AUTO-ZAITHINK-1: same one-field-at-a-time treatment as
                # reasoning/chat_template_kwargs above — a provider (e.g. a
                # strict-schema gateway) that rejects the `thinking` object
                # outright gets just that field disabled and remembered;
                # reasoning_effort/reasoning keep working if the server
                # accepted those.
                if (_BARE_THINKING_FIELD_RE.search(detail)
                        and not _ServerCaps.think_field_rejected(
                            self.base_url, "thinking")):
                    newly_rejected.append("thinking")

                if newly_rejected or reasoning_effort_fixed:
                    for f in newly_rejected:
                        _ServerCaps.mark_think_field_rejected(self.base_url, f)
                    if newly_rejected:
                        _dbg(f"chat(): HTTP 400 naming think-field(s) "
                            f"{newly_rejected} — disabling just these for "
                            f"this client and retrying "
                            f"(detail={detail[:200]!r})")
                    return self.chat(system, user, temperature=temperature,
                                     max_tokens=max_tokens, json_mode=json_mode)
            # HTTP-RETRY: any other error status — 402 "out of credits",
            # 5xx, and so on. These used to fail immediately with no attempt
            # to wait. router.huggingface.co returns 402 for short-lived
            # billing glitches on their side, not only for a truly exhausted
            # quota, so a fixed pause and a repeat are worth their cost even
            # though they will not save a genuinely exhausted monthly quota.
            #
            # RATE-2: if this is the SECOND consecutive 429 on the same call
            # (RATE-1 above already spent its single retry), do not sleep
            # error_retry_wait_sec blindly — Groq usually states the real
            # delay in the body ("Please try again in 820ms"), and waiting a
            # fixed minute instead is pure loss. Other codes (402/5xx)
            # rarely carry a figure, so they keep error_retry_wait_sec.
            if _http_retry_n < self.error_retries:
                wait_s = self.error_retry_wait_sec
                if e.code == 429:
                    precise = _parse_retry_after(e, detail)
                    if precise is not None:
                        wait_s = precise
                # RATE-3: same ceiling as RATE-1 above. THIS is the branch
                # where the real 1754s hang happened — the second
                # consecutive 429 on a Groq daily limit lands here.
                if wait_s > self.max_retry_after_sec:
                    msg = (f"HTTP {e.code} от {url}: сервер просит ждать "
                          f"{wait_s:.0f} сек — это дольше потолка "
                          f"{self.max_retry_after_sec} сек, не жду")
                    on_retry = getattr(self, "on_retry", None)
                    if on_retry:
                        on_retry(msg)
                    _dbg(f"chat(): {msg}")
                    raise RuntimeError(msg) from None
                on_retry = getattr(self, "on_retry", None)
                msg = (f"HTTP {e.code} от {url} ({detail or e.reason}), жду "
                       f"{wait_s:.1f} сек и повторяю тот же запрос (попытка "
                       f"{_http_retry_n + 1}/{self.error_retries})")
                if on_retry:
                    on_retry(msg)
                _dbg(f"chat(): {msg}")
                time.sleep(wait_s)
                return self.chat(system, user, temperature=temperature,
                                 max_tokens=max_tokens, json_mode=json_mode,
                                 _retried_429=_retried_429,
                                 _http_retry_n=_http_retry_n + 1)
            raise RuntimeError(f"HTTP {e.code} от {url}: {detail or e.reason}") from None
        except urllib.error.URLError as e:
            raise RuntimeError(
                f"Не удалось подключиться к {url} ({e.reason}). "
                f"Проверьте, что сервер запущен (ollama serve / Jan API server) "
                f"и base_url в config.ini указан верно."
            ) from None

        text = self._extract_content(raw)
        _dbg(f"chat(): raw text len={len(text)}, repr(first 300): {text[:300]!r}")
        stripped = strip_think(text)
        _dbg(f"chat(): after strip_think len={len(stripped)}, repr(first 300): {stripped[:300]!r}")
        if not stripped and text:
            _dbg("chat(): WARNING — strip_think() returned empty string! "
                 "The model likely returned ONLY a <think> block with no content after it.")
            # DIAG-CTX-3: this is the point where context overflow can be
            # told apart from "the model got stuck reasoning". The check
            # used to live in _extract_content(), where content is NOT yet
            # empty (it is the literal "<think>..." that strip_think() only
            # ate here), so it could never fire for the case that actually
            # happened (Groq, qwen3.6: finish_reason='length',
            # completion_tokens=700=max_tokens, content one unterminated
            # <think> block). Use the usage/finish_reason (openai) or
            # prompt_eval_count/eval_count (ollama) stashed earlier.
            if self.api_format == "openai":
                usage = getattr(self, "_last_usage", None) or {}
                finish_reason = getattr(self, "_last_finish_reason", "")
                completion_tokens = usage.get("completion_tokens")
                if completion_tokens is not None:
                    if finish_reason == "length":
                        _dbg(
                            f"  DIAGNOSIS: finish_reason='length', "
                            f"completion_tokens={completion_tokens} spent "
                            f"entirely inside the <think> block above — the "
                            f"model burned its whole max_tokens budget on "
                            f"reasoning and never reached an actual answer. "
                            f"Raise max_tokens or force reasoning "
                            f"suppression (think=false) for this provider."
                        )
                    else:
                        _dbg(
                            f"  DIAGNOSIS: finish_reason={finish_reason!r} "
                            f"with completion_tokens={completion_tokens} — "
                            f"the model stopped on its own INSIDE its "
                            f"<think> block without producing an answer "
                            f"(not a token-budget issue — investigate "
                            f"prompt/sampling)."
                        )
            elif self.api_format == "ollama":
                diag = getattr(self, "_last_ollama_diag", None) or {}
                pe = diag.get("prompt_eval_count")
                ec = diag.get("eval_count")
                nc = diag.get("num_ctx")
                near_limit = bool(nc and pe is not None and pe >= nc - 64)
                _dbg(
                    f"  DIAGNOSIS: prompt_eval_count={pe}/{nc}, "
                    f"eval_count={ec}. "
                    + ("Prompt has filled the context window — this is "
                       "context overflow, NOT a thinking-only glitch. "
                       "Shrink dsyn/checklist/history sizes or raise num_ctx."
                       if near_limit else
                       "Not near the context limit — the model stopped "
                       "inside its own thinking without an answer; "
                       "investigate server-side (stop token / filter?).")
                )
        return stripped

    def _json_format_off(self) -> bool:
        """getattr: stubs built without __init__ have no base_url."""
        return _ServerCaps.rejects_json_format(getattr(self, "base_url", ""))

    @classmethod
    def configure_breaker(cls, threshold: int):
        _Breaker.threshold = max(1, int(threshold))
        _Breaker.failures = 0

    @classmethod
    def reset_breaker(cls):
        _Breaker.failures = 0

    def chat_json(self, system: str, user: str, temperature: float = 0.4,
                  max_tokens: int = 400) -> dict:
        """
        MODEL-FALLBACK-1/2: if self.models holds more than one model, then when
        the current one exhausts its own retries (RETRY-1 inside
        _chat_json_one_model) we move to the next model and give it a full set of
        attempts from scratch, instead of raising immediately. Only when every
        model in the pass is exhausted is the last error raised, exactly as a
        single model did before.

        LLMUnavailable also lets the next model try: six failures in a row on ONE
        model can mean that model is throttled (Groq/Gemini count TPM/TPD per
        model, not per server), and a sibling may answer fine. It propagates only
        if it was the last error of the LAST model, so the external contract does
        not change.

        MODEL-FALLBACK-2: every call used to restart the scan at models[0] even
        if that model had been declared dead a minute earlier with a daily quota
        — 6-7 minutes wasted per call before reaching a working one. The starting
        position now comes from _ServerCaps, shared by all clients on this
        base_url. If the whole pass fails anyway, the position is reset to 0 so
        the NEXT external call does not stick to the last model of a failed pass.

        RATE-ROT: a pass that ends in 429 on every model says the limit is per
        KEY, and only then is sleeping worth it — see the loop below.
        """
        models = getattr(self, "models", None) or [getattr(self, "model", None)]
        n = len(models)
        base_url = getattr(self, "base_url", "")
        on_retry = getattr(self, "on_retry", None)
        last_exc = None
        # RATE-ROT: how many passes over the list to make in total. A second
        # pass happens only if the previous one was 429 all the way round —
        # i.e. the limit looks per-KEY and switching models cannot dodge it.
        # Then, and only then, sleeping is worth it.
        cycles = max(0, int(getattr(self, "rate_limit_cycles", 0))) + 1
        for cycle in range(cycles):
            start = _ServerCaps.get_model_index(base_url) % n
            rl_hits = []            # [(model, wait_s)] — 429s in this pass
            only_rate_limits = True  # False once a failure of another kind
            pass_t0 = time.monotonic()
            for step in range(n):
                idx = (start + step) % n
                model = models[idx]
                self.model = model
                try:
                    result = self._chat_json_one_model(
                        system, user, temperature=temperature,
                        max_tokens=max_tokens)
                    _ServerCaps.set_model_index(base_url, idx)
                    return result
                except Exception as e:
                    last_exc = e
                    if isinstance(e, RateLimited):
                        rl_hits.append((model, getattr(e, "wait_s", None)))
                    else:
                        only_rate_limits = False
                    if step < n - 1:
                        next_model = models[(idx + 1) % n]
                        msg = (f"model {model!r} failed ({type(e).__name__}: {e}), "
                              f"switching to next model in list: {next_model!r}")
                        if on_retry:
                            on_retry(msg)
                        _dbg(f"chat_json(): {msg}")
                        continue
                    # A full pass, and nothing answered.
                    if (n > 1 and only_rate_limits and rl_hits
                            and cycle < cycles - 1):
                        # Every one returned 429 — that answers "per key or
                        # per model": per key. Sleep the MINIMUM asked (the
                        # first model to free up will do), capped by
                        # max_retry_after_sec.
                        waits = [w for _, w in rl_hits if w]
                        wait_s = (min(waits) if waits
                                  else float(getattr(self,
                                                     "error_retry_wait_sec", 60)))
                        ceiling = float(getattr(self, "max_retry_after_sec", 180))
                        wait_s = max(0.5, min(wait_s, ceiling))
                        elapsed = time.monotonic() - pass_t0
                        seq = ", ".join(
                            f"{m}({w:.1f}s)" if w else f"{m}(?)"
                            for m, w in rl_hits)
                        msg = (f"RATE-ROT: круг {cycle + 1}/{cycles} — все "
                               f"{n} модели вернули 429 за {elapsed:.1f} сек "
                               f"[{seq}]; похоже, лимит общий на КЛЮЧ, а не "
                               f"на модель. Жду {wait_s:.1f} сек и начинаю "
                               f"круг заново с {models[start]!r}")
                        if on_retry:
                            on_retry(msg)
                        _dbg(f"chat_json(): {msg}")
                        time.sleep(wait_s)
                        break       # → next pass of the outer loop
                    _ServerCaps.set_model_index(base_url, 0)
                    raise
        raise last_exc   # unreachable (models is never empty), but explicit

    def _chat_json_one_model(self, system: str, user: str,
                             temperature: float = 0.4,
                             max_tokens: int = 400) -> dict:
        """
        Like chat(), but parses the answer as JSON (stripping ``` fences if
        present). Works on self.model as it is at call time; switching between
        models is chat_json()'s job, this function knows nothing about the list.

        RETRY-1: on failure makes self.retries more attempts. Anything that might
        pass on a second try is retried — a timeout, broken JSON, a transient
        server error. LLMUnavailable is NOT retried: the breaker has already
        decided the server is gone (though chat_json() above still gives the next
        model a chance).

        The cost of a retry on a CPU laptop is another wall-clock timeout. That is
        a deliberate trade: a skipped call silently becomes a stub answer and is
        indistinguishable from a real one in the log.
        """
        # getattr, not self.retries: stub subclasses in tests and stub mode
        # are built without __init__, and a hard attribute access would
        # break them. Without it, the old behaviour: a single attempt.
        retries = getattr(self, "retries", 0)
        attempts = retries + 1
        last = None
        for attempt in range(1, attempts + 1):
            try:
                # EOS-2 (replaces EOS-1). EOS-1 LOWERED temperature on
                # retry and thereby guaranteed the same failure: if EOS is
                # already the argmax of the first token, a lower temperature
                # makes it even more certain. The log shows it: three
                # attempts at temp 0.8 -> 0.48 -> 0.288, all eval_count=1,
                # the last two in 0.8s (same prefix in KV cache, same argmax).
                #
                # Now the opposite: raise the temperature and perturb the
                # prompt tail. Changing the last prompt token shifts both the
                # distribution and cache reuse, so the attempt is no longer
                # an exact copy of the previous one.
                attempt_temp = temperature
                attempt_user = user
                if attempt > 1:
                    attempt_temp = min(temperature + 0.15 * (attempt - 1), 1.0)
                    attempt_user = (
                        user + "\n\n(Ответь ТОЛЬКО объектом JSON, "
                        "начиная с символа '{'. Попытка "
                        f"{attempt}.)"
                    )
                # EOS-2/STUB: stubs in tests and stub mode replace chat()
                # with an old-signature function that has no json_mode.
                # Passing the kwarg blindly would raise TypeError inside an
                # except branch, where it would read as "the model did not
                # answer". Inspect the signature instead of catching
                # TypeError, so a real TypeError from chat() is not swallowed.
                if _chat_takes_json_mode(self.chat):
                    text = self.chat(system, attempt_user,
                                     temperature=attempt_temp,
                                     max_tokens=max_tokens, json_mode=True)
                else:
                    text = self.chat(system, attempt_user,
                                     temperature=attempt_temp,
                                     max_tokens=max_tokens)
                _dbg(f"chat_json() attempt {attempt}: text len={len(text)}, "
                     f"temperature={attempt_temp:.3f}")
                cleaned = strip_json_fence(text)
                _dbg(f"chat_json() after strip_json_fence len={len(cleaned)}, "
                     f"repr(first 300): {cleaned[:300]!r}")
                if not cleaned:
                    raise json.JSONDecodeError(
                        "chat_json got empty string after stripping — "
                        "model sampled EOS as first token or spent its "
                        "whole budget on hidden reasoning (0-1 completion "
                        "tokens; see LLM_DEBUG for finish_reason/usage "
                        "[openai-format providers] or prompt_eval_count/"
                        "eval_count [ollama] for context-overflow vs "
                        "sampling-glitch diagnosis)",
                        "", 0)
                result = json.loads(cleaned)
            except LLMUnavailable:
                raise
            except RateLimited:
                # RATE-ROT: retrying the SAME model that just said 429 is
                # pointless — RETRY-1 changes temperature and prompt tail,
                # the limit does not move. Hand it up; the next model waits.
                raise
            except Exception as e:
                last = e
                if attempt < attempts:
                    on_retry = getattr(self, "on_retry", None)
                    # HTTP-RETRY: pause before retry. This used to apply
                    # ONLY to HTTPError inside chat() (402/5xx), while
                    # broken JSON / EOS-as-first-token went through THIS
                    # loop and retried instantly, even though
                    # error_retries/error_retry_wait_sec already implied a
                    # pause on ANY failure. getattr for the same reason as
                    # `retries` above: stubs without __init__ must not break
                    # here, just not pause.
                    error_retries = getattr(self, "error_retries", 0)
                    wait_s = getattr(self, "error_retry_wait_sec", 60)
                    if error_retries > 0:
                        msg = (f"LLM call failed ({type(e).__name__}: {e}), "
                               f"жду {wait_s} сек и повторяю (попытка "
                               f"{attempt}/{retries})")
                        if on_retry:
                            on_retry(msg)
                        _dbg(f"chat_json(): {msg}")
                        time.sleep(wait_s)
                    elif on_retry:
                        kind = "timeout" if _is_timeout(e) else type(e).__name__
                        on_retry(f"LLM call failed ({kind}: {e}), "
                                 f"retry {attempt}/{retries}")
                    continue
                # FIX-17: a dropped connection and broken JSON are counted
                # differently. Broken JSON means the server is alive and
                # answering — the model just missed the format, which is
                # routine. RETRY-1 added timeouts to that: a slow server is
                # alive too (see _is_timeout).
                if not isinstance(e, (json.JSONDecodeError, ValueError)):
                    _note_failure(e)
                raise
            _Breaker.failures = 0
            return result
        raise last   # unreachable, but explicit
