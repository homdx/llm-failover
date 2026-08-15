"""
test_llm_parity.py — тесты на то, что было ДОБАВЛЕНО и на то, что было
НЕ ПОКРЫТО в старом наборе. python3 test_llm_parity.py

Две группы:

  * PARITY-1/2/3 — клиент, собранный для пула, обязан быть настроен ТОЧНО
    так же, как одиночный клиент из того же конфига. Именно здесь была
    дыра: пул строил LLMClient своими руками, и каждая новая настройка
    [api] доезжала до одиночного режима и молча не доезжала до пула.
    Старый test_llm_pool.py ловил это по одной настройке за раз
    (error_retries, потом max_retry_after_sec — двумя заплатками задним
    числом) и не поймал rotate_on_429/rate_limit_cycles вообще.
    Здесь проверяется не список известных полей, а СОВПАДЕНИЕ целиком.

  * RATE-ROT — механизм ротации моделей на 429. В старом наборе тестов на
    него нет НИ ОДНОГО: он появился в llm_client.py позже, чем писался
    test_model_fallback.py, и остался непокрытым.
"""

import configparser
import json
import unittest
import urllib.error

import llm_client
import llm_pool
from llm_client import LLMClient, LLMUnavailable, RateLimited, _ServerCaps
from llm_pool import PooledClient


def http_error(code, body=b"", headers=None):
    return urllib.error.HTTPError(
        "http://x/v1/chat/completions", code, "err", headers or {},
        __import__("io").BytesIO(body))


CFG_TEXT = """
[api]
active = remote
verify_ssl = false
timeout_seconds = 77
retries = 3
error_retries = 5
error_retry_wait_sec = 11
max_retry_after_sec = 222
rotate_on_429 = true
rate_limit_cycles = 2
max_failover = 1

[api_remote]
base_url = https://api.one/v1
api_key = k1
model = a, b, b, c
api_format = openai
num_ctx = 4096
think = false

[api_remote2]
base_url = https://api.two/v1
api_key = k2
model = x, y
api_format = openai
num_ctx = 8192
think = false
"""


def cfg(pool=None):
    c = configparser.ConfigParser()
    c.read_string(CFG_TEXT)
    if pool:
        c.set("api", "pool", pool)
    return c


# Все настроечные атрибуты клиента. Список намеренно ПОЛНЫЙ, а не
# «интересные поля»: смысл теста в том, чтобы новое поле, добавленное в
# LLMClient и забытое в пути пула, роняло тест само, без правки теста.
SETTINGS = ("base_url api_key api_format num_ctx think timeout retries "
            "error_retries error_retry_wait_sec max_retry_after_sec "
            "rotate_on_429 rate_limit_cycles models model").split()


class TestPoolSingleParity(unittest.TestCase):
    """PARITY-1: одна секция — два пути сборки — один результат."""

    def setUp(self):
        _ServerCaps.reset()
        llm_pool.reset_shared_pool()

    def test_every_setting_matches_single_client(self):
        c = cfg()
        single = LLMClient.from_config(c)                     # путь [api].active
        pooled = llm_pool._client_for_section(c, "api_remote")  # путь пула
        for f in SETTINGS:
            self.assertEqual(getattr(single, f), getattr(pooled, f),
                             f"настройка {f!r} расходится между одиночным "
                             f"клиентом и клиентом пула")

    def test_verify_ssl_reaches_pool_clients_too(self):
        """verify_ssl живёт в [api], а не в секции сервера — легко потерять."""
        c = cfg()
        self.assertIsNotNone(LLMClient.from_config(c)._ssl_context)
        self.assertIsNotNone(llm_pool._client_for_section(c, "api_remote")
                             ._ssl_context)

    def test_rotate_flags_reach_pool_clients(self):
        """
        КОНКРЕТНО та дыра, из-за которой всё затевалось: в режиме пула 429
        засыпал внутри chat() и обходил ротацию моделей, ради которой
        список моделей и заводился.
        """
        pc = llm_pool.build_client(cfg(pool="api_remote, api_remote2"))
        self.assertIsInstance(pc, PooledClient)
        for ep in pc.pool._eps:
            self.assertTrue(ep.client.rotate_on_429)
            self.assertEqual(ep.client.rate_limit_cycles, 2)
            self.assertEqual(ep.client.max_retry_after_sec, 222)
            self.assertEqual(ep.client.error_retries, 5)

    def test_per_section_values_are_not_cross_contaminated(self):
        """
        КОНТРФАКТ к предыдущему: общие настройки [api] обязаны совпадать, а
        СОБСТВЕННЫЕ настройки секции — обязаны различаться. Иначе «паритет»
        достигнут тем, что все клиенты стали одинаковыми.
        """
        pc = llm_pool.build_client(cfg(pool="api_remote, api_remote2"))
        a, b = (ep.client for ep in pc.pool._eps)
        self.assertEqual(a.num_ctx, 4096)
        self.assertEqual(b.num_ctx, 8192)
        self.assertEqual(a.models, ["a", "b", "c"])   # дедуп сработал и тут
        self.assertEqual(b.models, ["x", "y"])
        self.assertNotEqual(a.base_url, b.base_url)

    def test_on_retry_propagates_to_underlying_clients(self):
        """
        PARITY-2: в режиме пула молча пропадали ВСЕ сообщения самого
        LLMClient (429 с паузой, RATE-ROT, HTTP-RETRY) — on_retry ставился
        только на обёртку. При одном сервере они видны; расхождение в
        наблюдаемости при одном и том же конфиге.
        """
        seen = []
        # именованная функция, а не seen.append: связанный метод — НОВЫЙ
        # объект на каждое обращение, assertIs на нём падает всегда.
        def log(msg):
            seen.append(msg)
        pc = llm_pool.build_client(cfg(pool="api_remote, api_remote2"),
                                   on_retry=log)
        for ep in pc.pool._eps:
            self.assertIs(ep.client.on_retry, log)
        one = llm_pool.build_client(cfg(), on_retry=log)
        self.assertIs(one.on_retry, log)   # прежний путь не сломан

    def test_shared_client_also_propagates_on_retry(self):
        seen = []
        def log(msg):
            seen.append(msg)
        sc = llm_pool.shared_client(cfg(pool="api_remote, api_remote2"),
                                    on_retry=log)
        for ep in sc.pool._eps:
            self.assertIs(ep.client.on_retry, log)

    def test_pooled_client_exposes_models(self):
        """PARITY-3: чтение client.models падало AttributeError'ом ровно
        при переходе с одной секции на пул."""
        pc = llm_pool.build_client(cfg(pool="api_remote, api_remote2"))
        self.assertEqual(pc.models, ["a", "b", "c"])
        self.assertEqual(pc.model, "a")
        self.assertEqual(pc.base_url, "https://api.one/v1")


class TestFromConfigBackwardCompat(unittest.TestCase):
    """api_section добавлен как НЕОБЯЗАТЕЛЬНЫЙ — старые вызовы не трогаем."""

    def test_positional_section_is_still_ignored_as_before(self):
        c = cfg()
        self.assertEqual(LLMClient.from_config(c, "player").base_url,
                         "https://api.one/v1")

    def test_no_args_still_follows_api_active(self):
        self.assertEqual(LLMClient.from_config(cfg()).base_url,
                         "https://api.one/v1")

    def test_explicit_api_section_overrides_active(self):
        self.assertEqual(
            LLMClient.from_config(cfg(), api_section="api_remote2").base_url,
            "https://api.two/v1")


class RotBase(unittest.TestCase):
    def setUp(self):
        _ServerCaps.reset()
        LLMClient.reset_breaker()
        self.slept = []
        self._real_sleep = llm_client.time.sleep
        llm_client.time.sleep = self.slept.append

    def tearDown(self):
        llm_client.time.sleep = self._real_sleep
        _ServerCaps.reset()
        LLMClient.reset_breaker()

    @staticmethod
    def client(model, **kw):
        kw.setdefault("rotate_on_429", True)
        kw.setdefault("rate_limit_cycles", 0)
        kw.setdefault("retries", 2)
        kw.setdefault("error_retries", 0)
        return LLMClient(base_url="http://x", api_key="k", model=model,
                         api_format="openai", **kw)


class TestRateRot(RotBase):
    """RATE-ROT: 429 на одной модели не усыпляет вызов, а отдаёт ход соседней."""

    def test_429_switches_to_next_model_without_sleeping(self):
        c = self.client("m1, m2")
        seen = []

        def fake_open(req, **kw):
            seen.append(json.loads(req.data)["model"])
            if seen[-1] == "m1":
                raise http_error(429, b'{"error":"try again in 30s"}')
            return FakeResp({"choices": [{"message": {"content": '{"ok":1}'},
                                          "finish_reason": "stop"}]})

        llm_client.urllib.request.urlopen = fake_open
        try:
            self.assertEqual(c.chat_json("s", "u"), {"ok": 1})
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(seen, ["m1", "m2"])
        self.assertEqual(self.slept, [],
                         "на 429 при списке моделей спать нельзя — соседняя "
                         "модель отвечает прямо сейчас")

    def test_429_does_not_burn_retries_on_the_same_model(self):
        """
        Повторять ТУ ЖЕ модель после 429 бессмысленно: RETRY-1 меняет
        температуру и хвост промпта, а лимит от этого не двигается.
        retries=2 не должны превратиться в 3 вызова m1.
        """
        c = self.client("m1, m2")
        seen = []

        def fake_open(req, **kw):
            seen.append(json.loads(req.data)["model"])
            raise http_error(429, b"rate limited")

        llm_client.urllib.request.urlopen = fake_open
        try:
            with self.assertRaises(RateLimited):
                c.chat_json("s", "u")
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(seen, ["m1", "m2"])

    def test_single_model_still_sleeps_as_before(self):
        """
        КОНТРФАКТ: с ОДНОЙ моделью переключаться некуда, и прежнее
        поведение (подождать Retry-After и повторить тот же запрос)
        обязано сохраниться — иначе rotate_on_429 молча сломал бы всех,
        у кого в конфиге одна модель.
        """
        c = self.client("m1")
        calls = []

        def fake_open(req, **kw):
            calls.append(1)
            if len(calls) == 1:
                raise http_error(429, b'{"error":"try again in 2s"}')
            return FakeResp({"choices": [{"message": {"content": '{"ok":1}'},
                                          "finish_reason": "stop"}]})

        llm_client.urllib.request.urlopen = fake_open
        try:
            self.assertEqual(c.chat_json("s", "u"), {"ok": 1})
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(self.slept, [2.0])

    def test_rotate_off_restores_old_behaviour(self):
        """rotate_on_429=false в конфиге — прежнее поведение одной строкой."""
        c = self.client("m1, m2", rotate_on_429=False)
        calls = []

        def fake_open(req, **kw):
            calls.append(json.loads(req.data)["model"])
            if len(calls) == 1:
                raise http_error(429, b'{"error":"try again in 5s"}')
            return FakeResp({"choices": [{"message": {"content": '{"ok":1}'},
                                          "finish_reason": "stop"}]})

        llm_client.urllib.request.urlopen = fake_open
        try:
            self.assertEqual(c.chat_json("s", "u"), {"ok": 1})
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(self.slept, [5.0])
        self.assertEqual(calls, ["m1", "m1"], "без ротации повторяется та же модель")

    def test_whole_cycle_of_429_sleeps_the_minimum_then_retries(self):
        """
        Весь круг лёг в 429 — значит лимит общий на КЛЮЧ, и только тогда
        имеет смысл спать. Ждём МИНИМУМ из запрошенного: первая
        освободившаяся модель нас устроит.
        """
        c = self.client("m1, m2", rate_limit_cycles=1)
        seen = []
        waits = {"m1": b'{"error":"retry in 40s"}',
                 "m2": b'{"error":"retry in 9s"}'}

        def fake_open(req, **kw):
            m = json.loads(req.data)["model"]
            seen.append(m)
            if len(seen) <= 2:
                raise http_error(429, waits[m])
            return FakeResp({"choices": [{"message": {"content": '{"ok":1}'},
                                          "finish_reason": "stop"}]})

        llm_client.urllib.request.urlopen = fake_open
        try:
            self.assertEqual(c.chat_json("s", "u"), {"ok": 1})
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(self.slept, [9.0], "спать надо по самой быстрой, не по самой медленной")
        self.assertEqual(seen[:3], ["m1", "m2", "m1"])

    def test_sleep_between_cycles_is_capped(self):
        """RATE-3-потолок действует и здесь: дневной лимит не должен
        блокировать вызов на полчаса."""
        c = self.client("m1, m2", rate_limit_cycles=1, max_retry_after_sec=30)
        seen = []

        def fake_open(req, **kw):
            seen.append(json.loads(req.data)["model"])
            if len(seen) <= 2:
                raise http_error(429, b'{"error":"retry in 1800s"}')
            return FakeResp({"choices": [{"message": {"content": '{"ok":1}'},
                                          "finish_reason": "stop"}]})

        llm_client.urllib.request.urlopen = fake_open
        try:
            c.chat_json("s", "u")
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(self.slept, [30.0])

    def test_non_429_failure_in_the_cycle_prevents_the_sleep(self):
        """
        КОНТРФАКТ: спать между кругами можно, ТОЛЬКО если круг лёг целиком
        в 429. Если хоть одна модель упала по другой причине — вывод «лимит
        общий на ключ» не следует, и сон был бы ложным.
        """
        c = self.client("m1, m2", rate_limit_cycles=1, retries=0)
        seen = []

        def fake_open(req, **kw):
            m = json.loads(req.data)["model"]
            seen.append(m)
            raise http_error(429 if m == "m1" else 500, b"boom")

        llm_client.urllib.request.urlopen = fake_open
        try:
            with self.assertRaises(Exception):
                c.chat_json("s", "u")
        finally:
            llm_client.urllib.request.urlopen = _REAL_URLOPEN
        self.assertEqual(self.slept, [])

    def test_rate_limited_is_a_runtime_error(self):
        """Вызывающий код ловит RuntimeError — RateLimited не должна из
        него выпасть и стать необработанной."""
        self.assertTrue(issubclass(RateLimited, RuntimeError))
        e = RateLimited("m", model="m1", wait_s=3.0, url="u")
        self.assertEqual((e.model, e.wait_s), ("m1", 3.0))


class TestRateRotInPool(RotBase):
    def test_rate_limited_endpoint_fails_over_to_another_server(self):
        """
        Сцепка двух механизмов: модели на сервере кончились по лимиту →
        наверх уходит RateLimited → пул обязан считать это обычным сбоем
        эндпоинта и уйти на соседний сервер, а не уронить вызов.
        """
        class Bad:
            base_url, model, models, on_retry = "http://bad", "m", ["m"], None
            def chat_json(self, *a, **k):
                raise RateLimited("429", model="m", wait_s=60.0)

        class Good:
            base_url, model, models, on_retry = "http://good", "g", ["g"], None
            def chat_json(self, *a, **k):
                return {"ok": "good"}

        pool = llm_pool.LLMPool([Bad(), Good()], ["bad", "good"])
        self.assertEqual(PooledClient(pool, max_failover=1).chat_json("s", "u"),
                         {"ok": "good"})

    def test_llm_unavailable_still_not_failed_over(self):
        """Регресс-щит: предохранитель глобален, другой сервер его не лечит."""
        class Dead:
            base_url, model, models, on_retry = "http://d", "m", ["m"], None
            def chat_json(self, *a, **k):
                raise LLMUnavailable("dead")

        pool = llm_pool.LLMPool([Dead(), Dead()], ["a", "b"])
        with self.assertRaises(LLMUnavailable):
            PooledClient(pool, max_failover=1).chat_json("s", "u")


class FakeResp:
    def __init__(self, payload):
        self._b = json.dumps(payload).encode()
    def read(self):
        return self._b
    def __enter__(self):
        return self
    def __exit__(self, *a):
        return False


_REAL_URLOPEN = llm_client.urllib.request.urlopen


if __name__ == "__main__":
    unittest.main(verbosity=2)
