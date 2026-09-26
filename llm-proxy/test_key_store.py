"""Unit tests for key_store.py — the sqlite-backed per-key upstream lookup.

Uses the real api_manager.ApiStore to write the database, so this exercises
the same file main.py reads against, not a hand-rolled schema.

    python3 -m pytest test_key_store.py
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import api_manager as am
import key_store


@pytest.fixture
def db_with_one_entry(tmp_path):
    db_path = tmp_path / "entries.db"
    with am.ApiStore(db_path) as store:
        store.add(
            "openrouter",
            api_key="sk-or-v1-abcdefghijklmnop",
            base_url="https://openrouter.ai/api/v1",
            host="openrouter.ai",
        )
    return db_path


# --------------------------------------------------------------------------- extract_api_key


def test_extract_bearer_token():
    headers = {"Authorization": "Bearer sk-abc123", "Content-Type": "application/json"}
    assert key_store.extract_api_key(headers) == "sk-abc123"


def test_extract_bearer_is_case_insensitive_on_header_name_and_scheme():
    headers = {"authorization": "bearer sk-abc123"}
    assert key_store.extract_api_key(headers) == "sk-abc123"


def test_extract_authorization_without_bearer_prefix_is_used_as_is():
    headers = {"Authorization": "sk-abc123"}
    assert key_store.extract_api_key(headers) == "sk-abc123"


def test_extract_falls_back_to_api_key_header():
    headers = {"api-key": "sk-abc123"}
    assert key_store.extract_api_key(headers) == "sk-abc123"


def test_extract_falls_back_to_x_api_key_header():
    headers = {"x-api-key": "sk-abc123"}
    assert key_store.extract_api_key(headers) == "sk-abc123"


def test_extract_returns_none_when_no_key_present():
    assert key_store.extract_api_key({"Content-Type": "application/json"}) is None


def test_extract_accepts_http_message_like_items():
    # http.client.HTTPMessage (what self.headers actually is in the proxy)
    # yields duplicate-tolerant (name, value) pairs via .items() same as a
    # dict does for our purposes here.
    class FakeHeaders:
        def items(self):
            return [("Authorization", "Bearer sk-xyz")]

    assert key_store.extract_api_key(FakeHeaders()) == "sk-xyz"


# --------------------------------------------------------------------------- db_available / resolve_db_path


def test_db_available_false_for_missing_file(tmp_path):
    assert key_store.db_available(tmp_path / "nope.db") is False


def test_db_available_true_for_existing_file(db_with_one_entry):
    assert key_store.db_available(db_with_one_entry) is True


def test_resolve_db_path_prefers_explicit_argument(monkeypatch, tmp_path):
    monkeypatch.setenv(key_store.DB_ENV_VAR, str(tmp_path / "env.db"))
    explicit = tmp_path / "explicit.db"
    assert key_store.resolve_db_path(str(explicit)) == explicit


def test_resolve_db_path_falls_back_to_env_var(monkeypatch, tmp_path):
    env_path = tmp_path / "env.db"
    monkeypatch.setenv(key_store.DB_ENV_VAR, str(env_path))
    assert key_store.resolve_db_path(None) == env_path


def test_resolve_db_path_falls_back_to_default_when_nothing_set(monkeypatch):
    monkeypatch.delenv(key_store.DB_ENV_VAR, raising=False)
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    assert key_store.resolve_db_path(None) == key_store.default_db_path()


# --------------------------------------------------------------------------- lookup


def test_lookup_returns_entry_for_a_matching_key(db_with_one_entry):
    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    assert entry is not None
    assert entry.name == "openrouter"
    assert entry.host == "openrouter.ai"
    assert entry.scheme == "https"
    assert entry.api_key == "sk-or-v1-abcdefghijklmnop"


def test_lookup_returns_none_for_a_key_not_in_the_db(db_with_one_entry):
    assert key_store.lookup("sk-not-a-real-key", db_with_one_entry) is None


def test_lookup_returns_none_when_db_file_does_not_exist(tmp_path):
    assert key_store.lookup("sk-anything", tmp_path / "missing.db") is None


def test_lookup_returns_none_for_empty_api_key(db_with_one_entry):
    assert key_store.lookup("", db_with_one_entry) is None
    assert key_store.lookup(None, db_with_one_entry) is None


def test_lookup_returns_none_for_a_file_that_is_not_a_sqlite_db(tmp_path):
    bogus = tmp_path / "entries.db"
    bogus.write_bytes(b"this is not a sqlite file at all")
    assert key_store.lookup("sk-anything", bogus) is None


def test_lookup_derives_scheme_from_base_url_for_http_upstreams(tmp_path):
    db_path = tmp_path / "entries.db"
    with am.ApiStore(db_path) as store:
        store.add("local-ollama", api_key="unused", base_url="http://localhost:11434", host="localhost:11434")
    entry = key_store.lookup("unused", db_path)
    assert entry.scheme == "http"
    assert entry.host == "localhost:11434"


def test_lookup_does_not_write_to_the_database(db_with_one_entry):
    before = db_with_one_entry.stat().st_mtime_ns
    key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    key_store.lookup("sk-not-a-real-key", db_with_one_entry)
    after = db_with_one_entry.stat().st_mtime_ns
    assert before == after


def test_lookup_works_even_while_another_connection_holds_the_db_open(db_with_one_entry):
    # api_manager keeps its own connection open only for the duration of a
    # `with ApiStore(...)`, but the proxy's lookup must never need a write
    # lock -- confirm a plain read-only sqlite3 connection can coexist.
    keep_open = sqlite3.connect(str(db_with_one_entry))
    try:
        entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
        assert entry is not None
    finally:
        keep_open.close()


# --------------------------------------------------------------------------- lookup: the entry_keys failover pool


def test_lookup_returns_empty_pool_when_nothing_was_ever_added(db_with_one_entry):
    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    assert entry.api_keys == ()


def test_lookup_returns_pool_in_the_order_keys_were_added(db_with_one_entry):
    with am.ApiStore(db_with_one_entry) as store:
        store.add_key("openrouter", "sk-proxy-c")
        store.add_key("openrouter", "sk-proxy-a")
        store.add_key("openrouter", "sk-proxy-b")
    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    assert entry.api_keys == ("sk-proxy-c", "sk-proxy-a", "sk-proxy-b")


def test_lookup_pool_is_scoped_to_the_matched_entry_only(tmp_path):
    db_path = tmp_path / "entries.db"
    with am.ApiStore(db_path) as store:
        store.add("openrouter", api_key="sk-client-or", base_url="https://openrouter.ai/api/v1", host="openrouter.ai")
        store.add("openai", api_key="sk-client-oa", base_url="https://api.openai.com/v1", host="api.openai.com")
        store.add_key("openrouter", "sk-or-pool-1")
        store.add_key("openrouter", "sk-or-pool-2")
        # openai's pool is left empty on purpose.

    or_entry = key_store.lookup("sk-client-or", db_path)
    oa_entry = key_store.lookup("sk-client-oa", db_path)
    assert or_entry.api_keys == ("sk-or-pool-1", "sk-or-pool-2")
    assert oa_entry.api_keys == ()


def test_lookup_pool_survives_removing_one_key(db_with_one_entry):
    with am.ApiStore(db_with_one_entry) as store:
        store.add_key("openrouter", "sk-a")
        store.add_key("openrouter", "sk-b")
        store.remove_key("openrouter", "sk-a")
    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    assert entry.api_keys == ("sk-b",)


def test_lookup_pool_empty_again_after_clear(db_with_one_entry):
    with am.ApiStore(db_with_one_entry) as store:
        store.add_key("openrouter", "sk-a")
        store.clear_keys("openrouter")
    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_with_one_entry)
    assert entry.api_keys == ()


def test_lookup_tolerates_a_db_with_no_entry_keys_table(tmp_path):
    # A database written by a version of api_manager.py from before the
    # entry_keys table existed. lookup() must still resolve the host --
    # it just reports an empty pool instead of erroring.
    db_path = tmp_path / "entries.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        CREATE TABLE entries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            host TEXT NOT NULL,
            base_url TEXT NOT NULL,
            api_key TEXT NOT NULL,
            note TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
    """)
    conn.execute(
        "INSERT INTO entries (name, host, base_url, api_key, note, created_at, updated_at) "
        "VALUES ('openrouter', 'openrouter.ai', 'https://openrouter.ai/api/v1', "
        "'sk-or-v1-abcdefghijklmnop', NULL, '2020-01-01T00:00:00Z', '2020-01-01T00:00:00Z')"
    )
    conn.commit()
    conn.close()

    entry = key_store.lookup("sk-or-v1-abcdefghijklmnop", db_path)
    assert entry is not None
    assert entry.host == "openrouter.ai"
    assert entry.api_keys == ()
