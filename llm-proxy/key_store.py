"""Resolve an upstream host/scheme from a client's API key via SQLite.

This reads the same SQLite file `api_manager.py` writes (table `entries`,
columns name/host/base_url/api_key) — it never writes to it. Everything
here fails toward "not found": a missing DB file, a DB that can't be
opened, or a key that isn't in it all come back as None so the caller
falls back to config.toml's static upstream. Nothing here raises for
those cases; that's what makes the support "transparent" — a proxy
with no DB configured behaves exactly as it did before this existed.
"""

from __future__ import annotations

import os
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

DB_ENV_VAR = "API_MANAGER_DB"

_BEARER_RE = re.compile(r"^\s*Bearer\s+(.+)$", re.IGNORECASE)


@dataclass(frozen=True)
class UpstreamEntry:
    name: str
    host: str
    scheme: str
    api_key: str


def default_db_path() -> Path:
    """Same default api_manager.py uses, so both tools agree with no config."""
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "api-manager" / "entries.db"


def resolve_db_path(explicit: str | None = None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(DB_ENV_VAR)
    if env:
        return Path(env)
    return default_db_path()


def db_available(db_path: Path) -> bool:
    return db_path.is_file()


def extract_api_key(headers) -> str | None:
    """Pull a bearer/api key out of request headers.

    Accepts anything that yields (name, value) pairs via .items() (an
    http.client.HTTPMessage, or a plain dict). Checks Authorization first
    (stripping a "Bearer " prefix if present), then api-key / x-api-key.
    """
    lowered = {k.lower(): v for k, v in headers.items()}
    auth = lowered.get("authorization")
    if auth:
        m = _BEARER_RE.match(auth)
        return (m.group(1) if m else auth).strip()
    for name in ("api-key", "x-api-key"):
        value = lowered.get(name)
        if value:
            return value.strip()
    return None


def mask_key(api_key: str | None) -> str:
    """Shorten a key for logging without ever printing it in full."""
    if not api_key:
        return "<none>"
    if len(api_key) <= 8:
        return "*" * len(api_key)
    return f"{api_key[:6]}...{api_key[-4:]}"


def lookup(api_key: str, db_path: Path) -> UpstreamEntry | None:
    """Look an api_key up in the sqlite store.

    Returns None on any kind of miss: no key given, no file at db_path, a
    file that isn't a readable sqlite database, or no matching row.
    """
    if not api_key or not db_available(db_path):
        return None
    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None
    try:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT name, host, base_url, api_key FROM entries WHERE api_key = ?",
            (api_key,),
        ).fetchone()
    except sqlite3.DatabaseError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    scheme = urlsplit(row["base_url"]).scheme or "https"
    return UpstreamEntry(name=row["name"], host=row["host"], scheme=scheme, api_key=row["api_key"])
