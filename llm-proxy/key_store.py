#!/usr/bin/env python3
"""Lightweight, read-only lookup against the api_manager.py sqlite store.

This module is deliberately independent of api_manager.py: the proxy
(main.py) only ever needs to READ, never write, so it talks to the
database directly through a plain read-only connection instead of
importing the full CLI module. It shares the same environment variable,
default path, and table layout as api_manager.py so the two point at the
same file with zero extra configuration.

Public surface:
    DB_ENV_VAR          -- env var name that overrides the default db path
    default_db_path()   -- ~/.config/api-manager/entries.db (or $XDG_CONFIG_HOME)
    resolve_db_path()   -- explicit path > env var > default
    db_available()      -- cheap existence check, no connection opened
    extract_api_key()   -- pull a client-sent key out of request headers
    mask_key()          -- redact a key for logging
    lookup()            -- api_key -> KeyEntry | None
"""

from __future__ import annotations

import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

# Same value as api_manager.DB_ENV_VAR -- intentionally duplicated (not
# imported) so this module has zero dependency on api_manager.py. Keep
# the two in sync if this ever changes.
DB_ENV_VAR = "API_MANAGER_DB"


@dataclass(frozen=True)
class KeyEntry:
    """One row from `entries`, plus its `entry_keys` failover pool."""

    name: str
    host: str
    scheme: str
    base_url: str
    api_key: str
    api_keys: tuple[str, ...]


# --------------------------------------------------------------------------- path resolution


def default_db_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "api-manager" / "entries.db"


def resolve_db_path(explicit: str | Path | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(DB_ENV_VAR)
    if env:
        return Path(env)
    return default_db_path()


def db_available(db_path: Path) -> bool:
    """True if a file exists at db_path. Does not open a connection."""
    try:
        return Path(db_path).is_file()
    except OSError:
        return False


# --------------------------------------------------------------------------- key helpers


def mask_key(api_key: str | None) -> str:
    """Show a short, unambiguous hint without exposing the secret."""
    if not api_key:
        return ""
    if len(api_key) <= 10:
        return "*" * len(api_key)
    return f"{api_key[:6]}{'*' * 4}{api_key[-4:]}"


def extract_api_key(headers) -> str | None:
    """Pull a client-sent API key out of request headers.

    Accepts anything exposing `.items()` -- a plain dict, an
    `http.client.HTTPMessage`, etc. Checks, in order:
      1. Authorization: Bearer <key>   (a value with no "Bearer" prefix
         is used as-is)
      2. api-key: <key>
      3. x-api-key: <key>
    Header names and the "Bearer" scheme are matched case-insensitively.
    The first header of each name wins if duplicates are present.
    Returns None if no candidate header is present or all are empty.
    """
    lowered: dict[str, str] = {}
    for name, value in headers.items():
        key = name.lower()
        if key not in lowered:
            lowered[key] = value

    auth = lowered.get("authorization")
    if auth:
        auth = auth.strip()
        if auth[:7].lower() == "bearer ":
            token = auth[7:].strip()
            if token:
                return token
        elif auth:
            return auth

    for header_name in ("api-key", "x-api-key"):
        value = lowered.get(header_name)
        if value:
            value = value.strip()
            if value:
                return value

    return None


def _scheme_from_base_url(base_url: str) -> str:
    return urlsplit(base_url).scheme or "https"


# --------------------------------------------------------------------------- lookup


def lookup(api_key: str | None, db_path: Path) -> KeyEntry | None:
    """Look up the entry (if any) whose client-facing api_key matches.

    Read-only by construction: opens the database with sqlite's `mode=ro`
    URI so it never blocks on, or competes for, a write lock, and never
    mutates the file (safe to call from every request). Tolerates:
      - a missing db file (returns None)
      - a file that isn't a valid sqlite database (returns None)
      - a schema from an older api_manager.py with no entry_keys table
        yet (returns the matched entry with an empty pool, not an error)
    """
    if not api_key:
        return None
    db_path = Path(db_path)
    if not db_available(db_path):
        return None

    try:
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    except sqlite3.OperationalError:
        return None

    try:
        conn.row_factory = sqlite3.Row
        try:
            row = conn.execute(
                "SELECT id, name, host, base_url, api_key FROM entries WHERE api_key = ?",
                (api_key,),
            ).fetchone()
        except sqlite3.DatabaseError:
            # Not a sqlite file, corrupt, or missing the entries table.
            return None
        if row is None:
            return None

        api_keys: tuple[str, ...] = ()
        try:
            pool_rows = conn.execute(
                "SELECT api_key FROM entry_keys WHERE entry_id = ? ORDER BY position",
                (row["id"],),
            ).fetchall()
            api_keys = tuple(r["api_key"] for r in pool_rows)
        except sqlite3.DatabaseError:
            # Older db, predates the entry_keys table -- empty pool.
            api_keys = ()

        return KeyEntry(
            name=row["name"],
            host=row["host"],
            scheme=_scheme_from_base_url(row["base_url"]),
            base_url=row["base_url"],
            api_key=row["api_key"],
            api_keys=api_keys,
        )
    finally:
        conn.close()
