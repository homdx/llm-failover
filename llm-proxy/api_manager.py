#!/usr/bin/env python3
"""Store, list, retrieve, update, and delete API credentials in a SQLite file."""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import re
import sqlite3
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

PROGRAM = "api-manager"
DB_ENV_VAR = "API_MANAGER_DB"
NAME_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

SCHEMA = """
CREATE TABLE IF NOT EXISTS entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL UNIQUE,
    host TEXT NOT NULL,
    base_url TEXT NOT NULL,
    api_key TEXT NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

-- An entry's OWN pool of upstream failover keys -- separate from
-- entries.api_key above, which is the CLIENT-facing key used only to
-- look the entry up (see key_store.py). When this pool is non-empty,
-- the proxy's multi-key failover rotates through these instead of the
-- static [upstream] api_keys in config.toml, tried in `position` order.
-- Deleting an entry cascades and drops its whole pool with it.
CREATE TABLE IF NOT EXISTS entry_keys (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    entry_id INTEGER NOT NULL REFERENCES entries(id) ON DELETE CASCADE,
    api_key TEXT NOT NULL,
    position INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (entry_id, api_key)
);
"""


# --------------------------------------------------------------------------- errors


class ApiManagerError(Exception):
    exit_code = 1


class ValidationError(ApiManagerError):
    exit_code = 2


class NotFoundError(ApiManagerError):
    exit_code = 1


class ConflictError(ApiManagerError):
    exit_code = 3


# --------------------------------------------------------------------------- model


@dataclass(frozen=True)
class ApiEntry:
    id: int
    name: str
    host: str
    base_url: str
    api_key: str
    note: str | None
    created_at: str
    updated_at: str
    # The entry's own pool of upstream failover keys (entry_keys table),
    # in try order. Empty for a brand-new entry or one nothing was ever
    # added to. NOT populated by _from_row -- ApiStore attaches it via
    # _attach_pool() after the row is fetched, since it lives in a
    # separate table. Bug fix: get/list used to report only the single
    # client-facing api_key above and silently drop this pool from their
    # output, even with --show-keys, so an entry with several failover
    # keys looked identical to one with none.
    failover_keys: tuple[str, ...] = dataclasses.field(default_factory=tuple)

    def masked(self) -> ApiEntry:
        return dataclasses.replace(
            self,
            api_key=mask_api_key(self.api_key),
            failover_keys=tuple(mask_api_key(k) for k in self.failover_keys),
        )

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)

    @classmethod
    def _from_row(cls, row: sqlite3.Row) -> ApiEntry:
        return cls(
            id=row["id"],
            name=row["name"],
            host=row["host"],
            base_url=row["base_url"],
            api_key=row["api_key"],
            note=row["note"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )


# --------------------------------------------------------------------------- helpers


def mask_api_key(api_key: str | None) -> str:
    """Show a short, unambiguous hint without exposing the secret."""
    if not api_key:
        return ""
    if len(api_key) <= 10:
        return "*" * len(api_key)
    return f"{api_key[:6]}{'*' * 4}{api_key[-4:]}"


def validate_name(name: str) -> str:
    if not NAME_RE.match(name):
        raise ValidationError(
            f"invalid name {name!r}: use 1-64 chars of letters, digits, '.', '_' or '-'"
        )
    return name


def validate_base_url(base_url: str) -> str:
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or not parts.hostname:
        raise ValidationError(f"invalid --base-url {base_url!r}: need an http(s) URL with a host")
    return base_url


def derive_host(base_url: str) -> str:
    return urlsplit(base_url).hostname or ""


def resolve_host(base_url: str, host: str | None, no_host_check: bool) -> str:
    derived = derive_host(base_url)
    if host is None:
        return derived
    if not no_host_check and host != derived:
        raise ValidationError(
            f"--host {host!r} does not match host {derived!r} embedded in --base-url "
            f"(pass --no-host-check to allow this)"
        )
    return host


def default_db_path() -> Path:
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "api-manager" / "entries.db"


def resolve_db_path(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit)
    env = os.environ.get(DB_ENV_VAR)
    if env:
        return Path(env)
    return default_db_path()


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- store


class ApiStore:
    """Context manager wrapping a SQLite-backed collection of API entries."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.conn: sqlite3.Connection | None = None

    def __enter__(self) -> ApiStore:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        existed = self.db_path.exists()
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.conn.commit()
        if not existed:
            # Credentials are sensitive: keep the file owner-only.
            try:
                os.chmod(self.db_path, 0o600)
            except OSError:
                pass
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self.conn is not None:
            self.conn.close()
            self.conn = None

    def add(
        self,
        name: str,
        api_key: str,
        base_url: str,
        host: str,
        note: str | None = None,
    ) -> ApiEntry:
        assert self.conn is not None
        self._reject_duplicate_client_key(api_key)
        ts = now_iso()
        try:
            cur = self.conn.execute(
                "INSERT INTO entries (name, host, base_url, api_key, note, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (name, host, base_url, api_key, note, ts, ts),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"an entry named {name!r} already exists") from exc
        self.conn.commit()
        return self._get_by_id(cur.lastrowid)

    def _reject_duplicate_client_key(self, api_key: str, *, exclude_name: str | None = None) -> None:
        """Refuse a client-facing api_key already used by a different entry.

        entries.api_key has no UNIQUE constraint, so nothing previously
        stopped two entries from sharing one. key_store.lookup() matches
        with a plain `WHERE api_key = ?` and `.fetchone()`, so with a
        collision it silently returns whichever row sqlite happens to
        return first and the other entry (and its whole failover pool)
        becomes unreachable through the proxy without any error.
        """
        assert self.conn is not None
        row = self.conn.execute(
            "SELECT name FROM entries WHERE api_key = ?", (api_key,)
        ).fetchone()
        if row is not None and row["name"] != exclude_name:
            raise ConflictError(
                f"that --api-key is already used by entry {row['name']!r} -- "
                f"lookups match on this key alone, so sharing it would make "
                f"{row['name']!r} (and its failover pool) unreachable"
            )

    def _attach_pool(self, entry: ApiEntry) -> ApiEntry:
        """Fill in an entry's failover_keys from the entry_keys table."""
        assert self.conn is not None
        rows = self.conn.execute(
            "SELECT api_key FROM entry_keys WHERE entry_id = ? ORDER BY position",
            (entry.id,),
        ).fetchall()
        return dataclasses.replace(entry, failover_keys=tuple(r["api_key"] for r in rows))

    def _get_by_id(self, entry_id: int) -> ApiEntry:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return self._attach_pool(ApiEntry._from_row(row))

    def get(self, name: str) -> ApiEntry:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM entries WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise NotFoundError(f"no entry named {name!r}")
        return self._attach_pool(ApiEntry._from_row(row))

    def list(self) -> list[ApiEntry]:
        assert self.conn is not None
        rows = self.conn.execute("SELECT * FROM entries ORDER BY name").fetchall()
        return [self._attach_pool(ApiEntry._from_row(row)) for row in rows]

    def update(
        self,
        name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        host: str | None = None,
        note: str | None = None,
    ) -> ApiEntry:
        current = self.get(name)
        new_base_url = base_url if base_url is not None else current.base_url
        new_host = host if host is not None else current.host
        new_api_key = api_key if api_key is not None else current.api_key
        new_note = note if note is not None else current.note
        assert self.conn is not None
        if new_api_key != current.api_key:
            self._reject_duplicate_client_key(new_api_key, exclude_name=name)
        self.conn.execute(
            "UPDATE entries SET host = ?, base_url = ?, api_key = ?, note = ?, updated_at = ? "
            "WHERE name = ?",
            (new_host, new_base_url, new_api_key, new_note, now_iso(), name),
        )
        self.conn.commit()
        return self.get(name)

    def delete(self, name: str) -> None:
        assert self.conn is not None
        cur = self.conn.execute("DELETE FROM entries WHERE name = ?", (name,))
        if cur.rowcount == 0:
            raise NotFoundError(f"no entry named {name!r}")
        self.conn.commit()

    # ----------------------------------------------------------- failover key pool

    def add_key(self, name: str, api_key: str) -> list[str]:
        """Append one failover key to an entry's pool. Returns the pool after adding."""
        entry = self.get(name)  # raises NotFoundError if the entry doesn't exist
        assert self.conn is not None
        next_pos = self.conn.execute(
            "SELECT COALESCE(MAX(position), -1) + 1 FROM entry_keys WHERE entry_id = ?",
            (entry.id,),
        ).fetchone()[0]
        try:
            self.conn.execute(
                "INSERT INTO entry_keys (entry_id, api_key, position, created_at) "
                "VALUES (?, ?, ?, ?)",
                (entry.id, api_key, next_pos, now_iso()),
            )
        except sqlite3.IntegrityError as exc:
            raise ConflictError(f"entry {name!r} already has that api key in its pool") from exc
        self.conn.commit()
        return self.list_keys(name)

    def list_keys(self, name: str) -> list[str]:
        """Return an entry's failover pool, in the order keys are tried."""
        entry = self.get(name)  # get() already attaches the pool
        return list(entry.failover_keys)

    def remove_key(self, name: str, api_key: str) -> None:
        """Remove exactly one key from an entry's pool."""
        entry = self.get(name)
        assert self.conn is not None
        cur = self.conn.execute(
            "DELETE FROM entry_keys WHERE entry_id = ? AND api_key = ?",
            (entry.id, api_key),
        )
        if cur.rowcount == 0:
            raise NotFoundError(f"entry {name!r} has no such key in its pool")
        self.conn.commit()

    def clear_keys(self, name: str) -> int:
        """Remove every key from an entry's pool. Returns how many were removed."""
        entry = self.get(name)
        assert self.conn is not None
        cur = self.conn.execute("DELETE FROM entry_keys WHERE entry_id = ?", (entry.id,))
        self.conn.commit()
        return cur.rowcount


# --------------------------------------------------------------------------- rendering


def render_table(entries: Sequence[ApiEntry]) -> str:
    if not entries:
        return "(no entries)"
    # FAILOVER KEYS shows just the pool size here to keep the table
    # readable; `get NAME --show-keys` or `keys list NAME` gives the
    # actual values.
    headers = ["NAME", "HOST", "BASE URL", "API KEY", "FAILOVER KEYS", "NOTE"]
    rows = [
        [e.name, e.host, e.base_url, e.api_key,
         str(len(e.failover_keys)) if e.failover_keys else "-", e.note or "-"]
        for e in entries
    ]
    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows)) for i in range(len(headers))
    ]
    lines = ["  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))]
    lines.append("  ".join("-" * w for w in widths))
    for r in rows:
        lines.append("  ".join(c.ljust(widths[i]) for i, c in enumerate(r)))
    return "\n".join(lines)


def render_detail(entry: ApiEntry) -> str:
    d = entry.to_dict()
    width = max(len(k) for k in d)
    lines = []
    for k, v in d.items():
        if isinstance(v, (list, tuple)):
            v = ", ".join(v) if v else "(none -- falls back to [upstream] api_keys in config.toml)"
        lines.append(f"{k.ljust(width)} : {v if v not in (None, '') else '-'}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- confirmation


def confirm(prompt: str) -> bool:
    """Read a y/N answer from stdin; EOF or a non-interactive stream aborts."""
    if not sys.stdin.isatty():
        raise ApiManagerError("confirmation required: pass --yes (stdin is not a terminal)")
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


# --------------------------------------------------------------------------- CLI commands


def cmd_add(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    validate_base_url(args.base_url)
    host = resolve_host(args.base_url, args.host, args.no_host_check)
    with ApiStore(resolve_db_path(args.db)) as store:
        entry = store.add(name, args.api_key, args.base_url, host, note=args.note)
    if args.json:
        print(json.dumps(entry.masked().to_dict(), indent=2, sort_keys=True))
    else:
        print(f"added {entry.name!r} (id {entry.id})")
    return 0


def cmd_list(args: argparse.Namespace) -> int:
    with ApiStore(resolve_db_path(args.db)) as store:
        entries = store.list()
    shown = entries if args.show_keys else [e.masked() for e in entries]
    if args.json:
        print(json.dumps([e.to_dict() for e in shown], indent=2, sort_keys=True))
    else:
        print(render_table(shown))
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    with ApiStore(resolve_db_path(args.db)) as store:
        entry = store.get(name)
    shown = entry if args.show_keys else entry.masked()
    if args.json:
        print(json.dumps(shown.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_detail(shown))
    return 0


def cmd_update(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    if args.base_url is not None:
        validate_base_url(args.base_url)
    host = args.host
    if args.base_url is not None and args.host is None and not args.no_host_check:
        host = derive_host(args.base_url)
    elif args.host is not None and args.base_url is not None and not args.no_host_check:
        host = resolve_host(args.base_url, args.host, args.no_host_check)
    with ApiStore(resolve_db_path(args.db)) as store:
        entry = store.update(
            name,
            api_key=args.api_key,
            base_url=args.base_url,
            host=host,
            note=args.note,
        )
    if args.json:
        print(json.dumps(entry.masked().to_dict(), indent=2, sort_keys=True))
    else:
        print(f"updated {entry.name!r}")
    return 0


def cmd_delete(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    if not args.yes:
        if not confirm(f"Delete entry {name!r}? [y/N] "):
            print("aborted", file=sys.stderr)
            return 1
    with ApiStore(resolve_db_path(args.db)) as store:
        store.delete(name)
    print(f"deleted {name!r}")
    return 0


def _validate_pool_key(raw: str) -> str:
    key = raw.strip()
    if not key:
        raise ValidationError("--api-key must not be empty")
    return key


def cmd_keys_add(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    api_key = _validate_pool_key(args.api_key)
    with ApiStore(resolve_db_path(args.db)) as store:
        pool = store.add_key(name, api_key)
    if args.json:
        print(json.dumps([mask_api_key(k) for k in pool], indent=2))
    else:
        print(f"added key to {name!r} (pool now has {len(pool)} key(s))")
    return 0


def cmd_keys_list(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    with ApiStore(resolve_db_path(args.db)) as store:
        pool = store.list_keys(name)
    shown = pool if args.show_keys else [mask_api_key(k) for k in pool]
    if args.json:
        print(json.dumps(shown, indent=2))
    elif not shown:
        print(f"(no failover keys stored for {name!r} -- falls back to "
              f"[upstream] api_keys in config.toml)")
    else:
        for i, key in enumerate(shown, 1):
            print(f"{i}. {key}")
    return 0


def cmd_keys_rm(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    api_key = _validate_pool_key(args.api_key)
    with ApiStore(resolve_db_path(args.db)) as store:
        store.remove_key(name, api_key)
    print(f"removed key from {name!r}")
    return 0


def cmd_keys_clear(args: argparse.Namespace) -> int:
    name = validate_name(args.name)
    if not args.yes:
        if not confirm(f"Remove ALL failover keys stored for {name!r}? [y/N] "):
            print("aborted", file=sys.stderr)
            return 1
    with ApiStore(resolve_db_path(args.db)) as store:
        removed = store.clear_keys(name)
    print(f"removed {removed} key(s) from {name!r}")
    return 0


# --------------------------------------------------------------------------- argument parsing


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog=PROGRAM, description=__doc__)
    parser.add_argument(
        "--db",
        default=None,
        metavar="PATH",
        help=f"SQLite database file (default: ${DB_ENV_VAR} or {default_db_path()})",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 1.0")

    sub = parser.add_subparsers(dest="command", metavar="COMMAND", required=True)

    p_add = sub.add_parser("add", help="store a new entry")
    p_add.add_argument("name", help="unique name for this entry")
    p_add.add_argument("--api-key", required=True, dest="api_key", metavar="KEY")
    p_add.add_argument("--base-url", required=True, dest="base_url", metavar="URL")
    p_add.add_argument("--host", default=None, metavar="HOST", help="derived from --base-url when omitted")
    p_add.add_argument("--note", default=None, metavar="TEXT")
    p_add.add_argument(
        "--no-host-check",
        action="store_true",
        dest="no_host_check",
        help="allow --host to differ from the host embedded in --base-url",
    )
    p_add.add_argument("--json", action="store_true", help="print the new entry as JSON")
    p_add.set_defaults(handler=cmd_add)

    p_list = sub.add_parser("list", aliases=["ls"], help="show every stored entry")
    p_list.add_argument("--json", action="store_true", help="print as JSON")
    p_list.add_argument("--show-keys", action="store_true", dest="show_keys", help="reveal api keys in plain text")
    p_list.set_defaults(handler=cmd_list)

    p_get = sub.add_parser("get", help="show one entry by name")
    p_get.add_argument("name")
    p_get.add_argument("--json", action="store_true", help="print as JSON")
    p_get.add_argument("--show-keys", action="store_true", dest="show_keys", help="reveal the api key in plain text")
    p_get.set_defaults(handler=cmd_get)

    p_update = sub.add_parser("update", help="change an existing entry")
    p_update.add_argument("name")
    p_update.add_argument("--api-key", default=None, dest="api_key", metavar="KEY")
    p_update.add_argument("--base-url", default=None, dest="base_url", metavar="URL")
    p_update.add_argument("--host", default=None, metavar="HOST")
    p_update.add_argument("--note", default=None, metavar="TEXT")
    p_update.add_argument("--no-host-check", action="store_true", dest="no_host_check")
    p_update.add_argument("--json", action="store_true", help="print the updated entry as JSON")
    p_update.set_defaults(handler=cmd_update)

    p_delete = sub.add_parser("delete", aliases=["rm"], help="remove one entry")
    p_delete.add_argument("name")
    p_delete.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_delete.set_defaults(handler=cmd_delete)

    p_keys = sub.add_parser(
        "keys",
        help="manage an entry's own pool of upstream failover keys",
        description=(
            "Manage an entry's own pool of upstream failover keys. When this pool "
            "is non-empty the proxy rotates through it (in the order added) "
            "instead of the static [upstream] api_keys in config.toml -- exactly "
            "the same 4xx/5xx-triggers-the-next-key behaviour, just scoped to "
            "this one entry's host."
        ),
    )
    keys_sub = p_keys.add_subparsers(dest="keys_command", metavar="ACTION", required=True)

    p_keys_add = keys_sub.add_parser("add", help="add one key to an entry's pool")
    p_keys_add.add_argument("name", help="name of an existing entry")
    p_keys_add.add_argument("--api-key", required=True, dest="api_key", metavar="KEY")
    p_keys_add.add_argument("--json", action="store_true", help="print the resulting pool (masked) as JSON")
    p_keys_add.set_defaults(handler=cmd_keys_add)

    p_keys_list = keys_sub.add_parser("list", aliases=["ls"], help="show an entry's pool, in try order")
    p_keys_list.add_argument("name")
    p_keys_list.add_argument("--json", action="store_true", help="print as JSON")
    p_keys_list.add_argument("--show-keys", action="store_true", dest="show_keys", help="reveal keys in plain text")
    p_keys_list.set_defaults(handler=cmd_keys_list)

    p_keys_rm = keys_sub.add_parser(
        "rm", aliases=["remove"], help="remove one specific key from an entry's pool"
    )
    p_keys_rm.add_argument("name")
    p_keys_rm.add_argument("--api-key", required=True, dest="api_key", metavar="KEY")
    p_keys_rm.set_defaults(handler=cmd_keys_rm)

    p_keys_clear = keys_sub.add_parser("clear", help="remove EVERY key from an entry's pool")
    p_keys_clear.add_argument("name")
    p_keys_clear.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    p_keys_clear.set_defaults(handler=cmd_keys_clear)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.handler(args)
    except ApiManagerError as exc:
        print(f"{PROGRAM}: error: {exc}", file=sys.stderr)
        return exc.exit_code
    except BrokenPipeError:
        return 1


if __name__ == "__main__":
    sys.exit(main())
