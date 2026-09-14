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

    def masked(self) -> ApiEntry:
        return dataclasses.replace(self, api_key=mask_api_key(self.api_key))

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

    def _get_by_id(self, entry_id: int) -> ApiEntry:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM entries WHERE id = ?", (entry_id,)).fetchone()
        return ApiEntry._from_row(row)

    def get(self, name: str) -> ApiEntry:
        assert self.conn is not None
        row = self.conn.execute("SELECT * FROM entries WHERE name = ?", (name,)).fetchone()
        if row is None:
            raise NotFoundError(f"no entry named {name!r}")
        return ApiEntry._from_row(row)

    def list(self) -> list[ApiEntry]:
        assert self.conn is not None
        rows = self.conn.execute("SELECT * FROM entries ORDER BY name").fetchall()
        return [ApiEntry._from_row(row) for row in rows]

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


# --------------------------------------------------------------------------- rendering


def render_table(entries: Sequence[ApiEntry]) -> str:
    if not entries:
        return "(no entries)"
    headers = ["NAME", "HOST", "BASE URL", "API KEY", "NOTE"]
    rows = [
        [e.name, e.host, e.base_url, e.api_key, e.note or "-"]
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
    return "\n".join(f"{k.ljust(width)} : {v if v is not None else '-'}" for k, v in d.items())


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
