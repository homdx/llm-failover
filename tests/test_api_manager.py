from __future__ import annotations

import builtins
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "api_manager.py"
sys.path.insert(0, str(ROOT))

import api_manager as am

OPENROUTER = {
    "name": "openrouter",
    "api_key": "sk-or-v1-abcdefghijklmnop",
    "base_url": "https://openrouter.ai/api/v1",
    "host": "openrouter.ai",
}
OPENAI = {
    "name": "openai",
    "api_key": "sk-openai-abcdefghijklmnop",
    "base_url": "https://api.openai.com/v1",
    "host": "api.openai.com",
}


@pytest.fixture
def db_path(tmp_path: Path) -> str:
    return str(tmp_path / "entries.db")


def add_openrouter(db: str, **overrides) -> int:
    args = {**OPENROUTER, **overrides}
    return am.main(
        [
            "--db",
            db,
            "add",
            args["name"],
            "--api-key",
            args["api_key"],
            "--base-url",
            args["base_url"],
        ]
    )


# --------------------------------------------------------------------------- pure helpers


class TestMaskApiKey:
    def test_empty_and_none(self):
        assert am.mask_api_key(None) == ""
        assert am.mask_api_key("") == ""

    def test_short_key_fully_masked(self):
        assert am.mask_api_key("short") == "*****"

    def test_long_key_shows_prefix_and_suffix(self):
        assert am.mask_api_key("sk-or-v1-abcdefghijklmnop") == "sk-or-****mnop"

    def test_boundary_length_ten(self):
        key = "1234567890"
        assert am.mask_api_key(key) == "*" * 10


class TestValidateName:
    @pytest.mark.parametrize("name", ["openrouter", "my.key-1", "a_b", "x" * 64])
    def test_valid_names_pass_through(self, name):
        assert am.validate_name(name) == name

    @pytest.mark.parametrize("name", ["", "has space", "slash/es", "x" * 65, "émoji"])
    def test_invalid_names_raise(self, name):
        with pytest.raises(am.ValidationError):
            am.validate_name(name)


class TestValidateBaseUrl:
    def test_valid_https_url(self):
        assert am.validate_base_url("https://api.openai.com/v1") == "https://api.openai.com/v1"

    @pytest.mark.parametrize(
        "url", ["not-a-url", "ftp://example.com", "https://", "http:///nohostname"]
    )
    def test_invalid_urls_raise(self, url):
        with pytest.raises(am.ValidationError):
            am.validate_base_url(url)


class TestResolveHost:
    def test_derives_when_host_omitted(self):
        assert am.resolve_host("https://api.openai.com/v1", None, False) == "api.openai.com"

    def test_matching_host_accepted(self):
        assert am.resolve_host("https://api.openai.com/v1", "api.openai.com", False) == "api.openai.com"

    def test_mismatched_host_rejected(self):
        with pytest.raises(am.ValidationError):
            am.resolve_host("https://api.openai.com/v1", "example.com", False)

    def test_mismatched_host_allowed_with_no_host_check(self):
        assert am.resolve_host("https://api.openai.com/v1", "example.com", True) == "example.com"


class TestDbPathResolution:
    def test_explicit_wins(self, monkeypatch):
        monkeypatch.setenv(am.DB_ENV_VAR, "/env/path.db")
        assert am.resolve_db_path("/explicit/path.db") == Path("/explicit/path.db")

    def test_env_var_used_when_no_explicit(self, monkeypatch):
        monkeypatch.setenv(am.DB_ENV_VAR, "/env/path.db")
        assert am.resolve_db_path(None) == Path("/env/path.db")

    def test_default_uses_xdg_config_home(self, monkeypatch):
        monkeypatch.delenv(am.DB_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
        assert am.default_db_path() == Path("/xdg/api-manager/entries.db")

    def test_default_falls_back_to_home(self, monkeypatch):
        monkeypatch.delenv(am.DB_ENV_VAR, raising=False)
        monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
        monkeypatch.setenv("HOME", "/home/x")
        assert am.default_db_path() == Path("/home/x/.config/api-manager/entries.db")

    def test_resolve_db_path_falls_back_to_default_when_unset(self, monkeypatch):
        monkeypatch.delenv(am.DB_ENV_VAR, raising=False)
        monkeypatch.setenv("XDG_CONFIG_HOME", "/xdg")
        assert am.resolve_db_path(None) == Path("/xdg/api-manager/entries.db")


# --------------------------------------------------------------------------- ApiStore


class TestApiStore:
    def test_context_manager_creates_db_file(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            assert store.conn is not None
        assert Path(db_path).exists()

    @pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file permission bits")
    def test_new_db_file_is_chmod_0600(self, db_path):
        with am.ApiStore(Path(db_path)):
            pass
        mode = Path(db_path).stat().st_mode & 0o777
        assert mode == 0o600

    def test_chmod_failure_on_new_db_is_swallowed(self, db_path, monkeypatch):
        def raise_oserror(*a, **k):
            raise OSError("not permitted")

        monkeypatch.setattr(am.os, "chmod", raise_oserror)
        with am.ApiStore(Path(db_path)) as store:
            assert store.conn is not None
        assert Path(db_path).exists()

    def test_reopening_existing_db_does_not_touch_permissions(self, db_path):
        with am.ApiStore(Path(db_path)):
            pass
        os.chmod(db_path, 0o644)
        with am.ApiStore(Path(db_path)):
            pass
        mode = Path(db_path).stat().st_mode & 0o777
        assert mode == 0o644

    def test_add_and_get_roundtrip(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            added = store.add(**OPENROUTER, note="primary")
            fetched = store.get(OPENROUTER["name"])
        assert added.name == fetched.name == OPENROUTER["name"]
        assert fetched.api_key == OPENROUTER["api_key"]
        assert fetched.note == "primary"
        assert fetched.created_at == fetched.updated_at

    def test_add_duplicate_name_raises_conflict(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            store.add(**OPENROUTER)
            with pytest.raises(am.ConflictError):
                store.add(**OPENROUTER)

    def test_get_missing_raises_not_found(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            with pytest.raises(am.NotFoundError):
                store.get("nope")

    def test_list_empty(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            assert store.list() == []

    def test_list_orders_by_name(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            store.add(**OPENROUTER)
            store.add(**OPENAI)
            names = [e.name for e in store.list()]
        assert names == sorted(names)

    def test_update_partial_fields(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            store.add(**OPENROUTER)
            updated = store.update(OPENROUTER["name"], note="rotated")
        assert updated.note == "rotated"
        assert updated.api_key == OPENROUTER["api_key"]
        assert updated.updated_at >= updated.created_at

    def test_update_missing_raises_not_found(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            with pytest.raises(am.NotFoundError):
                store.update("nope", note="x")

    def test_delete_removes_entry(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            store.add(**OPENROUTER)
            store.delete(OPENROUTER["name"])
            assert store.list() == []

    def test_delete_missing_raises_not_found(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            with pytest.raises(am.NotFoundError):
                store.delete("nope")


# --------------------------------------------------------------------------- rendering


class TestRendering:
    def test_render_table_empty(self):
        assert am.render_table([]) == "(no entries)"

    def test_render_table_masks_are_passed_through(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            entry = store.add(**OPENROUTER)
        table = am.render_table([entry.masked()])
        assert OPENROUTER["api_key"] not in table
        assert "sk-or-" in table

    def test_render_detail_lists_all_fields(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            entry = store.add(**OPENROUTER)
        detail = am.render_detail(entry)
        for field in ("id", "name", "host", "base_url", "api_key", "note", "created_at", "updated_at"):
            assert field in detail

    def test_render_detail_none_note_shown_as_dash(self, db_path):
        with am.ApiStore(Path(db_path)) as store:
            entry = store.add(**OPENROUTER)
        assert "-" in am.render_detail(entry).splitlines()[4]


# --------------------------------------------------------------------------- confirm()


class TestConfirm:
    def test_non_tty_raises(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        with pytest.raises(am.ApiManagerError):
            am.confirm("? ")

    @pytest.mark.parametrize("answer,expected", [("y\n", True), ("yes\n", True), ("n\n", False), ("\n", False)])
    def test_tty_reads_answer(self, monkeypatch, answer, expected):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda prompt="": answer.strip())
        assert am.confirm("? ") is expected

    def test_eof_aborts(self, monkeypatch):
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)

        def raise_eof(prompt=""):
            raise EOFError

        monkeypatch.setattr(builtins, "input", raise_eof)
        assert am.confirm("? ") is False


# --------------------------------------------------------------------------- CLI: add


class TestCliAdd:
    def test_add_success_prints_confirmation(self, db_path, capsys):
        rc = add_openrouter(db_path)
        out = capsys.readouterr().out
        assert rc == 0
        assert "added 'openrouter'" in out

    def test_add_json_output_masks_key(self, db_path, capsys):
        rc = am.main(
            [
                "--db", db_path, "add", OPENROUTER["name"],
                "--api-key", OPENROUTER["api_key"],
                "--base-url", OPENROUTER["base_url"],
                "--json",
            ]
        )
        payload = json.loads(capsys.readouterr().out)
        assert rc == 0
        assert payload["api_key"] != OPENROUTER["api_key"]
        assert payload["host"] == OPENROUTER["host"]

    def test_add_derives_host_from_base_url(self, db_path):
        add_openrouter(db_path)
        with am.ApiStore(Path(db_path)) as store:
            assert store.get("openrouter").host == "openrouter.ai"

    def test_add_explicit_host_must_match(self, db_path, capsys):
        rc = am.main(
            [
                "--db", db_path, "add", "openrouter",
                "--api-key", OPENROUTER["api_key"],
                "--base-url", OPENROUTER["base_url"],
                "--host", "example.com",
            ]
        )
        assert rc == am.ValidationError.exit_code
        assert "does not match" in capsys.readouterr().err

    def test_add_no_host_check_allows_mismatch(self, db_path):
        rc = am.main(
            [
                "--db", db_path, "add", "openrouter",
                "--api-key", OPENROUTER["api_key"],
                "--base-url", OPENROUTER["base_url"],
                "--host", "example.com",
                "--no-host-check",
            ]
        )
        assert rc == 0
        with am.ApiStore(Path(db_path)) as store:
            assert store.get("openrouter").host == "example.com"

    def test_add_duplicate_name_exits_with_conflict_code(self, db_path, capsys):
        add_openrouter(db_path)
        rc = add_openrouter(db_path)
        assert rc == am.ConflictError.exit_code
        assert "already exists" in capsys.readouterr().err

    def test_add_invalid_name_rejected(self, db_path, capsys):
        rc = am.main(
            [
                "--db", db_path, "add", "bad name",
                "--api-key", "sk-x", "--base-url", OPENROUTER["base_url"],
            ]
        )
        assert rc == am.ValidationError.exit_code

    def test_add_invalid_base_url_rejected(self, db_path):
        rc = am.main(
            ["--db", db_path, "add", "openrouter", "--api-key", "sk-x", "--base-url", "not-a-url"]
        )
        assert rc == am.ValidationError.exit_code

    def test_add_missing_required_flag_errors_via_argparse(self, db_path):
        with pytest.raises(SystemExit) as exc:
            am.main(["--db", db_path, "add", "openrouter", "--base-url", OPENROUTER["base_url"]])
        assert exc.value.code == 2

    def test_add_with_note(self, db_path):
        am.main(
            [
                "--db", db_path, "add", "openrouter",
                "--api-key", OPENROUTER["api_key"],
                "--base-url", OPENROUTER["base_url"],
                "--note", "primary account",
            ]
        )
        with am.ApiStore(Path(db_path)) as store:
            assert store.get("openrouter").note == "primary account"


# --------------------------------------------------------------------------- CLI: list


class TestCliList:
    def test_list_empty_db(self, db_path, capsys):
        rc = am.main(["--db", db_path, "list"])
        assert rc == 0
        assert "no entries" in capsys.readouterr().out

    def test_list_masks_keys_by_default(self, db_path, capsys):
        add_openrouter(db_path)
        am.main(["--db", db_path, "list"])
        out = capsys.readouterr().out
        assert OPENROUTER["api_key"] not in out

    def test_list_show_keys_reveals_key(self, db_path, capsys):
        add_openrouter(db_path)
        am.main(["--db", db_path, "list", "--show-keys"])
        out = capsys.readouterr().out
        assert OPENROUTER["api_key"] in out

    def test_list_json_shape(self, db_path, capsys):
        add_openrouter(db_path)
        capsys.readouterr()
        am.main(["--db", db_path, "list", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert isinstance(payload, list)
        assert payload[0]["name"] == "openrouter"

    def test_list_alias_ls(self, db_path, capsys):
        rc = am.main(["--db", db_path, "ls"])
        assert rc == 0

    def test_list_multiple_entries_sorted(self, db_path, capsys):
        add_openrouter(db_path)
        am.main(
            [
                "--db", db_path, "add", OPENAI["name"],
                "--api-key", OPENAI["api_key"], "--base-url", OPENAI["base_url"],
            ]
        )
        capsys.readouterr()
        am.main(["--db", db_path, "list"])
        out = capsys.readouterr().out
        assert out.index("openai") < out.index("openrouter")


# --------------------------------------------------------------------------- CLI: get


class TestCliGet:
    def test_get_masks_by_default(self, db_path, capsys):
        add_openrouter(db_path)
        rc = am.main(["--db", db_path, "get", "openrouter"])
        out = capsys.readouterr().out
        assert rc == 0
        assert OPENROUTER["api_key"] not in out

    def test_get_show_keys_reveals_key(self, db_path, capsys):
        add_openrouter(db_path)
        am.main(["--db", db_path, "get", "openrouter", "--show-keys"])
        out = capsys.readouterr().out
        assert OPENROUTER["api_key"] in out

    def test_get_json(self, db_path, capsys):
        add_openrouter(db_path)
        capsys.readouterr()
        am.main(["--db", db_path, "get", "openrouter", "--json", "--show-keys"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["api_key"] == OPENROUTER["api_key"]

    def test_get_missing_entry_exits_1(self, db_path, capsys):
        rc = am.main(["--db", db_path, "get", "nope"])
        assert rc == 1
        assert "no entry named" in capsys.readouterr().err

    def test_get_invalid_name_exits_2(self, db_path):
        rc = am.main(["--db", db_path, "get", "bad name"])
        assert rc == am.ValidationError.exit_code


# --------------------------------------------------------------------------- CLI: update


class TestCliUpdate:
    def test_update_api_key_only(self, db_path):
        add_openrouter(db_path)
        rc = am.main(["--db", db_path, "update", "openrouter", "--api-key", "sk-new-key-value"])
        assert rc == 0
        with am.ApiStore(Path(db_path)) as store:
            entry = store.get("openrouter")
        assert entry.api_key == "sk-new-key-value"
        assert entry.base_url == OPENROUTER["base_url"]

    def test_update_base_url_rederives_host(self, db_path):
        add_openrouter(db_path)
        rc = am.main(["--db", db_path, "update", "openrouter", "--base-url", OPENAI["base_url"]])
        assert rc == 0
        with am.ApiStore(Path(db_path)) as store:
            entry = store.get("openrouter")
        assert entry.host == OPENAI["host"]

    def test_update_host_mismatch_rejected(self, db_path):
        add_openrouter(db_path)
        rc = am.main(
            ["--db", db_path, "update", "openrouter", "--base-url", OPENAI["base_url"], "--host", "example.com"]
        )
        assert rc == am.ValidationError.exit_code

    def test_update_missing_entry_exits_1(self, db_path):
        rc = am.main(["--db", db_path, "update", "nope", "--note", "x"])
        assert rc == 1

    def test_update_json_output(self, db_path, capsys):
        add_openrouter(db_path)
        capsys.readouterr()
        am.main(["--db", db_path, "update", "openrouter", "--note", "x", "--json"])
        payload = json.loads(capsys.readouterr().out)
        assert payload["note"] == "x"


# --------------------------------------------------------------------------- CLI: delete


class TestCliDelete:
    def test_delete_with_yes_flag(self, db_path, capsys):
        add_openrouter(db_path)
        rc = am.main(["--db", db_path, "delete", "openrouter", "--yes"])
        assert rc == 0
        assert "deleted" in capsys.readouterr().out
        with am.ApiStore(Path(db_path)) as store:
            assert store.list() == []

    def test_delete_alias_rm(self, db_path):
        add_openrouter(db_path)
        rc = am.main(["--db", db_path, "rm", "openrouter", "--yes"])
        assert rc == 0

    def test_delete_prompts_and_confirms(self, db_path, monkeypatch, capsys):
        add_openrouter(db_path)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda prompt="": "y")
        rc = am.main(["--db", db_path, "delete", "openrouter"])
        assert rc == 0

    def test_delete_prompt_declined_aborts(self, db_path, monkeypatch, capsys):
        add_openrouter(db_path)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(builtins, "input", lambda prompt="": "n")
        rc = am.main(["--db", db_path, "delete", "openrouter"])
        assert rc == 1
        with am.ApiStore(Path(db_path)) as store:
            assert store.get("openrouter") is not None

    def test_delete_without_tty_and_without_yes_errors(self, db_path, monkeypatch, capsys):
        add_openrouter(db_path)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
        rc = am.main(["--db", db_path, "delete", "openrouter"])
        assert rc == am.ApiManagerError.exit_code
        assert "confirmation required" in capsys.readouterr().err

    def test_delete_missing_entry_exits_1(self, db_path):
        rc = am.main(["--db", db_path, "delete", "nope", "--yes"])
        assert rc == 1

    def test_delete_all_entries_one_at_a_time(self, db_path):
        add_openrouter(db_path)
        am.main(
            [
                "--db", db_path, "add", OPENAI["name"],
                "--api-key", OPENAI["api_key"], "--base-url", OPENAI["base_url"],
            ]
        )
        am.main(["--db", db_path, "delete", "openrouter", "--yes"])
        with am.ApiStore(Path(db_path)) as store:
            assert [e.name for e in store.list()] == ["openai"]
        am.main(["--db", db_path, "delete", "openai", "--yes"])
        with am.ApiStore(Path(db_path)) as store:
            assert store.list() == []


# --------------------------------------------------------------------------- CLI: argument handling


class TestCliArgumentHandling:
    def test_no_subcommand_errors(self):
        with pytest.raises(SystemExit) as exc:
            am.main([])
        assert exc.value.code == 2

    def test_help_exits_zero(self, capsys):
        with pytest.raises(SystemExit) as exc:
            am.main(["--help"])
        assert exc.value.code == 0
        assert "COMMAND" in capsys.readouterr().out

    def test_version_flag(self, capsys):
        with pytest.raises(SystemExit) as exc:
            am.main(["--version"])
        assert exc.value.code == 0

    def test_unknown_command_errors(self):
        with pytest.raises(SystemExit):
            am.main(["frobnicate"])

    def test_db_env_var_used_when_flag_omitted(self, monkeypatch, tmp_path, capsys):
        env_db = tmp_path / "env.db"
        monkeypatch.setenv(am.DB_ENV_VAR, str(env_db))
        rc = am.main(
            [
                "add", "openrouter",
                "--api-key", OPENROUTER["api_key"], "--base-url", OPENROUTER["base_url"],
            ]
        )
        assert rc == 0
        assert env_db.exists()

    def test_explicit_db_flag_wins_over_env(self, monkeypatch, tmp_path):
        env_db = tmp_path / "env.db"
        explicit_db = tmp_path / "explicit.db"
        monkeypatch.setenv(am.DB_ENV_VAR, str(env_db))
        am.main(
            [
                "--db", str(explicit_db), "add", "openrouter",
                "--api-key", OPENROUTER["api_key"], "--base-url", OPENROUTER["base_url"],
            ]
        )
        assert explicit_db.exists()
        assert not env_db.exists()


class TestBrokenPipe:
    def test_broken_pipe_during_output_exits_1(self, db_path, monkeypatch):
        add_openrouter(db_path)

        def raise_bpe(*a, **k):
            raise BrokenPipeError

        monkeypatch.setattr(builtins, "print", raise_bpe)
        rc = am.main(["--db", db_path, "list"])
        assert rc == 1


# --------------------------------------------------------------------------- subprocess smoke tests


def run_cli(args, db=None):
    cmd = [sys.executable, str(SCRIPT)]
    if db is not None:
        cmd += ["--db", str(db)]
    cmd += args
    return subprocess.run(cmd, capture_output=True, text=True)


class TestSubprocessEntryPoint:
    def test_full_lifecycle_as_a_real_process(self, tmp_path):
        db = tmp_path / "sub.db"

        added = run_cli(
            ["add", "openrouter", "--api-key", OPENROUTER["api_key"], "--base-url", OPENROUTER["base_url"]],
            db=db,
        )
        assert added.returncode == 0, added.stderr

        listed = run_cli(["list"], db=db)
        assert listed.returncode == 0
        assert "openrouter" in listed.stdout

        got = run_cli(["get", "openrouter", "--show-keys"], db=db)
        assert OPENROUTER["api_key"] in got.stdout

        deleted = run_cli(["delete", "openrouter", "--yes"], db=db)
        assert deleted.returncode == 0

        empty = run_cli(["list"], db=db)
        assert "no entries" in empty.stdout

    def test_failure_exit_code_is_one(self, tmp_path):
        db = tmp_path / "sub.db"
        result = run_cli(["get", "missing"], db=db)
        assert result.returncode == 1
        assert "no entry named" in result.stderr

    def test_script_is_directly_executable_guard(self, tmp_path):
        db = tmp_path / "sub.db"
        result = run_cli(["--version"], db=db)
        assert result.returncode == 0
