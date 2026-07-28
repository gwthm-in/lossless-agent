"""Unit tests for the Claude Code capture integration (no database required)."""
import json

from lossless_agent.integrations import claude_code as cc


def test_project_db_name_is_deterministic_and_slugged():
    a = cc.project_db_name("/Users/x/Documents/repos/My-Project")
    b = cc.project_db_name("/Users/x/Documents/repos/My-Project")
    assert a == b                             # deterministic
    assert a.startswith("lcm_my_project_")    # basename lower-cased + slugged
    assert len(a.rsplit("_", 1)[-1]) == 8     # 8-hex path suffix
    # different paths with the same basename must not collide
    assert cc.project_db_name("/a/My-Project") != cc.project_db_name("/b/My-Project")
    # must stay within Postgres's 63-byte identifier limit even for very long basenames
    assert len(cc.project_db_name("/x/" + "a" * 200)) <= 63


def test_build_config_default_is_shared_sqlite(monkeypatch):
    # The zero-config path must be SQLite (no Postgres) AND the library default path — the same
    # store the MCP server defaults to, so recall sees captures out of the box.
    monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.delenv("LCM_CAPTURE_POSTGRES", raising=False)
    config = cc.build_config({"cwd": "/whatever"})
    assert not config.database_dsn                                   # SQLite, not Postgres
    assert config.resolved_db_path.endswith("/.lossless-agent/lcm.db")


def test_build_config_postgres_optin(monkeypatch):
    # LCM_CAPTURE_POSTGRES derives a per-project Postgres DSN (matches the launcher / kb_serve scheme).
    monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setenv("LCM_CAPTURE_POSTGRES", "1")
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/x/repos/widget")
    config = cc.build_config({"cwd": "/ignored"})
    expected = f"postgresql://localhost:5432/{cc.project_db_name('/Users/x/repos/widget')}"
    assert config.database_dsn == expected


def test_build_config_uses_explicit_dsn(monkeypatch):
    monkeypatch.setenv("LCM_DATABASE_DSN", "postgresql://localhost:5432/explicit_store")
    config = cc.build_config({"cwd": "/whatever"})
    assert config.database_dsn == "postgresql://localhost:5432/explicit_store"


def test_build_config_respects_explicit_sqlite_path(monkeypatch):
    monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
    monkeypatch.setenv("LCM_DATABASE_PATH", "/tmp/custom-lcm.db")
    config = cc.build_config({"cwd": "/whatever"})
    assert not config.database_dsn
    assert config.resolved_db_path == "/tmp/custom-lcm.db"


def test_parse_transcript_extracts_user_and_assistant_text(tmp_path):
    p = tmp_path / "transcript.jsonl"
    p.write_text("\n".join(json.dumps(o) for o in [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
        {"type": "assistant", "message": {"role": "assistant",
                                          "content": [{"type": "text", "text": "hi there"},
                                                      {"type": "tool_use", "name": "x"}]}},
        {"type": "summary", "message": {"role": "system", "content": "ignored"}},  # non user/assistant
        {"type": "user", "message": {"role": "user", "content": "   "}},           # whitespace -> dropped
        {"not": "json-but-a-dict-without-message"},
    ]) + "\ngarbage-not-json\n")
    msgs = cc.parse_transcript(str(p))
    assert msgs == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi there"},   # only the text block, tool_use dropped
    ]


def test_guard_env_short_circuits(monkeypatch):
    # When a summarizer's own subprocess triggers the hook, LCM_SUMMARIZING makes it a no-op.
    monkeypatch.setenv("LCM_SUMMARIZING", "1")
    assert cc.main([]) == 0


def test_ensure_database_prefers_target_over_admin(monkeypatch):
    # A reachable target DB must NOT require an admin (`postgres`) connection — least-privilege
    # roles often can't reach it. Inject a fake psycopg2 so the test needs no real driver.
    import sys
    import types

    calls = []

    class _FakeConn:
        autocommit = False

        def close(self):
            pass

    fake = types.ModuleType("psycopg2")
    fake.connect = lambda dsn, **kw: (calls.append(dsn), _FakeConn())[1]
    monkeypatch.setitem(sys.modules, "psycopg2", fake)

    assert cc.ensure_database("postgresql://user@host:5432/mydb") is True
    assert calls == ["postgresql://user@host:5432/mydb"]   # target only; never /postgres


def test_pg_target_strips_query_from_dbname():
    # A query-bearing DSN must not leak options into the created DB name or the admin path.
    dbname, admin = cc._pg_target("postgresql://user:pw@host:5432/lcm?sslmode=require")
    assert dbname == "lcm"
    assert admin == "postgresql://user:pw@host:5432/postgres?sslmode=require"


def test_prepare_store_sqlite_bare_filename():
    # A bare filename SQLite path has no parent dir; prepare_store must not crash on makedirs("").
    class _Cfg:
        database_dsn = ""
        resolved_db_path = "lcm.db"

    assert cc.prepare_store(_Cfg()) is True
