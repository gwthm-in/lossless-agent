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


def test_default_store_path_is_per_project_sqlite():
    p = cc.default_store_path("/Users/x/repos/widget")
    assert p.endswith(".db")
    assert "/stores/" in p
    assert cc.project_db_name("/Users/x/repos/widget") in p


def test_build_config_default_is_sqlite_no_server(monkeypatch):
    # The documented "seamless" path (no env) must NOT require Postgres.
    monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
    monkeypatch.delenv("LCM_DATABASE_PATH", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/x/repos/widget")
    config = cc.build_config({"cwd": "/ignored/when/project/dir/set"})
    assert not config.database_dsn                                   # SQLite, not Postgres
    assert config.db_path == cc.default_store_path("/Users/x/repos/widget")


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
