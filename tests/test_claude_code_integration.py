"""Unit tests for the Claude Code capture integration (no database required)."""
import json

from lossless_agent.integrations import claude_code as cc


def test_project_db_name_is_deterministic_and_slugged():
    a = cc.project_db_name("/Users/x/Documents/repos/My-Project")
    b = cc.project_db_name("/Users/x/Documents/repos/My-Project")
    assert a == b                       # deterministic
    assert a.startswith("lcm_my_project_")   # basename lower-cased + slugged
    assert len(a.rsplit("_", 1)[-1]) == 8     # 8-hex path suffix
    # different paths with the same basename must not collide
    assert cc.project_db_name("/a/My-Project") != cc.project_db_name("/b/My-Project")


def test_resolve_dsn_prefers_explicit_env(monkeypatch):
    monkeypatch.setenv("LCM_DATABASE_DSN", "postgresql://localhost:5432/explicit_store")
    assert cc.resolve_dsn({"cwd": "/whatever"}) == "postgresql://localhost:5432/explicit_store"


def test_resolve_dsn_derives_from_project_dir(monkeypatch):
    monkeypatch.delenv("LCM_DATABASE_DSN", raising=False)
    monkeypatch.setenv("CLAUDE_PROJECT_DIR", "/Users/x/repos/widget")
    dsn = cc.resolve_dsn({"cwd": "/ignored/when/project/dir/set"})
    assert dsn == f"postgresql://localhost:5432/{cc.project_db_name('/Users/x/repos/widget')}"


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


def test_guard_env_short_circuits(monkeypatch, capsys):
    # When a summarizer's own subprocess triggers the hook, LCM_SUMMARIZING makes it a no-op.
    monkeypatch.setenv("LCM_SUMMARIZING", "1")
    assert cc.main([]) == 0
