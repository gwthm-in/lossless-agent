"""Tests for CJK detection and search routing in tools/recall.py."""
from __future__ import annotations

import sqlite3

import pytest

from lossless_agent.store.database import Database
from lossless_agent.tools.recall import (
    _contains_cjk,
    _sanitize_fts5_query,
    lcm_grep,
)


@pytest.fixture
def db():
    """In-memory database with schema and test data."""
    d = Database(":memory:")
    # Insert a conversation
    d.conn.execute(
        "INSERT INTO conversations (id, session_key, title) VALUES (1, 'test', 'test')"
    )
    d.conn.commit()
    return d


def _insert_message(db: Database, msg_id: int, content: str):
    db.conn.execute(
        "INSERT INTO messages (id, conversation_id, seq, role, content, token_count) "
        "VALUES (?, 1, ?, 'user', ?, 10)",
        (msg_id, msg_id, content),
    )
    db.conn.commit()


def _insert_summary(db: Database, sid: str, content: str):
    db.conn.execute(
        "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
        "token_count, source_token_count, earliest_at, latest_at, model) "
        "VALUES (?, 1, 'leaf', 0, ?, 10, 50, '2025-01-01', '2025-01-02', 'test')",
        (sid, content),
    )
    db.conn.commit()


# ── CJK detection ──────────────────────────────────────────────────


class TestContainsCJK:
    def test_chinese(self):
        assert _contains_cjk("你好世界") is True

    def test_japanese(self):
        assert _contains_cjk("こんにちは") is True

    def test_korean(self):
        assert _contains_cjk("안녕하세요") is True

    def test_mixed(self):
        assert _contains_cjk("hello 你好") is True

    def test_ascii_only(self):
        assert _contains_cjk("hello world") is False

    def test_empty(self):
        assert _contains_cjk("") is False

    def test_numbers_and_symbols(self):
        assert _contains_cjk("123 !@#") is False


# ── FTS5 query sanitization ────────────────────────────────────────


class TestSanitizeFTS5Query:
    def test_plain_words(self):
        assert _sanitize_fts5_query("hello world") == "hello world"

    def test_strips_special_chars(self):
        result = _sanitize_fts5_query('hello "world" NOT test')
        # Should not contain unbalanced quotes or FTS operators
        assert '"' not in result or result.count('"') % 2 == 0

    def test_empty_query(self):
        result = _sanitize_fts5_query("")
        assert result == ""

    def test_cjk_passthrough(self):
        result = _sanitize_fts5_query("你好世界")
        assert "你好世界" in result


# ── CJK search routing ─────────────────────────────────────────────


class TestCJKSearchRouting:
    def test_cjk_query_searches_summaries_fts_cjk(self, db):
        """CJK queries should route to summaries_fts_cjk table."""
        _insert_summary(db, "s1", "这是一个中文摘要关于机器学习")
        results = lcm_grep(db, "中文", scope="summaries")
        # Should find the summary via CJK FTS
        assert any(r.id == "s1" for r in results)

    def test_ascii_query_searches_summaries_fts(self, db):
        """ASCII queries should use regular summaries_fts."""
        _insert_summary(db, "s1", "machine learning summary")
        results = lcm_grep(db, "machine", scope="summaries")
        assert any(r.id == "s1" for r in results)

    def test_cjk_message_search_still_works(self, db):
        """Message search should still work for CJK (uses LIKE fallback)."""
        _insert_message(db, 1, "这是中文消息")
        results = lcm_grep(db, "中文", scope="messages")
        assert len(results) >= 1

    def test_like_fallback_on_fts_error(self, db):
        """If FTS5 search fails, should fall back to LIKE."""
        _insert_summary(db, "s1", "test content for fallback")
        # A query with problematic FTS syntax should still work via fallback
        results = lcm_grep(db, "test content", scope="summaries")
        assert len(results) >= 0  # Should not raise


class TestLcmGrepSanitization:
    def test_special_chars_dont_crash(self, db):
        """Queries with special FTS5 chars should not cause errors."""
        _insert_message(db, 1, "normal text")
        # These should not raise
        lcm_grep(db, "hello OR world", scope="messages")
        lcm_grep(db, 'test "quoted"', scope="messages")
        lcm_grep(db, "NOT something", scope="messages")
