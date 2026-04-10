"""Tests for Postgres FTS dispatch in recall search functions."""
from __future__ import annotations

from unittest.mock import MagicMock

from lossless_agent.tools.recall import (
    _fts_search_messages,
    _fts_search_summaries,
)


def _make_mock_db(backend: str = "postgres") -> MagicMock:
    """Create a mock db with the given backend attribute."""
    db = MagicMock()
    db.backend = backend
    return db


class TestFtsSearchMessagesPostgres:
    """_fts_search_messages dispatches to Postgres path when backend == 'postgres'."""

    def test_dispatches_to_postgres_path(self):
        db = _make_mock_db("postgres")
        db.conn.execute.return_value.fetchall.return_value = [
            (1, 100, "hello world", "user", 1, "2025-01-01T00:00:00"),
        ]

        results = _fts_search_messages(db, "hello", None, 10)

        assert len(results) == 1
        assert results[0].type == "message"
        assert results[0].content_snippet == "hello world"
        # Verify SQL uses plainto_tsquery (Postgres path)
        sql_called = db.conn.execute.call_args[0][0]
        assert "plainto_tsquery" in sql_called
        assert "ts_rank" in sql_called

    def test_dispatches_to_postgres_with_filters(self):
        db = _make_mock_db("postgres")
        db.conn.execute.return_value.fetchall.return_value = []

        _fts_search_messages(
            db, "test", conversation_id=5, limit=10,
            since="2025-01-01", before="2025-12-31",
        )

        sql_called = db.conn.execute.call_args[0][0]
        assert "conversation_id" in sql_called
        assert "created_at >=" in sql_called
        assert "created_at <" in sql_called

    def test_like_fallback_when_fts_fails(self):
        db = _make_mock_db("postgres")
        # First call (FTS) raises, second call (LIKE fallback) succeeds
        db.conn.execute.side_effect = [
            Exception("tsvector error"),
            MagicMock(fetchall=MagicMock(return_value=[])),
        ]

        results = _fts_search_messages(db, "hello", None, 10)

        assert results == []
        # Should have been called twice: FTS attempt + LIKE fallback
        assert db.conn.execute.call_count == 2
        like_sql = db.conn.execute.call_args[0][0]
        assert "LIKE" in like_sql


class TestFtsSearchSummariesPostgres:
    """_fts_search_summaries dispatches to Postgres path when backend == 'postgres'."""

    def test_dispatches_to_postgres_path(self):
        db = _make_mock_db("postgres")
        db.conn.execute.return_value.fetchall.return_value = [
            ("sum_abc", 100, "summary content", "leaf", 0, "2025-01-01T00:00:00"),
        ]

        results = _fts_search_summaries(db, "summary", None, 10)

        assert len(results) == 1
        assert results[0].type == "summary"
        assert results[0].id == "sum_abc"
        sql_called = db.conn.execute.call_args[0][0]
        assert "plainto_tsquery" in sql_called
        assert "ts_rank" in sql_called

    def test_dispatches_to_postgres_with_filters(self):
        db = _make_mock_db("postgres")
        db.conn.execute.return_value.fetchall.return_value = []

        _fts_search_summaries(
            db, "test", conversation_id=3, limit=5,
            since="2025-06-01", before="2025-12-01",
        )

        sql_called = db.conn.execute.call_args[0][0]
        assert "conversation_id" in sql_called
        assert "created_at >=" in sql_called
        assert "created_at <" in sql_called

    def test_like_fallback_when_fts_fails(self):
        db = _make_mock_db("postgres")
        db.conn.execute.side_effect = [
            Exception("tsvector error"),
            MagicMock(fetchall=MagicMock(return_value=[])),
        ]

        results = _fts_search_summaries(db, "test", None, 10)

        assert results == []
        assert db.conn.execute.call_count == 2
        like_sql = db.conn.execute.call_args[0][0]
        assert "LIKE" in like_sql


class TestFtsSearchSQLitePath:
    """Verify SQLite path is still used when backend is not postgres."""

    def test_messages_uses_fts5_for_sqlite(self):
        db = _make_mock_db("sqlite")
        db.conn.execute.return_value.fetchall.return_value = [
            (1, 100, "hello world", "user", 1, "2025-01-01T00:00:00"),
        ]

        _fts_search_messages(db, "hello", None, 10)

        sql_called = db.conn.execute.call_args[0][0]
        assert "messages_fts MATCH" in sql_called
        assert "plainto_tsquery" not in sql_called
