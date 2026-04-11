"""Tests for VectorStore message embedding API (raw vector retrieval).

All tests mock psycopg2 — no real Postgres connection required.
"""
from __future__ import annotations

from unittest.mock import MagicMock

from lossless_agent.store.vector_store import VectorStore


def _make_store(dim=3, msg_dim=3):
    """Return a VectorStore with a mocked psycopg2 connection."""
    store = VectorStore.__new__(VectorStore)
    store._dsn = "postgresql://localhost/test"
    store._dim = dim
    store._msg_dim = msg_dim
    conn = MagicMock()
    conn.closed = 0
    cursor = MagicMock()
    conn.cursor.return_value = cursor
    store._conn = conn
    return store, conn, cursor


class TestStoreMessage:
    def test_inserts_single_message(self):
        store, conn, cursor = _make_store()
        store.store_message("msg_1", 42, [0.1, 0.2, 0.3])
        cursor.execute.assert_called_once()
        sql = cursor.execute.call_args[0][0]
        assert "message_embeddings" in sql
        assert "INSERT" in sql
        params = cursor.execute.call_args[0][1]
        assert params[0] == "msg_1"
        assert params[1] == 42
        conn.commit.assert_called_once()

    def test_upserts_on_conflict(self):
        store, conn, cursor = _make_store()
        store.store_message("msg_1", 42, [0.1, 0.2, 0.3])
        sql = cursor.execute.call_args[0][0]
        assert "ON CONFLICT" in sql


class TestStoreMessagesBatch:
    def test_batch_empty(self):
        store, conn, cursor = _make_store()
        store.store_messages_batch([])
        cursor.execute.assert_not_called()

    def test_batch_multiple(self):
        store, conn, cursor = _make_store()
        items = [
            ("msg_1", 1, [0.1, 0.2, 0.3]),
            ("msg_2", 1, [0.4, 0.5, 0.6]),
            ("msg_3", 2, [0.7, 0.8, 0.9]),
        ]
        store.store_messages_batch(items)
        assert cursor.execute.call_count == 3
        conn.commit.assert_called_once()

    def test_batch_rollback_on_error(self):
        store, conn, cursor = _make_store()
        cursor.execute.side_effect = RuntimeError("db error")
        try:
            store.store_messages_batch([("msg_1", 1, [0.1, 0.2, 0.3])])
        except RuntimeError:
            pass
        conn.rollback.assert_called_once()


class TestSearchMessages:
    def test_basic_search(self):
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = [
            ("msg_1", 0.92),
            ("msg_2", 0.85),
        ]
        results = store.search_messages([0.1, 0.2, 0.3], top_k=5)
        assert len(results) == 2
        assert results[0] == ("msg_1", 0.92)
        assert results[1] == ("msg_2", 0.85)

    def test_min_score_pushed_into_sql_where_clause(self):
        """min_score must appear in the SQL WHERE clause, not filtered in Python."""
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = [("msg_1", 0.92)]
        store.search_messages([0.1, 0.2, 0.3], min_score=0.35)
        sql = cursor.execute.call_args[0][0]
        # The threshold condition must be in the WHERE clause
        assert "WHERE" in sql
        # The params list must include the threshold value
        params = cursor.execute.call_args[0][1]
        assert 0.35 in params

    def test_min_score_zero_omits_threshold_clause(self):
        """min_score=0 should not add a WHERE clause for the threshold."""
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = []
        store.search_messages([0.1, 0.2, 0.3], min_score=0.0)
        sql = cursor.execute.call_args[0][0]
        params = cursor.execute.call_args[0][1]
        # No threshold value injected
        assert 0.0 not in params

    def test_filters_by_conversation_ids(self):
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = [("msg_1", 0.9)]
        results = store.search_messages(
            [0.1, 0.2, 0.3],
            conversation_ids=[1, 2, 3],
        )
        sql = cursor.execute.call_args[0][0]
        assert "conversation_id IN" in sql

    def test_exclude_conversation(self):
        store, conn, cursor = _make_store()
        cursor.fetchall.return_value = []
        results = store.search_messages(
            [0.1, 0.2, 0.3],
            exclude_conversation_id=5,
        )
        sql = cursor.execute.call_args[0][0]
        assert "conversation_id !=" in sql


class TestDeleteMessage:
    def test_deletes_by_id(self):
        store, conn, cursor = _make_store()
        store.delete_message("msg_1")
        sql = cursor.execute.call_args[0][0]
        assert "DELETE" in sql
        assert "message_embeddings" in sql
        conn.commit.assert_called_once()


class TestMessageEmbeddingCount:
    def test_returns_count(self):
        store, conn, cursor = _make_store()
        cursor.fetchone.return_value = (42,)
        assert store.message_embedding_count() == 42
