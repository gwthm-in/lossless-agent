"""Tests for VectorStore (pgvector semantic store).

All tests mock psycopg2 — no real Postgres connection required.
"""
from __future__ import annotations

from unittest.mock import MagicMock, call, patch, PropertyMock

import pytest

from lossless_agent.store.vector_store import VectorStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_conn():
    """Build a mock psycopg2 connection with a cursor that supports context mgr."""
    conn = MagicMock()
    conn.closed = 0  # psycopg2: 0 = open

    cursor = MagicMock()
    cursor.__enter__ = MagicMock(return_value=cursor)
    cursor.__exit__ = MagicMock(return_value=False)
    conn.cursor.return_value = cursor

    return conn, cursor


def _make_store(dsn="postgresql://localhost/test", dim=3):
    """Return a VectorStore with a mocked psycopg2 connection."""
    store = VectorStore.__new__(VectorStore)
    store._dsn = dsn
    store._dim = dim
    conn, cursor = _make_mock_conn()
    store._conn = conn
    return store, conn, cursor


# ---------------------------------------------------------------------------
# _ensure_schema
# ---------------------------------------------------------------------------

class TestEnsureSchema:
    def test_creates_extension_and_table(self):
        store, conn, _ = _make_store()
        # All DDL succeeds
        store._ensure_schema()

        calls = [str(c) for c in conn.cursor().execute.call_args_list]
        combined = " ".join(calls)
        assert "CREATE EXTENSION" in combined
        assert "summary_embeddings" in combined
        assert "vector" in combined

    def test_continues_when_create_extension_fails(self):
        """Permission error on CREATE EXTENSION should not abort schema setup."""
        store, conn, _ = _make_store()
        cursor = conn.cursor.return_value

        call_count = 0

        def side_effect(sql, *args):
            nonlocal call_count
            call_count += 1
            if "CREATE EXTENSION" in sql:
                raise Exception("permission denied to create extension")
            # All other calls succeed

        cursor.execute.side_effect = side_effect
        # Should not raise
        store._ensure_schema()
        # rollback called for the failed extension, then commit for table DDL
        assert conn.rollback.called

    def test_hnsw_fallback_to_ivfflat(self):
        """HNSW failure falls back to IVFFlat without raising."""
        store, conn, _ = _make_store()
        cursor = conn.cursor.return_value

        def side_effect(sql, *args):
            if "hnsw" in sql.lower():
                raise Exception("hnsw not supported")

        cursor.execute.side_effect = side_effect
        store._ensure_schema()  # must not raise
        # rollback called for HNSW failure
        assert conn.rollback.called


# ---------------------------------------------------------------------------
# store()
# ---------------------------------------------------------------------------

class TestStore:
    def test_upserts_embedding(self):
        store, conn, cursor = _make_store(dim=3)
        store.store("sum_abc", 42, [0.1, 0.2, 0.3])

        sql, params = cursor.execute.call_args[0]
        assert "INSERT INTO summary_embeddings" in sql
        assert "ON CONFLICT" in sql
        assert params[0] == "sum_abc"
        assert params[1] == 42
        assert params[2] == "[0.1,0.2,0.3]"
        conn.commit.assert_called()

    def test_vec_literal_format(self):
        assert VectorStore._vec_literal([1.0, -0.5, 0.0]) == "[1.0,-0.5,0.0]"

    def test_rollback_on_error(self):
        store, conn, cursor = _make_store()
        cursor.execute.side_effect = Exception("db error")
        with pytest.raises(Exception, match="db error"):
            store.store("sum_x", 1, [0.1, 0.2, 0.3])
        conn.rollback.assert_called()


# ---------------------------------------------------------------------------
# search()
# ---------------------------------------------------------------------------

class TestSearch:
    def test_returns_similarity_scores(self):
        store, conn, cursor = _make_store(dim=3)
        # SQL already computes 1.0 - distance AS similarity, so mock returns
        # similarity values directly (0.8 and 0.6)
        cursor.fetchall.return_value = [("sum_abc", 0.8), ("sum_xyz", 0.6)]

        results = store.search([0.1, 0.2, 0.3], top_k=2)

        assert len(results) == 2
        assert results[0] == ("sum_abc", pytest.approx(0.8))
        assert results[1] == ("sum_xyz", pytest.approx(0.6))

    def test_excludes_current_conversation(self):
        store, conn, cursor = _make_store(dim=3)
        cursor.fetchall.return_value = []

        store.search([0.1, 0.2, 0.3], top_k=5, exclude_conversation_id=7)

        sql = cursor.execute.call_args[0][0]
        assert "conversation_id != %s" in sql
        params = cursor.execute.call_args[0][1]
        assert 7 in params

    def test_no_exclude_when_not_specified(self):
        store, conn, cursor = _make_store(dim=3)
        cursor.fetchall.return_value = []

        store.search([0.1, 0.2, 0.3], top_k=5)

        sql = cursor.execute.call_args[0][0]
        assert "conversation_id" not in sql

    def test_returns_empty_list_when_no_hits(self):
        store, conn, cursor = _make_store(dim=3)
        cursor.fetchall.return_value = []
        assert store.search([0.1, 0.2, 0.3]) == []


# ---------------------------------------------------------------------------
# delete()
# ---------------------------------------------------------------------------

class TestDelete:
    def test_deletes_by_summary_id(self):
        store, conn, cursor = _make_store()
        store.delete("sum_abc")

        sql, params = cursor.execute.call_args[0]
        assert "DELETE FROM summary_embeddings" in sql
        assert params == ("sum_abc",)
        conn.commit.assert_called()

    def test_rollback_on_error(self):
        store, conn, cursor = _make_store()
        cursor.execute.side_effect = Exception("db error")
        with pytest.raises(Exception):
            store.delete("sum_abc")
        conn.rollback.assert_called()


# ---------------------------------------------------------------------------
# close()
# ---------------------------------------------------------------------------

class TestClose:
    def test_closes_connection(self):
        store, conn, _ = _make_store()
        store.close()
        conn.close.assert_called_once()
        assert store._conn is None

    def test_close_noop_when_already_none(self):
        store = VectorStore.__new__(VectorStore)
        store._conn = None
        store.close()  # should not raise
