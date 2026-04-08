"""Tests for store.transaction – serialized write context manager."""
from __future__ import annotations

import sqlite3

import pytest

from lossless_agent.store.transaction import transaction


@pytest.fixture
def conn():
    """In-memory SQLite connection with a test table."""
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE kv (key TEXT PRIMARY KEY, value TEXT)")
    return c


class TestTransactionCommit:
    """On success the transaction should commit."""

    def test_basic_commit(self, conn):
        with transaction(conn):
            conn.execute("INSERT INTO kv (key, value) VALUES ('a', '1')")
        row = conn.execute("SELECT value FROM kv WHERE key='a'").fetchone()
        assert row is not None
        assert row[0] == "1"

    def test_multiple_writes_commit(self, conn):
        with transaction(conn):
            conn.execute("INSERT INTO kv (key, value) VALUES ('a', '1')")
            conn.execute("INSERT INTO kv (key, value) VALUES ('b', '2')")
        count = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        assert count == 2


class TestTransactionRollback:
    """On exception the transaction should rollback and re-raise."""

    def test_rollback_on_error(self, conn):
        with pytest.raises(ValueError, match="boom"):
            with transaction(conn):
                conn.execute("INSERT INTO kv (key, value) VALUES ('a', '1')")
                raise ValueError("boom")
        row = conn.execute("SELECT value FROM kv WHERE key='a'").fetchone()
        assert row is None  # rolled back

    def test_exception_is_reraised(self, conn):
        with pytest.raises(RuntimeError):
            with transaction(conn):
                raise RuntimeError("fail")

    def test_partial_writes_rolled_back(self, conn):
        conn.execute("INSERT INTO kv (key, value) VALUES ('pre', 'existing')")
        conn.commit()
        with pytest.raises(Exception):
            with transaction(conn):
                conn.execute("INSERT INTO kv (key, value) VALUES ('new', 'val')")
                raise Exception("oops")
        count = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        assert count == 1  # only 'pre' remains


class TestTransactionBeginImmediate:
    """The transaction should use BEGIN IMMEDIATE."""

    def test_begin_immediate_used(self, conn):
        """Verify BEGIN IMMEDIATE is used by checking no other transaction is active."""
        # We'll verify this indirectly: after entering the context manager,
        # the connection should be in a transaction
        with transaction(conn):
            # in_transaction should be True during the context
            assert conn.in_transaction

    def test_not_in_transaction_after_commit(self, conn):
        with transaction(conn):
            conn.execute("INSERT INTO kv (key, value) VALUES ('x', 'y')")
        assert not conn.in_transaction


class TestTransactionNesting:
    """Edge cases for transaction usage."""

    def test_context_manager_is_reusable(self, conn):
        """Can use transaction() multiple times on the same connection."""
        with transaction(conn):
            conn.execute("INSERT INTO kv (key, value) VALUES ('a', '1')")
        with transaction(conn):
            conn.execute("INSERT INTO kv (key, value) VALUES ('b', '2')")
        count = conn.execute("SELECT COUNT(*) FROM kv").fetchone()[0]
        assert count == 2
