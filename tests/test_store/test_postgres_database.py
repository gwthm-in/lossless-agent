"""Tests for PostgresDatabase – RED phase: these should FAIL until implementation exists."""
import os

import pytest

# Skip entire module if no Postgres available
pg_available = False
try:
    import psycopg2
    conn = psycopg2.connect(dbname="lossless_agent_test", host="localhost")
    conn.close()
    pg_available = True
except Exception:
    pass

pytestmark = pytest.mark.skipif(
    not pg_available, reason="PostgreSQL not available locally"
)


@pytest.fixture
def pg_db():
    """Provide a fresh PostgresDatabase for each test, dropping all tables after."""
    from lossless_agent.store.postgres_database import PostgresDatabase

    # Clean up any leftover tables first
    import psycopg2 as _pg2
    _conn = _pg2.connect(dbname="lossless_agent_test", host="localhost")
    _conn.autocommit = True
    _cur = _conn.cursor()
    _cur.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
    for (t,) in _cur.fetchall():
        _cur.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
    _conn.close()

    db = PostgresDatabase(dsn="dbname=lossless_agent_test host=localhost")
    yield db

    # Teardown: drop all tables
    try:
        db._raw_conn.rollback()  # Clear any error state
    except Exception:
        pass
    try:
        _conn2 = _pg2.connect(dbname="lossless_agent_test", host="localhost")
        _conn2.autocommit = True
        _cur2 = _conn2.cursor()
        _cur2.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")
        for (t,) in _cur2.fetchall():
            _cur2.execute(f'DROP TABLE IF EXISTS "{t}" CASCADE')
        _conn2.close()
    except Exception:
        pass
    try:
        db.close()
    except Exception:
        pass


class TestPostgresDatabaseInit:
    """PostgresDatabase should create schema on init, just like SQLite Database."""

    def test_creates_schema_version_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT version FROM schema_version"
        ).fetchone()
        assert row is not None
        assert row[0] == 4

    def test_creates_conversations_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'conversations' AND table_schema = 'public'"
        ).fetchone()
        assert row[0] == 1

    def test_creates_messages_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'messages' AND table_schema = 'public'"
        ).fetchone()
        assert row[0] == 1

    def test_creates_summaries_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'summaries' AND table_schema = 'public'"
        ).fetchone()
        assert row[0] == 1

    def test_creates_context_items_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'context_items' AND table_schema = 'public'"
        ).fetchone()
        assert row[0] == 1

    def test_creates_message_parts_table(self, pg_db):
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM information_schema.tables "
            "WHERE table_name = 'message_parts' AND table_schema = 'public'"
        ).fetchone()
        assert row[0] == 1


class TestPostgresConnectionAdapter:
    """The conn adapter should translate SQLite-style SQL to Postgres."""

    def test_question_mark_placeholders_work(self, pg_db):
        """Stores use ? placeholders – adapter must translate to %s."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        row = pg_db.conn.execute(
            "SELECT session_key FROM conversations WHERE session_key = ?",
            ("sess1",),
        ).fetchone()
        assert row[0] == "sess1"

    def test_lastrowid_works_after_insert(self, pg_db):
        """MessageStore relies on cur.lastrowid after INSERT."""
        cur = pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        assert cur.lastrowid is not None
        assert cur.lastrowid > 0

    def test_insert_or_ignore_syntax(self, pg_db):
        """INSERT OR IGNORE should work (translated to ON CONFLICT DO NOTHING)."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        # This should not raise
        pg_db.conn.execute(
            "INSERT OR IGNORE INTO conversations (session_key, title, active) "
            "VALUES (?, ?, 1)",
            ("sess1", "Dupe"),
        )
        pg_db.conn.commit()

    def test_strftime_now_translated(self, pg_db):
        """strftime('%Y-%m-%dT%H:%M:%f', 'now') should become CURRENT_TIMESTAMP."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        # This UPDATE uses strftime – adapter must translate
        pg_db.conn.execute(
            "UPDATE conversations SET active = 0, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE session_key = ?",
            ("sess1",),
        )
        pg_db.conn.commit()
        row = pg_db.conn.execute(
            "SELECT updated_at FROM conversations WHERE session_key = ?",
            ("sess1",),
        ).fetchone()
        assert row[0] is not None

    def test_fetchone_returns_tuple(self, pg_db):
        """Results should be tuples like sqlite3, not DictRow."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        row = pg_db.conn.execute(
            "SELECT id, session_key FROM conversations WHERE session_key = ?",
            ("sess1",),
        ).fetchone()
        assert isinstance(row, tuple)

    def test_fetchall_returns_list_of_tuples(self, pg_db):
        """Results should be list of tuples."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        rows = pg_db.conn.execute(
            "SELECT id, session_key FROM conversations"
        ).fetchall()
        assert isinstance(rows, list)
        assert all(isinstance(r, tuple) for r in rows)


class TestPostgresSchemaConstraints:
    """Postgres schema should enforce the same constraints as SQLite."""

    def test_conversations_session_key_unique_when_active(self, pg_db):
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        with pytest.raises(Exception):
            pg_db.conn.execute(
                "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
                ("sess1", "Dupe"),
            )
        pg_db._raw_conn.rollback()  # Clear error state for next operations

    def test_message_role_check_constraint(self, pg_db):
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        conv_id = pg_db.conn.execute(
            "SELECT id FROM conversations WHERE session_key = ?", ("sess1",)
        ).fetchone()[0]
        with pytest.raises(Exception):
            pg_db.conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, 1, "invalid_role", "bad", 1),
            )
        pg_db._raw_conn.rollback()  # Clear error state

    def test_conversations_default_timestamp(self, pg_db):
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        row = pg_db.conn.execute(
            "SELECT created_at FROM conversations WHERE session_key = ?",
            ("sess1",),
        ).fetchone()
        assert row[0] is not None


class TestPostgresFTS:
    """Postgres should use tsvector/tsquery for full-text search."""

    def test_summaries_fts_index_exists(self, pg_db):
        """There should be a GIN index on summaries for FTS."""
        row = pg_db.conn.execute(
            "SELECT COUNT(*) FROM pg_indexes "
            "WHERE tablename = 'summaries' AND indexname = 'summaries_fts_idx'"
        ).fetchone()
        assert row[0] == 1

    def test_summaries_searchable_via_tsquery(self, pg_db):
        """Summaries should be searchable via to_tsquery."""
        pg_db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        pg_db.conn.commit()
        conv_id = pg_db.conn.execute(
            "SELECT id FROM conversations WHERE session_key = ?", ("sess1",)
        ).fetchone()[0]
        pg_db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES (?, ?, 'leaf', 0, 'quantum computing transforms cryptography', 5, 25, "
            "'2025-01-01', '2025-01-01', 'gpt-4')",
            ("sum_001", conv_id),
        )
        pg_db.conn.commit()
        rows = pg_db.conn.execute(
            "SELECT summary_id FROM summaries "
            "WHERE to_tsvector('english', content) @@ to_tsquery('english', ?)",
            ("quantum",),
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "sum_001"


class TestPostgresClose:
    def test_close_closes_connection(self, pg_db):
        pg_db.close()
        # After close, executing should raise
        with pytest.raises(Exception):
            pg_db.conn.execute("SELECT 1")


class TestPostgresBackendType:
    """Database objects should expose their backend type."""

    def test_postgres_backend_type(self, pg_db):
        assert pg_db.backend == "postgres"

    def test_sqlite_backend_type(self):
        from lossless_agent.store.database import Database
        db = Database(":memory:")
        assert db.backend == "sqlite"
        db.close()
