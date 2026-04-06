"""Tests for database schema creation and constraints."""
import sqlite3

import pytest

from lossless_agent.store.database import Database


class TestSchemaCreation:
    """Verify all tables, indices, and triggers exist after init."""

    def test_tables_exist(self, db):
        cur = db.conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cur.fetchall()}
        expected = {
            "schema_version",
            "conversations",
            "messages",
            "summaries",
            "summary_messages",
            "summary_parents",
            "messages_fts",
            "summaries_fts",
        }
        # FTS5 creates shadow tables; just check our expected ones are present
        assert expected.issubset(tables), f"Missing tables: {expected - tables}"

    def test_schema_version_is_one(self, db):
        row = db.conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 1

    def test_wal_mode_on_file_db(self, db_file):
        mode = db_file.conn.execute("PRAGMA journal_mode").fetchone()[0]
        assert mode == "wal"

    def test_foreign_keys_enabled(self, db):
        fk = db.conn.execute("PRAGMA foreign_keys").fetchone()[0]
        assert fk == 1


class TestConversationsTable:
    def test_insert_conversation(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        row = db.conn.execute("SELECT * FROM conversations WHERE session_key='sess1'").fetchone()
        assert row is not None

    def test_session_key_unique(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
                ("sess1", "Dupe"),
            )

    def test_active_defaults_to_true(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        row = db.conn.execute(
            "SELECT active FROM conversations WHERE session_key='sess1'"
        ).fetchone()
        assert row[0] == 1  # SQLite stores bools as int


class TestMessagesTable:
    def _insert_conversation(self, db, session_key="sess1"):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            (session_key, "Test"),
        )
        return db.conn.execute(
            "SELECT id FROM conversations WHERE session_key=?", (session_key,)
        ).fetchone()[0]

    def test_insert_message(self, db):
        conv_id = self._insert_conversation(db)
        db.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, 1, "user", "hello", 1),
        )
        row = db.conn.execute("SELECT * FROM messages").fetchone()
        assert row is not None

    def test_role_check_constraint(self, db):
        conv_id = self._insert_conversation(db)
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, 1, "invalid_role", "bad", 1),
            )

    def test_seq_unique_per_conversation(self, db):
        conv_id = self._insert_conversation(db)
        db.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, 1, "user", "first", 1),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (conv_id, 1, "user", "dupe seq", 1),
            )

    def test_foreign_key_constraint(self, db):
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
                "VALUES (?, ?, ?, ?, ?)",
                (9999, 1, "user", "orphan", 1),
            )


class TestSummariesTable:
    def _insert_conversation(self, db, session_key="sess1"):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            (session_key, "Test"),
        )
        return db.conn.execute(
            "SELECT id FROM conversations WHERE session_key=?", (session_key,)
        ).fetchone()[0]

    def test_insert_summary(self, db):
        conv_id = self._insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("sum_0001", conv_id, "leaf", 0, "summary text", 10, 100,
             "2025-01-01T00:00:00", "2025-01-01T01:00:00", "gpt-4"),
        )
        row = db.conn.execute("SELECT * FROM summaries").fetchone()
        assert row is not None

    def test_kind_check_constraint(self, db):
        conv_id = self._insert_conversation(db)
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
                "token_count, source_token_count, earliest_at, latest_at, model) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("sum_0001", conv_id, "bad_kind", 0, "text", 10, 100,
                 "2025-01-01T00:00:00", "2025-01-01T01:00:00", "gpt-4"),
            )


class TestFTS5:
    def _setup_data(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            ("sess1", "Test"),
        )
        conv_id = db.conn.execute(
            "SELECT id FROM conversations WHERE session_key='sess1'"
        ).fetchone()[0]
        db.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (conv_id, 1, "user", "the quick brown fox jumps over the lazy dog", 9),
        )
        return conv_id

    def test_messages_fts_populated_on_insert(self, db):
        self._setup_data(db)
        rows = db.conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH 'fox'"
        ).fetchall()
        assert len(rows) == 1

    def test_messages_fts_updated_on_update(self, db):
        self._setup_data(db)
        msg_id = db.conn.execute("SELECT id FROM messages").fetchone()[0]
        db.conn.execute(
            "UPDATE messages SET content = 'the slow red cat' WHERE id = ?",
            (msg_id,),
        )
        # Old term gone
        rows = db.conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH 'fox'"
        ).fetchall()
        assert len(rows) == 0
        # New term present
        rows = db.conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH 'cat'"
        ).fetchall()
        assert len(rows) == 1

    def test_messages_fts_cleaned_on_delete(self, db):
        self._setup_data(db)
        msg_id = db.conn.execute("SELECT id FROM messages").fetchone()[0]
        db.conn.execute("DELETE FROM messages WHERE id = ?", (msg_id,))
        rows = db.conn.execute(
            "SELECT * FROM messages_fts WHERE messages_fts MATCH 'fox'"
        ).fetchall()
        assert len(rows) == 0

    def test_summaries_fts_populated_on_insert(self, db):
        conv_id = self._setup_data(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            ("sum_0001", conv_id, "leaf", 0, "unique searchable summary content", 5, 50,
             "2025-01-01T00:00:00", "2025-01-01T01:00:00", "gpt-4"),
        )
        rows = db.conn.execute(
            "SELECT * FROM summaries_fts WHERE summaries_fts MATCH 'searchable'"
        ).fetchall()
        assert len(rows) == 1
