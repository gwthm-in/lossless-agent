"""Tests for schema v4 changes – closing gaps with lossless-claw."""
import json
import sqlite3

import pytest



# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _col_names(db, table):
    """Return set of column names for a table."""
    rows = db.conn.execute(f"PRAGMA table_info({table})").fetchall()
    return {r[1] for r in rows}


def _index_names(db):
    """Return set of index names in the database."""
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='index'"
    ).fetchall()
    return {r[0] for r in rows}


def _table_names(db):
    """Return set of table names."""
    rows = db.conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()
    return {r[0] for r in rows}


def _insert_conversation(db, session_key="sess1", title="Test"):
    db.conn.execute(
        "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
        (session_key, title),
    )
    return db.conn.execute(
        "SELECT id FROM conversations WHERE session_key=?", (session_key,)
    ).fetchone()[0]


# ---------------------------------------------------------------------------
# Schema version
# ---------------------------------------------------------------------------

class TestSchemaVersion:
    def test_schema_version_is_4(self, db):
        row = db.conn.execute("SELECT version FROM schema_version").fetchone()
        assert row[0] == 4


# ---------------------------------------------------------------------------
# 1. Conversations table new columns & indexes
# ---------------------------------------------------------------------------

class TestConversationsV4:
    def test_session_id_column_exists(self, db):
        assert "session_id" in _col_names(db, "conversations")

    def test_archived_at_column_exists(self, db):
        assert "archived_at" in _col_names(db, "conversations")

    def test_bootstrapped_at_column_exists(self, db):
        assert "bootstrapped_at" in _col_names(db, "conversations")

    def test_session_id_nullable(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES ('s1', 'T')"
        )
        row = db.conn.execute(
            "SELECT session_id FROM conversations WHERE session_key='s1'"
        ).fetchone()
        assert row[0] is None

    def test_session_id_stores_value(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title, session_id) "
            "VALUES ('s1', 'T', 'sid-abc')"
        )
        row = db.conn.execute(
            "SELECT session_id FROM conversations WHERE session_key='s1'"
        ).fetchone()
        assert row[0] == "sid-abc"

    def test_archived_at_nullable(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES ('s1', 'T')"
        )
        row = db.conn.execute(
            "SELECT archived_at FROM conversations WHERE session_key='s1'"
        ).fetchone()
        assert row[0] is None

    def test_bootstrapped_at_nullable(self, db):
        db.conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES ('s1', 'T')"
        )
        row = db.conn.execute(
            "SELECT bootstrapped_at FROM conversations WHERE session_key='s1'"
        ).fetchone()
        assert row[0] is None

    def test_partial_unique_index_allows_duplicate_inactive(self, db):
        """Two inactive conversations can share the same session_key."""
        db.conn.execute(
            "INSERT INTO conversations (session_key, title, active) VALUES ('s1', 'T', 0)"
        )
        # Should NOT raise
        db.conn.execute(
            "INSERT INTO conversations (session_key, title, active) VALUES ('s1', 'T2', 0)"
        )

    def test_partial_unique_index_blocks_duplicate_active(self, db):
        """Two active conversations cannot share the same session_key."""
        db.conn.execute(
            "INSERT INTO conversations (session_key, title, active) VALUES ('s1', 'T', 1)"
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO conversations (session_key, title, active) "
                "VALUES ('s1', 'T2', 1)"
            )

    def test_composite_index_exists(self, db):
        idxs = _index_names(db)
        assert "conversations_session_key_active_created_idx" in idxs

    def test_active_session_key_index_exists(self, db):
        idxs = _index_names(db)
        assert "conversations_active_session_key_idx" in idxs


# ---------------------------------------------------------------------------
# 2. Summaries table new columns
# ---------------------------------------------------------------------------

class TestSummariesV4:
    def test_file_ids_column_exists(self, db):
        assert "file_ids" in _col_names(db, "summaries")

    def test_descendant_count_column_exists(self, db):
        assert "descendant_count" in _col_names(db, "summaries")

    def test_descendant_token_count_column_exists(self, db):
        assert "descendant_token_count" in _col_names(db, "summaries")

    def test_descendant_count_defaults_to_zero(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s1', ?, 'leaf', 0, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        row = db.conn.execute(
            "SELECT descendant_count FROM summaries WHERE summary_id='s1'"
        ).fetchone()
        assert row[0] == 0

    def test_descendant_token_count_defaults_to_zero(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s1', ?, 'leaf', 0, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        row = db.conn.execute(
            "SELECT descendant_token_count FROM summaries WHERE summary_id='s1'"
        ).fetchone()
        assert row[0] == 0

    def test_file_ids_stores_json(self, db):
        cid = _insert_conversation(db)
        ids = json.dumps(["f1", "f2"])
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model, file_ids) "
            "VALUES ('s1', ?, 'leaf', 0, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm', ?)",
            (cid, ids),
        )
        row = db.conn.execute(
            "SELECT file_ids FROM summaries WHERE summary_id='s1'"
        ).fetchone()
        assert json.loads(row[0]) == ["f1", "f2"]


# ---------------------------------------------------------------------------
# 3. summary_messages ordinal
# ---------------------------------------------------------------------------

class TestSummaryMessagesV4:
    def test_ordinal_column_exists(self, db):
        assert "ordinal" in _col_names(db, "summary_messages")

    def test_ordinal_defaults_to_zero(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content) "
            "VALUES (?, 1, 'user', 'hi')",
            (cid,),
        )
        mid = db.conn.execute("SELECT id FROM messages").fetchone()[0]
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s1', ?, 'leaf', 0, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        db.conn.execute(
            "INSERT INTO summary_messages (summary_id, message_id) VALUES ('s1', ?)",
            (mid,),
        )
        row = db.conn.execute(
            "SELECT ordinal FROM summary_messages WHERE summary_id='s1'"
        ).fetchone()
        assert row[0] == 0


# ---------------------------------------------------------------------------
# 4. summary_parents ordinal
# ---------------------------------------------------------------------------

class TestSummaryParentsV4:
    def test_ordinal_column_exists(self, db):
        assert "ordinal" in _col_names(db, "summary_parents")

    def test_ordinal_defaults_to_zero(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s1', ?, 'leaf', 0, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s2', ?, 'condensed', 1, 'x', 1, 1, '2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        db.conn.execute(
            "INSERT INTO summary_parents (parent_id, child_id) VALUES ('s2', 's1')"
        )
        row = db.conn.execute(
            "SELECT ordinal FROM summary_parents WHERE parent_id='s2'"
        ).fetchone()
        assert row[0] == 0


# ---------------------------------------------------------------------------
# 5. message_parts new columns
# ---------------------------------------------------------------------------

class TestMessagePartsV4:
    def test_new_columns_exist(self, db):
        cols = _col_names(db, "message_parts")
        for col in [
            "session_id", "tool_error", "tool_title",
            "patch_old", "patch_new", "file_name", "file_content",
            "snapshot_hash", "compaction_auto",
        ]:
            assert col in cols, f"missing column: {col}"

    def test_compaction_auto_defaults_to_zero(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content) "
            "VALUES (?, 1, 'user', 'hi')",
            (cid,),
        )
        mid = db.conn.execute("SELECT id FROM messages").fetchone()[0]
        db.conn.execute(
            "INSERT INTO message_parts (part_id, message_id, part_type) "
            "VALUES ('p1', ?, 'text')",
            (mid,),
        )
        row = db.conn.execute(
            "SELECT compaction_auto FROM message_parts WHERE part_id='p1'"
        ).fetchone()
        assert row[0] == 0


# ---------------------------------------------------------------------------
# 6. large_files_v2 table
# ---------------------------------------------------------------------------

class TestLargeFilesV2:
    def test_large_files_v2_table_exists(self, db):
        assert "large_files_v2" in _table_names(db)

    def test_large_files_v2_columns(self, db):
        cols = _col_names(db, "large_files_v2")
        expected = {
            "file_id", "conversation_id", "file_name", "mime_type",
            "byte_size", "storage_uri", "exploration_summary", "created_at",
        }
        assert expected.issubset(cols), f"Missing: {expected - cols}"

    def test_large_files_v2_insert(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO large_files_v2 (file_id, conversation_id, file_name, "
            "mime_type, byte_size, storage_uri, exploration_summary) "
            "VALUES ('f1', ?, 'test.py', 'text/plain', 1234, '/store/f1', 'a file')",
            (cid,),
        )
        row = db.conn.execute(
            "SELECT file_id, byte_size FROM large_files_v2 WHERE file_id='f1'"
        ).fetchone()
        assert row == ("f1", 1234)

    def test_large_files_v2_file_id_is_primary_key(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO large_files_v2 (file_id, conversation_id) VALUES ('f1', ?)",
            (cid,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO large_files_v2 (file_id, conversation_id) VALUES ('f1', ?)",
                (cid,),
            )

    def test_old_large_files_table_still_exists(self, db):
        """Backward compat: old table kept around."""
        assert "large_files" in _table_names(db)


# ---------------------------------------------------------------------------
# 7. conversation_bootstrap_state table
# ---------------------------------------------------------------------------

class TestConversationBootstrapState:
    def test_table_exists(self, db):
        assert "conversation_bootstrap_state" in _table_names(db)

    def test_columns(self, db):
        cols = _col_names(db, "conversation_bootstrap_state")
        expected = {
            "conversation_id", "session_file_path",
            "last_seen_size", "last_seen_mtime_ms",
            "last_processed_offset", "last_processed_entry_hash",
            "updated_at",
        }
        assert expected.issubset(cols), f"Missing: {expected - cols}"

    def test_insert_and_defaults(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO conversation_bootstrap_state "
            "(conversation_id, session_file_path) VALUES (?, '/tmp/session.jsonl')",
            (cid,),
        )
        row = db.conn.execute(
            "SELECT last_seen_size, last_seen_mtime_ms, last_processed_offset "
            "FROM conversation_bootstrap_state WHERE conversation_id=?",
            (cid,),
        ).fetchone()
        assert row == (0, 0, 0)

    def test_primary_key_on_conversation_id(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO conversation_bootstrap_state "
            "(conversation_id, session_file_path) VALUES (?, '/tmp/a.jsonl')",
            (cid,),
        )
        with pytest.raises(sqlite3.IntegrityError):
            db.conn.execute(
                "INSERT INTO conversation_bootstrap_state "
                "(conversation_id, session_file_path) VALUES (?, '/tmp/b.jsonl')",
                (cid,),
            )

    def test_bootstrap_state_path_index_exists(self, db):
        assert "bootstrap_state_path_idx" in _index_names(db)


# ---------------------------------------------------------------------------
# 8. summaries_fts_cjk FTS table
# ---------------------------------------------------------------------------

class TestSummariesFtsCjk:
    def test_fts_cjk_table_exists(self, db):
        tables = _table_names(db)
        assert "summaries_fts_cjk" in tables

    def test_fts_cjk_syncs_on_insert(self, db):
        cid = _insert_conversation(db)
        db.conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES ('s1', ?, 'leaf', 0, 'trigram test content', 3, 10, "
            "'2025-01-01', '2025-01-01', 'm')",
            (cid,),
        )
        rows = db.conn.execute(
            "SELECT summary_id FROM summaries_fts_cjk WHERE content MATCH 'trigram'"
        ).fetchall()
        assert len(rows) == 1
        assert rows[0][0] == "s1"


# ---------------------------------------------------------------------------
# 9. Missing indexes
# ---------------------------------------------------------------------------

class TestMissingIndexes:
    def test_messages_conv_seq_idx(self, db):
        assert "messages_conv_seq_idx" in _index_names(db)

    def test_summaries_conv_created_idx(self, db):
        assert "summaries_conv_created_idx" in _index_names(db)

    def test_large_files_conv_idx(self, db):
        assert "large_files_conv_idx" in _index_names(db)
