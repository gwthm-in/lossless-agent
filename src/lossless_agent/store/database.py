"""SQLite database initialization and schema management."""
from __future__ import annotations

import sqlite3

_SCHEMA_SQL = """\
-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Conversations (v4: session_id, archived_at, bootstrapped_at; no table-level UNIQUE on session_key)
CREATE TABLE IF NOT EXISTS conversations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key     TEXT    NOT NULL,
    session_id      TEXT,
    title           TEXT    NOT NULL DEFAULT '',
    active          INTEGER NOT NULL DEFAULT 1,
    archived_at     TEXT,
    bootstrapped_at TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
-- Partial unique: only one active conversation per session_key
CREATE UNIQUE INDEX IF NOT EXISTS conversations_active_session_key_idx
    ON conversations(session_key) WHERE active=1;
-- Composite lookup index
CREATE INDEX IF NOT EXISTS conversations_session_key_active_created_idx
    ON conversations(session_key, active, created_at);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT    NOT NULL DEFAULT '',
    token_count     INTEGER NOT NULL DEFAULT 0,
    tool_call_id    TEXT,
    tool_name       TEXT,
    created_at      TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    UNIQUE (conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_conv_seq_idx ON messages(conversation_id, seq);

-- Summaries (DAG nodes) – v4: file_ids, descendant_count, descendant_token_count
CREATE TABLE IF NOT EXISTS summaries (
    summary_id              TEXT    PRIMARY KEY,
    conversation_id         INTEGER NOT NULL REFERENCES conversations(id),
    kind                    TEXT    NOT NULL CHECK (kind IN ('leaf', 'condensed')),
    depth                   INTEGER NOT NULL DEFAULT 0,
    content                 TEXT    NOT NULL DEFAULT '',
    token_count             INTEGER NOT NULL DEFAULT 0,
    source_token_count      INTEGER NOT NULL DEFAULT 0,
    earliest_at             TEXT    NOT NULL,
    latest_at               TEXT    NOT NULL,
    model                   TEXT    NOT NULL DEFAULT '',
    file_ids                TEXT,
    descendant_count        INTEGER NOT NULL DEFAULT 0,
    descendant_token_count  INTEGER NOT NULL DEFAULT 0,
    created_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
CREATE INDEX IF NOT EXISTS summaries_conv_created_idx ON summaries(conversation_id, created_at);

-- Leaf summary -> source messages (many-to-many) – v4: ordinal
CREATE TABLE IF NOT EXISTS summary_messages (
    summary_id TEXT    NOT NULL REFERENCES summaries(summary_id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    ordinal    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (summary_id, message_id)
);

-- Condensed summary -> child summaries (DAG edges) – v4: ordinal
CREATE TABLE IF NOT EXISTS summary_parents (
    parent_id TEXT NOT NULL REFERENCES summaries(summary_id),
    child_id  TEXT NOT NULL REFERENCES summaries(summary_id),
    ordinal   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (parent_id, child_id)
);

-- Large file storage (legacy – kept for backward compatibility)
CREATE TABLE IF NOT EXISTS large_files (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id     INTEGER REFERENCES conversations(id),
    message_id          INTEGER REFERENCES messages(id),
    content             TEXT    NOT NULL DEFAULT '',
    token_count         INTEGER NOT NULL DEFAULT 0,
    summary             TEXT    NOT NULL DEFAULT '',
    summary_token_count INTEGER NOT NULL DEFAULT 0,
    mime_type           TEXT,
    file_path           TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
CREATE INDEX IF NOT EXISTS large_files_conv_idx ON large_files(conversation_id);

-- Large files v2 – URI-based approach matching lossless-claw
CREATE TABLE IF NOT EXISTS large_files_v2 (
    file_id             TEXT PRIMARY KEY,
    conversation_id     INTEGER REFERENCES conversations(id),
    file_name           TEXT,
    mime_type           TEXT,
    byte_size           INTEGER NOT NULL DEFAULT 0,
    storage_uri         TEXT,
    exploration_summary TEXT,
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);
CREATE INDEX IF NOT EXISTS large_files_v2_conv_idx ON large_files_v2(conversation_id);

-- FTS5 virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    content, content='summaries', content_rowid='rowid'
);

-- CJK trigram FTS for summaries (v4)
CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts_cjk USING fts5(
    content, summary_id UNINDEXED, tokenize='trigram'
);

-- Triggers to keep FTS in sync
CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_ad AFTER DELETE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
END;

CREATE TRIGGER IF NOT EXISTS messages_au AFTER UPDATE ON messages BEGIN
    INSERT INTO messages_fts(messages_fts, rowid, content) VALUES('delete', old.id, old.content);
    INSERT INTO messages_fts(rowid, content) VALUES (new.id, new.content);
END;

CREATE TRIGGER IF NOT EXISTS summaries_ai AFTER INSERT ON summaries BEGIN
    INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
    INSERT INTO summaries_fts_cjk(content, summary_id) VALUES (new.content, new.summary_id);
END;

CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    DELETE FROM summaries_fts_cjk WHERE summary_id = old.summary_id;
END;

CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
    DELETE FROM summaries_fts_cjk WHERE summary_id = old.summary_id;
    INSERT INTO summaries_fts_cjk(content, summary_id) VALUES (new.content, new.summary_id);
END;

-- Context items: ordered context window tracking
CREATE TABLE IF NOT EXISTS context_items (
    conversation_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL,
    item_type TEXT NOT NULL CHECK(item_type IN ('message', 'summary')),
    message_id TEXT,
    summary_id TEXT,
    PRIMARY KEY (conversation_id, ordinal),
    CHECK ((item_type='message' AND message_id IS NOT NULL AND summary_id IS NULL) OR
           (item_type='summary' AND summary_id IS NOT NULL AND message_id IS NULL))
);
CREATE INDEX IF NOT EXISTS context_items_conv_idx ON context_items(conversation_id);

-- Message parts: structured storage for multi-part messages (v4: many new columns)
CREATE TABLE IF NOT EXISTS message_parts (
    part_id TEXT PRIMARY KEY,
    message_id TEXT NOT NULL,
    part_type TEXT NOT NULL CHECK(part_type IN ('text', 'tool_call', 'tool_result', 'reasoning', 'file', 'patch', 'snapshot', 'image', 'audio', 'video', 'code', 'other')),
    ordinal INTEGER NOT NULL DEFAULT 0,
    text_content TEXT,
    tool_call_id TEXT,
    tool_name TEXT,
    tool_input TEXT,
    tool_output TEXT,
    tool_status TEXT,
    metadata TEXT,
    session_id TEXT,
    tool_error TEXT,
    tool_title TEXT,
    patch_old TEXT,
    patch_new TEXT,
    file_name TEXT,
    file_content TEXT,
    snapshot_hash TEXT,
    compaction_auto INTEGER DEFAULT 0,
    FOREIGN KEY (message_id) REFERENCES messages(id)
);
CREATE INDEX IF NOT EXISTS message_parts_message_idx ON message_parts(message_id);
CREATE INDEX IF NOT EXISTS message_parts_type_idx ON message_parts(part_type);

-- Conversation bootstrap state (v4)
CREATE TABLE IF NOT EXISTS conversation_bootstrap_state (
    conversation_id         INTEGER NOT NULL REFERENCES conversations(id),
    session_file_path       TEXT    NOT NULL,
    last_seen_size          INTEGER NOT NULL DEFAULT 0,
    last_seen_mtime_ms      INTEGER NOT NULL DEFAULT 0,
    last_processed_offset   INTEGER NOT NULL DEFAULT 0,
    last_processed_entry_hash TEXT,
    updated_at              TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    PRIMARY KEY (conversation_id)
);
CREATE INDEX IF NOT EXISTS bootstrap_state_path_idx ON conversation_bootstrap_state(session_file_path);
"""


class Database:
    """Manages a SQLite connection with the lossless-agent schema."""

    backend = "sqlite"

    def __init__(self, path: str = ":memory:") -> None:
        self.path = path
        self.conn = sqlite3.connect(path)
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.execute("PRAGMA busy_timeout = 5000")
        if path != ":memory:":
            self.conn.execute("PRAGMA journal_mode = WAL")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist and set schema version."""
        self.conn.executescript(_SCHEMA_SQL)
        row = self.conn.execute("SELECT version FROM schema_version").fetchone()
        if row is None:
            self.conn.execute("INSERT INTO schema_version (version) VALUES (4)")
            self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
