"""SQLite database initialization and schema management."""
from __future__ import annotations

import sqlite3
from pathlib import Path

_SCHEMA_SQL = """\
-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_key TEXT    NOT NULL UNIQUE,
    title       TEXT    NOT NULL DEFAULT '',
    active      INTEGER NOT NULL DEFAULT 1,
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now')),
    updated_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

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

-- Summaries (DAG nodes)
CREATE TABLE IF NOT EXISTS summaries (
    summary_id          TEXT    PRIMARY KEY,
    conversation_id     INTEGER NOT NULL REFERENCES conversations(id),
    kind                TEXT    NOT NULL CHECK (kind IN ('leaf', 'condensed')),
    depth               INTEGER NOT NULL DEFAULT 0,
    content             TEXT    NOT NULL DEFAULT '',
    token_count         INTEGER NOT NULL DEFAULT 0,
    source_token_count  INTEGER NOT NULL DEFAULT 0,
    earliest_at         TEXT    NOT NULL,
    latest_at           TEXT    NOT NULL,
    model               TEXT    NOT NULL DEFAULT '',
    created_at          TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%f', 'now'))
);

-- Leaf summary -> source messages (many-to-many)
CREATE TABLE IF NOT EXISTS summary_messages (
    summary_id TEXT    NOT NULL REFERENCES summaries(summary_id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    PRIMARY KEY (summary_id, message_id)
);

-- Condensed summary -> child summaries (DAG edges)
CREATE TABLE IF NOT EXISTS summary_parents (
    parent_id TEXT NOT NULL REFERENCES summaries(summary_id),
    child_id  TEXT NOT NULL REFERENCES summaries(summary_id),
    PRIMARY KEY (parent_id, child_id)
);

-- Large file storage (intercepted oversized content)
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

-- FTS5 virtual tables
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content, content='messages', content_rowid='id'
);

CREATE VIRTUAL TABLE IF NOT EXISTS summaries_fts USING fts5(
    content, content='summaries', content_rowid='rowid'
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
END;

CREATE TRIGGER IF NOT EXISTS summaries_ad AFTER DELETE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS summaries_au AFTER UPDATE ON summaries BEGIN
    INSERT INTO summaries_fts(summaries_fts, rowid, content) VALUES('delete', old.rowid, old.content);
    INSERT INTO summaries_fts(rowid, content) VALUES (new.rowid, new.content);
END;
"""


class Database:
    """Manages a SQLite connection with the lossless-agent schema."""

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
            self.conn.execute("INSERT INTO schema_version (version) VALUES (2)")
            self.conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
