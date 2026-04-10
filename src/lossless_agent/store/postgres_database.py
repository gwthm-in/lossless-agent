"""PostgreSQL database initialization and schema management.

Provides a PostgresDatabase class that is a drop-in replacement for the
SQLite Database class. A connection adapter translates SQLite-style SQL
(? placeholders, strftime, INSERT OR IGNORE) so that existing stores
work unchanged.
"""
from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

import psycopg2
import psycopg2.extensions

# ---------------------------------------------------------------------------
# Postgres schema (equivalent to SQLite schema in database.py)
# ---------------------------------------------------------------------------

_PG_SCHEMA_SQL = """\
-- Version tracking
CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER NOT NULL
);

-- Conversations
CREATE TABLE IF NOT EXISTS conversations (
    id              SERIAL PRIMARY KEY,
    session_key     TEXT    NOT NULL,
    session_id      TEXT,
    title           TEXT    NOT NULL DEFAULT '',
    active          INTEGER NOT NULL DEFAULT 1,
    archived_at     TEXT,
    bootstrapped_at TEXT,
    created_at      TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US'),
    updated_at      TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US')
);

-- Partial unique: only one active conversation per session_key
CREATE UNIQUE INDEX IF NOT EXISTS conversations_active_session_key_idx
    ON conversations(session_key) WHERE active=1;
-- Composite lookup index
CREATE INDEX IF NOT EXISTS conversations_session_key_active_created_idx
    ON conversations(session_key, active, created_at);

-- Messages
CREATE TABLE IF NOT EXISTS messages (
    id              SERIAL PRIMARY KEY,
    conversation_id INTEGER NOT NULL REFERENCES conversations(id),
    seq             INTEGER NOT NULL,
    role            TEXT    NOT NULL CHECK (role IN ('system', 'user', 'assistant', 'tool')),
    content         TEXT    NOT NULL DEFAULT '',
    token_count     INTEGER NOT NULL DEFAULT 0,
    tool_call_id    TEXT,
    tool_name       TEXT,
    created_at      TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US'),
    UNIQUE (conversation_id, seq)
);
CREATE INDEX IF NOT EXISTS messages_conv_seq_idx ON messages(conversation_id, seq);

-- Summaries (DAG nodes)
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
    created_at              TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US')
);
CREATE INDEX IF NOT EXISTS summaries_conv_created_idx ON summaries(conversation_id, created_at);

-- GIN index for full-text search on summaries
CREATE INDEX IF NOT EXISTS summaries_fts_idx
    ON summaries USING gin(to_tsvector('english', content));

-- Leaf summary -> source messages (many-to-many)
CREATE TABLE IF NOT EXISTS summary_messages (
    summary_id TEXT    NOT NULL REFERENCES summaries(summary_id),
    message_id INTEGER NOT NULL REFERENCES messages(id),
    ordinal    INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (summary_id, message_id)
);

-- Condensed summary -> child summaries (DAG edges)
CREATE TABLE IF NOT EXISTS summary_parents (
    parent_id TEXT NOT NULL REFERENCES summaries(summary_id),
    child_id  TEXT NOT NULL REFERENCES summaries(summary_id),
    ordinal   INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (parent_id, child_id)
);

-- Large file storage (legacy)
CREATE TABLE IF NOT EXISTS large_files (
    id                  SERIAL PRIMARY KEY,
    conversation_id     INTEGER REFERENCES conversations(id),
    message_id          INTEGER REFERENCES messages(id),
    content             TEXT    NOT NULL DEFAULT '',
    token_count         INTEGER NOT NULL DEFAULT 0,
    summary             TEXT    NOT NULL DEFAULT '',
    summary_token_count INTEGER NOT NULL DEFAULT 0,
    mime_type           TEXT,
    file_path           TEXT,
    created_at          TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US')
);
CREATE INDEX IF NOT EXISTS large_files_conv_idx ON large_files(conversation_id);

-- Large files v2
CREATE TABLE IF NOT EXISTS large_files_v2 (
    file_id             TEXT PRIMARY KEY,
    conversation_id     INTEGER REFERENCES conversations(id),
    file_name           TEXT,
    mime_type           TEXT,
    byte_size           INTEGER NOT NULL DEFAULT 0,
    storage_uri         TEXT,
    exploration_summary TEXT,
    created_at          TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US')
);
CREATE INDEX IF NOT EXISTS large_files_v2_conv_idx ON large_files_v2(conversation_id);

-- Context items
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

-- Message parts
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
    compaction_auto INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS message_parts_message_idx ON message_parts(message_id);
CREATE INDEX IF NOT EXISTS message_parts_type_idx ON message_parts(part_type);

-- Conversation bootstrap state
CREATE TABLE IF NOT EXISTS conversation_bootstrap_state (
    conversation_id         INTEGER NOT NULL REFERENCES conversations(id),
    session_file_path       TEXT    NOT NULL,
    last_seen_size          INTEGER NOT NULL DEFAULT 0,
    last_seen_mtime_ms      INTEGER NOT NULL DEFAULT 0,
    last_processed_offset   INTEGER NOT NULL DEFAULT 0,
    last_processed_entry_hash TEXT,
    updated_at              TEXT    NOT NULL DEFAULT TO_CHAR(CURRENT_TIMESTAMP, 'YYYY-MM-DD"T"HH24:MI:SS.US'),
    PRIMARY KEY (conversation_id)
);
CREATE INDEX IF NOT EXISTS bootstrap_state_path_idx ON conversation_bootstrap_state(session_file_path);
"""


# ---------------------------------------------------------------------------
# SQL translation helpers
# ---------------------------------------------------------------------------

# Pattern matching for SQLite → Postgres translation
_STRFTIME_RE = re.compile(
    r"strftime\s*\(\s*'[^']*'\s*,\s*'now'\s*\)",
    re.IGNORECASE,
)

_INSERT_OR_IGNORE_RE = re.compile(
    r"INSERT\s+OR\s+IGNORE\s+INTO",
    re.IGNORECASE,
)


def _translate_sql(sql: str) -> str:
    """Translate SQLite-flavored SQL to Postgres-compatible SQL."""
    # ? → %s  (only outside of string literals - simple approach works for our SQL)
    translated = sql.replace("?", "%s")

    # strftime('%Y-%m-%dT%H:%M:%f', 'now') → CURRENT_TIMESTAMP
    translated = _STRFTIME_RE.sub("CURRENT_TIMESTAMP", translated)

    # INSERT OR IGNORE → INSERT ... ON CONFLICT DO NOTHING
    translated = _INSERT_OR_IGNORE_RE.sub("INSERT INTO", translated)
    if "ON CONFLICT DO NOTHING" not in translated and "INSERT OR IGNORE" in sql.upper():
        # Append ON CONFLICT DO NOTHING before any RETURNING clause
        translated = translated.rstrip(";") + " ON CONFLICT DO NOTHING"

    return translated


def _needs_returning(sql: str) -> bool:
    """Check if this is an INSERT into a SERIAL table that needs RETURNING id."""
    upper = sql.strip().upper()
    if not upper.startswith("INSERT"):
        return False
    # Tables with SERIAL id columns
    serial_tables = ("conversations", "messages", "large_files")
    for table in serial_tables:
        if table.upper() in upper and "RETURNING" not in upper:
            return True
    return False


# ---------------------------------------------------------------------------
# Cursor adapter: makes psycopg2 cursor look like sqlite3 cursor
# ---------------------------------------------------------------------------

class _CursorAdapter:
    """Wraps a psycopg2 cursor to provide sqlite3-compatible interface."""

    def __init__(self, pg_cursor):
        self._cursor = pg_cursor
        self._lastrowid = None

    @property
    def lastrowid(self):
        return self._lastrowid

    @property
    def rowcount(self):
        return self._cursor.rowcount

    def fetchone(self) -> Optional[tuple]:
        row = self._cursor.fetchone()
        if row is None:
            return None
        return tuple(row)

    def fetchall(self) -> List[tuple]:
        rows = self._cursor.fetchall()
        return [tuple(r) for r in rows]

    @property
    def description(self):
        return self._cursor.description


# ---------------------------------------------------------------------------
# Connection adapter: makes psycopg2 connection look like sqlite3 connection
# ---------------------------------------------------------------------------

class _ConnectionAdapter:
    """Wraps a psycopg2 connection to accept SQLite-style SQL."""

    def __init__(self, pg_conn):
        self._conn = pg_conn

    def execute(self, sql: str, params=None) -> _CursorAdapter:
        translated = _translate_sql(sql)
        add_returning = _needs_returning(sql)
        if add_returning:
            translated = translated.rstrip(";") + " RETURNING id"

        cursor = self._conn.cursor()
        adapter = _CursorAdapter(cursor)

        if params is not None:
            # Handle both tuple and list params
            cursor.execute(translated, tuple(params))
        else:
            cursor.execute(translated)

        if add_returning:
            try:
                row = cursor.fetchone()
                if row:
                    adapter._lastrowid = row[0]
            except psycopg2.ProgrammingError:
                pass

        return adapter

    def executescript(self, sql: str) -> None:
        """Execute multiple SQL statements. Used for schema init."""
        cursor = self._conn.cursor()
        cursor.execute(sql)

    def commit(self) -> None:
        self._conn.commit()

    def rollback(self) -> None:
        self._conn.rollback()

    def close(self) -> None:
        self._conn.close()

    @property
    def in_transaction(self) -> bool:
        return self._conn.status == psycopg2.extensions.TRANSACTION_STATUS_INTRANS


# ---------------------------------------------------------------------------
# PostgresDatabase
# ---------------------------------------------------------------------------

class PostgresDatabase:
    """Manages a PostgreSQL connection with the lossless-agent schema.

    Drop-in replacement for Database (SQLite). The connection adapter
    translates SQLite-style SQL so existing stores work unchanged.
    """

    backend = "postgres"

    def __init__(self, dsn: str = "dbname=lossless_agent host=localhost") -> None:
        self.dsn = dsn
        self._raw_conn = psycopg2.connect(dsn)
        self._raw_conn.autocommit = False
        self.conn = _ConnectionAdapter(self._raw_conn)
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist and set schema version."""
        # Execute schema DDL statement by statement
        # (Postgres doesn't support executescript like SQLite)
        cursor = self._raw_conn.cursor()

        # Split and execute each statement
        for statement in _PG_SCHEMA_SQL.split(";"):
            statement = statement.strip()
            if statement:
                try:
                    cursor.execute(statement)
                except psycopg2.errors.DuplicateTable:
                    self._raw_conn.rollback()
                    cursor = self._raw_conn.cursor()
                    continue
                except psycopg2.errors.DuplicateObject:
                    self._raw_conn.rollback()
                    cursor = self._raw_conn.cursor()
                    continue
                except Exception as e:
                    # For "IF NOT EXISTS" we shouldn't get here, but be safe
                    if "already exists" in str(e).lower():
                        self._raw_conn.rollback()
                        cursor = self._raw_conn.cursor()
                        continue
                    raise
        self._raw_conn.commit()

        # Set schema version
        cursor.execute("SELECT version FROM schema_version")
        row = cursor.fetchone()
        if row is None:
            cursor.execute("INSERT INTO schema_version (version) VALUES (4)")
            self._raw_conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        self._raw_conn.close()
