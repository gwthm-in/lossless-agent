"""Transaction context manager for serialized SQLite writes.

Usage::

    from lossless_agent.store.transaction import transaction

    with transaction(conn):
        conn.execute("INSERT INTO ...")
        conn.execute("UPDATE ...")
    # auto-COMMIT on success, ROLLBACK on exception

Should be applied to critical write paths in the stores:
- ConversationStore.get_or_create (insert + potential update)
- MessageStore.append (insert with seq computation)
- SummaryStore.save (insert summary + edges)
- Any bulk operations that must be atomic
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from typing import Generator


@contextmanager
def transaction(conn: sqlite3.Connection) -> Generator[sqlite3.Connection, None, None]:
    """Execute a block inside a BEGIN IMMEDIATE transaction.

    Uses BEGIN IMMEDIATE to acquire a reserved lock immediately,
    preventing concurrent write conflicts in WAL mode.

    On success: COMMIT
    On exception: ROLLBACK, then re-raise
    """
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
    except BaseException:
        conn.rollback()
        raise
    else:
        conn.commit()
