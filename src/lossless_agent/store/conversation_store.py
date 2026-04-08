"""CRUD operations for conversations."""
from __future__ import annotations

from typing import Optional

from .abc import AbstractConversationStore
from .database import Database
from .models import Conversation


class ConversationStore(AbstractConversationStore):
    """Manage conversation lifecycle."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _row_to_conversation(self, row: tuple) -> Conversation:
        return Conversation(
            id=row[0],
            session_key=row[1],
            title=row[2],
            active=bool(row[3]),
            created_at=row[4],
            updated_at=row[5],
            session_id=row[6] if len(row) > 6 else None,
            archived_at=row[7] if len(row) > 7 else None,
            bootstrapped_at=row[8] if len(row) > 8 else None,
        )

    def get_or_create(self, session_key: str, title: str = "") -> Conversation:
        """Return existing conversation for session_key, or create a new one."""
        conn = self._db.conn
        row = conn.execute(
            "SELECT id, session_key, title, active, created_at, updated_at, "
            "session_id, archived_at, bootstrapped_at "
            "FROM conversations WHERE session_key = ? AND active = 1",
            (session_key,),
        ).fetchone()
        if row is not None:
            return self._row_to_conversation(row)

        conn.execute(
            "INSERT INTO conversations (session_key, title) VALUES (?, ?)",
            (session_key, title),
        )
        conn.commit()
        row = conn.execute(
            "SELECT id, session_key, title, active, created_at, updated_at, "
            "session_id, archived_at, bootstrapped_at "
            "FROM conversations WHERE session_key = ? AND active = 1",
            (session_key,),
        ).fetchone()
        return self._row_to_conversation(row)

    def get_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """Fetch a conversation by its ID."""
        row = self._db.conn.execute(
            "SELECT id, session_key, title, active, created_at, updated_at, "
            "session_id, archived_at, bootstrapped_at "
            "FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_conversation(row)

    def deactivate(self, conversation_id: int) -> None:
        """Mark a conversation as inactive."""
        self._db.conn.execute(
            "UPDATE conversations SET active = 0, "
            "updated_at = strftime('%Y-%m-%dT%H:%M:%f', 'now') "
            "WHERE id = ?",
            (conversation_id,),
        )
        self._db.conn.commit()
