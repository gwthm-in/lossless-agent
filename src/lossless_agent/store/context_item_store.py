"""CRUD operations for context items."""
from __future__ import annotations

from typing import List

from .abc import AbstractContextItemStore
from .database import Database
from .models import ContextItem


class ContextItemStore(AbstractContextItemStore):
    """Track ordered context window items (messages and summaries)."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _row_to_item(self, row: tuple) -> ContextItem:
        return ContextItem(
            conversation_id=row[0],
            ordinal=row[1],
            item_type=row[2],
            message_id=row[3],
            summary_id=row[4],
        )

    def add_message(self, conversation_id: str, ordinal: int, message_id: str) -> ContextItem:
        """Add a message item to the context window."""
        conn = self._db.conn
        conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
            "VALUES (?, ?, 'message', ?)",
            (conversation_id, ordinal, message_id),
        )
        conn.commit()
        return ContextItem(
            conversation_id=conversation_id,
            ordinal=ordinal,
            item_type="message",
            message_id=message_id,
        )

    def add_summary(self, conversation_id: str, ordinal: int, summary_id: str) -> ContextItem:
        """Add a summary item to the context window."""
        conn = self._db.conn
        conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id) "
            "VALUES (?, ?, 'summary', ?)",
            (conversation_id, ordinal, summary_id),
        )
        conn.commit()
        return ContextItem(
            conversation_id=conversation_id,
            ordinal=ordinal,
            item_type="summary",
            summary_id=summary_id,
        )

    def get_items(self, conversation_id: str) -> List[ContextItem]:
        """Get all context items for a conversation, ordered by ordinal."""
        rows = self._db.conn.execute(
            "SELECT conversation_id, ordinal, item_type, message_id, summary_id "
            "FROM context_items WHERE conversation_id = ? ORDER BY ordinal ASC",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def remove_by_message_ids(self, conversation_id: str, message_ids: List[str]) -> None:
        """Remove context items that reference the given message IDs."""
        if not message_ids:
            return
        placeholders = ",".join("?" for _ in message_ids)
        self._db.conn.execute(
            f"DELETE FROM context_items "
            f"WHERE conversation_id = ? AND item_type = 'message' "
            f"AND message_id IN ({placeholders})",
            [conversation_id] + list(message_ids),
        )
        self._db.conn.commit()

    def replace_messages_with_summary(
        self,
        conversation_id: str,
        message_ids: List[str],
        summary_id: str,
        new_ordinal: int,
    ) -> None:
        """Atomically remove message items and insert a summary item."""
        conn = self._db.conn
        if message_ids:
            placeholders = ",".join("?" for _ in message_ids)
            conn.execute(
                f"DELETE FROM context_items "
                f"WHERE conversation_id = ? AND item_type = 'message' "
                f"AND message_id IN ({placeholders})",
                [conversation_id] + list(message_ids),
            )
        conn.execute(
            "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id) "
            "VALUES (?, ?, 'summary', ?)",
            (conversation_id, new_ordinal, summary_id),
        )
        conn.commit()

    def get_max_ordinal(self, conversation_id: str) -> int:
        """Return the highest ordinal for a conversation, or 0 if empty."""
        row = self._db.conn.execute(
            "SELECT COALESCE(MAX(ordinal), 0) FROM context_items WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return int(row[0])
