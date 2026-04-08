"""CRUD operations for message parts."""
from __future__ import annotations

from typing import List, Optional

from .abc import AbstractMessagePartStore
from .database import Database
from .models import MessagePart


class MessagePartStore(AbstractMessagePartStore):
    """Structured storage for multi-part messages."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _row_to_part(self, row: tuple) -> MessagePart:
        return MessagePart(
            part_id=row[0],
            message_id=row[1],
            part_type=row[2],
            ordinal=row[3],
            text_content=row[4],
            tool_call_id=row[5],
            tool_name=row[6],
            tool_input=row[7],
            tool_output=row[8],
            tool_status=row[9],
            metadata=row[10],
        )

    _SELECT_COLS = (
        "part_id, message_id, part_type, ordinal, text_content, "
        "tool_call_id, tool_name, tool_input, tool_output, tool_status, metadata"
    )

    def add(self, part: MessagePart) -> MessagePart:
        """Insert a message part."""
        conn = self._db.conn
        conn.execute(
            "INSERT INTO message_parts (part_id, message_id, part_type, ordinal, "
            "text_content, tool_call_id, tool_name, tool_input, tool_output, "
            "tool_status, metadata) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                part.part_id, part.message_id, part.part_type, part.ordinal,
                part.text_content, part.tool_call_id, part.tool_name,
                part.tool_input, part.tool_output, part.tool_status, part.metadata,
            ),
        )
        conn.commit()
        return part

    def get_by_message(self, message_id: str) -> List[MessagePart]:
        """Get all parts for a message, ordered by ordinal."""
        rows = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM message_parts "
            "WHERE message_id = ? ORDER BY ordinal ASC",
            (message_id,),
        ).fetchall()
        return [self._row_to_part(r) for r in rows]

    def get_by_id(self, part_id: str) -> Optional[MessagePart]:
        """Get a single part by its ID."""
        row = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM message_parts WHERE part_id = ?",
            (part_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_part(row)

    def get_by_type(self, message_id: str, part_type: str) -> List[MessagePart]:
        """Get parts of a specific type for a message."""
        rows = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM message_parts "
            "WHERE message_id = ? AND part_type = ? ORDER BY ordinal ASC",
            (message_id, part_type),
        ).fetchall()
        return [self._row_to_part(r) for r in rows]

    def delete_by_message(self, message_id: str) -> int:
        """Delete all parts for a message. Return count deleted."""
        conn = self._db.conn
        cur = conn.execute(
            "DELETE FROM message_parts WHERE message_id = ?",
            (message_id,),
        )
        conn.commit()
        return cur.rowcount
