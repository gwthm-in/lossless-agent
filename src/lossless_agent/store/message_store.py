"""CRUD operations for messages."""
from __future__ import annotations

from typing import List, Optional

from .abc import AbstractMessageStore
from .database import Database
from .models import Message


class MessageStore(AbstractMessageStore):
    """Append-only message log per conversation."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _row_to_message(self, row: tuple) -> Message:
        return Message(
            id=row[0],
            conversation_id=row[1],
            seq=row[2],
            role=row[3],
            content=row[4],
            token_count=row[5],
            tool_call_id=row[6],
            tool_name=row[7],
            created_at=row[8],
        )

    def append(
        self,
        conversation_id: int,
        role: str,
        content: str,
        token_count: int = 0,
        tool_call_id: Optional[str] = None,
        tool_name: Optional[str] = None,
    ) -> Message:
        """Append a message, auto-assigning the next seq number."""
        conn = self._db.conn
        # Get next seq for this conversation
        row = conn.execute(
            "SELECT COALESCE(MAX(seq), 0) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        next_seq = row[0] + 1

        cur = conn.execute(
            "INSERT INTO messages (conversation_id, seq, role, content, token_count, "
            "tool_call_id, tool_name) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (conversation_id, next_seq, role, content, token_count,
             tool_call_id, tool_name),
        )
        conn.commit()
        msg_id = cur.lastrowid

        row = conn.execute(
            "SELECT id, conversation_id, seq, role, content, token_count, "
            "tool_call_id, tool_name, created_at FROM messages WHERE id = ?",
            (msg_id,),
        ).fetchone()
        return self._row_to_message(row)

    def get_messages(
        self,
        conversation_id: int,
        after_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages for a conversation, optionally after a seq and with a limit."""
        sql = (
            "SELECT id, conversation_id, seq, role, content, token_count, "
            "tool_call_id, tool_name, created_at FROM messages "
            "WHERE conversation_id = ?"
        )
        params: list = [conversation_id]
        if after_seq is not None:
            sql += " AND seq > ?"
            params.append(after_seq)
        sql += " ORDER BY seq ASC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        rows = self._db.conn.execute(sql, params).fetchall()
        return [self._row_to_message(r) for r in rows]

    def count(self, conversation_id: int) -> int:
        """Count messages in a conversation."""
        row = self._db.conn.execute(
            "SELECT COUNT(*) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row[0]

    def total_tokens(self, conversation_id: int) -> int:
        """Sum of token_count for all messages in a conversation."""
        row = self._db.conn.execute(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        return row[0]

    def tail(self, conversation_id: int, n: int) -> List[Message]:
        """Get the last n messages in a conversation, ordered by seq ascending."""
        rows = self._db.conn.execute(
            "SELECT id, conversation_id, seq, role, content, token_count, "
            "tool_call_id, tool_name, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY seq DESC LIMIT ?",
            (conversation_id, n),
        ).fetchall()
        # Reverse to get ascending order
        rows.reverse()
        return [self._row_to_message(r) for r in rows]
