"""CRUD operations for summaries (DAG of leaf and condensed nodes)."""
from __future__ import annotations

import secrets
from typing import List, Optional

from .database import Database
from .models import Summary


def _generate_summary_id() -> str:
    """Generate a random summary ID like sum_xxxxxxxxxxxx (16 chars total)."""
    return "sum_" + secrets.token_hex(6)


class SummaryStore:
    """Manage the summary DAG: leaf summaries over messages, condensed over children."""

    def __init__(self, db: Database) -> None:
        self._db = db

    def _row_to_summary(self, row: tuple) -> Summary:
        return Summary(
            summary_id=row[0],
            conversation_id=row[1],
            kind=row[2],
            depth=row[3],
            content=row[4],
            token_count=row[5],
            source_token_count=row[6],
            earliest_at=row[7],
            latest_at=row[8],
            model=row[9],
            created_at=row[10],
        )

    _SELECT_COLS = (
        "summary_id, conversation_id, kind, depth, content, token_count, "
        "source_token_count, earliest_at, latest_at, model, created_at"
    )

    def create_leaf(
        self,
        conversation_id: int,
        content: str,
        token_count: int,
        source_token_count: int,
        message_ids: List[int],
        earliest_at: str,
        latest_at: str,
        model: str,
    ) -> Summary:
        """Create a leaf summary that covers specific messages."""
        conn = self._db.conn
        summary_id = _generate_summary_id()

        conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES (?, ?, 'leaf', 0, ?, ?, ?, ?, ?, ?)",
            (summary_id, conversation_id, content, token_count,
             source_token_count, earliest_at, latest_at, model),
        )
        # Link to source messages
        for msg_id in message_ids:
            conn.execute(
                "INSERT INTO summary_messages (summary_id, message_id) VALUES (?, ?)",
                (summary_id, msg_id),
            )
        conn.commit()

        row = conn.execute(
            f"SELECT {self._SELECT_COLS} FROM summaries WHERE summary_id = ?",
            (summary_id,),
        ).fetchone()
        return self._row_to_summary(row)

    def create_condensed(
        self,
        conversation_id: int,
        content: str,
        token_count: int,
        child_ids: List[str],
        earliest_at: str,
        latest_at: str,
        model: str,
    ) -> Summary:
        """Create a condensed summary over child summaries. Depth = max(child depths) + 1."""
        conn = self._db.conn
        summary_id = _generate_summary_id()

        # Calculate depth from children
        placeholders = ",".join("?" for _ in child_ids)
        row = conn.execute(
            f"SELECT COALESCE(MAX(depth), -1) FROM summaries WHERE summary_id IN ({placeholders})",
            child_ids,
        ).fetchone()
        depth = row[0] + 1

        # Calculate source_token_count as sum of children's token_count
        row = conn.execute(
            f"SELECT COALESCE(SUM(token_count), 0) FROM summaries WHERE summary_id IN ({placeholders})",
            child_ids,
        ).fetchone()
        source_token_count = row[0]

        conn.execute(
            "INSERT INTO summaries (summary_id, conversation_id, kind, depth, content, "
            "token_count, source_token_count, earliest_at, latest_at, model) "
            "VALUES (?, ?, 'condensed', ?, ?, ?, ?, ?, ?, ?)",
            (summary_id, conversation_id, depth, content, token_count,
             source_token_count, earliest_at, latest_at, model),
        )
        # Link parent -> children
        for child_id in child_ids:
            conn.execute(
                "INSERT INTO summary_parents (parent_id, child_id) VALUES (?, ?)",
                (summary_id, child_id),
            )
        conn.commit()

        row = conn.execute(
            f"SELECT {self._SELECT_COLS} FROM summaries WHERE summary_id = ?",
            (summary_id,),
        ).fetchone()
        return self._row_to_summary(row)

    def get_by_id(self, summary_id: str) -> Optional[Summary]:
        """Fetch a summary by its ID."""
        row = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM summaries WHERE summary_id = ?",
            (summary_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_summary(row)

    def get_by_conversation(self, conversation_id: int) -> List[Summary]:
        """Get all summaries for a conversation."""
        rows = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM summaries WHERE conversation_id = ? "
            "ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_by_depth(self, conversation_id: int, depth: int) -> List[Summary]:
        """Get summaries at a specific depth for a conversation."""
        rows = self._db.conn.execute(
            f"SELECT {self._SELECT_COLS} FROM summaries "
            "WHERE conversation_id = ? AND depth = ? ORDER BY created_at ASC",
            (conversation_id, depth),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]

    def get_source_message_ids(self, summary_id: str) -> List[int]:
        """Get message IDs linked to a leaf summary."""
        rows = self._db.conn.execute(
            "SELECT message_id FROM summary_messages WHERE summary_id = ?",
            (summary_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_child_ids(self, summary_id: str) -> List[str]:
        """Get child summary IDs for a condensed summary."""
        rows = self._db.conn.execute(
            "SELECT child_id FROM summary_parents WHERE parent_id = ?",
            (summary_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_compacted_message_ids(self, conversation_id: int) -> List[int]:
        """Get all message IDs that are already covered by a leaf summary."""
        rows = self._db.conn.execute(
            "SELECT DISTINCT sm.message_id "
            "FROM summary_messages sm "
            "JOIN summaries s ON sm.summary_id = s.summary_id "
            "WHERE s.conversation_id = ?",
            (conversation_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def get_orphan_ids(self, conversation_id: int, depth: int) -> List[str]:
        """Get summary IDs at a given depth that are not children of any higher summary."""
        rows = self._db.conn.execute(
            "SELECT s.summary_id FROM summaries s "
            "WHERE s.conversation_id = ? AND s.depth = ? "
            "AND s.summary_id NOT IN (SELECT child_id FROM summary_parents) "
            "ORDER BY s.created_at ASC",
            (conversation_id, depth),
        ).fetchall()
        return [r[0] for r in rows]

    def search(self, query: str) -> List[Summary]:
        """Full-text search across summary content."""
        rows = self._db.conn.execute(
            f"SELECT s.{self._SELECT_COLS.replace(', ', ', s.')} "
            "FROM summaries s JOIN summaries_fts f ON s.rowid = f.rowid "
            "WHERE summaries_fts MATCH ? ORDER BY rank",
            (query,),
        ).fetchall()
        return [self._row_to_summary(r) for r in rows]
