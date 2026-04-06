"""Large file interception and separate storage.

When a message contains very large content (e.g. a 50K token tool result),
it is intercepted, stored separately, and replaced with a compact summary
reference to avoid bloating the conversation context.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from lossless_agent.store.database import Database


@dataclass
class LargeFileConfig:
    """Thresholds for large-file interception."""

    token_threshold: int = 25_000
    summary_target_tokens: int = 500


class LargeFileInterceptor:
    """Intercepts large content blocks and stores them separately."""

    def __init__(
        self,
        db: Database,
        summarize_fn: Callable[[str, int], Awaitable[str]],
        config: Optional[LargeFileConfig] = None,
    ) -> None:
        self.db = db
        self.summarize_fn = summarize_fn
        self.config = config or LargeFileConfig()

    async def intercept(
        self,
        conversation_id: int,
        message_content: str,
        token_count: int,
    ) -> Tuple[str, Optional[int]]:
        """Intercept large content, storing it separately.

        Returns (content, file_id).  If below threshold the original
        content is returned unchanged with file_id=None.
        """
        if token_count < self.config.token_threshold:
            return message_content, None

        # Generate summary via the provided callable
        summary = await self.summarize_fn(
            message_content, self.config.summary_target_tokens
        )

        # Estimate summary token count (rough: ~4 chars per token)
        summary_token_count = max(1, len(summary) // 4)

        # Store in large_files table
        cur = self.db.conn.execute(
            "INSERT INTO large_files "
            "(conversation_id, content, token_count, summary, summary_token_count) "
            "VALUES (?, ?, ?, ?, ?)",
            (conversation_id, message_content, token_count, summary, summary_token_count),
        )
        self.db.conn.commit()
        file_id = cur.lastrowid

        replacement = (
            f"[Large content stored as file_id={file_id}. "
            f"Summary: {summary}. "
            f"Use lcm_expand to retrieve full content.]"
        )
        return replacement, file_id

    def get_file(self, file_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a stored large file by ID."""
        self.db.conn.row_factory = None
        cur = self.db.conn.execute(
            "SELECT id, conversation_id, message_id, content, token_count, "
            "summary, summary_token_count, mime_type, file_path, created_at "
            "FROM large_files WHERE id = ?",
            (file_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        return self._row_to_dict(row)

    def get_files_for_conversation(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Retrieve all stored large files for a conversation."""
        cur = self.db.conn.execute(
            "SELECT id, conversation_id, message_id, content, token_count, "
            "summary, summary_token_count, mime_type, file_path, created_at "
            "FROM large_files WHERE conversation_id = ? ORDER BY id",
            (conversation_id,),
        )
        return [self._row_to_dict(row) for row in cur.fetchall()]

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        keys = [
            "id", "conversation_id", "message_id", "content", "token_count",
            "summary", "summary_token_count", "mime_type", "file_path", "created_at",
        ]
        return dict(zip(keys, row))
