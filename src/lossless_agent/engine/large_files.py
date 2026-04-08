"""Large file interception and separate storage.

When a message contains very large content (e.g. a 50K token tool result),
it is intercepted, stored separately, and replaced with a compact summary
reference to avoid bloating the conversation context.
"""
from __future__ import annotations

import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable, Dict, List, Optional, Tuple

from lossless_agent.store.database import Database


@dataclass
class LargeFileConfig:
    """Thresholds for large-file interception."""

    token_threshold: int = 25_000
    summary_target_tokens: int = 500
    file_storage_dir: str = "~/.lossless-agent/files/"


def _generate_file_id() -> str:
    """Generate a file_id as 'file_' + 16 hex chars."""
    return "file_" + secrets.token_hex(8)


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

    def _storage_dir(self) -> str:
        """Return the resolved storage directory, creating it if needed."""
        d = os.path.expanduser(self.config.file_storage_dir)
        os.makedirs(d, exist_ok=True)
        return d

    async def intercept(
        self,
        conversation_id: int,
        message_content: str,
        token_count: int,
        file_name: Optional[str] = None,
        mime_type: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
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

        file_id = _generate_file_id()

        # Save content to file storage
        storage_dir = self._storage_dir()
        storage_path = os.path.join(storage_dir, file_id)
        with open(storage_path, "w", encoding="utf-8") as f:
            f.write(message_content)

        byte_size = len(message_content.encode("utf-8"))
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

        # Store in large_files_v2 table
        self.db.conn.execute(
            "INSERT INTO large_files_v2 "
            "(file_id, conversation_id, file_name, mime_type, byte_size, "
            "storage_uri, exploration_summary, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (file_id, conversation_id, file_name or "untitled",
             mime_type or "text/plain", byte_size,
             storage_path, summary, now),
        )
        self.db.conn.commit()

        replacement = (
            f"[Large content stored as file_id={file_id}. "
            f"Summary: {summary}. "
            f"Use lcm_expand to retrieve full content.]"
        )
        return replacement, file_id

    def get_file(self, file_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored large file by ID."""
        cur = self.db.conn.execute(
            "SELECT file_id, conversation_id, file_name, mime_type, byte_size, "
            "storage_uri, exploration_summary, created_at "
            "FROM large_files_v2 WHERE file_id = ?",
            (file_id,),
        )
        row = cur.fetchone()
        if row is None:
            return None
        result = self._row_to_dict(row)
        # Load content from storage_uri
        storage_uri = result.get("storage_uri", "")
        if storage_uri and os.path.exists(storage_uri):
            with open(storage_uri, "r", encoding="utf-8") as f:
                result["content"] = f.read()
        else:
            result["content"] = None
        return result

    def get_files_for_conversation(self, conversation_id: int) -> List[Dict[str, Any]]:
        """Retrieve all stored large files for a conversation."""
        cur = self.db.conn.execute(
            "SELECT file_id, conversation_id, file_name, mime_type, byte_size, "
            "storage_uri, exploration_summary, created_at "
            "FROM large_files_v2 WHERE conversation_id = ? ORDER BY created_at",
            (conversation_id,),
        )
        results = []
        for row in cur.fetchall():
            d = self._row_to_dict(row)
            results.append(d)
        return results

    @staticmethod
    def _row_to_dict(row: tuple) -> Dict[str, Any]:
        keys = [
            "file_id", "conversation_id", "file_name", "mime_type", "byte_size",
            "storage_uri", "exploration_summary", "created_at",
        ]
        return dict(zip(keys, row))
