"""Abstract base classes for the store layer.

These ABCs define the contract that any storage backend must implement,
making the database layer swappable (e.g. SQLite, PostgreSQL, DynamoDB).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from .models import Conversation, ContextItem, Message, MessagePart, Summary


class AbstractConversationStore(ABC):
    """Contract for conversation persistence."""

    @abstractmethod
    def get_or_create(self, session_key: str, title: str = "") -> Conversation:
        """Return existing conversation for session_key, or create a new one."""

    @abstractmethod
    def get_by_id(self, conversation_id: int) -> Optional[Conversation]:
        """Fetch a conversation by its ID."""

    @abstractmethod
    def deactivate(self, conversation_id: int) -> None:
        """Mark a conversation as inactive."""


class AbstractMessageStore(ABC):
    """Contract for message persistence."""

    @abstractmethod
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

    @abstractmethod
    def get_messages(
        self,
        conversation_id: int,
        after_seq: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> List[Message]:
        """Get messages for a conversation, optionally filtered."""

    @abstractmethod
    def count(self, conversation_id: int) -> int:
        """Count messages in a conversation."""

    @abstractmethod
    def total_tokens(self, conversation_id: int) -> int:
        """Sum of token_count for all messages in a conversation."""

    @abstractmethod
    def tail(self, conversation_id: int, n: int) -> List[Message]:
        """Get the last n messages in a conversation, ordered by seq ascending."""


class AbstractSummaryStore(ABC):
    """Contract for summary DAG persistence."""

    @abstractmethod
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

    @abstractmethod
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
        """Create a condensed summary over child summaries."""

    @abstractmethod
    def get_by_id(self, summary_id: str) -> Optional[Summary]:
        """Fetch a summary by its ID."""

    @abstractmethod
    def get_by_conversation(self, conversation_id: int) -> List[Summary]:
        """Get all summaries for a conversation."""

    @abstractmethod
    def get_by_depth(self, conversation_id: int, depth: int) -> List[Summary]:
        """Get summaries at a specific depth for a conversation."""

    @abstractmethod
    def get_source_message_ids(self, summary_id: str) -> List[int]:
        """Get message IDs linked to a leaf summary."""

    @abstractmethod
    def get_child_ids(self, summary_id: str) -> List[str]:
        """Get child summary IDs for a condensed summary."""

    @abstractmethod
    def get_compacted_message_ids(self, conversation_id: int) -> List[int]:
        """Get all message IDs that are already covered by a leaf summary."""

    @abstractmethod
    def get_orphan_ids(self, conversation_id: int, depth: int) -> List[str]:
        """Get summary IDs at a given depth that are not children of any higher summary."""

    @abstractmethod
    def search(self, query: str) -> List[Summary]:
        """Full-text search across summary content."""


class AbstractContextItemStore(ABC):
    """Contract for context items persistence."""

    @abstractmethod
    def add_message(self, conversation_id: str, ordinal: int, message_id: str) -> ContextItem:
        """Add a message item to the context window."""

    @abstractmethod
    def add_summary(self, conversation_id: str, ordinal: int, summary_id: str) -> ContextItem:
        """Add a summary item to the context window."""

    @abstractmethod
    def get_items(self, conversation_id: str) -> List[ContextItem]:
        """Get all context items for a conversation, ordered by ordinal."""

    @abstractmethod
    def remove_by_message_ids(self, conversation_id: str, message_ids: List[str]) -> None:
        """Remove context items that reference the given message IDs."""

    @abstractmethod
    def replace_messages_with_summary(
        self,
        conversation_id: str,
        message_ids: List[str],
        summary_id: str,
        new_ordinal: int,
    ) -> None:
        """Atomically remove message items and insert a summary item."""

    @abstractmethod
    def get_max_ordinal(self, conversation_id: str) -> int:
        """Return the highest ordinal for a conversation, or 0 if empty."""


class AbstractMessagePartStore(ABC):
    """Contract for message parts persistence."""

    @abstractmethod
    def add(self, part: MessagePart) -> MessagePart:
        """Insert a message part."""

    @abstractmethod
    def get_by_message(self, message_id: str) -> List[MessagePart]:
        """Get all parts for a message, ordered by ordinal."""

    @abstractmethod
    def get_by_id(self, part_id: str) -> Optional[MessagePart]:
        """Get a single part by its ID."""

    @abstractmethod
    def get_by_type(self, message_id: str, part_type: str) -> List[MessagePart]:
        """Get parts of a specific type for a message."""

    @abstractmethod
    def delete_by_message(self, message_id: str) -> int:
        """Delete all parts for a message. Return count deleted."""
