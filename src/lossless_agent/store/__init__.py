"""Store layer for conversations, messages, and summaries."""

from .abc import (
    AbstractConversationStore,
    AbstractContextItemStore,
    AbstractMessagePartStore,
    AbstractMessageStore,
    AbstractSummaryStore,
)
from .database import Database
from .context_item_store import ContextItemStore
from .conversation_store import ConversationStore
from .message_part_store import MessagePartStore
from .message_store import MessageStore
from .summary_store import SummaryStore

# Postgres backend (optional – requires psycopg2)
try:
    from .postgres_database import PostgresDatabase
except ImportError:
    PostgresDatabase = None  # type: ignore[misc,assignment]

__all__ = [
    "AbstractConversationStore",
    "AbstractContextItemStore",
    "AbstractMessagePartStore",
    "AbstractMessageStore",
    "AbstractSummaryStore",
    "Database",
    "PostgresDatabase",
    "ContextItemStore",
    "ConversationStore",
    "MessagePartStore",
    "MessageStore",
    "SummaryStore",
]
