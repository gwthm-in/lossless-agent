"""Dataclass models for the store layer."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Conversation:
    id: int
    session_key: str
    title: str
    active: bool
    created_at: str
    updated_at: str
    session_id: Optional[str] = None
    archived_at: Optional[str] = None
    bootstrapped_at: Optional[str] = None


@dataclass
class Message:
    id: int
    conversation_id: int
    seq: int
    role: str
    content: str
    token_count: int
    tool_call_id: Optional[str]
    tool_name: Optional[str]
    created_at: str


@dataclass
class Summary:
    summary_id: str
    conversation_id: int
    kind: str
    depth: int
    content: str
    token_count: int
    source_token_count: int
    earliest_at: str
    latest_at: str
    model: str
    created_at: str
    file_ids: Optional[str] = None
    descendant_count: int = 0
    descendant_token_count: int = 0


@dataclass
class ContextItem:
    """A single item in the ordered context window."""
    conversation_id: str
    ordinal: int
    item_type: str  # 'message' or 'summary'
    message_id: Optional[str] = None
    summary_id: Optional[str] = None


@dataclass
class MessagePart:
    """A structured part of a message (text, tool call, media, etc.)."""
    part_id: str
    message_id: str
    part_type: str
    ordinal: int = 0
    text_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    tool_name: Optional[str] = None
    tool_input: Optional[str] = None
    tool_output: Optional[str] = None
    tool_status: Optional[str] = None
    metadata: Optional[str] = None
    session_id: Optional[str] = None
    tool_error: Optional[str] = None
    tool_title: Optional[str] = None
    patch_old: Optional[str] = None
    patch_new: Optional[str] = None
    file_name: Optional[str] = None
    file_content: Optional[str] = None
    snapshot_hash: Optional[str] = None
    compaction_auto: int = 0
