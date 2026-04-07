"""Dataclass models for the store layer."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class Conversation:
    id: int
    session_key: str
    title: str
    active: bool
    created_at: str
    updated_at: str


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
