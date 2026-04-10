"""Heartbeat OK pruning: detect and remove heartbeat turn cycles."""
from __future__ import annotations

import logging
from typing import List, Protocol

from lossless_agent.store.models import Message

logger = logging.getLogger(__name__)

HEARTBEAT_OK_TOKEN = "heartbeat_ok"
HEARTBEAT_TURN_MARKER = "heartbeat.md"


class MessageDeleter(Protocol):
    """Protocol for message deletion (subset of MessageStore)."""

    def delete(self, message_id: int) -> None: ...


class HeartbeatPruner:
    """Detect and prune heartbeat turn cycles from conversation messages.

    A heartbeat turn cycle is:
      1. user message containing the heartbeat turn marker (heartbeat.md)
      2. any tool_call / tool_result messages in between
      3. assistant message with content == 'heartbeat_ok'
    """

    @staticmethod
    def is_heartbeat_content(content: str) -> bool:
        """Return True if *content* is the heartbeat_ok token."""
        return content.strip().lower() == HEARTBEAT_OK_TOKEN

    @staticmethod
    def is_heartbeat_turn_marker(content: str) -> bool:
        """Return True if *content* contains the heartbeat turn marker."""
        return HEARTBEAT_TURN_MARKER in content

    @staticmethod
    def find_heartbeat_turns(messages: List[Message]) -> List[List[int]]:
        """Find complete heartbeat turn cycles.

        Returns a list of groups, each group being a list of message IDs
        that form a complete heartbeat cycle (user prompt + tool calls +
        assistant heartbeat_ok reply).
        """
        groups: List[List[int]] = []
        i = 0
        while i < len(messages):
            msg = messages[i]
            # Look for a user message with the heartbeat turn marker
            if msg.role == "user" and HeartbeatPruner.is_heartbeat_turn_marker(msg.content):
                group = [msg.id]
                j = i + 1
                # Collect tool calls/results until we find assistant heartbeat_ok
                while j < len(messages):
                    next_msg = messages[j]
                    if next_msg.role == "assistant" and HeartbeatPruner.is_heartbeat_content(next_msg.content):
                        group.append(next_msg.id)
                        groups.append(group)
                        j += 1
                        break
                    elif next_msg.role in ("tool", "tool_call", "tool_result") or next_msg.tool_call_id is not None:
                        group.append(next_msg.id)
                        j += 1
                    else:
                        # Not a complete heartbeat cycle
                        break
                i = j
            else:
                i += 1
        return groups

    @staticmethod
    def prune(messages: List[Message], message_store) -> int:
        """Delete heartbeat turns from the message store. Return count pruned."""
        groups = HeartbeatPruner.find_heartbeat_turns(messages)
        count = 0
        for group in groups:
            for msg_id in group:
                message_store.delete(msg_id)
                count += 1
        return count
