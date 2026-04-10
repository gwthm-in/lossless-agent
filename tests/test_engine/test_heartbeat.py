"""Tests for heartbeat OK pruning."""
from __future__ import annotations


from lossless_agent.store.models import Message
from lossless_agent.engine.heartbeat import (
    HeartbeatPruner,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _msg(id, role, content, tool_call_id=None, seq=None):
    """Shortcut to create a Message for testing."""
    return Message(
        id=id,
        conversation_id=1,
        seq=seq or id,
        role=role,
        content=content,
        token_count=10,
        tool_call_id=tool_call_id,
        tool_name=None,
        created_at="2024-01-01",
    )


class FakeMessageStore:
    """Minimal store that tracks deletes."""
    def __init__(self):
        self.deleted: list[int] = []

    def delete(self, message_id: int) -> None:
        self.deleted.append(message_id)


# ------------------------------------------------------------------
# is_heartbeat_content
# ------------------------------------------------------------------

class TestIsHeartbeatContent:
    def test_exact_match(self):
        assert HeartbeatPruner.is_heartbeat_content("heartbeat_ok") is True

    def test_case_insensitive(self):
        assert HeartbeatPruner.is_heartbeat_content("HEARTBEAT_OK") is True
        assert HeartbeatPruner.is_heartbeat_content("Heartbeat_Ok") is True

    def test_strips_whitespace(self):
        assert HeartbeatPruner.is_heartbeat_content("  heartbeat_ok  ") is True

    def test_rejects_other_content(self):
        assert HeartbeatPruner.is_heartbeat_content("hello") is False
        assert HeartbeatPruner.is_heartbeat_content("") is False
        assert HeartbeatPruner.is_heartbeat_content("heartbeat_ok extra") is False


# ------------------------------------------------------------------
# is_heartbeat_turn_marker
# ------------------------------------------------------------------

class TestIsHeartbeatTurnMarker:
    def test_contains_marker(self):
        assert HeartbeatPruner.is_heartbeat_turn_marker("Read heartbeat.md now") is True

    def test_exact_marker(self):
        assert HeartbeatPruner.is_heartbeat_turn_marker("heartbeat.md") is True

    def test_rejects_without_marker(self):
        assert HeartbeatPruner.is_heartbeat_turn_marker("heartbeat") is False
        assert HeartbeatPruner.is_heartbeat_turn_marker("") is False


# ------------------------------------------------------------------
# find_heartbeat_turns
# ------------------------------------------------------------------

class TestFindHeartbeatTurns:
    def test_simple_heartbeat_cycle(self):
        msgs = [
            _msg(1, "user", "check heartbeat.md"),
            _msg(2, "assistant", "heartbeat_ok"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == [[1, 2]]

    def test_heartbeat_with_tool_calls(self):
        msgs = [
            _msg(1, "user", "check heartbeat.md"),
            _msg(2, "tool", "tool result", tool_call_id="tc1"),
            _msg(3, "assistant", "heartbeat_ok"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == [[1, 2, 3]]

    def test_no_heartbeat(self):
        msgs = [
            _msg(1, "user", "hello"),
            _msg(2, "assistant", "hi there"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == []

    def test_incomplete_cycle_no_assistant(self):
        msgs = [
            _msg(1, "user", "check heartbeat.md"),
            _msg(2, "user", "something else"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == []

    def test_multiple_cycles(self):
        msgs = [
            _msg(1, "user", "check heartbeat.md"),
            _msg(2, "assistant", "heartbeat_ok"),
            _msg(3, "user", "real question"),
            _msg(4, "assistant", "real answer"),
            _msg(5, "user", "heartbeat.md check"),
            _msg(6, "assistant", "HEARTBEAT_OK"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == [[1, 2], [5, 6]]

    def test_mixed_with_non_heartbeat(self):
        msgs = [
            _msg(1, "user", "hello"),
            _msg(2, "assistant", "hi"),
            _msg(3, "user", "read heartbeat.md"),
            _msg(4, "assistant", "heartbeat_ok"),
        ]
        groups = HeartbeatPruner.find_heartbeat_turns(msgs)
        assert groups == [[3, 4]]

    def test_empty_messages(self):
        assert HeartbeatPruner.find_heartbeat_turns([]) == []


# ------------------------------------------------------------------
# prune
# ------------------------------------------------------------------

class TestPrune:
    def test_prune_deletes_heartbeat_messages(self):
        msgs = [
            _msg(1, "user", "check heartbeat.md"),
            _msg(2, "assistant", "heartbeat_ok"),
            _msg(3, "user", "real question"),
            _msg(4, "assistant", "real answer"),
        ]
        store = FakeMessageStore()
        count = HeartbeatPruner.prune(msgs, store)
        assert count == 2
        assert store.deleted == [1, 2]

    def test_prune_returns_zero_when_nothing(self):
        msgs = [
            _msg(1, "user", "hello"),
            _msg(2, "assistant", "hi"),
        ]
        store = FakeMessageStore()
        count = HeartbeatPruner.prune(msgs, store)
        assert count == 0
        assert store.deleted == []

    def test_prune_with_tool_calls(self):
        msgs = [
            _msg(1, "user", "heartbeat.md"),
            _msg(2, "tool", "result", tool_call_id="tc1"),
            _msg(3, "tool", "result2", tool_call_id="tc2"),
            _msg(4, "assistant", "heartbeat_ok"),
        ]
        store = FakeMessageStore()
        count = HeartbeatPruner.prune(msgs, store)
        assert count == 4
        assert store.deleted == [1, 2, 3, 4]
