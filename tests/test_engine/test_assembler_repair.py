"""Tests for transcript repair integration into ContextAssembler."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from lossless_agent.engine.assembler import AssemblerConfig, ContextAssembler
from lossless_agent.store.models import Message, Summary


def _msg(seq, role="user", content="hi", tool_call_id=None, tool_name=None, token_count=5):
    return Message(
        id=seq,
        conversation_id=1,
        seq=seq,
        role=role,
        content=content,
        token_count=token_count,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        created_at="2025-01-01",
    )


def _summary(sid, depth=0, token_count=10):
    return Summary(
        summary_id=sid,
        conversation_id=1,
        kind="leaf",
        depth=depth,
        content=f"summary-{sid}",
        token_count=token_count,
        source_token_count=50,
        earliest_at="2025-01-01",
        latest_at="2025-01-02",
        model="test",
        created_at="2025-01-01",
    )


@pytest.fixture
def msg_store():
    store = MagicMock()
    return store


@pytest.fixture
def sum_store():
    store = MagicMock()
    store.get_by_conversation.return_value = []
    store.get_child_ids.return_value = []
    return store


class TestAssemblerConfigHasRepairFlag:
    """AssemblerConfig should have a repair_transcripts flag defaulting to True."""

    def test_default_true(self):
        cfg = AssemblerConfig(max_context_tokens=1000)
        assert cfg.repair_transcripts is True

    def test_can_disable(self):
        cfg = AssemblerConfig(max_context_tokens=1000, repair_transcripts=False)
        assert cfg.repair_transcripts is False


class TestAssembleCallsRepair:
    """assemble() should call TranscriptRepairer.repair when enabled."""

    def test_repair_called_when_enabled(self, msg_store, sum_store):
        tail = [
            _msg(1, role="user"),
            _msg(2, role="assistant", tool_call_id="tc1", tool_name="grep"),
            _msg(3, role="tool", tool_call_id="tc1", tool_name="grep"),
        ]
        msg_store.tail.return_value = tail

        cfg = AssemblerConfig(max_context_tokens=1000, repair_transcripts=True)
        assembler = ContextAssembler(msg_store, sum_store, cfg)

        with patch("lossless_agent.engine.assembler.TranscriptRepairer") as MockRepairer:
            mock_instance = MockRepairer.return_value
            mock_instance.repair.return_value = tail  # pass through
            assembler.assemble(conv_id=1)
            mock_instance.repair.assert_called_once_with(tail)

    def test_repair_not_called_when_disabled(self, msg_store, sum_store):
        tail = [_msg(1), _msg(2, role="assistant")]
        msg_store.tail.return_value = tail

        cfg = AssemblerConfig(max_context_tokens=1000, repair_transcripts=False)
        assembler = ContextAssembler(msg_store, sum_store, cfg)

        with patch("lossless_agent.engine.assembler.TranscriptRepairer") as MockRepairer:
            assembler.assemble(conv_id=1)
            MockRepairer.return_value.repair.assert_not_called()

    def test_repaired_messages_used_in_result(self, msg_store, sum_store):
        """The assembled result should use the repaired messages, not originals."""
        original_tail = [
            _msg(1, role="user"),
            _msg(2, role="assistant", tool_call_id="tc1", tool_name="grep"),
            # Missing tool result - repairer will add one
        ]
        repaired_tail = [
            _msg(1, role="user"),
            _msg(2, role="assistant", tool_call_id="tc1", tool_name="grep"),
            _msg(3, role="tool", content="[Tool result missing]", tool_call_id="tc1"),
        ]
        msg_store.tail.return_value = original_tail

        cfg = AssemblerConfig(max_context_tokens=1000, repair_transcripts=True)
        assembler = ContextAssembler(msg_store, sum_store, cfg)

        with patch("lossless_agent.engine.assembler.TranscriptRepairer") as MockRepairer:
            mock_instance = MockRepairer.return_value
            mock_instance.repair.return_value = repaired_tail
            result = assembler.assemble(conv_id=1)
            # Result should have 3 messages (the repaired set)
            assert len(result.messages) == 3

    def test_token_count_uses_repaired_messages(self, msg_store, sum_store):
        """Token accounting should be based on repaired messages."""
        original_tail = [_msg(1, token_count=10)]
        repaired_tail = [
            _msg(1, token_count=10),
            _msg(2, role="tool", content="synthetic", token_count=7),
        ]
        msg_store.tail.return_value = original_tail

        cfg = AssemblerConfig(max_context_tokens=1000, repair_transcripts=True)
        assembler = ContextAssembler(msg_store, sum_store, cfg)

        with patch("lossless_agent.engine.assembler.TranscriptRepairer") as MockRepairer:
            MockRepairer.return_value.repair.return_value = repaired_tail
            result = assembler.assemble(conv_id=1)
            # total_tokens should include the synthetic message's tokens
            assert result.total_tokens >= 17  # 10 + 7
