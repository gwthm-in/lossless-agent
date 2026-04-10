"""Tests for engine.transcript_repair – TranscriptRepairer."""
from __future__ import annotations

import pytest

from lossless_agent.store.models import Message
from lossless_agent.engine.transcript_repair import TranscriptRepairer


def _msg(seq, role="user", content="hi", tool_call_id=None, tool_name=None):
    """Helper to create a Message with minimal boilerplate."""
    return Message(
        id=seq,
        conversation_id=1,
        seq=seq,
        role=role,
        content=content,
        token_count=5,
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        created_at="2025-01-01",
    )


@pytest.fixture
def repairer():
    return TranscriptRepairer()


# ── Basic pass-through ──────────────────────────────────────────────

class TestPassThrough:
    def test_empty(self, repairer):
        assert repairer.repair([]) == []

    def test_no_tools(self, repairer):
        msgs = [_msg(1), _msg(2, role="assistant", content="ok")]
        result = repairer.repair(msgs)
        assert len(result) == 2
        assert [m.seq for m in result] == [1, 2]


# ── 1. Reorder tool results ────────────────────────────────────────

class TestReorder:
    def test_result_moved_after_call(self, repairer):
        msgs = [
            _msg(1, role="user"),
            _msg(2, role="tool", content="result A", tool_call_id="tc_1"),
            _msg(3, role="assistant", content="call A", tool_call_id="tc_1", tool_name="fn"),
        ]
        result = repairer.repair(msgs)
        [(m.role, m.tool_call_id) for m in result]
        # assistant call should come before tool result
        call_idx = next(i for i, m in enumerate(result) if m.role == "assistant" and m.tool_call_id == "tc_1")
        result_idx = next(i for i, m in enumerate(result) if m.role == "tool" and m.tool_call_id == "tc_1")
        assert call_idx < result_idx
        assert result_idx == call_idx + 1

    def test_already_ordered(self, repairer):
        msgs = [
            _msg(1, role="user"),
            _msg(2, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
            _msg(3, role="tool", content="result", tool_call_id="tc_1"),
        ]
        result = repairer.repair(msgs)
        assert len(result) == 3
        assert result[1].role == "assistant"
        assert result[2].role == "tool"


# ── 2. Synthetic missing results ───────────────────────────────────

class TestSyntheticResults:
    def test_missing_result_synthesised(self, repairer):
        msgs = [
            _msg(1, role="user"),
            _msg(2, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
        ]
        result = repairer.repair(msgs)
        assert len(result) == 3
        synthetic = result[2]
        assert synthetic.role == "tool"
        assert synthetic.tool_call_id == "tc_1"
        assert "missing" in synthetic.content.lower()

    def test_no_synthetic_when_result_exists(self, repairer):
        msgs = [
            _msg(1, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
            _msg(2, role="tool", content="result", tool_call_id="tc_1"),
        ]
        result = repairer.repair(msgs)
        assert len(result) == 2
        assert all("missing" not in m.content.lower() for m in result)


# ── 3. Duplicate tool results ──────────────────────────────────────

class TestDuplicateResults:
    def test_duplicate_dropped(self, repairer):
        msgs = [
            _msg(1, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
            _msg(2, role="tool", content="result 1", tool_call_id="tc_1"),
            _msg(3, role="tool", content="result 2 (dup)", tool_call_id="tc_1"),
        ]
        result = repairer.repair(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].content == "result 1"


# ── 4. Orphaned tool results ───────────────────────────────────────

class TestOrphanedResults:
    def test_orphan_dropped(self, repairer):
        msgs = [
            _msg(1, role="user"),
            _msg(2, role="tool", content="orphan result", tool_call_id="tc_no_call"),
        ]
        result = repairer.repair(msgs)
        assert len(result) == 1
        assert result[0].role == "user"

    def test_orphan_dropped_mixed(self, repairer):
        msgs = [
            _msg(1, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
            _msg(2, role="tool", content="good result", tool_call_id="tc_1"),
            _msg(3, role="tool", content="orphan", tool_call_id="tc_other"),
        ]
        result = repairer.repair(msgs)
        tool_msgs = [m for m in result if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "tc_1"


# ── Combined scenario ──────────────────────────────────────────────

class TestCombined:
    def test_complex_repair(self, repairer):
        """Scenario: misordered result, missing result, duplicate, orphan."""
        msgs = [
            _msg(1, role="user", content="do stuff"),
            # Tool result before its call (misordered)
            _msg(2, role="tool", content="result for tc_1", tool_call_id="tc_1"),
            _msg(3, role="assistant", content="call 1", tool_call_id="tc_1", tool_name="fn1"),
            # Tool call with no result (missing)
            _msg(4, role="assistant", content="call 2", tool_call_id="tc_2", tool_name="fn2"),
            # Duplicate result
            _msg(5, role="tool", content="dup 1", tool_call_id="tc_1"),
            # Orphan result
            _msg(6, role="tool", content="orphan", tool_call_id="tc_ghost"),
            _msg(7, role="user", content="thanks"),
        ]
        result = repairer.repair(msgs)

        # Check ordering
        [(m.role, m.tool_call_id) for m in result]

        # tc_1 call should be followed by its result
        tc1_call_idx = next(
            i for i, m in enumerate(result)
            if m.role == "assistant" and m.tool_call_id == "tc_1"
        )
        tc1_result_idx = next(
            i for i, m in enumerate(result)
            if m.role == "tool" and m.tool_call_id == "tc_1"
        )
        assert tc1_result_idx == tc1_call_idx + 1

        # tc_2 should have a synthetic result
        tc2_call_idx = next(
            i for i, m in enumerate(result)
            if m.role == "assistant" and m.tool_call_id == "tc_2"
        )
        tc2_result = result[tc2_call_idx + 1]
        assert tc2_result.role == "tool"
        assert tc2_result.tool_call_id == "tc_2"
        assert "missing" in tc2_result.content.lower()

        # No orphan
        assert not any(
            m.tool_call_id == "tc_ghost" for m in result
        )

        # No duplicate tc_1 results
        tc1_results = [m for m in result if m.role == "tool" and m.tool_call_id == "tc_1"]
        assert len(tc1_results) == 1

    def test_does_not_mutate_input(self, repairer):
        msgs = [
            _msg(1, role="assistant", content="call", tool_call_id="tc_1", tool_name="fn"),
        ]
        original_content = msgs[0].content
        repairer.repair(msgs)
        assert msgs[0].content == original_content
        assert len(msgs) == 1  # original list not modified


# ── Edge cases ──────────────────────────────────────────────────────

class TestEdgeCases:
    def test_tool_message_without_tool_call_id(self, repairer):
        """Tool messages without tool_call_id are left alone."""
        msgs = [
            _msg(1, role="user"),
            _msg(2, role="tool", content="no id"),
        ]
        result = repairer.repair(msgs)
        assert len(result) == 2

    def test_multiple_calls_multiple_results(self, repairer):
        msgs = [
            _msg(1, role="assistant", content="call A", tool_call_id="tc_a", tool_name="fnA"),
            _msg(2, role="assistant", content="call B", tool_call_id="tc_b", tool_name="fnB"),
            _msg(3, role="tool", content="result B", tool_call_id="tc_b"),
            _msg(4, role="tool", content="result A", tool_call_id="tc_a"),
        ]
        result = repairer.repair(msgs)
        # call A -> result A, call B -> result B
        assert result[0].role == "assistant" and result[0].tool_call_id == "tc_a"
        assert result[1].role == "tool" and result[1].tool_call_id == "tc_a"
        assert result[2].role == "assistant" and result[2].tool_call_id == "tc_b"
        assert result[3].role == "tool" and result[3].tool_call_id == "tc_b"
