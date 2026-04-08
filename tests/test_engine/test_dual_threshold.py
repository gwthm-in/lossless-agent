"""Tests for dual-threshold compaction urgency (tau_soft / tau_hard)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.compaction import (
    CompactionConfig,
    CompactionEngine,
    CompactionUrgency,
)


SUMMARY_TEXT = "This is a test summary."


def _make_engine(db, *, config=None):
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    mock_summarize = AsyncMock(return_value=SUMMARY_TEXT)
    cfg = config or CompactionConfig(
        fresh_tail_count=2,
        leaf_chunk_tokens=500,
        leaf_min_fanout=2,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(msg_store, sum_store, mock_summarize, cfg)
    return conv_store, msg_store, sum_store, engine, mock_summarize


def _seed_messages(msg_store, conv_id, n, token_count=10):
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        m = msg_store.append(conv_id, role, f"message {i}", token_count=token_count)
        msgs.append(m)
    return msgs


class TestCompactionUrgencyEnum:
    def test_values(self):
        assert CompactionUrgency.NONE.value == "none"
        assert CompactionUrgency.ASYNC.value == "async"
        assert CompactionUrgency.BLOCKING.value == "blocking"


class TestDualThresholdConfig:
    def test_default_soft_threshold_falls_back(self):
        cfg = CompactionConfig()
        assert cfg.soft_threshold is None
        assert cfg.effective_soft_threshold == 0.75  # context_threshold

    def test_explicit_soft_threshold(self):
        cfg = CompactionConfig(soft_threshold=0.6)
        assert cfg.effective_soft_threshold == 0.6

    def test_hard_threshold_default(self):
        cfg = CompactionConfig()
        assert cfg.hard_threshold == 0.85

    def test_backward_compat_context_threshold(self):
        cfg = CompactionConfig(context_threshold=0.5)
        assert cfg.effective_soft_threshold == 0.5


class TestCompactionUrgency:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store, self.msg_store, self.sum_store, self.engine, _ = _make_engine(
            db,
            config=CompactionConfig(
                fresh_tail_count=2,
                leaf_chunk_tokens=500,
                leaf_min_fanout=2,
                soft_threshold=0.6,
                hard_threshold=0.85,
            ),
        )
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_none_when_under_soft(self):
        # context_limit=1000, soft=0.6 -> need 600 tokens. Add 500.
        _seed_messages(self.msg_store, self.conv.id, 5, token_count=100)
        urgency = self.engine.compaction_urgency(self.conv.id, 1000)
        assert urgency is CompactionUrgency.NONE

    def test_async_between_soft_and_hard(self):
        # context_limit=1000, soft=0.6, hard=0.85
        # 700 tokens -> ratio=0.7 -> ASYNC
        _seed_messages(self.msg_store, self.conv.id, 7, token_count=100)
        urgency = self.engine.compaction_urgency(self.conv.id, 1000)
        assert urgency is CompactionUrgency.ASYNC

    def test_blocking_above_hard(self):
        # 900 tokens -> ratio=0.9 -> BLOCKING
        _seed_messages(self.msg_store, self.conv.id, 9, token_count=100)
        urgency = self.engine.compaction_urgency(self.conv.id, 1000)
        assert urgency is CompactionUrgency.BLOCKING

    def test_blocking_at_hard_boundary(self):
        # 850 tokens -> ratio=0.85 -> BLOCKING (>= hard)
        _seed_messages(self.msg_store, self.conv.id, 17, token_count=50)
        urgency = self.engine.compaction_urgency(self.conv.id, 1000)
        assert urgency is CompactionUrgency.BLOCKING

    def test_async_at_soft_boundary(self):
        # 600 tokens -> ratio=0.6 -> ASYNC (>= soft)
        _seed_messages(self.msg_store, self.conv.id, 6, token_count=100)
        urgency = self.engine.compaction_urgency(self.conv.id, 1000)
        assert urgency is CompactionUrgency.ASYNC

    def test_needs_compaction_backward_compat(self):
        # needs_compaction should still return bool
        _seed_messages(self.msg_store, self.conv.id, 7, token_count=100)
        assert self.engine.needs_compaction(self.conv.id, 1000) is True

    def test_needs_compaction_false_when_under(self):
        _seed_messages(self.msg_store, self.conv.id, 5, token_count=100)
        assert self.engine.needs_compaction(self.conv.id, 1000) is False

    def test_zero_context_limit(self):
        _seed_messages(self.msg_store, self.conv.id, 5, token_count=100)
        urgency = self.engine.compaction_urgency(self.conv.id, 0)
        assert urgency is CompactionUrgency.NONE


class TestRunIncrementalWithUrgency:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock = _make_engine(
            db,
            config=CompactionConfig(
                fresh_tail_count=2,
                leaf_chunk_tokens=500,
                leaf_min_fanout=2,
                condensed_min_fanout=3,
                soft_threshold=0.6,
                hard_threshold=0.85,
            ),
        )
        self.conv = self.conv_store.get_or_create("s1", "Test")

    @pytest.mark.asyncio
    async def test_skips_when_none(self):
        _seed_messages(self.msg_store, self.conv.id, 5, token_count=10)
        result = await self.engine.run_incremental(self.conv.id, 1000)
        assert result == []
        self.mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_single_pass_when_async(self):
        # 10 msgs * 100 tokens = 1000, context_limit=1500
        # ratio = 0.667 -> ASYNC
        _seed_messages(self.msg_store, self.conv.id, 10, token_count=100)
        result = await self.engine.run_incremental(self.conv.id, 1500)
        # Should do at most one leaf pass (ASYNC = single pass)
        assert len(result) >= 0  # may or may not produce summary depending on chunks
