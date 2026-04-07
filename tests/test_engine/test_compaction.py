"""Tests for the compaction engine."""
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
    _format_messages,
    _format_summaries,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SUMMARY_TEXT = "This is a test summary."


def _make_engine(db, *, config=None):
    """Build stores + engine backed by the given DB."""
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
    """Append *n* messages and return the list."""
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        m = msg_store.append(conv_id, role, f"message {i}", token_count=token_count)
        msgs.append(m)
    return msgs


# ===================================================================
# select_chunk
# ===================================================================

class TestSelectChunk:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store, self.msg_store, self.sum_store, self.engine, _ = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_selects_oldest_uncompacted(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        chunk = self.engine.select_chunk(self.conv.id)
        # fresh_tail_count=2 -> exclude last 2, so eligible = msgs[0..3]
        # all uncompacted, min_fanout=2 -> should return msgs[0..3]
        assert len(chunk) == 4
        assert chunk[0].id == msgs[0].id
        assert chunk[-1].id == msgs[3].id

    def test_protects_tail(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        chunk = self.engine.select_chunk(self.conv.id)
        chunk_ids = {m.id for m in chunk}
        # Last 2 messages must not be in chunk
        assert msgs[-1].id not in chunk_ids
        assert msgs[-2].id not in chunk_ids

    def test_respects_token_limit(self):
        # Each message = 100 tokens, limit = 500 -> max 5
        conv_store, msg_store, sum_store, engine, _ = _make_engine(
            Database(":memory:"),
            config=CompactionConfig(
                fresh_tail_count=2,
                leaf_chunk_tokens=250,
                leaf_min_fanout=2,
            ),
        )
        conv = conv_store.get_or_create("s2", "Test")
        _seed_messages(msg_store, conv.id, 10, token_count=100)
        chunk = engine.select_chunk(conv.id)
        total = sum(m.token_count for m in chunk)
        assert total <= 250

    def test_respects_min_fanout(self):
        # Only 3 total messages with tail=2 -> 1 eligible < min_fanout=2
        _seed_messages(self.msg_store, self.conv.id, 3)
        chunk = self.engine.select_chunk(self.conv.id)
        assert chunk == []

    def test_skips_compacted_messages(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 8)
        # Mark first 2 as compacted via a leaf summary
        self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="old summary",
            token_count=5,
            source_token_count=20,
            message_ids=[msgs[0].id, msgs[1].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )
        chunk = self.engine.select_chunk(self.conv.id)
        chunk_ids = {m.id for m in chunk}
        assert msgs[0].id not in chunk_ids
        assert msgs[1].id not in chunk_ids

    def test_returns_empty_when_all_tail(self):
        # Only 2 messages with tail=2 -> nothing eligible
        _seed_messages(self.msg_store, self.conv.id, 2)
        assert self.engine.select_chunk(self.conv.id) == []

    def test_returns_empty_when_no_messages(self):
        assert self.engine.select_chunk(self.conv.id) == []


# ===================================================================
# needs_compaction
# ===================================================================

class TestNeedsCompaction:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store, self.msg_store, _, self.engine, _ = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_true_when_over_threshold(self):
        # 10 msgs * 100 tokens = 1000 > 0.75 * 1000 = 750
        _seed_messages(self.msg_store, self.conv.id, 10, token_count=100)
        assert self.engine.needs_compaction(self.conv.id, 1000) is True

    def test_false_when_under_threshold(self):
        # 5 msgs * 10 tokens = 50 < 0.75 * 1000 = 750
        _seed_messages(self.msg_store, self.conv.id, 5, token_count=10)
        assert self.engine.needs_compaction(self.conv.id, 1000) is False

    def test_false_when_empty(self):
        assert self.engine.needs_compaction(self.conv.id, 1000) is False


# ===================================================================
# compact_leaf
# ===================================================================

class TestCompactLeaf:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    @pytest.mark.asyncio
    async def test_creates_summary(self):
        _seed_messages(self.msg_store, self.conv.id, 6)
        result = await self.engine.compact_leaf(self.conv.id)
        assert result is not None
        assert result.kind == "leaf"
        assert result.depth == 0
        assert result.content == SUMMARY_TEXT

    @pytest.mark.asyncio
    async def test_links_messages(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        result = await self.engine.compact_leaf(self.conv.id)
        linked = self.sum_store.get_source_message_ids(result.summary_id)
        # Should cover msgs[0..3] (6 - tail 2 = 4 eligible)
        assert set(linked) == {msgs[0].id, msgs[1].id, msgs[2].id, msgs[3].id}

    @pytest.mark.asyncio
    async def test_returns_none_when_nothing_to_do(self):
        # Only 2 messages, all protected by tail
        _seed_messages(self.msg_store, self.conv.id, 2)
        result = await self.engine.compact_leaf(self.conv.id)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_summarize_fn_with_formatted_text(self):
        _seed_messages(self.msg_store, self.conv.id, 6)
        await self.engine.compact_leaf(self.conv.id)
        self.mock_fn.assert_called_once()
        call_text = self.mock_fn.call_args[0][0]
        assert "[user]" in call_text
        assert "[assistant]" in call_text

    @pytest.mark.asyncio
    async def test_source_token_count_matches(self):
        _seed_messages(self.msg_store, self.conv.id, 6, token_count=25)
        result = await self.engine.compact_leaf(self.conv.id)
        # 4 msgs * 25 tokens
        assert result.source_token_count == 100


# ===================================================================
# compact_condensed
# ===================================================================

class TestCompactCondensed:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def _make_leaf(self, msg):
        return self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content=f"Summary of {msg.content}",
            token_count=5,
            source_token_count=msg.token_count,
            message_ids=[msg.id],
            earliest_at=msg.created_at,
            latest_at=msg.created_at,
            model="test",
        )

    @pytest.mark.asyncio
    async def test_creates_condensed_node(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 5)
        for m in msgs[:3]:
            self._make_leaf(m)
        result = await self.engine.compact_condensed(self.conv.id, depth=0)
        assert result is not None
        assert result.kind == "condensed"
        assert result.depth == 1
        assert result.content == SUMMARY_TEXT

    @pytest.mark.asyncio
    async def test_returns_none_when_not_enough_summaries(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 5)
        # Only 2 leaf summaries, need condensed_min_fanout=3
        for m in msgs[:2]:
            self._make_leaf(m)
        result = await self.engine.compact_condensed(self.conv.id, depth=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_skips_non_orphan_summaries(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        leaves = [self._make_leaf(m) for m in msgs[:4]]
        # Condense first 3 leaves -> they become non-orphans
        self.sum_store.create_condensed(
            conversation_id=self.conv.id,
            content="existing condensed",
            token_count=5,
            child_ids=[leaf.summary_id for leaf in leaves[:3]],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[2].created_at,
            model="test",
        )
        # Only 1 orphan at depth 0 now (leaves[3]) -> not enough
        result = await self.engine.compact_condensed(self.conv.id, depth=0)
        assert result is None

    @pytest.mark.asyncio
    async def test_calls_summarize_fn(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 5)
        for m in msgs[:3]:
            self._make_leaf(m)
        await self.engine.compact_condensed(self.conv.id, depth=0)
        self.mock_fn.assert_called_once()
        call_text = self.mock_fn.call_args[0][0]
        assert "[summary depth=0]" in call_text


# ===================================================================
# run_incremental
# ===================================================================

class TestRunIncremental:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    @pytest.mark.asyncio
    async def test_does_nothing_below_threshold(self):
        _seed_messages(self.msg_store, self.conv.id, 3, token_count=1)
        result = await self.engine.run_incremental(self.conv.id, 100_000)
        assert result == []

    @pytest.mark.asyncio
    async def test_runs_leaf_pass(self):
        _seed_messages(self.msg_store, self.conv.id, 10, token_count=100)
        result = await self.engine.run_incremental(self.conv.id, 1000)
        # Should produce at least a leaf
        assert len(result) >= 1
        assert result[0].kind == "leaf"

    @pytest.mark.asyncio
    async def test_runs_leaf_and_condensed(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 10, token_count=100)
        # Pre-create 2 orphan leaves so after the leaf pass we have 3 orphans
        for m in msgs[:2]:
            self.sum_store.create_leaf(
                conversation_id=self.conv.id,
                content="pre-existing",
                token_count=5,
                source_token_count=m.token_count,
                message_ids=[m.id],
                earliest_at=m.created_at,
                latest_at=m.created_at,
                model="test",
            )
        result = await self.engine.run_incremental(self.conv.id, 1000)
        kinds = [s.kind for s in result]
        assert "leaf" in kinds
        assert "condensed" in kinds


# ===================================================================
# SummaryStore new methods
# ===================================================================

class TestSummaryStoreNewMethods:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.sum_store = SummaryStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")
        self.msgs = _seed_messages(self.msg_store, self.conv.id, 5)

    def test_get_compacted_message_ids(self):
        self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="leaf",
            token_count=5,
            source_token_count=20,
            message_ids=[self.msgs[0].id, self.msgs[1].id],
            earliest_at=self.msgs[0].created_at,
            latest_at=self.msgs[1].created_at,
            model="test",
        )
        compacted = self.sum_store.get_compacted_message_ids(self.conv.id)
        assert set(compacted) == {self.msgs[0].id, self.msgs[1].id}

    def test_get_compacted_message_ids_empty(self):
        assert self.sum_store.get_compacted_message_ids(self.conv.id) == []

    def test_get_orphan_ids(self):
        leaves = []
        for m in self.msgs[:3]:
            leaves.append(self.sum_store.create_leaf(
                conversation_id=self.conv.id,
                content="leaf",
                token_count=5,
                source_token_count=10,
                message_ids=[m.id],
                earliest_at=m.created_at,
                latest_at=m.created_at,
                model="test",
            ))
        # All 3 are orphans
        orphans = self.sum_store.get_orphan_ids(self.conv.id, 0)
        assert len(orphans) == 3

        # Condense first 2 -> they are no longer orphans
        self.sum_store.create_condensed(
            conversation_id=self.conv.id,
            content="condensed",
            token_count=5,
            child_ids=[leaves[0].summary_id, leaves[1].summary_id],
            earliest_at=self.msgs[0].created_at,
            latest_at=self.msgs[1].created_at,
            model="test",
        )
        orphans = self.sum_store.get_orphan_ids(self.conv.id, 0)
        assert len(orphans) == 1
        assert orphans[0] == leaves[2].summary_id


# ===================================================================
# Format helpers
# ===================================================================

class TestFormatHelpers:
    def test_format_messages(self):
        from lossless_agent.store.models import Message
        msgs = [
            Message(1, 1, 1, "user", "hello", 5, None, None, "2024-01-01"),
            Message(2, 1, 2, "assistant", "hi", 3, None, None, "2024-01-01"),
        ]
        text = _format_messages(msgs)
        assert "[user] hello" in text
        assert "[assistant] hi" in text

    def test_format_summaries(self):
        from lossless_agent.store.models import Summary
        sums = [
            Summary("s1", 1, "leaf", 0, "content A", 5, 10, "t1", "t2", "m", "c"),
            Summary("s2", 1, "leaf", 0, "content B", 5, 10, "t1", "t2", "m", "c"),
        ]
        text = _format_summaries(sums)
        assert "[summary depth=0] content A" in text
        assert "[summary depth=0] content B" in text
