"""Tests for resolve_prior_summary_context."""
from __future__ import annotations

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.context_item_store import ContextItemStore
from lossless_agent.engine.compaction import resolve_prior_summary_context


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _seed_messages(msg_store, conv_id, n, token_count=10):
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        m = msg_store.append(conv_id, role, f"message {i}", token_count=token_count)
        msgs.append(m)
    return msgs


# ------------------------------------------------------------------
# Without context_item_store (fallback path)
# ------------------------------------------------------------------

class TestResolvePriorSummaryFallback:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.sum_store = SummaryStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_returns_empty_when_no_summaries(self):
        result = resolve_prior_summary_context(
            self.conv.id, 1, self.sum_store
        )
        assert result == ""

    def test_returns_prior_leaf_summaries(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 10)
        # Create two leaf summaries
        s1 = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="Summary A",
            token_count=5,
            source_token_count=20,
            message_ids=[msgs[0].id, msgs[1].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )
        s2 = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="Summary B",
            token_count=5,
            source_token_count=20,
            message_ids=[msgs[2].id, msgs[3].id],
            earliest_at=msgs[2].created_at,
            latest_at=msgs[3].created_at,
            model="test",
        )
        result = resolve_prior_summary_context(
            self.conv.id, msgs[4].seq, self.sum_store
        )
        assert "Summary A" in result
        assert "Summary B" in result

    def test_respects_limit(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 10)
        for i in range(0, 6, 2):
            self.sum_store.create_leaf(
                conversation_id=self.conv.id,
                content=f"Summary {i}",
                token_count=5,
                source_token_count=20,
                message_ids=[msgs[i].id],
                earliest_at=msgs[i].created_at,
                latest_at=msgs[i].created_at,
                model="test",
            )
        result = resolve_prior_summary_context(
            self.conv.id, msgs[6].seq, self.sum_store, limit=1
        )
        # Should only contain the last summary
        assert "Summary 4" in result
        assert "Summary 0" not in result


# ------------------------------------------------------------------
# With context_item_store
# ------------------------------------------------------------------

class TestResolvePriorSummaryWithContextItems:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.sum_store = SummaryStore(db)
        self.ctx_store = ContextItemStore(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def test_returns_empty_when_no_context_items(self):
        result = resolve_prior_summary_context(
            self.conv.id, 10, self.sum_store, self.ctx_store
        )
        assert result == ""

    def test_returns_summary_from_context_items(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        # Create a leaf summary
        s = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="Prior summary content",
            token_count=5,
            source_token_count=20,
            message_ids=[msgs[0].id, msgs[1].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )
        # Add context items
        conv_id_str = str(self.conv.id)
        self.ctx_store.add_summary(conv_id_str, 1, s.summary_id)
        self.ctx_store.add_message(conv_id_str, 5, str(msgs[2].id))

        result = resolve_prior_summary_context(
            self.conv.id, 5, self.sum_store, self.ctx_store
        )
        assert "Prior summary content" in result

    def test_filters_by_ordinal(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 6)
        s = self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content="Later summary",
            token_count=5,
            source_token_count=20,
            message_ids=[msgs[4].id],
            earliest_at=msgs[4].created_at,
            latest_at=msgs[4].created_at,
            model="test",
        )
        conv_id_str = str(self.conv.id)
        # Summary at ordinal 10 - should not be included if chunk_start_seq=5
        self.ctx_store.add_summary(conv_id_str, 10, s.summary_id)

        result = resolve_prior_summary_context(
            self.conv.id, 5, self.sum_store, self.ctx_store
        )
        assert result == ""
