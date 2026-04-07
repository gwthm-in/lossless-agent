"""Tests for the context assembler."""
import pytest

from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.assembler import (
    AssemblerConfig,
    ContextAssembler,
)


@pytest.fixture
def stores(db):
    """Provide MessageStore and SummaryStore on a shared in-memory DB."""
    return MessageStore(db), SummaryStore(db)


@pytest.fixture
def conv_id(db):
    """Create a conversation and return its ID."""
    cur = db.conn.execute(
        "INSERT INTO conversations (session_key, title) VALUES ('s1', 'test')"
    )
    db.conn.commit()
    return cur.lastrowid


def _add_messages(msg_store, conv_id, count, token_count=10):
    """Helper: append ``count`` messages to a conversation."""
    for i in range(count):
        msg_store.append(conv_id, "user", f"msg {i}", token_count=token_count)


class TestAssembleReturnsTailMessages:
    def test_returns_last_n_messages(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 12, token_count=10)

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert len(result.messages) == 8
        # Should be the last 8 messages (seq 5..12)
        seqs = [m.seq for m in result.messages]
        assert seqs == [5, 6, 7, 8, 9, 10, 11, 12]


class TestAssembleIncludesSummaries:
    def test_includes_summaries_when_available(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 10, token_count=10)

        # Create a leaf summary
        msgs = msg_store.get_messages(conv_id, limit=4)
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Summary of first 4 messages",
            token_count=50,
            source_token_count=40,
            message_ids=[m.id for m in msgs],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[-1].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert len(result.summaries) == 1
        assert result.summaries[0].content == "Summary of first 4 messages"
        assert len(result.messages) == 8


class TestAssembleRespectsTokenBudget:
    def test_skips_summaries_exceeding_budget(self, stores, conv_id):
        msg_store, sum_store = stores
        # 8 messages x 100 tokens = 800 tail tokens
        _add_messages(msg_store, conv_id, 10, token_count=100)

        msgs = msg_store.get_messages(conv_id, limit=2)

        # max_context_tokens=1000, tail=800, remaining=200, budget=200*0.4=80
        # Summary with 90 tokens should be skipped
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Big summary",
            token_count=90,
            source_token_count=200,
            message_ids=[m.id for m in msgs],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[-1].created_at,
            model="test",
        )

        config = AssemblerConfig(
            max_context_tokens=1000, fresh_tail_count=8, summary_budget_ratio=0.4
        )
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert len(result.summaries) == 0
        assert result.total_tokens == 800  # just the tail

    def test_includes_summaries_within_budget(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 10, token_count=10)

        msgs = msg_store.get_messages(conv_id, limit=2)
        # max=10000, tail=80, remaining=9920, budget=9920*0.4=3968
        # 50 tokens fits easily
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Small summary",
            token_count=50,
            source_token_count=20,
            message_ids=[m.id for m in msgs],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[-1].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert len(result.summaries) == 1
        assert result.total_tokens == 80 + 50


class TestAssemblePrefersHigherDepth:
    def test_higher_depth_summaries_come_first(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 10, token_count=10)

        msgs = msg_store.get_messages(conv_id)

        # Create two leaf summaries
        leaf1 = sum_store.create_leaf(
            conversation_id=conv_id,
            content="Leaf 1",
            token_count=30,
            source_token_count=40,
            message_ids=[msgs[0].id, msgs[1].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )
        leaf2 = sum_store.create_leaf(
            conversation_id=conv_id,
            content="Leaf 2",
            token_count=30,
            source_token_count=40,
            message_ids=[msgs[2].id, msgs[3].id],
            earliest_at=msgs[2].created_at,
            latest_at=msgs[3].created_at,
            model="test",
        )
        # Condensed over both leaves (depth=1)
        sum_store.create_condensed(
            conversation_id=conv_id,
            content="Condensed over leaf1+leaf2",
            token_count=40,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[3].created_at,
            model="test",
        )

        # Budget large enough for all, but higher depth should come first
        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        # Condensed (depth=1) is included; leaves are children so skipped
        depths = [s.depth for s in result.summaries]
        assert depths == [1]
        assert result.summaries[0].content == "Condensed over leaf1+leaf2"


class TestAssembleSkipsRedundantChildren:
    def test_children_skipped_when_parent_included(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 10, token_count=10)

        msgs = msg_store.get_messages(conv_id)

        leaf1 = sum_store.create_leaf(
            conversation_id=conv_id,
            content="Leaf A",
            token_count=30,
            source_token_count=40,
            message_ids=[msgs[0].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[0].created_at,
            model="test",
        )
        leaf2 = sum_store.create_leaf(
            conversation_id=conv_id,
            content="Leaf B",
            token_count=30,
            source_token_count=40,
            message_ids=[msgs[1].id],
            earliest_at=msgs[1].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )
        condensed = sum_store.create_condensed(
            conversation_id=conv_id,
            content="Parent of A+B",
            token_count=40,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        included_ids = {s.summary_id for s in result.summaries}
        assert condensed.summary_id in included_ids
        assert leaf1.summary_id not in included_ids
        assert leaf2.summary_id not in included_ids


class TestFormatContext:
    def test_produces_correct_xml_structure(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 3, token_count=10)

        msgs = msg_store.get_messages(conv_id)
        leaf = sum_store.create_leaf(
            conversation_id=conv_id,
            content="A summary",
            token_count=10,
            source_token_count=20,
            message_ids=[msgs[0].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[0].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)
        formatted = assembler.format_context(result)

        # Check summary XML block
        assert f"<summary id='{leaf.summary_id}' depth=0>" in formatted
        assert "A summary</summary>" in formatted
        # Check messages
        assert "[user] msg 0" in formatted
        assert "[user] msg 1" in formatted
        assert "[user] msg 2" in formatted


class TestEmptyConversation:
    def test_empty_conversation_returns_empty(self, stores, conv_id):
        msg_store, sum_store = stores
        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert result.summaries == []
        assert result.messages == []
        assert result.total_tokens == 0


class TestNoSummaries:
    def test_returns_only_messages_when_no_summaries(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 5, token_count=10)

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=8)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        assert result.summaries == []
        assert len(result.messages) == 5
        assert result.total_tokens == 50
