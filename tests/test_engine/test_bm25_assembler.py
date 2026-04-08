"""Tests for BM25-lite relevance scoring in assembler."""
import pytest

from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.engine.assembler import (
    AssemblerConfig,
    ContextAssembler,
)


@pytest.fixture
def stores(db):
    return MessageStore(db), SummaryStore(db)


@pytest.fixture
def conv_id(db):
    cur = db.conn.execute(
        "INSERT INTO conversations (session_key, title) VALUES ('s1', 'test')"
    )
    db.conn.commit()
    return cur.lastrowid


def _add_messages(msg_store, conv_id, count, token_count=10):
    for i in range(count):
        msg_store.append(conv_id, "user", f"msg {i}", token_count=token_count)


class TestBM25Score:
    def test_score_with_matching_terms(self, stores, conv_id):
        msg_store, sum_store = stores
        config = AssemblerConfig(max_context_tokens=10000)
        assembler = ContextAssembler(msg_store, sum_store, config)

        score = assembler._bm25_score(
            ["python", "programming"],
            "python programming is fun and python is great",
            avg_doc_len=10.0,
        )
        assert score > 0

    def test_score_zero_for_no_match(self, stores, conv_id):
        msg_store, sum_store = stores
        config = AssemblerConfig(max_context_tokens=10000)
        assembler = ContextAssembler(msg_store, sum_store, config)

        score = assembler._bm25_score(
            ["python"], "java is a programming language", avg_doc_len=10.0
        )
        assert score == 0.0

    def test_higher_score_for_more_matches(self, stores, conv_id):
        msg_store, sum_store = stores
        config = AssemblerConfig(max_context_tokens=10000)
        assembler = ContextAssembler(msg_store, sum_store, config)

        score_low = assembler._bm25_score(
            ["python", "web"], "python is great", avg_doc_len=10.0
        )
        score_high = assembler._bm25_score(
            ["python", "web"], "python web framework for python web apps", avg_doc_len=10.0
        )
        assert score_high > score_low

    def test_empty_query_returns_zero(self, stores, conv_id):
        msg_store, sum_store = stores
        config = AssemblerConfig(max_context_tokens=10000)
        assembler = ContextAssembler(msg_store, sum_store, config)

        score = assembler._bm25_score([], "some document", avg_doc_len=10.0)
        assert score == 0.0


class TestAssembleWithPrompt:
    def test_prompt_reorders_by_relevance(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 4, token_count=10)
        msgs = msg_store.get_messages(conv_id)

        # Create summaries with different content
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Discussion about cooking recipes and food preparation",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[0].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[0].created_at,
            model="test",
        )
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Python programming and debugging techniques",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[1].id],
            earliest_at=msgs[1].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=4)
        assembler = ContextAssembler(msg_store, sum_store, config)

        # With prompt about python, python summary should come first
        result = assembler.assemble(conv_id, prompt="python debugging")
        assert len(result.summaries) == 2
        assert "python" in result.summaries[0].content.lower()

    def test_no_prompt_uses_chronological(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 4, token_count=10)
        msgs = msg_store.get_messages(conv_id)

        sum_store.create_leaf(
            conversation_id=conv_id,
            content="First topic",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[0].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[0].created_at,
            model="test",
        )
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="Second topic",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[1].id],
            earliest_at=msgs[1].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )

        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=4)
        assembler = ContextAssembler(msg_store, sum_store, config)

        # Without prompt, default ordering (depth DESC, earliest ASC)
        result = assembler.assemble(conv_id)
        assert len(result.summaries) == 2


class TestAssembleWithPromptRespectsBudget:
    def test_budget_still_respected_with_prompt(self, stores, conv_id):
        msg_store, sum_store = stores
        _add_messages(msg_store, conv_id, 8, token_count=100)
        msgs = msg_store.get_messages(conv_id)

        # Budget: max=1000, tail=800, remaining=200, summary_budget=80
        # Each summary=50, can fit 1
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="python topic",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[0].id],
            earliest_at=msgs[0].created_at,
            latest_at=msgs[0].created_at,
            model="test",
        )
        sum_store.create_leaf(
            conversation_id=conv_id,
            content="java topic",
            token_count=50,
            source_token_count=40,
            message_ids=[msgs[1].id],
            earliest_at=msgs[1].created_at,
            latest_at=msgs[1].created_at,
            model="test",
        )

        config = AssemblerConfig(
            max_context_tokens=1000, fresh_tail_count=8, summary_budget_ratio=0.4
        )
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id, prompt="python code")

        # Budget is 80, each summary is 50, so only 1 fits
        assert len(result.summaries) == 1
        # The python one should be picked (higher BM25 score)
        assert "python" in result.summaries[0].content.lower()
