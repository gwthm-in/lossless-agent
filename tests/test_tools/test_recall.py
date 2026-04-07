"""Tests for recall tools (lcm_grep, lcm_describe, lcm_expand)."""
from __future__ import annotations

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.recall import (
    lcm_grep,
    lcm_describe,
    lcm_expand,
)


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def stores(db):
    """Return (db, conv_store, msg_store, sum_store) with seeded data."""
    cs = ConversationStore(db)
    ms = MessageStore(db)
    ss = SummaryStore(db)
    return db, cs, ms, ss


@pytest.fixture
def seeded(stores):
    """Create two conversations with messages and summaries."""
    db, cs, ms, ss = stores

    c1 = cs.get_or_create("sess-1", "First chat")
    c2 = cs.get_or_create("sess-2", "Second chat")

    # Messages in conversation 1
    m1 = ms.append(c1.id, "user", "Tell me about quantum computing basics", token_count=10)
    m2 = ms.append(c1.id, "assistant", "Quantum computing uses qubits for parallel processing", token_count=12)
    m3 = ms.append(c1.id, "user", "What about entanglement?", token_count=8)

    # Messages in conversation 2
    m4 = ms.append(c2.id, "user", "Explain machine learning fundamentals", token_count=9)
    m5 = ms.append(c2.id, "assistant", "Machine learning trains models on data patterns", token_count=11)

    # Leaf summary in conversation 1
    leaf = ss.create_leaf(
        conversation_id=c1.id,
        content="Discussion about quantum computing covering qubits and entanglement",
        token_count=15,
        source_token_count=30,
        message_ids=[m1.id, m2.id, m3.id],
        earliest_at=m1.created_at,
        latest_at=m3.created_at,
        model="gpt-4",
    )

    # Another leaf in conversation 1
    leaf2 = ss.create_leaf(
        conversation_id=c1.id,
        content="Follow-up on quantum error correction",
        token_count=10,
        source_token_count=20,
        message_ids=[m1.id],
        earliest_at=m1.created_at,
        latest_at=m1.created_at,
        model="gpt-4",
    )

    # Condensed summary over the two leaves
    condensed = ss.create_condensed(
        conversation_id=c1.id,
        content="Comprehensive quantum computing overview including basics and error correction",
        token_count=20,
        child_ids=[leaf.summary_id, leaf2.summary_id],
        earliest_at=m1.created_at,
        latest_at=m3.created_at,
        model="gpt-4",
    )

    return {
        "db": db,
        "c1": c1, "c2": c2,
        "m1": m1, "m2": m2, "m3": m3, "m4": m4, "m5": m5,
        "leaf": leaf, "leaf2": leaf2, "condensed": condensed,
    }


# --- lcm_grep tests ---

class TestLcmGrep:
    def test_finds_messages_by_content(self, seeded):
        results = lcm_grep(seeded["db"], "quantum")
        msg_results = [r for r in results if r.type == "message"]
        assert len(msg_results) >= 1
        assert any("quantum" in r.content_snippet.lower() for r in msg_results)

    def test_finds_summaries_by_content(self, seeded):
        results = lcm_grep(seeded["db"], "quantum")
        sum_results = [r for r in results if r.type == "summary"]
        assert len(sum_results) >= 1

    def test_scope_messages_only(self, seeded):
        results = lcm_grep(seeded["db"], "quantum", scope="messages")
        assert all(r.type == "message" for r in results)
        assert len(results) >= 1

    def test_scope_summaries_only(self, seeded):
        results = lcm_grep(seeded["db"], "quantum", scope="summaries")
        assert all(r.type == "summary" for r in results)
        assert len(results) >= 1

    def test_conversation_id_filter(self, seeded):
        results = lcm_grep(seeded["db"], "machine", conversation_id=seeded["c2"].id)
        assert len(results) >= 1
        assert all(r.conversation_id == seeded["c2"].id for r in results)

        # Should not find conv2 messages when filtering to conv1
        results2 = lcm_grep(seeded["db"], "machine", conversation_id=seeded["c1"].id)
        assert len(results2) == 0

    def test_limit(self, seeded):
        results = lcm_grep(seeded["db"], "quantum", limit=1)
        assert len(results) <= 1

    def test_no_matches(self, seeded):
        results = lcm_grep(seeded["db"], "xyznonexistent")
        assert results == []

    def test_snippet_truncated(self, seeded):
        # All snippets should be <= 200 chars
        results = lcm_grep(seeded["db"], "quantum")
        for r in results:
            assert len(r.content_snippet) <= 200


# --- lcm_describe tests ---

class TestLcmDescribe:
    def test_returns_full_summary_info(self, seeded):
        result = lcm_describe(seeded["db"], seeded["leaf"].summary_id)
        assert result is not None
        assert result.summary_id == seeded["leaf"].summary_id
        assert result.kind == "leaf"
        assert result.depth == 0
        assert "quantum" in result.content.lower()
        assert result.token_count == 15
        assert result.source_token_count == 30
        assert result.earliest_at == seeded["leaf"].earliest_at
        assert result.latest_at == seeded["leaf"].latest_at
        assert result.source_message_count == 3
        assert result.child_ids == []

    def test_returns_child_ids_for_condensed(self, seeded):
        result = lcm_describe(seeded["db"], seeded["condensed"].summary_id)
        assert result is not None
        assert result.kind == "condensed"
        assert result.depth == 1
        assert set(result.child_ids) == {
            seeded["leaf"].summary_id,
            seeded["leaf2"].summary_id,
        }
        assert result.source_message_count == 0

    def test_returns_none_for_missing(self, seeded):
        result = lcm_describe(seeded["db"], "sum_doesnotexist")
        assert result is None


# --- lcm_expand tests ---

class TestLcmExpand:
    def test_returns_source_messages_for_leaf(self, seeded):
        result = lcm_expand(seeded["db"], seeded["leaf"].summary_id)
        assert result is not None
        assert result.summary_id == seeded["leaf"].summary_id
        assert result.kind == "leaf"
        assert len(result.children) == 3
        # Children should be Message objects
        from lossless_agent.store.models import Message
        assert all(isinstance(c, Message) for c in result.children)
        contents = [c.content for c in result.children]
        assert "Tell me about quantum computing basics" in contents

    def test_returns_child_summaries_for_condensed(self, seeded):
        result = lcm_expand(seeded["db"], seeded["condensed"].summary_id)
        assert result is not None
        assert result.summary_id == seeded["condensed"].summary_id
        assert result.kind == "condensed"
        assert len(result.children) == 2
        from lossless_agent.store.models import Summary
        assert all(isinstance(c, Summary) for c in result.children)
        child_ids = {c.summary_id for c in result.children}
        assert child_ids == {
            seeded["leaf"].summary_id,
            seeded["leaf2"].summary_id,
        }

    def test_returns_none_for_missing(self, seeded):
        result = lcm_expand(seeded["db"], "sum_doesnotexist")
        assert result is None
