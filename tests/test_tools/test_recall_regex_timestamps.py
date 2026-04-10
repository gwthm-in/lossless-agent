"""Tests for lcm_grep regex mode and timestamp filters (Feature 6)."""
from __future__ import annotations


import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.recall import lcm_grep


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def seeded(db):
    """Create conversations with messages at different timestamps."""
    cs = ConversationStore(db)
    ms = MessageStore(db)
    ss = SummaryStore(db)

    conv = cs.get_or_create("sess-regex", "Regex test chat")

    m1 = ms.append(conv.id, "user", "Tell me about quantum-123 computing", token_count=10)
    m2 = ms.append(conv.id, "assistant", "Quantum computing uses qubits-456 for processing", token_count=12)
    m3 = ms.append(conv.id, "user", "What about error-789 correction?", token_count=8)

    leaf = ss.create_leaf(
        conversation_id=conv.id,
        content="Discussion about quantum-ABC error correction",
        token_count=15,
        source_token_count=30,
        message_ids=[m1.id, m2.id, m3.id],
        earliest_at=m1.created_at,
        latest_at=m3.created_at,
        model="gpt-4",
    )

    return {
        "db": db,
        "conv": conv,
        "m1": m1, "m2": m2, "m3": m3,
        "leaf": leaf,
    }


class TestRegexMode:
    def test_regex_finds_pattern_in_messages(self, seeded):
        results = lcm_grep(seeded["db"], r"quantum-\d+", mode="regex")
        msg_results = [r for r in results if r.type == "message"]
        assert len(msg_results) >= 1
        assert any("quantum-123" in r.content_snippet for r in msg_results)

    def test_regex_finds_pattern_in_summaries(self, seeded):
        results = lcm_grep(seeded["db"], r"quantum-[A-Z]+", mode="regex")
        sum_results = [r for r in results if r.type == "summary"]
        assert len(sum_results) >= 1

    def test_regex_no_match(self, seeded):
        results = lcm_grep(seeded["db"], r"^ZZZZZ$", mode="regex")
        assert results == []

    def test_regex_scope_messages_only(self, seeded):
        results = lcm_grep(seeded["db"], r"\d{3}", mode="regex", scope="messages")
        assert all(r.type == "message" for r in results)
        assert len(results) >= 1

    def test_regex_scope_summaries_only(self, seeded):
        results = lcm_grep(seeded["db"], r"quantum", mode="regex", scope="summaries")
        assert all(r.type == "summary" for r in results)

    def test_regex_respects_limit(self, seeded):
        results = lcm_grep(seeded["db"], r".*", mode="regex", limit=2)
        assert len(results) <= 2

    def test_regex_conversation_id_filter(self, seeded):
        results = lcm_grep(
            seeded["db"], r"quantum", mode="regex",
            conversation_id=seeded["conv"].id,
        )
        assert all(r.conversation_id == seeded["conv"].id for r in results)

    def test_regex_results_have_created_at(self, seeded):
        results = lcm_grep(seeded["db"], r"quantum", mode="regex")
        for r in results:
            assert r.created_at is not None


class TestTimestampFilters:
    def test_since_filter_messages(self, seeded):
        # Use a timestamp before all messages — should get all
        results = lcm_grep(seeded["db"], "quantum", since="2000-01-01T00:00:00")
        assert len(results) >= 1

    def test_since_filter_excludes_old(self, seeded):
        # Use a future timestamp — should get nothing
        results = lcm_grep(seeded["db"], "quantum", since="2099-01-01T00:00:00")
        assert len(results) == 0

    def test_before_filter_messages(self, seeded):
        # Use a future timestamp — should get all
        results = lcm_grep(seeded["db"], "quantum", before="2099-01-01T00:00:00")
        assert len(results) >= 1

    def test_before_filter_excludes_future(self, seeded):
        # Use a past timestamp — should get nothing
        results = lcm_grep(seeded["db"], "quantum", before="2000-01-01T00:00:00")
        assert len(results) == 0

    def test_since_and_before_combined(self, seeded):
        results = lcm_grep(
            seeded["db"], "quantum",
            since="2000-01-01T00:00:00",
            before="2099-01-01T00:00:00",
        )
        assert len(results) >= 1

    def test_timestamp_filter_with_regex_mode(self, seeded):
        results = lcm_grep(
            seeded["db"], r"quantum",
            mode="regex",
            since="2000-01-01T00:00:00",
            before="2099-01-01T00:00:00",
        )
        assert len(results) >= 1

    def test_timestamp_filter_with_regex_excludes(self, seeded):
        results = lcm_grep(
            seeded["db"], r"quantum",
            mode="regex",
            since="2099-01-01T00:00:00",
        )
        assert len(results) == 0


class TestCreatedAtInResults:
    def test_fulltext_results_have_created_at(self, seeded):
        results = lcm_grep(seeded["db"], "quantum")
        for r in results:
            assert r.created_at is not None

    def test_default_mode_is_full_text(self, seeded):
        """Ensure backward compatibility - default mode is full_text."""
        results = lcm_grep(seeded["db"], "quantum")
        assert len(results) >= 1
