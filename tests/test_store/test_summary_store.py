"""Tests for SummaryStore."""
import pytest

from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.models import Summary


class TestSummaryStore:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store = ConversationStore(db)
        self.msg_store = MessageStore(db)
        self.store = SummaryStore(db)
        self.conv = self.conv_store.get_or_create("sess1", "Test")
        # Insert some messages to reference
        self.m1 = self.msg_store.append(self.conv.id, "user", "hello", token_count=10)
        self.m2 = self.msg_store.append(self.conv.id, "assistant", "hi there", token_count=15)
        self.m3 = self.msg_store.append(self.conv.id, "user", "how are you", token_count=12)

    def test_create_leaf(self):
        summary = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="User greeted assistant",
            token_count=5,
            source_token_count=37,
            message_ids=[self.m1.id, self.m2.id, self.m3.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m3.created_at,
            model="gpt-4",
        )
        assert isinstance(summary, Summary)
        assert summary.kind == "leaf"
        assert summary.depth == 0
        assert summary.content == "User greeted assistant"
        assert summary.summary_id.startswith("sum_")
        assert len(summary.summary_id) == 16  # sum_ + 12 hex chars

    def test_create_leaf_links_messages(self):
        summary = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Summary",
            token_count=5,
            source_token_count=37,
            message_ids=[self.m1.id, self.m2.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        msg_ids = self.store.get_source_message_ids(summary.summary_id)
        assert set(msg_ids) == {self.m1.id, self.m2.id}

    def test_create_condensed(self):
        leaf1 = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf 1",
            token_count=5,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        leaf2 = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf 2",
            token_count=5,
            source_token_count=12,
            message_ids=[self.m2.id],
            earliest_at=self.m2.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        condensed = self.store.create_condensed(
            conversation_id=self.conv.id,
            content="Combined summary",
            token_count=8,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at=self.m1.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        assert condensed.kind == "condensed"
        assert condensed.depth == 1  # one above leaf (depth 0)

    def test_condensed_depth_auto_calc(self):
        """Condensed summary depth = max(child depths) + 1."""
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf",
            token_count=5,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        c1 = self.store.create_condensed(
            conversation_id=self.conv.id,
            content="Level 1",
            token_count=5,
            child_ids=[leaf.summary_id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        assert c1.depth == 1

        c2 = self.store.create_condensed(
            conversation_id=self.conv.id,
            content="Level 2",
            token_count=5,
            child_ids=[c1.summary_id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        assert c2.depth == 2

    def test_get_by_id(self):
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Test",
            token_count=5,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        fetched = self.store.get_by_id(leaf.summary_id)
        assert fetched is not None
        assert fetched.summary_id == leaf.summary_id
        assert fetched.content == "Test"

    def test_get_by_id_not_found(self):
        assert self.store.get_by_id("sum_nonexistent") is None

    def test_get_by_conversation(self):
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf 1",
            token_count=5,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf 2",
            token_count=5,
            source_token_count=12,
            message_ids=[self.m2.id],
            earliest_at=self.m2.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        summaries = self.store.get_by_conversation(self.conv.id)
        assert len(summaries) == 2

    def test_get_by_depth(self):
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Leaf",
            token_count=5,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        self.store.create_condensed(
            conversation_id=self.conv.id,
            content="Condensed",
            token_count=5,
            child_ids=[leaf.summary_id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        depth_0 = self.store.get_by_depth(self.conv.id, 0)
        depth_1 = self.store.get_by_depth(self.conv.id, 1)
        assert len(depth_0) == 1
        assert len(depth_1) == 1
        assert depth_0[0].kind == "leaf"
        assert depth_1[0].kind == "condensed"

    def test_get_child_ids(self):
        leaf1 = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="L1",
            token_count=5,
            source_token_count=10,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        leaf2 = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="L2",
            token_count=5,
            source_token_count=15,
            message_ids=[self.m2.id],
            earliest_at=self.m2.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        condensed = self.store.create_condensed(
            conversation_id=self.conv.id,
            content="C",
            token_count=5,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at=self.m1.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        child_ids = self.store.get_child_ids(condensed.summary_id)
        assert set(child_ids) == {leaf1.summary_id, leaf2.summary_id}

    def test_search_fts(self):
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="The quantum computing revolution is transforming cryptography",
            token_count=8,
            source_token_count=25,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Weather patterns indicate sunny skies tomorrow",
            token_count=7,
            source_token_count=12,
            message_ids=[self.m2.id],
            earliest_at=self.m2.created_at,
            latest_at=self.m2.created_at,
            model="gpt-4",
        )
        results = self.store.search("quantum")
        assert len(results) == 1
        assert "quantum" in results[0].content

    def test_search_no_results(self):
        results = self.store.search("nonexistent_term_xyz")
        assert results == []

    def test_source_token_count_stored(self):
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Test",
            token_count=5,
            source_token_count=42,
            message_ids=[self.m1.id],
            earliest_at=self.m1.created_at,
            latest_at=self.m1.created_at,
            model="gpt-4",
        )
        fetched = self.store.get_by_id(leaf.summary_id)
        assert fetched.source_token_count == 42
