"""Backend-agnostic store tests — run against both SQLite and Postgres.

These tests verify that ConversationStore, MessageStore, SummaryStore,
ContextItemStore, and MessagePartStore work identically on both backends.
"""
import pytest

from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.context_item_store import ContextItemStore
from lossless_agent.store.message_part_store import MessagePartStore
from lossless_agent.store.models import Conversation, Message, Summary, MessagePart


# ===========================================================================
# ConversationStore
# ===========================================================================

class TestConversationStoreMultiBackend:
    @pytest.fixture(autouse=True)
    def setup(self, any_db):
        self.store = ConversationStore(any_db)
        self.db = any_db

    def test_get_or_create_creates_new(self):
        conv = self.store.get_or_create("sess1", "My Chat")
        assert isinstance(conv, Conversation)
        assert conv.session_key == "sess1"
        assert conv.title == "My Chat"
        assert conv.active is True
        assert conv.id is not None

    def test_get_or_create_returns_existing(self):
        conv1 = self.store.get_or_create("sess1", "Chat 1")
        conv2 = self.store.get_or_create("sess1", "Chat 1 again")
        assert conv1.id == conv2.id

    def test_get_or_create_preserves_original_title(self):
        self.store.get_or_create("sess1", "Original")
        conv = self.store.get_or_create("sess1", "New Title")
        assert conv.title == "Original"

    def test_get_by_id(self):
        conv = self.store.get_or_create("sess1", "Test")
        fetched = self.store.get_by_id(conv.id)
        assert fetched is not None
        assert fetched.id == conv.id
        assert fetched.session_key == "sess1"

    def test_get_by_id_not_found(self):
        result = self.store.get_by_id(9999)
        assert result is None

    def test_deactivate(self):
        conv = self.store.get_or_create("sess1", "Test")
        self.store.deactivate(conv.id)
        fetched = self.store.get_by_id(conv.id)
        assert fetched.active is False

    def test_active_is_bool_not_int(self):
        conv = self.store.get_or_create("sess1", "Test")
        assert type(conv.active) is bool
        self.store.deactivate(conv.id)
        fetched = self.store.get_by_id(conv.id)
        assert type(fetched.active) is bool


# ===========================================================================
# MessageStore
# ===========================================================================

class TestMessageStoreMultiBackend:
    @pytest.fixture(autouse=True)
    def setup(self, any_db):
        self.db = any_db
        self.conv_store = ConversationStore(any_db)
        self.store = MessageStore(any_db)
        self.conv = self.conv_store.get_or_create("sess1", "Test")

    def test_append_returns_message(self):
        msg = self.store.append(self.conv.id, "user", "hello", token_count=1)
        assert isinstance(msg, Message)
        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.seq == 1
        assert msg.token_count == 1

    def test_append_auto_increments_seq(self):
        m1 = self.store.append(self.conv.id, "user", "first", token_count=1)
        m2 = self.store.append(self.conv.id, "assistant", "second", token_count=2)
        assert m1.seq == 1
        assert m2.seq == 2

    def test_append_with_tool_fields(self):
        msg = self.store.append(
            self.conv.id, "tool", "result", token_count=5,
            tool_call_id="call_123", tool_name="search"
        )
        assert msg.tool_call_id == "call_123"
        assert msg.tool_name == "search"

    def test_append_invalid_role_raises(self):
        with pytest.raises(Exception):
            self.store.append(self.conv.id, "invalid", "bad", token_count=1)

    def test_get_messages_all(self):
        self.store.append(self.conv.id, "user", "one", token_count=1)
        self.store.append(self.conv.id, "assistant", "two", token_count=1)
        self.store.append(self.conv.id, "user", "three", token_count=1)
        msgs = self.store.get_messages(self.conv.id)
        assert len(msgs) == 3
        assert msgs[0].content == "one"
        assert msgs[2].content == "three"

    def test_get_messages_after_seq(self):
        self.store.append(self.conv.id, "user", "one", token_count=1)
        self.store.append(self.conv.id, "assistant", "two", token_count=1)
        self.store.append(self.conv.id, "user", "three", token_count=1)
        msgs = self.store.get_messages(self.conv.id, after_seq=1)
        assert len(msgs) == 2
        assert msgs[0].content == "two"

    def test_get_messages_with_limit(self):
        for i in range(10):
            self.store.append(self.conv.id, "user", f"msg {i}", token_count=1)
        msgs = self.store.get_messages(self.conv.id, limit=3)
        assert len(msgs) == 3

    def test_count(self):
        assert self.store.count(self.conv.id) == 0
        self.store.append(self.conv.id, "user", "one", token_count=1)
        self.store.append(self.conv.id, "assistant", "two", token_count=2)
        assert self.store.count(self.conv.id) == 2

    def test_total_tokens(self):
        self.store.append(self.conv.id, "user", "one", token_count=10)
        self.store.append(self.conv.id, "assistant", "two", token_count=20)
        assert self.store.total_tokens(self.conv.id) == 30

    def test_total_tokens_empty(self):
        assert self.store.total_tokens(self.conv.id) == 0

    def test_tail(self):
        for i in range(10):
            self.store.append(self.conv.id, "user", f"msg {i}", token_count=1)
        msgs = self.store.tail(self.conv.id, 3)
        assert len(msgs) == 3
        assert msgs[0].content == "msg 7"
        assert msgs[2].content == "msg 9"

    def test_tail_more_than_available(self):
        self.store.append(self.conv.id, "user", "only one", token_count=1)
        msgs = self.store.tail(self.conv.id, 5)
        assert len(msgs) == 1

    def test_seq_independent_per_conversation(self):
        conv2 = self.conv_store.get_or_create("sess2", "Other")
        m1 = self.store.append(self.conv.id, "user", "conv1 msg", token_count=1)
        m2 = self.store.append(conv2.id, "user", "conv2 msg", token_count=1)
        assert m1.seq == 1
        assert m2.seq == 1


# ===========================================================================
# SummaryStore
# ===========================================================================

class TestSummaryStoreMultiBackend:
    @pytest.fixture(autouse=True)
    def setup(self, any_db):
        self.db = any_db
        self.conv_store = ConversationStore(any_db)
        self.msg_store = MessageStore(any_db)
        self.store = SummaryStore(any_db)
        self.conv = self.conv_store.get_or_create("sess1", "Test")
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
            conversation_id=self.conv.id, content="Leaf 1", token_count=5,
            source_token_count=25, message_ids=[self.m1.id],
            earliest_at=self.m1.created_at, latest_at=self.m1.created_at, model="gpt-4",
        )
        leaf2 = self.store.create_leaf(
            conversation_id=self.conv.id, content="Leaf 2", token_count=5,
            source_token_count=12, message_ids=[self.m2.id],
            earliest_at=self.m2.created_at, latest_at=self.m2.created_at, model="gpt-4",
        )
        condensed = self.store.create_condensed(
            conversation_id=self.conv.id, content="Combined summary", token_count=8,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at=self.m1.created_at, latest_at=self.m2.created_at, model="gpt-4",
        )
        assert condensed.kind == "condensed"
        assert condensed.depth == 1

    def test_get_by_id(self):
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id, content="Test", token_count=5,
            source_token_count=25, message_ids=[self.m1.id],
            earliest_at=self.m1.created_at, latest_at=self.m1.created_at, model="gpt-4",
        )
        fetched = self.store.get_by_id(leaf.summary_id)
        assert fetched is not None
        assert fetched.summary_id == leaf.summary_id
        assert fetched.content == "Test"

    def test_get_by_id_not_found(self):
        assert self.store.get_by_id("sum_nonexistent") is None

    def test_get_by_conversation(self):
        self.store.create_leaf(
            conversation_id=self.conv.id, content="Leaf 1", token_count=5,
            source_token_count=25, message_ids=[self.m1.id],
            earliest_at=self.m1.created_at, latest_at=self.m1.created_at, model="gpt-4",
        )
        self.store.create_leaf(
            conversation_id=self.conv.id, content="Leaf 2", token_count=5,
            source_token_count=12, message_ids=[self.m2.id],
            earliest_at=self.m2.created_at, latest_at=self.m2.created_at, model="gpt-4",
        )
        summaries = self.store.get_by_conversation(self.conv.id)
        assert len(summaries) == 2

    def test_search_fts(self):
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="The quantum computing revolution is transforming cryptography",
            token_count=8, source_token_count=25, message_ids=[self.m1.id],
            earliest_at=self.m1.created_at, latest_at=self.m1.created_at, model="gpt-4",
        )
        self.store.create_leaf(
            conversation_id=self.conv.id,
            content="Weather patterns indicate sunny skies tomorrow",
            token_count=7, source_token_count=12, message_ids=[self.m2.id],
            earliest_at=self.m2.created_at, latest_at=self.m2.created_at, model="gpt-4",
        )
        results = self.store.search("quantum")
        assert len(results) == 1
        assert "quantum" in results[0].content

    def test_search_no_results(self):
        results = self.store.search("nonexistent_term_xyz")
        assert results == []

    def test_get_compacted_message_ids(self):
        self.store.create_leaf(
            conversation_id=self.conv.id, content="Test", token_count=5,
            source_token_count=25, message_ids=[self.m1.id, self.m2.id],
            earliest_at=self.m1.created_at, latest_at=self.m2.created_at, model="gpt-4",
        )
        compacted = self.store.get_compacted_message_ids(self.conv.id)
        assert set(compacted) == {self.m1.id, self.m2.id}

    def test_get_orphan_ids(self):
        leaf = self.store.create_leaf(
            conversation_id=self.conv.id, content="Leaf", token_count=5,
            source_token_count=25, message_ids=[self.m1.id],
            earliest_at=self.m1.created_at, latest_at=self.m1.created_at, model="gpt-4",
        )
        orphans = self.store.get_orphan_ids(self.conv.id, 0)
        assert leaf.summary_id in orphans


# ===========================================================================
# ContextItemStore
# ===========================================================================

CONV_ID = "conv_001"


class TestContextItemStoreMultiBackend:
    @pytest.fixture(autouse=True)
    def setup(self, any_db):
        self.store = ContextItemStore(any_db)
        self.db = any_db

    def test_add_message_returns_context_item(self):
        item = self.store.add_message(CONV_ID, 1, "msg_001")
        assert item.conversation_id == CONV_ID
        assert item.ordinal == 1
        assert item.item_type == "message"
        assert item.message_id == "msg_001"
        assert item.summary_id is None

    def test_add_summary_returns_context_item(self):
        item = self.store.add_summary(CONV_ID, 1, "sum_001")
        assert item.item_type == "summary"
        assert item.summary_id == "sum_001"
        assert item.message_id is None

    def test_get_items_ordered_by_ordinal(self):
        self.store.add_message(CONV_ID, 3, "msg_003")
        self.store.add_summary(CONV_ID, 1, "sum_001")
        self.store.add_message(CONV_ID, 2, "msg_002")
        items = self.store.get_items(CONV_ID)
        assert [i.ordinal for i in items] == [1, 2, 3]

    def test_remove_by_message_ids(self):
        self.store.add_message(CONV_ID, 1, "msg_001")
        self.store.add_message(CONV_ID, 2, "msg_002")
        self.store.add_message(CONV_ID, 3, "msg_003")
        self.store.remove_by_message_ids(CONV_ID, ["msg_001", "msg_002"])
        items = self.store.get_items(CONV_ID)
        assert len(items) == 1
        assert items[0].message_id == "msg_003"

    def test_replace_messages_with_summary(self):
        self.store.add_message(CONV_ID, 1, "msg_001")
        self.store.add_message(CONV_ID, 2, "msg_002")
        self.store.add_message(CONV_ID, 3, "msg_003")
        self.store.replace_messages_with_summary(
            CONV_ID, ["msg_001", "msg_002"], "sum_001", new_ordinal=1,
        )
        items = self.store.get_items(CONV_ID)
        assert len(items) == 2
        assert items[0].item_type == "summary"
        assert items[1].item_type == "message"

    def test_get_max_ordinal(self):
        self.store.add_message(CONV_ID, 5, "msg_001")
        self.store.add_message(CONV_ID, 10, "msg_002")
        assert self.store.get_max_ordinal(CONV_ID) == 10

    def test_get_max_ordinal_empty(self):
        assert self.store.get_max_ordinal("nonexistent") == 0


# ===========================================================================
# MessagePartStore
# ===========================================================================

def _seed_message(db, msg_int_id=1, conversation_id=1):
    """Create parent conversation + message so FK constraints pass."""
    db.conn.execute(
        "INSERT OR IGNORE INTO conversations (id, session_key, title, active) "
        "VALUES (?, ?, 'Test', 1)",
        (conversation_id, f"sess_{conversation_id}"),
    )
    db.conn.execute(
        "INSERT OR IGNORE INTO messages (id, conversation_id, seq, role, content, token_count) "
        "VALUES (?, ?, ?, 'user', 'test', 10)",
        (msg_int_id, conversation_id, msg_int_id),
    )
    db.conn.commit()


def _make_part(part_id="part_001", message_id="1", part_type="text",
               ordinal=0, text_content="Hello world", **kwargs):
    return MessagePart(
        part_id=part_id, message_id=message_id, part_type=part_type,
        ordinal=ordinal, text_content=text_content, **kwargs,
    )


class TestMessagePartStoreMultiBackend:
    @pytest.fixture(autouse=True)
    def setup(self, any_db):
        self.store = MessagePartStore(any_db)
        self.db = any_db

    def test_add_text_part(self):
        _seed_message(self.db, 1)
        part = _make_part()
        result = self.store.add(part)
        assert result.part_id == "part_001"
        assert result.part_type == "text"

    def test_get_by_message_ordered(self):
        _seed_message(self.db, 1)
        self.store.add(_make_part(part_id="p3", ordinal=2))
        self.store.add(_make_part(part_id="p1", ordinal=0))
        self.store.add(_make_part(part_id="p2", ordinal=1))
        parts = self.store.get_by_message("1")
        assert [p.part_id for p in parts] == ["p1", "p2", "p3"]

    def test_get_by_id(self):
        _seed_message(self.db, 1)
        self.store.add(_make_part(part_id="p1"))
        result = self.store.get_by_id("p1")
        assert result is not None
        assert result.part_id == "p1"

    def test_get_by_id_not_found(self):
        assert self.store.get_by_id("nonexistent") is None

    def test_get_by_type(self):
        _seed_message(self.db, 1)
        self.store.add(_make_part(part_id="p1", part_type="text", ordinal=0))
        self.store.add(_make_part(
            part_id="p2", part_type="tool_call", ordinal=1,
            text_content=None, tool_call_id="c1", tool_name="test"))
        texts = self.store.get_by_type("1", "text")
        assert len(texts) == 1
        tools = self.store.get_by_type("1", "tool_call")
        assert len(tools) == 1

    def test_delete_by_message(self):
        _seed_message(self.db, 1)
        self.store.add(_make_part(part_id="p1", ordinal=0))
        self.store.add(_make_part(part_id="p2", ordinal=1))
        count = self.store.delete_by_message("1")
        assert count == 2
        assert self.store.get_by_message("1") == []
