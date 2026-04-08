"""Tests for ContextItemStore — context window tracking with ordinals."""

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.context_item_store import ContextItemStore


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def store(db):
    return ContextItemStore(db)


CONV_ID = "conv_001"


class TestAddMessage:
    """RED: add_message should insert a message-type context item."""

    def test_add_message_returns_context_item(self, store):
        item = store.add_message(CONV_ID, 1, "msg_001")
        assert item.conversation_id == CONV_ID
        assert item.ordinal == 1
        assert item.item_type == "message"
        assert item.message_id == "msg_001"
        assert item.summary_id is None

    def test_add_multiple_messages_with_ordinals(self, store):
        store.add_message(CONV_ID, 1, "msg_001")
        store.add_message(CONV_ID, 2, "msg_002")
        store.add_message(CONV_ID, 3, "msg_003")
        items = store.get_items(CONV_ID)
        assert len(items) == 3
        assert [i.ordinal for i in items] == [1, 2, 3]


class TestAddSummary:
    """RED: add_summary should insert a summary-type context item."""

    def test_add_summary_returns_context_item(self, store):
        item = store.add_summary(CONV_ID, 1, "sum_001")
        assert item.conversation_id == CONV_ID
        assert item.ordinal == 1
        assert item.item_type == "summary"
        assert item.summary_id == "sum_001"
        assert item.message_id is None


class TestGetItems:
    """RED: get_items should return items ordered by ordinal."""

    def test_empty_conversation_returns_empty(self, store):
        assert store.get_items("nonexistent") == []

    def test_items_returned_in_ordinal_order(self, store):
        store.add_message(CONV_ID, 3, "msg_003")
        store.add_summary(CONV_ID, 1, "sum_001")
        store.add_message(CONV_ID, 2, "msg_002")
        items = store.get_items(CONV_ID)
        assert [i.ordinal for i in items] == [1, 2, 3]
        assert items[0].item_type == "summary"
        assert items[1].item_type == "message"
        assert items[2].item_type == "message"

    def test_items_isolated_by_conversation(self, store):
        store.add_message("conv_a", 1, "msg_001")
        store.add_message("conv_b", 1, "msg_002")
        assert len(store.get_items("conv_a")) == 1
        assert len(store.get_items("conv_b")) == 1


class TestRemoveByMessageIds:
    """RED: remove_by_message_ids should delete matching message items."""

    def test_removes_specific_messages(self, store):
        store.add_message(CONV_ID, 1, "msg_001")
        store.add_message(CONV_ID, 2, "msg_002")
        store.add_message(CONV_ID, 3, "msg_003")
        store.remove_by_message_ids(CONV_ID, ["msg_001", "msg_002"])
        items = store.get_items(CONV_ID)
        assert len(items) == 1
        assert items[0].message_id == "msg_003"

    def test_does_not_remove_summaries(self, store):
        store.add_summary(CONV_ID, 1, "sum_001")
        store.add_message(CONV_ID, 2, "msg_001")
        store.remove_by_message_ids(CONV_ID, ["sum_001"])  # Wrong type
        items = store.get_items(CONV_ID)
        assert len(items) == 2  # Summary unaffected

    def test_empty_list_is_noop(self, store):
        store.add_message(CONV_ID, 1, "msg_001")
        store.remove_by_message_ids(CONV_ID, [])
        assert len(store.get_items(CONV_ID)) == 1


class TestReplaceMessagesWithSummary:
    """RED: replace_messages_with_summary should atomically swap messages for a summary."""

    def test_replaces_messages_with_summary(self, store):
        store.add_message(CONV_ID, 1, "msg_001")
        store.add_message(CONV_ID, 2, "msg_002")
        store.add_message(CONV_ID, 3, "msg_003")
        store.replace_messages_with_summary(
            CONV_ID,
            ["msg_001", "msg_002"],
            "sum_001",
            new_ordinal=1,
        )
        items = store.get_items(CONV_ID)
        # Should have summary at ordinal 1 and message at ordinal 3
        assert len(items) == 2
        assert items[0].item_type == "summary"
        assert items[0].summary_id == "sum_001"
        assert items[1].item_type == "message"
        assert items[1].message_id == "msg_003"

    def test_replace_with_empty_message_ids_just_adds_summary(self, store):
        store.replace_messages_with_summary(CONV_ID, [], "sum_001", new_ordinal=1)
        items = store.get_items(CONV_ID)
        assert len(items) == 1
        assert items[0].item_type == "summary"


class TestGetMaxOrdinal:
    """RED: get_max_ordinal should return highest ordinal or 0."""

    def test_empty_conversation_returns_zero(self, store):
        assert store.get_max_ordinal("nonexistent") == 0

    def test_returns_highest_ordinal(self, store):
        store.add_message(CONV_ID, 5, "msg_001")
        store.add_message(CONV_ID, 10, "msg_002")
        store.add_summary(CONV_ID, 15, "sum_001")
        assert store.get_max_ordinal(CONV_ID) == 15


class TestSchemaConstraints:
    """RED: schema CHECK constraints should be enforced."""

    def test_message_item_must_have_message_id(self, db):
        """A message-type item with summary_id should violate CHECK."""
        with pytest.raises(Exception):
            db.conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, summary_id) "
                "VALUES ('x', 1, 'message', 'sum_001')"
            )

    def test_summary_item_must_have_summary_id(self, db):
        """A summary-type item with message_id should violate CHECK."""
        with pytest.raises(Exception):
            db.conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
                "VALUES ('x', 1, 'summary', 'msg_001')"
            )

    def test_invalid_item_type_rejected(self, db):
        """Only 'message' and 'summary' are valid item types."""
        with pytest.raises(Exception):
            db.conn.execute(
                "INSERT INTO context_items (conversation_id, ordinal, item_type, message_id) "
                "VALUES ('x', 1, 'invalid', 'msg_001')"
            )
