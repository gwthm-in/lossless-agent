"""Tests for MessagePartStore — structured multi-part message storage."""

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.message_part_store import MessagePartStore
from lossless_agent.store.models import MessagePart


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def store(db):
    return MessagePartStore(db)


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


def _make_part(
    part_id="part_001",
    message_id="1",
    part_type="text",
    ordinal=0,
    text_content="Hello world",
    **kwargs,
) -> MessagePart:
    return MessagePart(
        part_id=part_id,
        message_id=message_id,
        part_type=part_type,
        ordinal=ordinal,
        text_content=text_content,
        **kwargs,
    )


class TestAdd:
    """RED: add should persist a MessagePart and return it."""

    def test_add_text_part(self, db, store):
        _seed_message(db, 1)
        part = _make_part()
        result = store.add(part)
        assert result.part_id == "part_001"
        assert result.message_id == "1"
        assert result.part_type == "text"
        assert result.text_content == "Hello world"

    def test_add_tool_call_part(self, db, store):
        _seed_message(db, 1)
        part = _make_part(
            part_id="part_tc",
            part_type="tool_call",
            text_content=None,
            tool_call_id="call_123",
            tool_name="search_files",
            tool_input='{"pattern": "*.py"}',
        )
        result = store.add(part)
        assert result.part_type == "tool_call"
        assert result.tool_call_id == "call_123"
        assert result.tool_name == "search_files"

    def test_add_tool_result_part(self, db, store):
        _seed_message(db, 1)
        part = _make_part(
            part_id="part_tr",
            part_type="tool_result",
            text_content=None,
            tool_call_id="call_123",
            tool_output='{"matches": []}',
            tool_status="success",
        )
        result = store.add(part)
        assert result.tool_output == '{"matches": []}'
        assert result.tool_status == "success"

    def test_add_with_metadata(self, db, store):
        _seed_message(db, 1)
        part = _make_part(part_id="part_meta", metadata='{"source": "clipboard"}')
        result = store.add(part)
        assert result.metadata == '{"source": "clipboard"}'


class TestGetByMessage:
    """RED: get_by_message should return parts ordered by ordinal."""

    def test_returns_parts_in_ordinal_order(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="p3", ordinal=2))
        store.add(_make_part(part_id="p1", ordinal=0))
        store.add(_make_part(part_id="p2", ordinal=1))
        parts = store.get_by_message("1")
        assert [p.part_id for p in parts] == ["p1", "p2", "p3"]
        assert [p.ordinal for p in parts] == [0, 1, 2]

    def test_empty_for_unknown_message(self, store):
        assert store.get_by_message("nonexistent") == []

    def test_isolated_by_message_id(self, db, store):
        _seed_message(db, 1)
        _seed_message(db, 2)
        store.add(_make_part(part_id="p1", message_id="1"))
        store.add(_make_part(part_id="p2", message_id="2"))
        assert len(store.get_by_message("1")) == 1
        assert len(store.get_by_message("2")) == 1


class TestGetById:
    """RED: get_by_id should return a single part or None."""

    def test_returns_existing_part(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="p1"))
        result = store.get_by_id("p1")
        assert result is not None
        assert result.part_id == "p1"

    def test_returns_none_for_missing(self, store):
        assert store.get_by_id("nonexistent") is None


class TestGetByType:
    """RED: get_by_type should filter parts by type."""

    def test_filters_by_type(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="p1", part_type="text", ordinal=0))
        store.add(_make_part(part_id="p2", part_type="tool_call", ordinal=1,
                             text_content=None, tool_call_id="c1", tool_name="test"))
        store.add(_make_part(part_id="p3", part_type="text", ordinal=2))

        texts = store.get_by_type("1", "text")
        assert len(texts) == 2
        assert all(p.part_type == "text" for p in texts)

        tools = store.get_by_type("1", "tool_call")
        assert len(tools) == 1
        assert tools[0].tool_name == "test"

    def test_empty_for_no_matches(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="p1", part_type="text"))
        assert store.get_by_type("1", "image") == []


class TestDeleteByMessage:
    """RED: delete_by_message should remove all parts and return count."""

    def test_deletes_all_parts_for_message(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="p1", ordinal=0))
        store.add(_make_part(part_id="p2", ordinal=1))
        store.add(_make_part(part_id="p3", ordinal=2))
        count = store.delete_by_message("1")
        assert count == 3
        assert store.get_by_message("1") == []

    def test_returns_zero_for_no_parts(self, store):
        assert store.delete_by_message("nonexistent") == 0

    def test_does_not_affect_other_messages(self, db, store):
        _seed_message(db, 1)
        _seed_message(db, 2)
        store.add(_make_part(part_id="p1", message_id="1"))
        store.add(_make_part(part_id="p2", message_id="2"))
        store.delete_by_message("1")
        assert len(store.get_by_message("2")) == 1


class TestSchemaConstraints:
    """RED: schema constraints should be enforced."""

    def test_invalid_part_type_rejected(self, db):
        """Only the 12 valid part types should be accepted."""
        _seed_message(db, 1)
        with pytest.raises(Exception):
            db.conn.execute(
                "INSERT INTO message_parts (part_id, message_id, part_type, ordinal) "
                "VALUES ('p1', '1', 'invalid_type', 0)"
            )

    def test_all_valid_part_types_accepted(self, db):
        """Verify all 12 part types work."""
        _seed_message(db, 1)
        valid_types = [
            "text", "tool_call", "tool_result", "reasoning", "file",
            "patch", "snapshot", "image", "audio", "video", "code", "other",
        ]
        for i, pt in enumerate(valid_types):
            db.conn.execute(
                "INSERT INTO message_parts (part_id, message_id, part_type, ordinal) "
                "VALUES (?, '1', ?, 0)",
                (f"p_{i}", pt),
            )
        db.conn.commit()
        rows = db.conn.execute("SELECT COUNT(*) FROM message_parts").fetchone()
        assert rows[0] == 12

    def test_duplicate_part_id_rejected(self, db, store):
        _seed_message(db, 1)
        store.add(_make_part(part_id="dup_id"))
        with pytest.raises(Exception):
            store.add(_make_part(part_id="dup_id", ordinal=1))
