"""Tests for MessageStore."""
import pytest

from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.models import Message


class TestMessageStore:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store = ConversationStore(db)
        self.store = MessageStore(db)
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
