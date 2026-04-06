"""Tests for ConversationStore."""
import pytest

from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.models import Conversation


class TestConversationStore:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.store = ConversationStore(db)
        self.db = db

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
