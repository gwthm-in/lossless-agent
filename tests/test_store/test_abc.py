"""Tests that SQLite store classes properly implement the ABCs."""
from __future__ import annotations

import pytest

from lossless_agent.store.abc import (
    AbstractConversationStore,
    AbstractMessageStore,
    AbstractSummaryStore,
)
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.store.database import Database


@pytest.fixture()
def db(tmp_path):
    return Database(str(tmp_path / "test.db"))


class TestConversationStoreABC:
    def test_is_subclass(self):
        assert issubclass(ConversationStore, AbstractConversationStore)

    def test_isinstance(self, db):
        store = ConversationStore(db)
        assert isinstance(store, AbstractConversationStore)


class TestMessageStoreABC:
    def test_is_subclass(self):
        assert issubclass(MessageStore, AbstractMessageStore)

    def test_isinstance(self, db):
        store = MessageStore(db)
        assert isinstance(store, AbstractMessageStore)


class TestSummaryStoreABC:
    def test_is_subclass(self):
        assert issubclass(SummaryStore, AbstractSummaryStore)

    def test_isinstance(self, db):
        store = SummaryStore(db)
        assert isinstance(store, AbstractSummaryStore)


class TestABCsCannotBeInstantiated:
    def test_conversation_store_abc(self):
        with pytest.raises(TypeError):
            AbstractConversationStore()

    def test_message_store_abc(self):
        with pytest.raises(TypeError):
            AbstractMessageStore()

    def test_summary_store_abc(self):
        with pytest.raises(TypeError):
            AbstractSummaryStore()
