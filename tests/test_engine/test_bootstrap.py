"""Tests for SessionBootstrap engine."""
import asyncio

import pytest

from lossless_agent.engine.bootstrap import BootstrapResult, SessionBootstrap
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.database import Database
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def stores(db):
    return {
        "conv": ConversationStore(db),
        "msg": MessageStore(db),
        "sum": SummaryStore(db),
    }


def _dummy_summarize(text: str) -> str:
    """Dummy summarize function that just returns the text truncated."""
    return text[:100]


def _token_count(text: str) -> int:
    """Simple token count approximation: split by spaces."""
    return len(text.split())


class TestBootstrapResult:
    def test_dataclass_fields(self):
        r = BootstrapResult(summaries_imported=3, messages_imported=5, tokens_used=1500)
        assert r.summaries_imported == 3
        assert r.messages_imported == 5
        assert r.tokens_used == 1500

    def test_defaults(self):
        r = BootstrapResult()
        assert r.summaries_imported == 0
        assert r.messages_imported == 0
        assert r.tokens_used == 0


class TestSessionBootstrap:
    def test_init(self, db):
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize)
        assert sb.bootstrap_max_tokens == 6000

    def test_custom_max_tokens(self, db):
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize, bootstrap_max_tokens=3000)
        assert sb.bootstrap_max_tokens == 3000

    def test_bootstrap_no_parent(self, db, stores):
        """Bootstrap should raise or return empty when parent doesn't exist."""
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize)
        new_conv = stores["conv"].get_or_create("new:session")
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "nonexistent:parent")
        )
        assert result.summaries_imported == 0
        assert result.messages_imported == 0
        assert result.tokens_used == 0

    def test_bootstrap_with_parent_messages(self, db, stores):
        """Bootstrap should import messages from parent conversation."""
        parent = stores["conv"].get_or_create("parent:session")
        for i in range(5):
            stores["msg"].append(
                parent.id, "user", f"Message number {i} with some content",
                token_count=10,
            )
        new_conv = stores["conv"].get_or_create("new:session")
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize, bootstrap_max_tokens=6000)
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )
        assert result.messages_imported > 0
        assert result.tokens_used > 0

    def test_bootstrap_with_parent_summaries(self, db, stores):
        """Bootstrap should prefer summaries over raw messages."""
        parent = stores["conv"].get_or_create("parent:session")
        # Add messages
        msg_ids = []
        for i in range(3):
            m = stores["msg"].append(
                parent.id, "user", f"Message {i}",
                token_count=10,
            )
            msg_ids.append(m.id)
        # Create a summary
        stores["sum"].create_leaf(
            conversation_id=parent.id,
            content="Summary of messages about testing",
            token_count=20,
            source_token_count=30,
            message_ids=msg_ids,
            earliest_at="2024-01-01T00:00:00",
            latest_at="2024-01-01T00:01:00",
            model="test-model",
        )

        new_conv = stores["conv"].get_or_create("new:session")
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize, bootstrap_max_tokens=6000)
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )
        assert result.summaries_imported > 0

    def test_bootstrap_respects_token_budget(self, db, stores):
        """Bootstrap should not exceed the token budget."""
        parent = stores["conv"].get_or_create("parent:session")
        for i in range(100):
            stores["msg"].append(
                parent.id, "user", f"Message {i} " * 20,
                token_count=100,
            )

        new_conv = stores["conv"].get_or_create("new:session")
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize, bootstrap_max_tokens=500)
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )
        assert result.tokens_used <= 500

    def test_bootstrap_updates_timestamp(self, db, stores):
        """Bootstrap should set bootstrapped_at on the new conversation."""
        parent = stores["conv"].get_or_create("parent:session")
        stores["msg"].append(parent.id, "user", "hello", token_count=5)

        new_conv = stores["conv"].get_or_create("new:session")
        assert new_conv.bootstrapped_at is None

        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize)
        asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )

        updated = stores["conv"].get_by_id(new_conv.id)
        assert updated.bootstrapped_at is not None

    def test_bootstrap_tracks_state(self, db, stores):
        """Bootstrap should write to conversation_bootstrap_state table."""
        parent = stores["conv"].get_or_create("parent:session")
        stores["msg"].append(parent.id, "user", "hello", token_count=5)

        new_conv = stores["conv"].get_or_create("new:session")
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize)
        asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )

        row = db.conn.execute(
            "SELECT conversation_id, session_file_path FROM conversation_bootstrap_state WHERE conversation_id = ?",
            (new_conv.id,),
        ).fetchone()
        assert row is not None
        assert row[0] == new_conv.id

    def test_bootstrap_highest_depth_summaries_first(self, db, stores):
        """Bootstrap should pick highest depth summaries first."""
        parent = stores["conv"].get_or_create("parent:session")
        msg_ids = []
        for i in range(4):
            m = stores["msg"].append(parent.id, "user", f"msg {i}", token_count=10)
            msg_ids.append(m.id)

        # Create leaf summaries
        leaf1 = stores["sum"].create_leaf(
            conversation_id=parent.id,
            content="Leaf summary 1",
            token_count=15,
            source_token_count=20,
            message_ids=msg_ids[:2],
            earliest_at="2024-01-01T00:00:00",
            latest_at="2024-01-01T00:01:00",
            model="test",
        )
        leaf2 = stores["sum"].create_leaf(
            conversation_id=parent.id,
            content="Leaf summary 2",
            token_count=15,
            source_token_count=20,
            message_ids=msg_ids[2:],
            earliest_at="2024-01-01T00:02:00",
            latest_at="2024-01-01T00:03:00",
            model="test",
        )

        # Create condensed summary (depth 1)
        stores["sum"].create_condensed(
            conversation_id=parent.id,
            content="High level condensed summary of everything",
            token_count=25,
            child_ids=[leaf1.summary_id, leaf2.summary_id],
            earliest_at="2024-01-01T00:00:00",
            latest_at="2024-01-01T00:03:00",
            model="test",
        )

        new_conv = stores["conv"].get_or_create("new:session")
        # Small budget: should prefer the condensed summary
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize, bootstrap_max_tokens=30)
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )
        assert result.summaries_imported >= 1
        assert result.tokens_used <= 30

    def test_bootstrap_empty_parent(self, db, stores):
        """Bootstrap from a parent with no messages or summaries."""
        stores["conv"].get_or_create("parent:session")
        new_conv = stores["conv"].get_or_create("new:session")
        sb = SessionBootstrap(db=db, summarize_fn=_dummy_summarize)
        result = asyncio.run(
            sb.bootstrap(new_conv.id, "parent:session")
        )
        assert result.summaries_imported == 0
        assert result.messages_imported == 0
        assert result.tokens_used == 0
