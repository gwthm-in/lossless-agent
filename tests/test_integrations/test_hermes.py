"""Tests for Hermes Memory Provider integration."""
from __future__ import annotations

import asyncio
import os

import pytest

from lossless_agent.integrations.hermes import LosslessMemoryProvider


@pytest.fixture
def provider(tmp_path):
    """Create a provider with a temp database."""
    db_path = str(tmp_path / "test_hermes.db")

    async def mock_summarize(text: str) -> str:
        return f"Summary of {len(text)} chars"

    p = LosslessMemoryProvider(
        summarize_fn=mock_summarize,
        db_path=db_path,
    )
    yield p
    asyncio.get_event_loop().run_until_complete(p.shutdown())


class TestInitialize:
    def test_creates_database(self, provider, tmp_path):
        asyncio.get_event_loop().run_until_complete(provider.initialize())
        assert provider._initialized is True
        assert os.path.exists(str(tmp_path / "test_hermes.db"))

    def test_idempotent(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())
        loop.run_until_complete(provider.initialize())
        assert provider._initialized is True

    def test_tilde_expansion(self, tmp_path):
        async def mock_summarize(text: str) -> str:
            return "summary"

        # Just verify the constructor expands ~
        p = LosslessMemoryProvider(
            summarize_fn=mock_summarize,
            db_path="~/test.db",
        )
        assert "~" not in p._db_path
        assert p._db_path.startswith("/")


class TestSystemPromptBlock:
    def test_returns_recall_instructions(self, provider):
        block = provider.system_prompt_block()
        assert "lcm_grep" in block
        assert "lcm_describe" in block
        assert "lcm_expand" in block
        assert "lossless" in block.lower()


class TestPrefetch:
    def test_returns_none_for_empty_session(self, provider):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(provider.prefetch("empty-session"))
        assert result is None

    def test_returns_context_after_ingestion(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        # Ingest messages
        loop.run_until_complete(provider.sync_turn("s1", [
            {"role": "user", "content": "How do I deploy?"},
            {"role": "assistant", "content": "Run make deploy in the project root."},
        ]))

        context = loop.run_until_complete(provider.prefetch("s1"))
        assert context is not None
        assert "deploy" in context

    def test_auto_initializes(self, provider):
        """prefetch should auto-initialize if not yet initialized."""
        loop = asyncio.get_event_loop()
        # Don't call initialize() — prefetch should do it
        loop.run_until_complete(provider.prefetch("new-session"))
        assert provider._initialized is True


class TestSyncTurn:
    def test_persists_messages(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        loop.run_until_complete(provider.sync_turn("s1", [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]))

        # Verify messages are in the store
        conv = provider._conv_store.get_or_create("s1")
        msgs = provider._msg_store.get_messages(conv.id)
        assert len(msgs) == 2
        assert msgs[0].role == "user"
        assert msgs[1].role == "assistant"

    def test_handles_tool_messages(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        loop.run_until_complete(provider.sync_turn("s1", [
            {"role": "assistant", "content": "Let me search...",
             "tool_call_id": "call_123", "tool_name": "search"},
            {"role": "tool", "content": '{"results": []}',
             "tool_call_id": "call_123"},
        ]))

        conv = provider._conv_store.get_or_create("s1")
        msgs = provider._msg_store.get_messages(conv.id)
        assert len(msgs) == 2
        assert msgs[0].tool_call_id == "call_123"


class TestOnPreCompress:
    def test_returns_true_to_skip_lossy(self, provider):
        """on_pre_compress should return True (skip Hermes's lossy compression)."""
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        result = loop.run_until_complete(provider.on_pre_compress("s1"))
        assert result is True

    def test_runs_compaction(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        # Add enough messages to trigger compaction
        messages = []
        for i in range(30):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} with some content to make tokens " * 5,
                "token_count": 50,
            })
        loop.run_until_complete(provider.sync_turn("s1", messages))

        # on_pre_compress should run without error
        result = loop.run_until_complete(provider.on_pre_compress("s1"))
        assert result is True


class TestOnSessionEnd:
    def test_runs_final_compaction(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        # Add messages
        loop.run_until_complete(provider.sync_turn("s1", [
            {"role": "user", "content": "test message " * 20, "token_count": 50},
            {"role": "assistant", "content": "response " * 20, "token_count": 50},
        ] * 10))

        # Should not error
        loop.run_until_complete(provider.on_session_end("s1"))


class TestTools:
    def test_get_tools_returns_schemas(self, provider):
        tools = provider.get_tools()
        assert len(tools) == 3
        names = {t["function"]["name"] for t in tools}
        assert names == {"lcm_grep", "lcm_describe", "lcm_expand"}

    def test_handle_tool_call_grep(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())

        # Ingest searchable content
        loop.run_until_complete(provider.sync_turn("s1", [
            {"role": "user", "content": "Deploy to kubernetes production"},
        ]))

        result = loop.run_until_complete(
            provider.handle_tool_call("lcm_grep", {"query": "kubernetes"})
        )
        parsed = __import__("json").loads(result)
        assert isinstance(parsed, list)

    def test_handle_tool_call_unknown(self, provider):
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            provider.handle_tool_call("unknown_tool", {})
        )
        parsed = __import__("json").loads(result)
        assert "error" in parsed


class TestShutdown:
    def test_shutdown_closes_db(self, provider):
        loop = asyncio.get_event_loop()
        loop.run_until_complete(provider.initialize())
        assert provider._initialized is True

        loop.run_until_complete(provider.shutdown())
        assert provider._initialized is False
        assert provider._db is None
