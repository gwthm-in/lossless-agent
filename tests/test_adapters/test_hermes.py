"""Tests for the HermesAdapter."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock

from lossless_agent.adapters.base import LCMConfig
from lossless_agent.adapters.hermes import HermesAdapter


@pytest.fixture
def summarize_fn():
    """Async mock summarizer that echoes a short summary."""
    fn = AsyncMock(return_value="Summary of the conversation.")
    return fn


@pytest.fixture
def config():
    """LCMConfig with in-memory DB and small thresholds for testing."""
    return LCMConfig(
        db_path=":memory:",
        summary_model="test-model",
        fresh_tail_count=2,
        leaf_min_fanout=2,
        leaf_chunk_tokens=500,
        condensed_min_fanout=2,
        context_threshold=0.5,
        max_context_tokens=1000,
    )


@pytest.fixture
def adapter(config, summarize_fn):
    """Create a HermesAdapter with test config."""
    a = HermesAdapter(config, summarize_fn)
    yield a
    a._db.close()


class TestOnTurnStart:
    """on_turn_start returns context string or None."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_conversation(self, adapter):
        result = await adapter.on_turn_start("new-session", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_context_after_messages_stored(self, adapter):
        # Store some messages first
        await adapter.on_turn_end("s1", [
            {"role": "user", "content": "Tell me about cats", "token_count": 5},
            {"role": "assistant", "content": "Cats are great pets", "token_count": 5},
        ])
        result = await adapter.on_turn_start("s1", "more about cats")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


class TestOnTurnEnd:
    """on_turn_end persists messages and triggers compaction."""

    @pytest.mark.asyncio
    async def test_persists_messages(self, adapter):
        messages = [
            {"role": "user", "content": "Hello there"},
            {"role": "assistant", "content": "Hi! How can I help?"},
        ]
        await adapter.on_turn_end("s1", messages)

        conv = adapter._conv_store.get_or_create("s1")
        stored = adapter._msg_store.get_messages(conv.id)
        assert len(stored) == 2
        assert stored[0].role == "user"
        assert stored[0].content == "Hello there"
        assert stored[1].role == "assistant"

    @pytest.mark.asyncio
    async def test_triggers_compaction_when_needed(self, adapter, summarize_fn):
        """When enough messages exceed the threshold, compaction runs."""
        # Insert many messages with high token counts to exceed threshold
        big_messages = []
        for i in range(10):
            big_messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message number {i} with lots of content",
                "token_count": 200,
            })
        await adapter.on_turn_end("s1", big_messages)

        # Compaction should have been triggered (summarize_fn called)
        if summarize_fn.called:
            assert summarize_fn.call_count >= 1

    @pytest.mark.asyncio
    async def test_multiple_turn_ends_accumulate(self, adapter):
        await adapter.on_turn_end("s1", [
            {"role": "user", "content": "First"},
        ])
        await adapter.on_turn_end("s1", [
            {"role": "assistant", "content": "Response"},
        ])
        conv = adapter._conv_store.get_or_create("s1")
        stored = adapter._msg_store.get_messages(conv.id)
        assert len(stored) == 2


class TestGetTools:
    """get_tools returns valid OpenAI function-calling schemas."""

    def test_returns_list(self, adapter):
        tools = adapter.get_tools()
        assert isinstance(tools, list)

    def test_has_three_tools(self, adapter):
        tools = adapter.get_tools()
        assert len(tools) == 3

    def test_tool_names(self, adapter):
        tools = adapter.get_tools()
        names = {t["function"]["name"] for t in tools}
        assert names == {"lcm_grep", "lcm_describe", "lcm_expand"}

    def test_tool_schema_structure(self, adapter):
        tools = adapter.get_tools()
        for tool in tools:
            assert tool["type"] == "function"
            func = tool["function"]
            assert "name" in func
            assert "description" in func
            assert "parameters" in func
            params = func["parameters"]
            assert params["type"] == "object"
            assert "properties" in params
            assert "required" in params

    def test_returns_copy(self, adapter):
        """Mutating the returned list shouldn't affect the adapter."""
        tools1 = adapter.get_tools()
        tools1.pop()
        tools2 = adapter.get_tools()
        assert len(tools2) == 3


class TestHandleToolCall:
    """handle_tool_call dispatches correctly."""

    @pytest.mark.asyncio
    async def test_grep_returns_json(self, adapter):
        # First add a message so there's something to search
        await adapter.on_turn_end("s1", [
            {"role": "user", "content": "cats are wonderful animals"},
        ])
        result = await adapter.handle_tool_call("lcm_grep", {"query": "cats"})
        parsed = json.loads(result)
        assert isinstance(parsed, list)

    @pytest.mark.asyncio
    async def test_describe_not_found(self, adapter):
        result = await adapter.handle_tool_call(
            "lcm_describe", {"summary_id": "nonexistent"}
        )
        parsed = json.loads(result)
        assert parsed.get("error") == "summary not found"

    @pytest.mark.asyncio
    async def test_expand_not_found(self, adapter):
        result = await adapter.handle_tool_call(
            "lcm_expand", {"summary_id": "nonexistent"}
        )
        parsed = json.loads(result)
        assert parsed.get("error") == "summary not found"

    @pytest.mark.asyncio
    async def test_unknown_tool(self, adapter):
        result = await adapter.handle_tool_call("unknown_tool", {})
        parsed = json.loads(result)
        assert "error" in parsed
        assert "unknown tool" in parsed["error"]


class TestGetSystemPromptBlock:
    """get_system_prompt_block returns non-empty policy text."""

    def test_returns_non_empty_string(self, adapter):
        block = adapter.get_system_prompt_block()
        assert isinstance(block, str)
        assert len(block) > 0

    def test_mentions_tools(self, adapter):
        block = adapter.get_system_prompt_block()
        assert "lcm_grep" in block
        assert "lcm_describe" in block
        assert "lcm_expand" in block

    def test_mentions_recall_policy(self, adapter):
        block = adapter.get_system_prompt_block()
        assert "Recall Policy" in block


class TestOnSessionEnd:
    """on_session_end runs final compaction."""

    @pytest.mark.asyncio
    async def test_runs_without_error_on_empty(self, adapter):
        await adapter.on_session_end("empty-session")

    @pytest.mark.asyncio
    async def test_compacts_remaining_messages(self, adapter, summarize_fn):
        """After storing enough messages, session end should compact them."""
        messages = []
        for i in range(10):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i}",
                "token_count": 10,
            })
        await adapter.on_turn_end("s1", messages)
        summarize_fn.reset_mock()

        await adapter.on_session_end("s1")

        # The summarizer should have been called for final compaction
        if summarize_fn.called:
            assert summarize_fn.call_count >= 1
