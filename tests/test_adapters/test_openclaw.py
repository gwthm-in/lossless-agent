"""Tests for the OpenClawAdapter and adapter factory."""
from __future__ import annotations

import json
import pytest
from unittest.mock import AsyncMock

from lossless_agent.adapters.base import AgentAdapter, LCMConfig
from lossless_agent.adapters.openclaw import OpenClawAdapter
from lossless_agent.adapters.hermes import HermesAdapter
from lossless_agent.adapters.factory import create_adapter


@pytest.fixture
def summarize_fn():
    """Async mock summarizer that echoes a short summary."""
    return AsyncMock(return_value="Summary of the conversation.")


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
    """Create an OpenClawAdapter with test config."""
    a = OpenClawAdapter(config, summarize_fn)
    yield a
    a._db.close()


# ------------------------------------------------------------------
# Inheritance
# ------------------------------------------------------------------

class TestInheritance:
    """OpenClawAdapter inherits from AgentAdapter."""

    def test_is_agent_adapter(self, adapter):
        assert isinstance(adapter, AgentAdapter)

    def test_is_openclaw_adapter(self, adapter):
        assert isinstance(adapter, OpenClawAdapter)


# ------------------------------------------------------------------
# on_turn_start
# ------------------------------------------------------------------

class TestOnTurnStart:
    """on_turn_start returns context string or None."""

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_conversation(self, adapter):
        result = await adapter.on_turn_start("new-session", "hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_returns_context_after_messages_stored(self, adapter):
        await adapter.on_turn_end("s1", [
            {"role": "user", "content": "Tell me about cats", "token_count": 5},
            {"role": "assistant", "content": "Cats are great pets", "token_count": 5},
        ])
        result = await adapter.on_turn_start("s1", "more about cats")
        assert result is not None
        assert isinstance(result, str)
        assert len(result) > 0


# ------------------------------------------------------------------
# on_turn_end
# ------------------------------------------------------------------

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


# ------------------------------------------------------------------
# get_tools
# ------------------------------------------------------------------

class TestGetTools:
    """get_tools returns schemas with openclaw_metadata."""

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

    def test_openclaw_metadata_present(self, adapter):
        tools = adapter.get_tools()
        for tool in tools:
            assert "openclaw_metadata" in tool
            meta = tool["openclaw_metadata"]
            assert meta["plugin_name"] == "lossless-claw"

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

    def test_returns_deep_copy(self, adapter):
        """Mutating the returned list shouldn't affect the adapter."""
        tools1 = adapter.get_tools()
        tools1.pop()
        tools1[0]["openclaw_metadata"]["plugin_name"] = "tampered"
        tools2 = adapter.get_tools()
        assert len(tools2) == 3
        assert tools2[0]["openclaw_metadata"]["plugin_name"] == "lossless-claw"


# ------------------------------------------------------------------
# handle_tool_call
# ------------------------------------------------------------------

class TestHandleToolCall:
    """handle_tool_call dispatches correctly."""

    @pytest.mark.asyncio
    async def test_grep_returns_json(self, adapter):
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


# ------------------------------------------------------------------
# get_system_prompt_block
# ------------------------------------------------------------------

class TestGetSystemPromptBlock:
    """get_system_prompt_block mentions OpenClaw."""

    def test_returns_non_empty_string(self, adapter):
        block = adapter.get_system_prompt_block()
        assert isinstance(block, str)
        assert len(block) > 0

    def test_mentions_openclaw(self, adapter):
        block = adapter.get_system_prompt_block()
        assert "OpenClaw" in block

    def test_mentions_tools(self, adapter):
        block = adapter.get_system_prompt_block()
        assert "lcm_grep" in block
        assert "lcm_describe" in block
        assert "lcm_expand" in block

    def test_mentions_recall_policy(self, adapter):
        block = adapter.get_system_prompt_block()
        assert "Recall Policy" in block


# ------------------------------------------------------------------
# on_session_end
# ------------------------------------------------------------------

class TestOnSessionEnd:
    """on_session_end runs final compaction."""

    @pytest.mark.asyncio
    async def test_runs_without_error_on_empty(self, adapter):
        await adapter.on_session_end("empty-session")


# ------------------------------------------------------------------
# Adapter Factory
# ------------------------------------------------------------------

class TestAdapterFactory:
    """create_adapter produces the correct adapter types."""

    def test_creates_hermes(self, config, summarize_fn):
        adapter = create_adapter("hermes", config, summarize_fn)
        assert isinstance(adapter, HermesAdapter)
        assert isinstance(adapter, AgentAdapter)
        adapter._db.close()

    def test_creates_openclaw(self, config, summarize_fn):
        adapter = create_adapter("openclaw", config, summarize_fn)
        assert isinstance(adapter, OpenClawAdapter)
        assert isinstance(adapter, AgentAdapter)
        adapter._db.close()

    def test_raises_for_unknown(self, config, summarize_fn):
        with pytest.raises(ValueError, match="Unknown agent type"):
            create_adapter("unknown", config, summarize_fn)

    def test_raises_for_empty_string(self, config, summarize_fn):
        with pytest.raises(ValueError):
            create_adapter("", config, summarize_fn)
