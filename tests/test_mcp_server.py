"""Tests for MCP server tools (Feature 9).

The mcp package requires Python 3.10+. These tests mock the mcp module
if it's not available, then test the tool handler logic directly.
"""
from __future__ import annotations

import json
import sys
import types as stdlib_types
from unittest.mock import MagicMock

import pytest

# ── Mock mcp module if not installed ──
_mcp_available = True
try:
    import mcp  # noqa: F401
except ImportError:
    _mcp_available = False
    # Create mock mcp module tree
    mock_mcp = stdlib_types.ModuleType("mcp")
    mock_server = stdlib_types.ModuleType("mcp.server")
    mock_stdio = stdlib_types.ModuleType("mcp.server.stdio")
    mock_types = stdlib_types.ModuleType("mcp.types")

    # Mock types
    class _TextContent:
        def __init__(self, type: str, text: str):
            self.type = type
            self.text = text

    class _Tool:
        def __init__(self, name: str, description: str, inputSchema: dict):
            self.name = name
            self.description = description
            self.inputSchema = inputSchema

    mock_types.TextContent = _TextContent
    mock_types.Tool = _Tool

    class _MockServer:
        def __init__(self, name: str):
            self.name = name
            self._list_tools_fn = None
            self._call_tool_fn = None

        def list_tools(self):
            def decorator(fn):
                self._list_tools_fn = fn
                return fn
            return decorator

        def call_tool(self):
            def decorator(fn):
                self._call_tool_fn = fn
                return fn
            return decorator

    mock_server.Server = _MockServer
    mock_stdio.stdio_server = MagicMock()

    sys.modules["mcp"] = mock_mcp
    sys.modules["mcp.server"] = mock_server
    sys.modules["mcp.server.stdio"] = mock_stdio
    sys.modules["mcp.types"] = mock_types

# Now we can import the mcp_server module
import lossless_agent.mcp_server as mcp_mod  # noqa: E402

from lossless_agent.store.database import Database  # noqa: E402
from lossless_agent.store.conversation_store import ConversationStore  # noqa: E402
from lossless_agent.store.message_store import MessageStore  # noqa: E402
from lossless_agent.store.summary_store import SummaryStore  # noqa: E402


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def seeded(db):
    """Seed data and set the module-level _db."""
    cs = ConversationStore(db)
    ms = MessageStore(db)
    ss = SummaryStore(db)

    conv = cs.get_or_create("sess-mcp", "MCP test chat")

    m1 = ms.append(conv.id, "user", "Tell me about quantum computing basics", token_count=10)
    m2 = ms.append(conv.id, "assistant", "Quantum computing uses qubits for parallel processing", token_count=12)
    m3 = ms.append(conv.id, "user", "What about entanglement?", token_count=8)

    leaf = ss.create_leaf(
        conversation_id=conv.id,
        content="Discussion about quantum computing covering qubits and entanglement",
        token_count=15,
        source_token_count=30,
        message_ids=[m1.id, m2.id, m3.id],
        earliest_at=m1.created_at,
        latest_at=m3.created_at,
        model="gpt-4",
    )

    leaf2 = ss.create_leaf(
        conversation_id=conv.id,
        content="Follow-up on quantum error correction",
        token_count=10,
        source_token_count=20,
        message_ids=[m1.id],
        earliest_at=m1.created_at,
        latest_at=m1.created_at,
        model="gpt-4",
    )

    condensed = ss.create_condensed(
        conversation_id=conv.id,
        content="Comprehensive quantum computing overview",
        token_count=20,
        child_ids=[leaf.summary_id, leaf2.summary_id],
        earliest_at=m1.created_at,
        latest_at=m3.created_at,
        model="gpt-4",
    )

    # Set the module-level _db
    mcp_mod._db = db

    return {
        "db": db, "conv": conv,
        "m1": m1, "m2": m2, "m3": m3,
        "leaf": leaf, "leaf2": leaf2, "condensed": condensed,
    }


class TestListTools:
    @pytest.mark.asyncio
    async def test_lists_five_tools(self, seeded):
        tools = await mcp_mod.list_tools()
        assert len(tools) == 5
        names = {t.name for t in tools}
        assert names == {"lcm_grep", "lcm_describe", "lcm_expand", "lcm_stats", "lcm_expand_query"}


class TestLcmGrepMCP:
    @pytest.mark.asyncio
    async def test_grep_returns_results(self, seeded):
        result = await mcp_mod.call_tool("lcm_grep", {"query": "quantum"})
        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert isinstance(payload, list)
        assert len(payload) >= 1

    @pytest.mark.asyncio
    async def test_grep_with_scope(self, seeded):
        result = await mcp_mod.call_tool("lcm_grep", {"query": "quantum", "scope": "messages"})
        payload = json.loads(result[0].text)
        assert all(r["type"] == "message" for r in payload)

    @pytest.mark.asyncio
    async def test_grep_with_limit(self, seeded):
        result = await mcp_mod.call_tool("lcm_grep", {"query": "quantum", "limit": 1})
        payload = json.loads(result[0].text)
        assert len(payload) <= 1


class TestLcmDescribeMCP:
    @pytest.mark.asyncio
    async def test_describe_returns_summary_info(self, seeded):
        result = await mcp_mod.call_tool("lcm_describe", {"summary_id": seeded["leaf"].summary_id})
        payload = json.loads(result[0].text)
        assert payload["summary_id"] == seeded["leaf"].summary_id
        assert payload["kind"] == "leaf"

    @pytest.mark.asyncio
    async def test_describe_missing_returns_error(self, seeded):
        result = await mcp_mod.call_tool("lcm_describe", {"summary_id": "sum_nonexistent"})
        payload = json.loads(result[0].text)
        assert "error" in payload


class TestLcmExpandMCP:
    @pytest.mark.asyncio
    async def test_expand_leaf_returns_children(self, seeded):
        result = await mcp_mod.call_tool("lcm_expand", {"summary_id": seeded["leaf"].summary_id})
        payload = json.loads(result[0].text)
        assert payload["kind"] == "leaf"
        assert len(payload["children"]) == 3

    @pytest.mark.asyncio
    async def test_expand_condensed_returns_children(self, seeded):
        result = await mcp_mod.call_tool("lcm_expand", {"summary_id": seeded["condensed"].summary_id})
        payload = json.loads(result[0].text)
        assert payload["kind"] == "condensed"
        assert len(payload["children"]) == 2

    @pytest.mark.asyncio
    async def test_expand_missing_returns_error(self, seeded):
        result = await mcp_mod.call_tool("lcm_expand", {"summary_id": "sum_nonexistent"})
        payload = json.loads(result[0].text)
        assert "error" in payload


class TestLcmStatsMCP:
    @pytest.mark.asyncio
    async def test_stats_returns_counts(self, seeded):
        result = await mcp_mod.call_tool("lcm_stats", {})
        payload = json.loads(result[0].text)
        assert "conversations" in payload
        assert payload["conversations"] >= 1
        assert "messages" in payload
        assert payload["messages"]["count"] >= 3
        assert "summaries" in payload
        assert payload["summaries"]["count"] >= 1


class TestLcmExpandQueryMCP:
    @pytest.mark.asyncio
    async def test_expand_query_returns_answer(self, seeded):
        result = await mcp_mod.call_tool("lcm_expand_query", {
            "conversation_id": seeded["conv"].id,
            "query": "quantum",
        })
        payload = json.loads(result[0].text)
        assert "answer" in payload
        assert "cited_summaries" in payload
        assert "steps_taken" in payload
        assert payload["steps_taken"] > 0

    @pytest.mark.asyncio
    async def test_expand_query_no_matches(self, seeded):
        result = await mcp_mod.call_tool("lcm_expand_query", {
            "conversation_id": seeded["conv"].id,
            "query": "xyznonexistent",
        })
        payload = json.loads(result[0].text)
        assert "answer" in payload


class TestUnknownTool:
    @pytest.mark.asyncio
    async def test_unknown_tool_returns_error(self, seeded):
        result = await mcp_mod.call_tool("unknown_tool", {})
        payload = json.loads(result[0].text)
        assert "error" in payload
