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
    async def test_lists_nine_tools(self, seeded):
        tools = await mcp_mod.list_tools()
        assert len(tools) == 9
        names = {t.name for t in tools}
        assert names == {
            "lcm_grep", "lcm_describe", "lcm_expand", "lcm_stats", "lcm_expand_query",
            "lcm_ingest", "lcm_compact", "lcm_get_context", "lcm_session_end",
        }


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


# ──────────────────────────────────────────────────────────────
# Tests for new lifecycle tools
# ──────────────────────────────────────────────────────────────


class TestLcmIngestMCP:
    @pytest.mark.asyncio
    async def test_ingest_creates_messages(self, db):
        mcp_mod._db = db
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-ingest-session",
            "messages": [
                {"role": "user", "content": "Hello, how are you?"},
                {"role": "assistant", "content": "I'm doing great, thanks!"},
            ],
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["messages_ingested"] == 2
        assert payload["conversation_id"] > 0
        assert len(payload["messages"]) == 2
        assert payload["messages"][0]["role"] == "user"
        assert payload["messages"][1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_ingest_with_token_count(self, db):
        mcp_mod._db = db
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-ingest-tokens",
            "messages": [
                {"role": "user", "content": "Hello", "token_count": 5},
            ],
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["messages_ingested"] == 1

        # Verify token count was stored
        ms = MessageStore(db)
        cs = ConversationStore(db)
        conv = cs.get_or_create("test-ingest-tokens")
        msgs = ms.get_messages(conv.id)
        assert msgs[0].token_count == 5

    @pytest.mark.asyncio
    async def test_ingest_estimates_tokens_when_not_provided(self, db):
        mcp_mod._db = db
        content = "This is a test message with some content for token estimation"
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-ingest-estimate",
            "messages": [
                {"role": "user", "content": content},
            ],
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"

        # Verify token count was estimated (~chars/4)
        ms = MessageStore(db)
        cs = ConversationStore(db)
        conv = cs.get_or_create("test-ingest-estimate")
        msgs = ms.get_messages(conv.id)
        assert msgs[0].token_count == max(1, len(content) // 4)

    @pytest.mark.asyncio
    async def test_ingest_same_session_appends(self, db):
        mcp_mod._db = db
        # First ingest
        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-ingest-append",
            "messages": [{"role": "user", "content": "First message"}],
        })
        # Second ingest
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-ingest-append",
            "messages": [{"role": "assistant", "content": "Second message"}],
        })
        payload = json.loads(result[0].text)
        assert payload["messages_ingested"] == 1

        # Verify both messages are in the same conversation
        ms = MessageStore(db)
        cs = ConversationStore(db)
        conv = cs.get_or_create("test-ingest-append")
        msgs = ms.get_messages(conv.id)
        assert len(msgs) == 2


class TestLcmCompactMCP:
    @pytest.mark.asyncio
    async def test_compact_empty_session(self, db):
        mcp_mod._db = db
        result = await mcp_mod.call_tool("lcm_compact", {
            "session_key": "test-compact-empty",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["summaries_created"] == 0

    @pytest.mark.asyncio
    async def test_compact_with_messages(self, db):
        """Compact a session that has enough messages for leaf compaction."""
        mcp_mod._db = db
        # Ingest enough messages to trigger leaf compaction (need > fresh_tail_count + leaf_min_fanout)
        messages = []
        for i in range(20):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message number {i} about various topics " * 10,
                "token_count": 50,
            })

        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-compact-full",
            "messages": messages,
        })

        result = await mcp_mod.call_tool("lcm_compact", {
            "session_key": "test-compact-full",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["summaries_created"] >= 1
        assert len(payload["summary_ids"]) >= 1


class TestLcmGetContextMCP:
    @pytest.mark.asyncio
    async def test_get_context_empty_session(self, db):
        mcp_mod._db = db
        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": "test-context-empty",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["total_tokens"] == 0
        assert payload["context"] == ""

    @pytest.mark.asyncio
    async def test_get_context_with_messages(self, db):
        mcp_mod._db = db
        # Ingest some messages
        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-context-msgs",
            "messages": [
                {"role": "user", "content": "What is Python?", "token_count": 5},
                {"role": "assistant", "content": "Python is a programming language.", "token_count": 8},
            ],
        })

        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": "test-context-msgs",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["message_count"] == 2
        assert "Python" in payload["context"]

    @pytest.mark.asyncio
    async def test_get_context_with_summaries(self, seeded):
        """Get context for session that already has summaries."""
        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": "sess-mcp",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["summary_count"] >= 1
        assert payload["message_count"] >= 1
        assert "quantum" in payload["context"].lower()

    @pytest.mark.asyncio
    async def test_get_context_respects_max_tokens(self, db):
        mcp_mod._db = db
        # Ingest messages
        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-context-budget",
            "messages": [
                {"role": "user", "content": "Hello", "token_count": 5},
            ],
        })

        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": "test-context-budget",
            "max_tokens": 50000,
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"


class TestLcmSessionEndMCP:
    @pytest.mark.asyncio
    async def test_session_end_empty(self, db):
        mcp_mod._db = db
        result = await mcp_mod.call_tool("lcm_session_end", {
            "session_key": "test-end-empty",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["session_closed"] is True

    @pytest.mark.asyncio
    async def test_session_end_marks_inactive(self, db):
        mcp_mod._db = db
        # Create session with messages
        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-end-inactive",
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        })

        # End session
        result = await mcp_mod.call_tool("lcm_session_end", {
            "session_key": "test-end-inactive",
        })
        payload = json.loads(result[0].text)
        assert payload["session_closed"] is True

        # Verify conversation is inactive
        row = db.conn.execute(
            "SELECT active FROM conversations WHERE session_key = ?",
            ("test-end-inactive",),
        ).fetchone()
        assert row[0] == 0

    @pytest.mark.asyncio
    async def test_session_end_compacts(self, db):
        """Session end should run compaction on sessions with enough messages."""
        mcp_mod._db = db
        messages = []
        for i in range(20):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} with enough content for compaction " * 10,
                "token_count": 50,
            })

        await mcp_mod.call_tool("lcm_ingest", {
            "session_key": "test-end-compact",
            "messages": messages,
        })

        result = await mcp_mod.call_tool("lcm_session_end", {
            "session_key": "test-end-compact",
        })
        payload = json.loads(result[0].text)
        assert payload["status"] == "ok"
        assert payload["summaries_created"] >= 1


class TestFullLifecycleMCP:
    """Integration tests that exercise the full ingest → context → compact → end cycle."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, db):
        mcp_mod._db = db
        session_key = "test-full-lifecycle"

        # 1. Get context (empty session)
        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": session_key,
        })
        payload = json.loads(result[0].text)
        assert payload["total_tokens"] == 0

        # 2. Ingest first turn
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": session_key,
            "messages": [
                {"role": "user", "content": "What is machine learning?"},
                {"role": "assistant", "content": "Machine learning is a subset of AI that learns from data."},
            ],
        })
        payload = json.loads(result[0].text)
        assert payload["messages_ingested"] == 2
        conv_id = payload["conversation_id"]

        # 3. Get context (should include messages now)
        result = await mcp_mod.call_tool("lcm_get_context", {
            "session_key": session_key,
        })
        payload = json.loads(result[0].text)
        assert payload["message_count"] == 2
        assert "machine learning" in payload["context"].lower()

        # 4. Ingest more turns
        result = await mcp_mod.call_tool("lcm_ingest", {
            "session_key": session_key,
            "messages": [
                {"role": "user", "content": "What about deep learning?"},
                {"role": "assistant", "content": "Deep learning uses neural networks with many layers."},
            ],
        })
        payload = json.loads(result[0].text)
        assert payload["conversation_id"] == conv_id  # same conversation

        # 5. End session
        result = await mcp_mod.call_tool("lcm_session_end", {
            "session_key": session_key,
        })
        payload = json.loads(result[0].text)
        assert payload["session_closed"] is True


class TestTruncationSummarizer:
    """Test the built-in truncation summarizer."""

    @pytest.mark.asyncio
    async def test_short_text_passes_through(self):
        summarize = mcp_mod._make_truncation_summarizer()
        result = await summarize("Short text")
        assert result == "Short text"

    @pytest.mark.asyncio
    async def test_long_text_truncated(self):
        summarize = mcp_mod._make_truncation_summarizer()
        long_text = "x" * 10000
        result = await summarize(long_text)
        assert len(result) < len(long_text)
        assert result.endswith("[Summary truncated — configure --summarize-command for LLM-quality summaries]")

    @pytest.mark.asyncio
    async def test_truncation_at_char_limit(self):
        summarize = mcp_mod._make_truncation_summarizer()
        # Default target_tokens = 1200, char_limit = 1200 * 4 = 4800
        text = "a" * 4801
        result = await summarize(text)
        assert result.startswith("a" * 4800)

    @pytest.mark.asyncio
    async def test_exact_limit_passes_through(self):
        summarize = mcp_mod._make_truncation_summarizer()
        text = "a" * 4800  # exactly at limit
        result = await summarize(text)
        assert result == text
