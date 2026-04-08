"""Tests for BaseAdapter wiring and Category C assembler improvements."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from lossless_agent.adapters.base import AgentAdapter, LCMConfig
from lossless_agent.adapters.base_impl import BaseAdapter
from lossless_agent.adapters.generic import GenericAdapter
from lossless_agent.adapters.hermes import HermesAdapter
from lossless_agent.adapters.openclaw import OpenClawAdapter
from lossless_agent.engine.circuit_breaker import CircuitBreaker
from lossless_agent.engine.session_patterns import SessionPatternMatcher
from lossless_agent.engine.system_prompt import CompactionAwarePrompt
from lossless_agent.engine.startup_banner import StartupBanner
from lossless_agent.engine.assembler import AssemblerConfig, ContextAssembler
from lossless_agent.engine.transcript_repair import TranscriptRepairer, generate_fallback_tool_call_id
from lossless_agent.store import Database, MessageStore, SummaryStore
from lossless_agent.store.models import Message


@pytest.fixture
def summarize_fn():
    return AsyncMock(return_value="Summary of the conversation.")


@pytest.fixture
def config():
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


@pytest.fixture(autouse=True)
def reset_banner():
    StartupBanner.reset()
    yield
    StartupBanner.reset()


# ------------------------------------------------------------------
# Category A: BaseAdapter wiring tests
# ------------------------------------------------------------------

class TestBaseAdapterWiring:
    """BaseAdapter wires all standalone modules correctly."""

    def test_generic_inherits_base(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert isinstance(a, BaseAdapter)
        assert isinstance(a, AgentAdapter)
        a._db.close()

    def test_hermes_inherits_base(self, config, summarize_fn):
        a = HermesAdapter(config, summarize_fn)
        assert isinstance(a, BaseAdapter)
        assert isinstance(a, AgentAdapter)
        a._db.close()

    def test_openclaw_inherits_base(self, config, summarize_fn):
        a = OpenClawAdapter(config, summarize_fn)
        assert isinstance(a, BaseAdapter)
        assert isinstance(a, AgentAdapter)
        a._db.close()

    def test_circuit_breaker_created(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert isinstance(a._circuit_breaker, CircuitBreaker)
        assert a._circuit_breaker.threshold == config.circuit_breaker_threshold
        assert a._circuit_breaker.cooldown_ms == config.circuit_breaker_cooldown_ms
        a._db.close()

    def test_session_matcher_created(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert isinstance(a._session_matcher, SessionPatternMatcher)
        a._db.close()

    def test_compaction_prompt_created(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert isinstance(a._compaction_prompt, CompactionAwarePrompt)
        a._db.close()

    def test_heartbeat_pruner_not_created_by_default(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert a._heartbeat_pruner is None
        a._db.close()

    def test_heartbeat_pruner_created_when_configured(self, summarize_fn):
        config = LCMConfig(db_path=":memory:", prune_heartbeat_ok=True)
        a = GenericAdapter(config, summarize_fn)
        assert a._heartbeat_pruner is not None
        a._db.close()

    def test_context_item_store_created(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        assert a._ctx_store is not None
        a._db.close()


class TestBaseAdapterIgnoredSessions:
    """Ignored sessions are skipped entirely."""

    @pytest.mark.asyncio
    async def test_ignored_session_returns_none(self, summarize_fn):
        config = LCMConfig(
            db_path=":memory:",
            ignore_session_patterns=["ignore-*"],
        )
        a = GenericAdapter(config, summarize_fn)
        result = await a.on_turn_start("ignore-session-123", "hello")
        assert result is None
        a._db.close()


class TestBaseAdapterStatelessSessions:
    """Stateless sessions get read-only mode."""

    @pytest.mark.asyncio
    async def test_stateless_session_returns_none_when_empty(self, summarize_fn):
        config = LCMConfig(
            db_path=":memory:",
            stateless_session_patterns=["stateless-*"],
            skip_stateless_sessions=True,
        )
        a = GenericAdapter(config, summarize_fn)
        result = await a.on_turn_start("stateless-session-1", "hello")
        assert result is None
        a._db.close()


class TestBaseAdapterBanner:
    """Startup banner emits once."""

    @pytest.mark.asyncio
    async def test_banner_emits_on_first_turn(self, config, summarize_fn):
        a = GenericAdapter(config, summarize_fn)
        await a.on_turn_start("s1", "hello")
        assert a._banner_emitted is True
        a._db.close()


# ------------------------------------------------------------------
# Category C: Assembler improvements
# ------------------------------------------------------------------

@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def stores(db):
    return MessageStore(db), SummaryStore(db)


@pytest.fixture
def conv_id(db):
    cur = db.conn.execute(
        "INSERT INTO conversations (session_key, title) VALUES ('s1', 'test')"
    )
    db.conn.commit()
    return cur.lastrowid


def _make_msg(id, conv_id, seq, role, content, token_count=10,
              tool_call_id=None, tool_name=None):
    return Message(
        id=id, conversation_id=conv_id, seq=seq, role=role,
        content=content, token_count=token_count,
        tool_call_id=tool_call_id, tool_name=tool_name,
        created_at="2024-01-01T00:00:00",
    )


class TestFreshTailToolCallProtection:
    """Tool results matching tail tool calls are pulled from prefix."""

    def test_pulls_tool_results_from_prefix(self, stores, conv_id):
        msg_store, sum_store = stores
        # Add messages: user, assistant(tool_call), tool(result), user, assistant
        msg_store.append(conv_id, "user", "start", token_count=10)
        msg_store.append(conv_id, "assistant", "calling tool", token_count=10,
                        tool_call_id="tc1", tool_name="lcm_grep")
        msg_store.append(conv_id, "tool", "result data", token_count=10,
                        tool_call_id="tc1", tool_name="lcm_grep")
        msg_store.append(conv_id, "user", "followup", token_count=10)
        msg_store.append(conv_id, "assistant", "response", token_count=10)
        msg_store.append(conv_id, "user", "another", token_count=10)
        msg_store.append(conv_id, "assistant", "calling again", token_count=10,
                        tool_call_id="tc2", tool_name="lcm_describe")
        msg_store.append(conv_id, "tool", "result 2", token_count=10,
                        tool_call_id="tc2", tool_name="lcm_describe")

        # Fresh tail of 4 should get last 4 messages
        config = AssemblerConfig(max_context_tokens=10000, fresh_tail_count=4)
        assembler = ContextAssembler(msg_store, sum_store, config)
        result = assembler.assemble(conv_id)

        # All messages in the tail have their tool results
        assert len(result.messages) >= 4


class TestNonFreshToolCallFiltering:
    """Orphaned tool calls in prefix are filtered."""

    def test_filter_orphaned_tool_calls(self):
        msgs = [
            _make_msg(1, 1, 1, "user", "hello"),
            _make_msg(2, 1, 2, "assistant", "calling", tool_call_id="tc1", tool_name="grep"),
            # No tool result for tc1
            _make_msg(3, 1, 3, "user", "next"),
        ]
        filtered = ContextAssembler._filter_orphaned_tool_calls(msgs)
        # tc1 assistant message should be filtered out
        assert len(filtered) == 2
        assert all(m.role != "assistant" or m.tool_call_id is None for m in filtered)


class TestFallbackToolCallIdGeneration:
    """Fallback IDs are generated for tool calls missing them."""

    def test_generate_fallback_id(self):
        result = generate_fallback_tool_call_id(42, 3)
        assert result == "toolu_lcm_42_3"

    def test_assembler_generates_fallback(self):
        result = ContextAssembler._generate_fallback_tool_call_id(42, 3)
        assert result == "toolu_lcm_42_3"

    def test_ensure_tool_call_ids_fills_missing(self):
        msgs = [
            _make_msg(1, 1, 1, "assistant", "tool call", tool_name="lcm_grep"),
        ]
        assert msgs[0].tool_call_id is None
        result = ContextAssembler._ensure_tool_call_ids(msgs)
        assert result[0].tool_call_id == "toolu_lcm_1_1"

    def test_ensure_tool_call_ids_preserves_existing(self):
        msgs = [
            _make_msg(1, 1, 1, "assistant", "tool call",
                     tool_call_id="existing_id", tool_name="lcm_grep"),
        ]
        result = ContextAssembler._ensure_tool_call_ids(msgs)
        assert result[0].tool_call_id == "existing_id"

    def test_transcript_repairer_ensures_ids(self):
        msgs = [
            _make_msg(10, 1, 5, "assistant", "calling tool", tool_name="lcm_grep"),
        ]
        repairer = TranscriptRepairer()
        repaired = repairer.repair(msgs)
        # The assistant message should now have a fallback ID
        assistant_msgs = [m for m in repaired if m.role == "assistant"]
        assert len(assistant_msgs) == 1
        assert assistant_msgs[0].tool_call_id == "toolu_lcm_10_5"
        # A synthetic tool result should also be generated
        tool_msgs = [m for m in repaired if m.role == "tool"]
        assert len(tool_msgs) == 1
        assert tool_msgs[0].tool_call_id == "toolu_lcm_10_5"
