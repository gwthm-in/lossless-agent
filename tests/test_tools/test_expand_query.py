"""Tests for sub-agent expansion (lcm_expand_query)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lossless_agent.store.database import Database
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore
from lossless_agent.tools.expand_query import (
    ExpandQueryConfig,
    ExpandQueryResult,
    ExpansionOrchestrator,
)


@pytest.fixture
def db():
    database = Database(":memory:")
    yield database
    database.close()


@pytest.fixture
def stores(db):
    cs = ConversationStore(db)
    ms = MessageStore(db)
    ss = SummaryStore(db)
    return db, cs, ms, ss


@pytest.fixture
def seeded(stores):
    """Create a conversation with messages and summaries for testing."""
    db, cs, ms, ss = stores

    conv = cs.get_or_create("sess-expand", "Expand test chat")

    m1 = ms.append(conv.id, "user", "Tell me about quantum computing basics", token_count=10)
    m2 = ms.append(conv.id, "assistant", "Quantum computing uses qubits for parallel processing", token_count=12)
    m3 = ms.append(conv.id, "user", "What about quantum entanglement?", token_count=8)
    m4 = ms.append(conv.id, "assistant", "Entanglement links qubits across distance", token_count=10)

    leaf1 = ss.create_leaf(
        conversation_id=conv.id,
        content="Discussion about quantum computing covering qubits and parallel processing",
        token_count=15,
        source_token_count=22,
        message_ids=[m1.id, m2.id],
        earliest_at=m1.created_at,
        latest_at=m2.created_at,
        model="gpt-4",
    )

    leaf2 = ss.create_leaf(
        conversation_id=conv.id,
        content="Discussion about quantum entanglement linking qubits across distance",
        token_count=14,
        source_token_count=18,
        message_ids=[m3.id, m4.id],
        earliest_at=m3.created_at,
        latest_at=m4.created_at,
        model="gpt-4",
    )

    condensed = ss.create_condensed(
        conversation_id=conv.id,
        content="Comprehensive quantum computing overview including basics and entanglement",
        token_count=20,
        child_ids=[leaf1.summary_id, leaf2.summary_id],
        earliest_at=m1.created_at,
        latest_at=m4.created_at,
        model="gpt-4",
    )

    return {
        "db": db,
        "conv": conv,
        "ms": ms,
        "ss": ss,
        "m1": m1, "m2": m2, "m3": m3, "m4": m4,
        "leaf1": leaf1, "leaf2": leaf2,
        "condensed": condensed,
    }


@pytest.fixture
def expand_fn():
    """Mock expand_fn that simulates LLM synthesis."""
    mock = AsyncMock()
    mock.return_value = "Quantum computing uses qubits for processing. See summary IDs for details."
    return mock


# --- Tests ---


class TestExpandQueryWithMatchingContent:
    @pytest.mark.asyncio
    async def test_returns_answer(self, seeded, expand_fn):
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert isinstance(result, ExpandQueryResult)
        assert len(result.answer) > 0
        assert "quantum" in result.answer.lower() or "Quantum" in result.answer


class TestExpandQueryCitesSummaryIDs:
    @pytest.mark.asyncio
    async def test_cites_relevant_summary_ids(self, seeded, expand_fn):
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert len(result.cited_summaries) > 0
        # All cited IDs should start with sum_
        for sid in result.cited_summaries:
            assert sid.startswith("sum_")


class TestExpandQueryNoMatches:
    @pytest.mark.asyncio
    async def test_no_matches_returns_minimal_answer(self, seeded, expand_fn):
        expand_fn.return_value = "No relevant information found."
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        result = await orch.expand_query(seeded["conv"].id, "xyznonexistent")
        assert isinstance(result, ExpandQueryResult)
        assert result.cited_summaries == []


class TestExpandQueryRespectsMaxSteps:
    @pytest.mark.asyncio
    async def test_respects_max_steps(self, seeded, expand_fn):
        config = ExpandQueryConfig(max_steps=2)
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
            config=config,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        # With max_steps=2: step 1 = grep, step 2 = one describe, then stops
        assert result.steps_taken <= config.max_steps


class TestExpandFnCalledWithContext:
    @pytest.mark.asyncio
    async def test_expand_fn_called_with_compiled_context(self, seeded, expand_fn):
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        await orch.expand_query(seeded["conv"].id, "quantum")
        expand_fn.assert_called_once()
        prompt_arg = expand_fn.call_args[0][0]
        # Prompt should contain the query
        assert "quantum" in prompt_arg.lower()
        # Prompt should contain recall agent strategy
        assert "recall agent" in prompt_arg.lower()
        # Prompt should contain retrieved context
        assert "Retrieved context" in prompt_arg or "context" in prompt_arg.lower()


class TestResultContainsStepsTaken:
    @pytest.mark.asyncio
    async def test_steps_taken_count(self, seeded, expand_fn):
        orch = ExpansionOrchestrator(
            db=seeded["db"],
            msg_store=seeded["ms"],
            sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert result.steps_taken > 0
        assert isinstance(result.steps_taken, int)
        # At minimum: 1 grep + at least 1 describe + synthesis call
        assert result.steps_taken >= 3
