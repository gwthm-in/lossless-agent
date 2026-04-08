"""Tests for expansion_auth + expansion_policy wiring into expand_query (Feature 7)."""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from lossless_agent.engine.expansion_auth import (
    ExpansionAuthManager,
    InvalidGrantError,
    ExhaustedBudgetError,
)
from lossless_agent.engine.expansion_policy import ExpansionPolicy, PolicyAction
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
def seeded(db):
    cs = ConversationStore(db)
    ms = MessageStore(db)
    ss = SummaryStore(db)

    conv = cs.get_or_create("sess-policy", "Policy test")

    m1 = ms.append(conv.id, "user", "Tell me about quantum computing", token_count=10)
    m2 = ms.append(conv.id, "assistant", "Quantum computing uses qubits", token_count=12)

    leaf = ss.create_leaf(
        conversation_id=conv.id,
        content="Discussion about quantum computing covering qubits",
        token_count=15,
        source_token_count=22,
        message_ids=[m1.id, m2.id],
        earliest_at=m1.created_at,
        latest_at=m2.created_at,
        model="gpt-4",
    )

    return {"db": db, "conv": conv, "ms": ms, "ss": ss, "leaf": leaf}


@pytest.fixture
def expand_fn():
    mock = AsyncMock()
    mock.return_value = "Quantum answer"
    return mock


class TestPolicyAnswerDirectly:
    @pytest.mark.asyncio
    async def test_answer_directly_returns_empty_with_reason(self, db, expand_fn):
        """When policy says ANSWER_DIRECTLY, return empty result with reason."""
        cs = ConversationStore(db)
        ms = MessageStore(db)
        ss = SummaryStore(db)
        conv = cs.get_or_create("sess-empty", "Empty conv")

        policy = ExpansionPolicy()
        orch = ExpansionOrchestrator(
            db=db, msg_store=ms, sum_store=ss,
            expand_fn=expand_fn, policy=policy,
        )
        # No summaries -> policy will say ANSWER_DIRECTLY
        result = await orch.expand_query(conv.id, "anything")
        assert result.answer == ""
        assert result.steps_taken == 0
        assert result.reason is not None
        assert "No candidates" in result.reason
        expand_fn.assert_not_called()


class TestPolicyExpandShallow:
    @pytest.mark.asyncio
    async def test_expand_shallow_proceeds(self, seeded, expand_fn):
        """When policy says EXPAND_SHALLOW, the pipeline proceeds normally."""
        policy = ExpansionPolicy()
        # With a large budget, few candidates -> EXPAND_SHALLOW
        config = ExpandQueryConfig(max_tokens=100000)
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, config=config, policy=policy,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert isinstance(result, ExpandQueryResult)
        assert result.steps_taken > 0
        # expand_fn should be called for synthesis
        expand_fn.assert_called_once()


class TestAuthValidation:
    @pytest.mark.asyncio
    async def test_invalid_grant_raises(self, seeded, expand_fn):
        auth = ExpansionAuthManager()
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, auth_manager=auth,
        )
        with pytest.raises(InvalidGrantError):
            await orch.expand_query(seeded["conv"].id, "quantum", grant_id="bad-grant")

    @pytest.mark.asyncio
    async def test_valid_grant_allows_expansion(self, seeded, expand_fn):
        auth = ExpansionAuthManager()
        grant = auth.create_grant(
            issuer_session_id="test",
            allowed_conversation_ids=[str(seeded["conv"].id)],
            token_cap=50000,
        )
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, auth_manager=auth,
        )
        result = await orch.expand_query(
            seeded["conv"].id, "quantum", grant_id=grant.grant_id,
        )
        assert isinstance(result, ExpandQueryResult)
        assert result.steps_taken > 0

    @pytest.mark.asyncio
    async def test_token_budget_consumed(self, seeded, expand_fn):
        auth = ExpansionAuthManager()
        grant = auth.create_grant(
            issuer_session_id="test",
            allowed_conversation_ids=[str(seeded["conv"].id)],
            token_cap=50000,
        )
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, auth_manager=auth,
        )
        initial_budget = auth.get_remaining_budget(grant.grant_id)
        await orch.expand_query(
            seeded["conv"].id, "quantum", grant_id=grant.grant_id,
        )
        remaining = auth.get_remaining_budget(grant.grant_id)
        assert remaining < initial_budget


class TestPolicyWithAuth:
    @pytest.mark.asyncio
    async def test_policy_and_auth_together(self, seeded, expand_fn):
        """Policy and auth can work together."""
        auth = ExpansionAuthManager()
        grant = auth.create_grant(
            issuer_session_id="test",
            allowed_conversation_ids=[str(seeded["conv"].id)],
            token_cap=100000,
        )
        policy = ExpansionPolicy()
        config = ExpandQueryConfig(max_tokens=100000)
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, config=config,
            auth_manager=auth, policy=policy,
        )
        result = await orch.expand_query(
            seeded["conv"].id, "quantum", grant_id=grant.grant_id,
        )
        assert isinstance(result, ExpandQueryResult)


class TestExpandQueryResultReason:
    @pytest.mark.asyncio
    async def test_normal_result_has_no_reason(self, seeded, expand_fn):
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn,
        )
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert result.reason is None


class TestGrantIdParameter:
    @pytest.mark.asyncio
    async def test_grant_id_none_skips_auth(self, seeded, expand_fn):
        """When grant_id is None, auth is skipped even if auth_manager is set."""
        auth = ExpansionAuthManager()
        orch = ExpansionOrchestrator(
            db=seeded["db"], msg_store=seeded["ms"], sum_store=seeded["ss"],
            expand_fn=expand_fn, auth_manager=auth,
        )
        # No grant_id -> should not raise
        result = await orch.expand_query(seeded["conv"].id, "quantum")
        assert isinstance(result, ExpandQueryResult)
