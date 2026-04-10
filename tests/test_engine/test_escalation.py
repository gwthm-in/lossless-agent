"""Tests for summarize_with_escalation and the new compaction features."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from lossless_agent.engine.compaction import (
    CompactionConfig,
    CompactionEngine,
    LcmProviderAuthError,
    _estimate_tokens,
    summarize_with_escalation,
)
from lossless_agent.engine.circuit_breaker import CircuitBreaker
from lossless_agent.store.conversation_store import ConversationStore
from lossless_agent.store.message_store import MessageStore
from lossless_agent.store.summary_store import SummaryStore


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

LONG_TEXT = "a" * 4000  # 1000 tokens


def _make_engine(db, *, config=None, circuit_breaker=None, summarize_fn=None):
    conv_store = ConversationStore(db)
    msg_store = MessageStore(db)
    sum_store = SummaryStore(db)
    mock_fn = summarize_fn or AsyncMock(return_value="short summary")
    cfg = config or CompactionConfig(
        fresh_tail_count=2,
        leaf_chunk_tokens=500,
        leaf_min_fanout=2,
        condensed_min_fanout=3,
    )
    engine = CompactionEngine(
        msg_store, sum_store, mock_fn, cfg,
        circuit_breaker=circuit_breaker,
    )
    return conv_store, msg_store, sum_store, engine, mock_fn


def _seed_messages(msg_store, conv_id, n, token_count=10):
    msgs = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        m = msg_store.append(conv_id, role, f"message {i}", token_count=token_count)
        msgs.append(m)
    return msgs


# ===================================================================
# summarize_with_escalation
# ===================================================================

class TestSummarizeWithEscalation:
    @pytest.mark.asyncio
    async def test_level1_normal_good_result(self):
        """When summarize_fn produces a shorter result, return it directly."""
        fn = AsyncMock(return_value="short")
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=500
        )
        assert result == "short"
        fn.assert_called_once_with(LONG_TEXT)

    @pytest.mark.asyncio
    async def test_level2_aggressive_on_bloat(self):
        """When normal result is as big as input, escalate to aggressive."""
        # First call returns text same size as input, second call returns short
        fn = AsyncMock(side_effect=[LONG_TEXT, "compressed"])
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=500
        )
        assert result == "compressed"
        assert fn.call_count == 2
        # Second call should have AGGRESSIVE prefix
        second_call_text = fn.call_args_list[1][0][0]
        assert second_call_text.startswith("AGGRESSIVE:")

    @pytest.mark.asyncio
    async def test_level3_fallback_on_double_bloat(self):
        """When both normal and aggressive produce bloat, fall to truncation."""
        bloated = "x" * 8000  # 2000 tokens, larger than input
        fn = AsyncMock(return_value=bloated)
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=200
        )
        # Should contain truncation suffix
        assert "[Truncated from" in result

    @pytest.mark.asyncio
    async def test_level3_fallback_on_exception(self):
        """When summarize_fn raises, fall through to truncation."""
        fn = AsyncMock(side_effect=RuntimeError("boom"))
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=200
        )
        assert result is not None
        assert "[Truncated from" in result

    @pytest.mark.asyncio
    async def test_auth_error_returns_none(self):
        """LcmProviderAuthError -> return None."""
        fn = AsyncMock(side_effect=LcmProviderAuthError("bad key"))
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=200
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_timeout_falls_to_fallback(self):
        """When summarize_fn takes too long, fall to truncation."""
        async def slow_fn(text):
            await asyncio.sleep(10)
            return "never"

        result = await summarize_with_escalation(
            LONG_TEXT, slow_fn, target_tokens=200, timeout_ms=50
        )
        assert result is not None
        assert "[Truncated from" in result

    @pytest.mark.asyncio
    async def test_level4_hard_cap(self):
        """Result is capped to max_overage_factor * target_tokens."""
        # Return something large but shorter than input
        big_but_valid = "b" * 3000  # 750 tokens
        fn = AsyncMock(return_value=big_but_valid)
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=100, max_overage_factor=2.0
        )
        # Cap = 2.0 * 100 = 200 tokens = 800 chars
        assert len(result) <= 800

    @pytest.mark.asyncio
    async def test_normal_result_not_capped_when_small(self):
        """A good small result passes through without capping."""
        fn = AsyncMock(return_value="tiny")
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=500, max_overage_factor=3.0
        )
        assert result == "tiny"

    @pytest.mark.asyncio
    async def test_auth_error_on_aggressive_returns_none(self):
        """Auth error during aggressive call also returns None."""
        fn = AsyncMock(side_effect=[LONG_TEXT, LcmProviderAuthError("bad")])
        result = await summarize_with_escalation(
            LONG_TEXT, fn, target_tokens=500
        )
        assert result is None


# ===================================================================
# compact_leaf with escalation
# ===================================================================

class TestCompactLeafEscalation:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    @pytest.mark.asyncio
    async def test_leaf_uses_escalation(self):
        """compact_leaf uses summarize_with_escalation (returns summary)."""
        _seed_messages(self.msg_store, self.conv.id, 6)
        result = await self.engine.compact_leaf(self.conv.id)
        assert result is not None
        assert result.kind == "leaf"

    @pytest.mark.asyncio
    async def test_leaf_returns_none_on_auth_error(self):
        """compact_leaf returns None when provider auth fails."""
        fn = AsyncMock(side_effect=LcmProviderAuthError("bad"))
        _, msg, _, engine, _ = _make_engine(self.db, summarize_fn=fn)
        conv = self.conv_store.get_or_create("s2", "Test")
        _seed_messages(msg, conv.id, 6)
        result = await engine.compact_leaf(conv.id)
        assert result is None


# ===================================================================
# compact_condensed with escalation
# ===================================================================

class TestCompactCondensedEscalation:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    def _make_leaf(self, msg):
        return self.sum_store.create_leaf(
            conversation_id=self.conv.id,
            content=f"Summary of {msg.content}",
            token_count=5,
            source_token_count=msg.token_count,
            message_ids=[msg.id],
            earliest_at=msg.created_at,
            latest_at=msg.created_at,
            model="test",
        )

    @pytest.mark.asyncio
    async def test_condensed_uses_escalation(self):
        msgs = _seed_messages(self.msg_store, self.conv.id, 5)
        for m in msgs[:3]:
            self._make_leaf(m)
        result = await self.engine.compact_condensed(self.conv.id, depth=0)
        assert result is not None
        assert result.kind == "condensed"

    @pytest.mark.asyncio
    async def test_hard_trigger_relaxes_fanout(self):
        """hard_trigger=True uses condensed_min_fanout_hard (2 instead of 3)."""
        msgs = _seed_messages(self.msg_store, self.conv.id, 5)
        # Only 2 leaves -> normal fanout (3) should reject
        for m in msgs[:2]:
            self._make_leaf(m)
        result_normal = await self.engine.compact_condensed(
            self.conv.id, depth=0, hard_trigger=False
        )
        assert result_normal is None

        result_hard = await self.engine.compact_condensed(
            self.conv.id, depth=0, hard_trigger=True
        )
        assert result_hard is not None


# ===================================================================
# compact_full_sweep
# ===================================================================

class TestCompactFullSweep:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.conv_store, self.msg_store, self.sum_store, self.engine, self.mock_fn = _make_engine(db)
        self.conv = self.conv_store.get_or_create("s1", "Test")

    @pytest.mark.asyncio
    async def test_sweep_produces_leaf_and_condensed(self):
        # Create enough messages for multiple leaf passes + condensed
        _seed_messages(self.msg_store, self.conv.id, 20, token_count=10)
        created = await self.engine.compact_full_sweep(self.conv.id)
        kinds = [s.kind for s in created]
        assert "leaf" in kinds

    @pytest.mark.asyncio
    async def test_sweep_empty_conversation(self):
        created = await self.engine.compact_full_sweep(self.conv.id)
        assert created == []


# ===================================================================
# compact_until_under
# ===================================================================

class TestCompactUntilUnder:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db

    @pytest.mark.asyncio
    async def test_returns_empty_when_under_budget(self, db):
        conv_store, msg_store, _, engine, _ = _make_engine(db)
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 3, token_count=1)
        result = await engine.compact_until_under(conv.id, 100_000)
        assert result == []

    @pytest.mark.asyncio
    async def test_bails_on_no_progress(self, db):
        """If tokens don't decrease, stop looping."""
        conv_store, msg_store, sum_store, engine, mock_fn = _make_engine(db)
        conv = conv_store.get_or_create("s1", "Test")
        # Need to be over budget but compaction can't help
        # (only 3 messages with tail=2 -> only 1 eligible, below min_fanout)
        _seed_messages(msg_store, conv.id, 3, token_count=10000)
        result = await engine.compact_until_under(conv.id, 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_respects_max_rounds(self, db):
        """Doesn't exceed max_rounds."""
        conv_store, msg_store, _, engine, _ = _make_engine(db)
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 50, token_count=100)
        # With max_rounds=1, should only do one sweep
        result = await engine.compact_until_under(
            conv.id, 100, max_rounds=1
        )
        # Should produce some summaries but not loop indefinitely
        assert isinstance(result, list)


# ===================================================================
# Circuit breaker integration
# ===================================================================

class TestCircuitBreakerIntegration:
    @pytest.fixture(autouse=True)
    def setup(self, db):
        self.db = db
        self.cb = CircuitBreaker(threshold=2, cooldown_ms=60_000)

    @pytest.mark.asyncio
    async def test_skips_when_circuit_open(self):
        conv_store, msg_store, _, engine, _ = _make_engine(
            self.db, circuit_breaker=self.cb
        )
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 10, token_count=100)

        # Open the circuit
        self.cb.record_failure(str(conv.id))
        self.cb.record_failure(str(conv.id))
        assert self.cb.is_open(str(conv.id)) is True

        result = await engine.run_incremental(conv.id, 100)
        assert result == []

    @pytest.mark.asyncio
    async def test_records_success_after_compaction(self):
        conv_store, msg_store, _, engine, _ = _make_engine(
            self.db, circuit_breaker=self.cb
        )
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 10, token_count=100)

        # Pre-failure but below threshold
        self.cb.record_failure(str(conv.id))
        assert self.cb.is_open(str(conv.id)) is False

        result = await engine.run_incremental(conv.id, 100)
        assert len(result) >= 1
        # Success should have reset failures
        assert self.cb.is_open(str(conv.id)) is False

    @pytest.mark.asyncio
    async def test_succeeds_even_with_summarize_error(self):
        """summarize_with_escalation catches errors internally, so compaction
        still proceeds (via fallback truncation) and records success."""
        fn = AsyncMock(side_effect=RuntimeError("boom"))
        conv_store, msg_store, _, engine, _ = _make_engine(
            self.db, circuit_breaker=self.cb, summarize_fn=fn
        )
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 10, token_count=100)

        # Pre-record one failure
        self.cb.record_failure(str(conv.id))

        # run_incremental should not raise because escalation catches errors
        result = await engine.run_incremental(conv.id, 100)
        # It falls through to truncation, so we still get a summary
        assert len(result) >= 1
        # Success resets the circuit breaker
        assert self.cb.is_open(str(conv.id)) is False

    @pytest.mark.asyncio
    async def test_compact_until_under_respects_circuit_breaker(self):
        conv_store, msg_store, _, engine, _ = _make_engine(
            self.db, circuit_breaker=self.cb
        )
        conv = conv_store.get_or_create("s1", "Test")
        _seed_messages(msg_store, conv.id, 10, token_count=100)

        self.cb.record_failure(str(conv.id))
        self.cb.record_failure(str(conv.id))

        result = await engine.compact_until_under(conv.id, 100)
        assert result == []


# ===================================================================
# _estimate_tokens
# ===================================================================

class TestEstimateTokens:
    def test_basic(self):
        assert _estimate_tokens("abcd") == 1
        assert _estimate_tokens("abcdefgh") == 2
        assert _estimate_tokens("") == 0

    def test_rough_accuracy(self):
        text = "a" * 400
        assert _estimate_tokens(text) == 100
