"""Tests for semantic layer integration: _maybe_embed hook and cross_session_context."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from lossless_agent.engine.assembler import AssemblerConfig, ContextAssembler
from lossless_agent.engine.compaction import CompactionEngine, CompactionConfig
from lossless_agent.store.models import Summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_summary(summary_id="sum_abc", conv_id=1, content="some content", token_count=10):
    return Summary(
        summary_id=summary_id,
        conversation_id=conv_id,
        kind="leaf",
        depth=0,
        content=content,
        token_count=token_count,
        source_token_count=40,
        earliest_at="2025-01-01T00:00:00",
        latest_at="2025-01-01T01:00:00",
        model="compaction",
        created_at="2025-01-01T00:00:00",
    )


# ---------------------------------------------------------------------------
# _maybe_embed
# ---------------------------------------------------------------------------

class TestMaybeEmbed:
    def _make_engine(self, embed_fn=None, vector_store=None):
        msg_store = MagicMock()
        sum_store = MagicMock()
        summarize_fn = AsyncMock(return_value="summary text")
        return CompactionEngine(
            msg_store, sum_store, summarize_fn,
            embed_fn=embed_fn, vector_store=vector_store,
        )

    def test_noop_when_embed_fn_none(self):
        engine = self._make_engine(embed_fn=None, vector_store=MagicMock())
        summary = _make_summary()
        asyncio.run(engine._maybe_embed(summary))  # must not raise

    def test_noop_when_vector_store_none(self):
        engine = self._make_engine(embed_fn=AsyncMock(return_value=[0.1]), vector_store=None)
        summary = _make_summary()
        asyncio.run(engine._maybe_embed(summary))  # must not raise

    def test_stores_embedding_when_configured(self):
        fake_embedding = [0.1, 0.2, 0.3]
        embed_fn = AsyncMock(return_value=fake_embedding)
        vector_store = MagicMock()
        engine = self._make_engine(embed_fn=embed_fn, vector_store=vector_store)

        summary = _make_summary("sum_xyz", conv_id=5, content="hello world")
        asyncio.run(engine._maybe_embed(summary))

        embed_fn.assert_awaited_once_with("hello world")
        vector_store.store.assert_called_once_with("sum_xyz", 5, fake_embedding)

    def test_swallows_embed_error(self):
        """Embedding failure must not propagate — compaction always succeeds."""
        embed_fn = AsyncMock(side_effect=RuntimeError("http error"))
        vector_store = MagicMock()
        engine = self._make_engine(embed_fn=embed_fn, vector_store=vector_store)

        summary = _make_summary()
        asyncio.run(engine._maybe_embed(summary))  # must not raise
        vector_store.store.assert_not_called()

    def test_swallows_store_error(self):
        embed_fn = AsyncMock(return_value=[0.1])
        vector_store = MagicMock()
        vector_store.store.side_effect = RuntimeError("pg error")
        engine = self._make_engine(embed_fn=embed_fn, vector_store=vector_store)

        asyncio.run(engine._maybe_embed(_make_summary()))  # must not raise


# ---------------------------------------------------------------------------
# cross_session_context
# ---------------------------------------------------------------------------

class TestCrossSessionContext:
    def _make_assembler(self, summaries_by_id=None):
        msg_store = MagicMock()
        sum_store = MagicMock()

        def get_by_id(sid):
            return (summaries_by_id or {}).get(sid)

        sum_store.get_by_id.side_effect = get_by_id
        config = AssemblerConfig(max_context_tokens=10_000)
        return ContextAssembler(msg_store, sum_store, config)

    def test_returns_empty_when_no_hits(self):
        assembler = self._make_assembler()
        vector_store = MagicMock()
        vector_store.search.return_value = []

        result = asyncio.run(
            assembler.cross_session_context([0.1], 1, vector_store)
        )
        assert result == ""

    def test_formats_cross_session_blocks(self):
        s = _make_summary("sum_abc", conv_id=99, content="I fixed the auth bug", token_count=5)
        assembler = self._make_assembler({"sum_abc": s})
        vector_store = MagicMock()
        vector_store.search.return_value = [("sum_abc", 0.92)]

        result = asyncio.run(
            assembler.cross_session_context([0.1], 1, vector_store, top_k=5)
        )
        assert "<cross_session_memory" in result
        assert "sum_abc" in result
        assert "0.920" in result
        assert "I fixed the auth bug" in result

    def test_respects_token_budget(self):
        s1 = _make_summary("sum_a", conv_id=99, content="aaa", token_count=1000)
        s2 = _make_summary("sum_b", conv_id=99, content="bbb", token_count=1000)
        assembler = self._make_assembler({"sum_a": s1, "sum_b": s2})
        vector_store = MagicMock()
        vector_store.search.return_value = [("sum_a", 0.9), ("sum_b", 0.8)]

        result = asyncio.run(
            assembler.cross_session_context([0.1], 1, vector_store, top_k=2, token_budget=1500)
        )
        # Only sum_a fits within budget of 1500 (sum_b would push to 2000)
        assert "sum_a" in result
        assert "sum_b" not in result

    def test_skips_missing_summaries(self):
        assembler = self._make_assembler(summaries_by_id={})  # no summaries found
        vector_store = MagicMock()
        vector_store.search.return_value = [("sum_ghost", 0.9)]

        result = asyncio.run(
            assembler.cross_session_context([0.1], 1, vector_store)
        )
        assert result == ""

    def test_excludes_current_conversation(self):
        s = _make_summary("sum_abc", conv_id=99)
        assembler = self._make_assembler({"sum_abc": s})
        vector_store = MagicMock()
        vector_store.search.return_value = []

        asyncio.run(
            assembler.cross_session_context([0.1], current_conv_id=7, vector_store=vector_store)
        )
        vector_store.search.assert_called_once()
        _, kwargs = vector_store.search.call_args
        assert kwargs.get("exclude_conversation_id") == 7
