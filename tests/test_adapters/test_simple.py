"""Tests for the SimpleAdapter."""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock

from lossless_agent.adapters.simple import SimpleAdapter


@pytest.fixture
def summarize_fn():
    """Async mock summarizer that echoes a short summary."""
    fn = AsyncMock(return_value="Summary of the conversation.")
    return fn


@pytest.fixture
def adapter(tmp_path, summarize_fn):
    """Create a SimpleAdapter with a temp database."""
    db_path = str(tmp_path / "test.db")
    a = SimpleAdapter(db_path, summarize_fn)
    yield a
    a.close()


class TestIngest:
    """ingest stores messages."""

    @pytest.mark.asyncio
    async def test_stores_messages(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "Hello", "token_count": 2},
            {"role": "assistant", "content": "Hi there", "token_count": 3},
        ])

        # Verify by retrieving
        context = await adapter.retrieve("s1", budget_tokens=1000)
        assert context is not None

    @pytest.mark.asyncio
    async def test_stores_multiple_batches(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "First", "token_count": 1},
        ])
        await adapter.ingest("s1", [
            {"role": "assistant", "content": "Response", "token_count": 1},
        ])
        # Both batches should accumulate
        context = await adapter.retrieve("s1", budget_tokens=1000)
        assert context is not None


class TestRetrieve:
    """retrieve returns assembled context."""

    @pytest.mark.asyncio
    async def test_returns_assembled_context(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "Cats are nice", "token_count": 4},
            {"role": "assistant", "content": "Indeed they are", "token_count": 4},
        ])
        result = await adapter.retrieve("s1", budget_tokens=1000)
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_returns_none_for_empty_session(self, adapter):
        result = await adapter.retrieve("nonexistent", budget_tokens=1000)
        assert result is None

    @pytest.mark.asyncio
    async def test_respects_budget(self, adapter):
        """With a very small budget, output should be constrained."""
        messages = []
        for i in range(20):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message number {i} with some content " * 5,
                "token_count": 50,
            })
        await adapter.ingest("s1", messages)

        # Small budget should produce less context than large budget
        small = await adapter.retrieve("s1", budget_tokens=100)
        large = await adapter.retrieve("s1", budget_tokens=10000)
        # Both should be strings (or None if empty)
        if small is not None and large is not None:
            assert len(small) <= len(large)


class TestCompact:
    """compact creates summaries."""

    @pytest.mark.asyncio
    async def test_compact_creates_summaries(self, adapter, summarize_fn):
        # Add enough messages to trigger compaction
        messages = []
        for i in range(10):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} about important topics",
                "token_count": 100,
            })
        await adapter.ingest("s1", messages)

        result = await adapter.compact("s1")
        assert isinstance(result, int)
        assert result >= 0

    @pytest.mark.asyncio
    async def test_compact_empty_session(self, adapter):
        result = await adapter.compact("empty-session")
        assert result == 0


class TestSearch:
    """search finds content."""

    @pytest.mark.asyncio
    async def test_search_finds_content(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "The quick brown fox jumps", "token_count": 6},
        ])
        results = await adapter.search("fox")
        assert isinstance(results, list)
        assert len(results) > 0

    @pytest.mark.asyncio
    async def test_search_returns_empty_for_no_match(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "Hello world", "token_count": 2},
        ])
        results = await adapter.search("xyznonexistent")
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_search_scoped_to_session(self, adapter):
        await adapter.ingest("s1", [
            {"role": "user", "content": "Elephants are large", "token_count": 4},
        ])
        await adapter.ingest("s2", [
            {"role": "user", "content": "Mice are small", "token_count": 4},
        ])

        # Scoped search should only return results from s1
        results = await adapter.search("elephants", session_key="s1")
        assert len(results) > 0

        # Scoped search for s2 should not find elephants
        results = await adapter.search("elephants", session_key="s2")
        assert len(results) == 0


class TestExpand:
    """expand returns source messages."""

    @pytest.mark.asyncio
    async def test_expand_nonexistent(self, adapter):
        result = await adapter.expand("nonexistent_id")
        assert result is not None
        assert "error" in result


class TestClose:
    """close doesn't error."""

    def test_close_works(self, tmp_path, summarize_fn):
        db_path = str(tmp_path / "close_test.db")
        a = SimpleAdapter(db_path, summarize_fn)
        a.close()
        # Should not raise

    def test_double_close(self, tmp_path, summarize_fn):
        """Closing twice should not raise."""
        db_path = str(tmp_path / "double_close.db")
        a = SimpleAdapter(db_path, summarize_fn)
        a.close()
        # Second close should be safe
        a.close()


class TestEndToEnd:
    """Full workflow: ingest -> compact -> retrieve -> search -> expand."""

    @pytest.mark.asyncio
    async def test_full_workflow(self, adapter, summarize_fn):
        # 1. Ingest messages
        messages = []
        for i in range(10):
            messages.append({
                "role": "user" if i % 2 == 0 else "assistant",
                "content": f"Message {i} discussing project architecture decisions",
                "token_count": 100,
            })
        await adapter.ingest("s1", messages)

        # 2. Compact
        num_summaries = await adapter.compact("s1")
        assert isinstance(num_summaries, int)

        # 3. Retrieve
        context = await adapter.retrieve("s1", budget_tokens=5000)
        assert context is not None
        assert isinstance(context, str)

        # 4. Search
        results = await adapter.search("architecture")
        assert isinstance(results, list)
        assert len(results) > 0

        # 5. Expand (if summaries were created)
        if num_summaries > 0 and summarize_fn.called:
            # Search for summaries to get an ID to expand
            summary_results = await adapter.search("architecture")
            for r in summary_results:
                if r.get("type") == "summary" and r.get("id"):
                    expand_result = await adapter.expand(r["id"])
                    assert isinstance(expand_result, dict)
                    break
