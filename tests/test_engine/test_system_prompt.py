"""Tests for dynamic system prompt injection."""
import pytest

from lossless_agent.engine.system_prompt import CompactionAwarePrompt
from lossless_agent.store.models import Summary


def _make_summary(summary_id="s1", kind="leaf", depth=0, content="test"):
    return Summary(
        summary_id=summary_id,
        conversation_id=1,
        kind=kind,
        depth=depth,
        content=content,
        token_count=100,
        source_token_count=200,
        earliest_at="2025-01-01",
        latest_at="2025-01-02",
        model="test",
        created_at="2025-01-02",
    )


@pytest.fixture
def prompt_gen():
    return CompactionAwarePrompt()


class TestGenerateNone:
    def test_shallow_summaries_return_none(self, prompt_gen):
        summaries = [_make_summary("s1", depth=0), _make_summary("s2", depth=0)]
        result = prompt_gen.generate(summaries)
        assert result is None

    def test_empty_summaries_return_none(self, prompt_gen):
        result = prompt_gen.generate([])
        assert result is None

    def test_single_leaf_returns_none(self, prompt_gen):
        result = prompt_gen.generate([_make_summary()])
        assert result is None


class TestGeneratePrompt:
    def test_high_depth_triggers_prompt(self, prompt_gen):
        summaries = [
            _make_summary("s1", depth=0),
            _make_summary("s2", kind="condensed", depth=2),
        ]
        result = prompt_gen.generate(summaries)
        assert result is not None
        assert "compacted" in result.lower()
        assert "lcm_grep" in result
        assert "lcm_describe" in result
        assert "lcm_expand_query" in result

    def test_many_condensed_triggers_prompt(self, prompt_gen):
        summaries = [
            _make_summary("s1", kind="condensed", depth=1),
            _make_summary("s2", kind="condensed", depth=1),
        ]
        result = prompt_gen.generate(summaries, condensed_threshold=2)
        assert result is not None
        assert "summary IDs" in result.lower() or "summary" in result.lower()

    def test_custom_thresholds(self, prompt_gen):
        summaries = [_make_summary("s1", kind="condensed", depth=1)]
        # With threshold=1, one condensed should trigger
        result = prompt_gen.generate(summaries, depth_threshold=5, condensed_threshold=1)
        assert result is not None

    def test_depth_threshold_boundary(self, prompt_gen):
        # Depth exactly at threshold should trigger
        summaries = [_make_summary("s1", depth=2)]
        result = prompt_gen.generate(summaries, depth_threshold=2)
        assert result is not None

    def test_below_threshold_returns_none(self, prompt_gen):
        summaries = [_make_summary("s1", depth=1)]
        result = prompt_gen.generate(summaries, depth_threshold=2, condensed_threshold=5)
        assert result is None


class TestGetCompactionStats:
    def test_stats_basic(self, prompt_gen):
        summaries = [
            _make_summary("s1", kind="leaf", depth=0),
            _make_summary("s2", kind="condensed", depth=1),
            _make_summary("s3", kind="condensed", depth=2),
            _make_summary("s4", kind="leaf", depth=0),
        ]
        stats = prompt_gen.get_compaction_stats(summaries)
        assert stats["max_depth"] == 2
        assert stats["leaf_count"] == 2
        assert stats["condensed_count"] == 2
        assert stats["total_summaries"] == 4

    def test_stats_empty(self, prompt_gen):
        stats = prompt_gen.get_compaction_stats([])
        assert stats["max_depth"] == 0
        assert stats["leaf_count"] == 0
        assert stats["condensed_count"] == 0
        assert stats["total_summaries"] == 0

    def test_stats_all_leaves(self, prompt_gen):
        summaries = [_make_summary(f"s{i}") for i in range(3)]
        stats = prompt_gen.get_compaction_stats(summaries)
        assert stats["max_depth"] == 0
        assert stats["leaf_count"] == 3
        assert stats["condensed_count"] == 0
